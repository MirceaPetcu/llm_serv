import asyncio
import os
import time

import aioboto3
from pydantic import Field
from rich import print

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.response import LLMResponse
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import (CredentialsException,
                                      InternalConversionException,
                                      ServiceCallException,
                                      ServiceCallThrottlingException)
from llm_serv.structured_response.model import StructuredResponse


class AWSLLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        required_variables = ["AWS_DEFAULT_REGION", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"]
        
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for AWS: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)

        AWSLLMProvider.check_credentials()

        self._context_window = model.max_tokens
        self._model_max_tokens = model.max_output_tokens

        from botocore.config import Config
        self._config = Config(retries={"max_attempts": 5, "mode": "adaptive"})
        
        self._session = aioboto3.Session(region_name=os.getenv("AWS_DEFAULT_REGION"))
        self._client = None
        self._client_cm = None
    
    async def start(self):
        """
        Initialize the boto3 client using the proper async context manager approach.
        """
        if self._client is None:
            self._client_cm = self._session.client(
                service_name="bedrock-runtime",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                config=self._config,
            )
            self._client = await self._client_cm.__aenter__()
            
    async def stop(self):
        if self._client and self._client_cm:
            await self._client_cm.__aexit__(None, None, None)
            self._client = None
            self._client_cm = None

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Ref here: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        returns processed request data as a dictionary

        Example of response:
        response = client.converse(
            modelId='string',
            messages=[
                {
                    'role': 'user'|'assistant',
                    'content': [
                        {
                            'text': 'string',
                            'image': {
                                'format': 'png'|'jpeg'|'gif'|'webp',
                                'source': {
                                    'bytes': b'bytes'
                                }
                            },
                            'document': {
                                'format': 'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
                                'name': 'string',
                                'source': {
                                    'bytes': b'bytes'
                                }
                            },


        You can include up to 20 images. Each image's size, height, and width must be no more than 3.75 MB, 8000 px, and 8000 px, respectively.
        You can include up to five documents. Each document's size must be no more than 4.5 MB.
        If you include a ContentBlock with a document field in the array, you must also include a ContentBlock with a text field.
        You can only include images and documents if the role is user.
        """
        try:
            messages = []
            for message in request.conversation.messages:
                _message = {"role": message.role.value}
                _content = []

                # Only user messages can contain images and documents
                # has_attachments = bool(message.images or message.documents)
                # if has_attachments and message.role != Role.USER:
                #    raise ValueError(f"Images and documents can only be included in user messages, not {message.role}")

                if message.text:
                    _content.append({"text": message.text})

                """if message.images:
                    # Check image count limit
                    if len(message.images) > 20:
                        raise ValueError(f"Maximum of 20 images allowed per message, got {len(message.images)}")
                    
                    for image in message.images:
                        # Check image size limit (3.75 MB = 3,932,160 bytes)
                        image_bytes = image._pil_to_bytes(image.image)
                        if len(image_bytes) > 3_932_160:
                            raise ValueError(f"Image size must be under 3.75 MB, got {len(image_bytes)/1_048_576:.2f} MB")
                        
                        # Check image dimensions
                        if image.width > 8000 or image.height > 8000:
                            raise ValueError(f"Image dimensions must be under 8000x8000 pixels, got {image.width}x{image.height}")
                        
                        _content.append({
                            "image": {
                                "format": image.format or "png",
                                "source": {
                                    "bytes": image_bytes
                                }
                            }
                        })
                
                if message.documents:
                    # Check document count limit
                    if len(message.documents) > 5:
                        raise ValueError(f"Maximum of 5 documents allowed per message, got {len(message.documents)}")
                    
                    # Check if there's a text content when documents are present
                    if not any(c.get("text") for c in _content):
                        raise ValueError("A text field is required when including documents")
                    
                    for document in message.documents:
                        # Check document size limit (4.5 MB = 4,718,592 bytes)
                        if len(document.content) > 4_718_592:
                            raise ValueError(f"Document size must be under 4.5 MB, got {len(document.content)/1_048_576:.2f} MB")
                        
                        _content.append({
                            "document": {
                                "format": document.extension,
                                "name": document.name or "",
                                "source": {
                                    "bytes": document.content
                                }
                            }
                        })
                """

                _message["content"] = _content
                messages.append(_message)

            system = (
                [
                    {
                        "text": request.conversation.system,
                    }
                ]
                if request.conversation.system is not None and len(request.conversation.system) > 0
                else None
            )

            config = {
                "maxTokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,
                "temperature": request.temperature,
                "topP": request.top_p,
            }

            return {
                "messages": messages,
                "system": system,
                "config": config
            }
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request for AWS: {str(e)}") from e

    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, ModelTokens]:
        """
        Make a call to AWS Bedrock with proper error handling.
        Returns a tuple of (output_text, tokens_info)
        """
        # Prepare request data
        try:
            processed = await self._convert(request)
            messages = processed["messages"]
            system = processed["system"]
            config = processed["config"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        try:            
            system = system if system else []
            api_response = await self._client.converse(
                modelId=self.model.internal_model_id,
                messages=messages,
                system=system,
                inferenceConfig=config,
            )

            output = api_response["output"]["message"]["content"][0]["text"]
            tokens = ModelTokens(
                input_tokens=api_response["usage"]["inputTokens"],
                output_tokens=api_response["usage"]["outputTokens"],
                input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
                cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
                reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
            )

            return output, tokens

        except Exception as e:
            if hasattr(e, "response"):
                status_code = e.response["ResponseMetadata"]["HTTPStatusCode"]
                error_msg = str(e)

                if status_code == 400:
                    raise ServiceCallException(f"ValidationException: The input fails to satisfy Bedrock constraints: {error_msg}")
                elif status_code == 403:
                    raise ServiceCallException(f"AccessDeniedException: Insufficient permissions to perform this action: {error_msg}")
                elif status_code == 404:
                    raise ServiceCallException(f"ResourceNotFoundException: The specified model was not found: {error_msg}")
                elif status_code == 408:
                    raise ServiceCallException(f"ModelTimeoutException: The request took too long to process: {error_msg}")
                elif status_code == 424:
                    raise ServiceCallException(f"ModelErrorException: The request failed due to a model processing error: {error_msg}")
                elif status_code == 429:
                    # Let the base class handle the retries for throttling exceptions
                    raise ServiceCallThrottlingException(
                        f"ThrottlingException: Request denied due to exceeding account quotas"
                    )
                elif status_code == 500:
                    raise ServiceCallException(f"InternalServerException: An internal server error occurred: {error_msg}")
                elif status_code == 503:
                    raise ServiceCallException(f"ServiceUnavailableException: The service is currently unavailable: {error_msg}")

            raise ServiceCallException(f"Unexpected AWS service error: {str(e)}")


if __name__ == "__main__":
    import asyncio

    from pydantic import Field

    from llm_serv import LLMService
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_aws():
        model = LLMService.get_model("AWS/claude-3-haiku")
        llm = AWSLLMProvider(model)
        
        class MyClass(StructuredResponse):
            example_string: str = Field(
                default="", description="A string field that should be filled with a random person name in Elven language"
            )
            example_int: int = Field(
                default=0, ge=0, le=10, description="An integer field with a random value, greater than 5."
            )
            example_float: float = Field(
                default=0, ge=0.0, le=10.0, description="A float field with a value exactly half of the integer value"
            )

        conversation = Conversation.from_prompt("Please fill in the following class respecting the following instructions.")
        conversation.add_text_message(role=Role.USER, content=MyClass.to_text())

        request = LLMRequest(conversation=conversation, response_model=MyClass)
            
        response = await llm(request)
        
        print(response)
        assert isinstance(response.output, MyClass)

        await llm.stop()

    # Run the test function with asyncio
    asyncio.run(test_aws())
