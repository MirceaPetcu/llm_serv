"""
TODO: Fix code, https://github.com/openai/openai-python/issues/874
Error codes: https://platform.openai.com/docs/guides/error-codes
"""
import asyncio
import os

from openai import AsyncOpenAI, RateLimitError
from pydantic import Field

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens, TokenTracker
from llm_serv.core.exceptions import (CredentialsException,
                                      InternalConversionException,
                                      ServiceCallException,
                                      ServiceCallThrottlingException)
from llm_serv.structured_response.model import StructuredResponse




class OpenAILLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        required_variables = ["OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"]
            
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for OpenAI: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)        
        
        OpenAILLMProvider.check_credentials()

        # The OpenAI client is already async-compatible
        self._client = AsyncOpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT")
        )

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Ref here: https://platform.openai.com/docs/api-reference/chat/object
        https://platform.openai.com/docs/guides/vision#multiple-image-inputs
        returns (messages, system, config)

        Example how to send multiple images as urls:
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "What are in these images? Is there any difference between them?",
            },
            {
                "type": "image_url",
                "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
            ],
        }
        ],
        max_tokens=300,
        )

        and example how to send an image as base64:

        import base64
        from openai import OpenAI

        client = OpenAI()

        # Function to encode the image
        def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

        # Path to your image
        image_path = "path_to_your_image.jpg"

        # Getting the base64 string
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What is in this image?",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        )
        """
        messages = []

        # Handle system message if present
        if request.conversation.system is not None and len(request.conversation.system) > 0:
            messages.append(
                {"role": Role.SYSTEM.value, "content": [{"type": "text", "text": request.conversation.system}]}
            )

        # Process each message
        for message in request.conversation.messages:
            content = []

            # Add text content if present
            if message.text:
                content.append({"type": "text", "text": message.text})

            # Add images if present
            for image in message.images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image.format or 'jpeg'};base64,{image.export_as_base64(image.image)}",
                            "detail": "high",
                        },
                    }
                )

            messages.append({"role": message.role.value, "content": content})

        """
        TODO: strict format handling
        "response_format": (
            {"type": "json_object"} if request.response_format == LLMResponseFormat.JSON else {"type": "text"}
        ),
        """
        
        config = {
            "max_completion_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "response_format": ({"type": "text"})
        }

        return {
            "messages": messages,            
            "config": config
        }
    
    # TODO https://platform.openai.com/docs/guides/responses-vs-chat-completions?api-mode=responses switch to responses api
    async def _llm_service_call(
        self,
        request: LLMRequest,
    ) -> tuple[str, ModelTokens]:
        # prepare request
        try:
            processed = await self._convert(request)
            config = processed["config"]
            messages = processed["messages"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        # call the LLM provider, no need to retry, it is handled in the base class
        try:            
            # Prepare parameters for the API call, excluding top_p initially.
            completion_params = {
                "model": self.model.internal_model_id,
                "messages": messages,
                "max_completion_tokens": config["max_completion_tokens"],
                "temperature": config["temperature"],
                "response_format": config["response_format"],
            }
            if config["top_p"] is not None:
                completion_params["top_p"] = config["top_p"]
            
            api_response = await self._client.chat.completions.create(**completion_params)
            
            output = api_response.choices[0].message.content
            tokens = ModelTokens(
                input_tokens=api_response.usage.prompt_tokens,
                #cached_input_tokens=api_response.usage.input_tokens_details.cached_tokens,
                output_tokens=api_response.usage.completion_tokens,
                #reasoning_output_tokens=api_response.usage.output_tokens_details.reasoning_tokens,
                total_tokens=api_response.usage.total_tokens,
                # Store current price rates for historical accuracy
                input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
                cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
                reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
            )

        except Exception as e:
            if isinstance(e, RateLimitError):  # package specific exception into our own for base class processing
                raise ServiceCallThrottlingException(f"OpenAI service is throttling requests: {str(e)}") from e
            
            # TODO: handle other error codes properly here
            
            raise ServiceCallException(f"OpenAI service error: {str(e)}")

        return output, tokens


if __name__ == "__main__":
    import asyncio

    from pydantic import Field

    from llm_serv import LLMService
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_openai():
        model = LLMService.get_model("OPENAI/o4-mini")
        llm = OpenAILLMProvider(model)

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
       
        response_model = StructuredResponse.from_basemodel(MyClass)
        conversation = Conversation.from_prompt("Please fill in the following class respecting the following instructions.")
        conversation.add_text_message(role=Role.USER, content=response_model.to_prompt())

        request = LLMRequest(conversation=conversation, response_model=response_model)

        response = await llm(request)
        
        print(response)
        assert isinstance(response.output, StructuredResponse)
    
        await llm.stop()    

    # Run the test function with asyncio
    asyncio.run(test_openai())
