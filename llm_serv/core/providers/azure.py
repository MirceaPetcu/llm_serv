import os

from openai import AsyncAzureOpenAI
from pydantic import Field

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import CredentialsException, ServiceCallException
from llm_serv.structured_response.model import StructuredResponse

class AzureOpenAILLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        required_variables = ["AZURE_OPENAI_API_KEY", "AZURE_OPEN_AI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]
        
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for Azure: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)     

        AzureOpenAILLMProvider.check_credentials()

        self._client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPEN_AI_API_VERSION"),
            azure_endpoint=f"https://{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}.openai.azure.com",
        )

    def _convert(self, request: LLMRequest) -> dict:
        """
        Ref here: https://platform.openai.com/docs/api-reference/chat/object
        https://platform.openai.com/docs/guides/vision#multiple-image-inputs
        returns processed request data as a dictionary
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

        config = {
            "max_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "response_format": ({"type": "text"})
        }

        return {
            "messages": messages,            
            "config": config
        }

    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, ModelTokens]:
        """
        Make a call to Azure OpenAI with proper error handling.
        Returns a tuple of (output_text, tokens_info)
        """
        try:
            # Convert the request to Azure OpenAI format
            processed = self._convert(request)
            messages = processed["messages"]
            config = processed["config"]

            # Make the API call - Azure's client supports async
            api_response = await self._client.chat.completions.create(
                model=self.model.internal_model_id,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                response_format=config["response_format"],
            )
            
            output = api_response.choices[0].message.content
            tokens = ModelTokens(
                input_tokens=api_response.usage.prompt_tokens,
                completion_tokens=api_response.usage.completion_tokens
            )

            return output, tokens

        except Exception as e:
            if hasattr(e, "status_code"):
                if e.status_code == 400:
                    raise ServiceCallException(f"Bad request: {str(e)}")
                # Other error checks can be added here if needed
            raise ServiceCallException(f"Azure service error: {str(e)}")


if __name__ == "__main__":
    import asyncio

    from pydantic import Field

    from llm_serv import LLMService
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_azure():
        model: Model = LLMService.get_model("AZURE/gpt-4o-mini")
        llm = AzureOpenAILLMProvider(model)

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
    asyncio.run(test_azure())
