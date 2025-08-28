"""
Together AI provider implementation.
"""
import asyncio
import os

from pydantic import BaseModel, Field
from together import AsyncTogether
from together.error import RateLimitError

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import CredentialsException, InternalConversionException, ServiceCallException, ServiceCallThrottlingException
from llm_serv.logger import logger
from llm_serv.structured_response.model import StructuredResponse


class TogetherLLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        required_variables = ["TOGETHER_API_KEY"]
            
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for Together AI: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)        
        
        TogetherLLMProvider.check_credentials()

        # Initialize OpenAI client with Together's base URL
        self._client = AsyncTogether(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Convert request to Together AI format using OpenAI-compatible API.
        Together uses the standard OpenAI chat completions format.
        """
        
        messages = []
        
        # Handle system message if present
        if request.conversation.system is not None and len(request.conversation.system) > 0:
            messages.append({"role": "system", "content": request.conversation.system})

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
            "max_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,  # noqa: E501
            "temperature": request.temperature,
            "top_p": request.top_p
        }

        return {
            "messages": messages,            
            "config": config
        }
    
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

        # Prepare parameters for the API call
        request_params = {
            "model": self.model.internal_model_id,
            "messages": messages,
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
        }

        if config["top_p"] is not None:
            request_params["top_p"] = config["top_p"]
        
        # call the LLM provider using chat completions API                 
        try: 
            response = await self._client.chat.completions.create(**request_params)

            # update the tokens
            tokens = ModelTokens(
                input_tokens=response.usage.prompt_tokens,
                cached_input_tokens=0,  # Together doesn't report cached tokens separately
                output_tokens=response.usage.completion_tokens,
                reasoning_output_tokens=0,  # Together doesn't report reasoning tokens separately
                total_tokens=response.usage.total_tokens,
                # Store current price rates for historical accuracy
                input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
                cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
                reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
            )

        except Exception as e:
            if isinstance(e, RateLimitError):  # package specific exception into our own for base class processing
                raise ServiceCallThrottlingException(f"Together service is throttling requests: {str(e)}") from e
            
            # TODO: handle other error codes properly here
            
            raise ServiceCallException(f"Together service error: {str(e)}") from e

        logger.info(f"'{response.model}' response, output:\n{response.choices[0].message.content}")           

        # check that we actually have an output        
        output = str(response.choices[0].message.content).strip()

        if len(output) == 0:
            raise ServiceCallException(f"Together service error, got an empty output! max_tokens={config['max_tokens']}, output_tokens={tokens.output_tokens}, total_tokens={tokens.total_tokens}")  # noqa: E501
        
        return output, tokens


if __name__ == "__main__":
    import asyncio

    from pydantic import Field

    from llm_serv import LLMService
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_together():
        model = LLMService.get_model("TOGETHER/Llama-3.2-3B-Instruct-Turbo")
        llm = TogetherLLMProvider(model)

        class MyClass(BaseModel):
            example_string: str = Field(
                default="", description="A string field that should be filled with a random person name."
            )
            example_int: int = Field(
                default=0, ge=0, le=10, description="An integer field with a random value, greater than 5."
            )
            example_float: float = Field(
                default=0, ge=0.0, le=10.0, description="A random float value."
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
    asyncio.run(test_together())
