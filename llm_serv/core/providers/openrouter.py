import os
import asyncio
from openai import AsyncOpenAI, RateLimitError
from pydantic import Field

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import (
    CredentialsException,
    InternalConversionException,
    ServiceCallException,
    ServiceCallThrottlingException,
)


class OpenRouterLLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        """
        Check required OpenRouter environment variables.
        """
        required_variables = ["OPENROUTER_API_KEY"]
        
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for OpenRouter: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)
        
        OpenRouterLLMProvider.check_credentials()
        
        # Initialize OpenAI client with OpenRouter base URL
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Optional site tracking headers
        self._site_url = os.getenv("OPENROUTER_SITE_URL")
        self._site_name = os.getenv("OPENROUTER_SITE_NAME")

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Convert internal LLMRequest to OpenRouter format.
        OpenRouter uses the same format as OpenAI chat completions.
        """
        try:
            messages = []

            # Handle system message if present
            if request.conversation.system is not None and len(request.conversation.system) > 0:
                messages.append(
                    {"role": Role.SYSTEM.value, "content": request.conversation.system}
                )

            # Process each message in the conversation
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

                # For text-only messages, use simple string format
                if len(content) == 1 and content[0]["type"] == "text":
                    messages.append({"role": message.role.value, "content": content[0]["text"]})
                else:
                    messages.append({"role": message.role.value, "content": content})

            # Configuration for the generation
            config = {
                "max_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
            
            # Remove None values from config
            config = {k: v for k, v in config.items() if v is not None}

            return {
                "messages": messages,
                "config": config
            }

        except Exception as e:
            raise InternalConversionException(f"Failed to convert request for OpenRouter: {str(e)}") from e

    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, ModelTokens]:
        """
        Make a call to OpenRouter using the OpenAI SDK.
        Returns a tuple of (output_text, tokens_info)
        """
        # Prepare request
        try:
            processed = await self._convert(request)
            messages = processed["messages"]
            config = processed["config"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        # Call the OpenRouter service
        try:
            # Prepare parameters for the API call
            completion_params = {
                "model": self.model.internal_model_id,
                "messages": messages,
                **config
            }
            
            # Prepare extra headers for OpenRouter
            extra_headers = {}
            if self._site_url:
                extra_headers["HTTP-Referer"] = self._site_url
            if self._site_name:
                extra_headers["X-Title"] = self._site_name
            
            # Make the API call
            if extra_headers:
                api_response = await self._client.chat.completions.create(
                    extra_headers=extra_headers,
                    **completion_params
                )
            else:
                api_response = await self._client.chat.completions.create(**completion_params)
            
            # Extract output text
            if not api_response.choices or not api_response.choices[0].message.content:
                raise ServiceCallException("OpenRouter returned empty response")
            
            output = api_response.choices[0].message.content

            # Extract token usage information
            usage = api_response.usage
            tokens = ModelTokens(
                input_tokens=getattr(usage, 'prompt_tokens', 0),
                output_tokens=getattr(usage, 'completion_tokens', 0),
                total_tokens=getattr(usage, 'total_tokens', 0),
                # Store current price rates for historical accuracy
                input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
                cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
                reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
            )

        except Exception as e:
            # Check for throttling/rate limiting errors
            if isinstance(e, RateLimitError):
                raise ServiceCallThrottlingException(f"OpenRouter service is throttling requests: {str(e)}") from e
            
            error_message = str(e).lower()
            if any(phrase in error_message for phrase in ['rate limit', 'quota', 'throttle', 'too many requests', '429']):
                raise ServiceCallThrottlingException(f"OpenRouter service is throttling requests: {str(e)}") from e
            
            # Check for authentication errors
            if any(phrase in error_message for phrase in ['unauthorized', 'invalid api key', '401', '403']):
                raise ServiceCallException(f"OpenRouter authentication error: {str(e)}") from e

            # General service error
            raise ServiceCallException(f"OpenRouter service error: {str(e)}") from e

        return output, tokens


if __name__ == "__main__":
    from llm_serv import LLMService
    from llm_serv.structured_response.model import StructuredResponse

    async def test_openrouter():
        """Test function for OpenRouterLLMProvider"""
        # Note: Replace with actual OpenRouter model ID
        model = LLMService.get_model("OPENROUTER/deepseek-r1-free")
        llm = OpenRouterLLMProvider(model)

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
    asyncio.run(test_openrouter())
