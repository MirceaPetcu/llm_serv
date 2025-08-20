"""
TODO: Fix code, https://github.com/openai/openai-python/issues/874
Error codes: https://platform.openai.com/docs/guides/error-codes
"""
import asyncio
import os

from openai import AsyncOpenAI, RateLimitError
from pydantic import Field

from llm_serv.logger import logger
from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import CredentialsException, InternalConversionException, ServiceCallException, ServiceCallThrottlingException
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
        Ref here: https://platform.openai.com/docs/api-reference/responses/create
        https://platform.openai.com/docs/guides/vision#multiple-image-inputs
        returns (input, config)
        """
        
        input_messages = []
        instructions = None
        # Handle system message if present
        if request.conversation.system is not None and len(request.conversation.system) > 0:
            instructions = request.conversation.system

        # Process each message
        for message in request.conversation.messages:
            content = []

            # Add text content if present
            if message.text:
                content.append({"type": "input_text", "text": message.text})

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

            input_messages.append({"role": message.role.value, "content": content})

        """
        TODO: strict format handling
        "response_format": (
            {"type": "json_object"} if request.response_format == LLMResponseFormat.JSON else {"type": "text"}
        ),
        """
        
        config = {
            "max_output_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,  # noqa: E501
            "temperature": request.temperature,
            "top_p": request.top_p
        }

        return {
            "instructions": instructions,
            "input": input_messages,            
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
            input_messages = processed["input"]
            instructions = processed["instructions"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        # Prepare parameters for the API call, excluding top_p initially.
        request_params = {
            "model": self.model.internal_model_id,
            "input": input_messages,
            "max_output_tokens": config["max_output_tokens"],
            "temperature": config["temperature"]                
        }

        if instructions is not None:
            request_params["instructions"] = instructions

        if config["top_p"] is not None:
            request_params["top_p"] = config["top_p"]
        
        # call the LLM provider using responses API, no need to retry, it is handled in the base class                   
        try: 
            response = await self._client.responses.create(**request_params)
        except Exception as e:
            if isinstance(e, RateLimitError):  # package specific exception into our own for base class processing
                raise ServiceCallThrottlingException(f"OpenAI service is throttling requests: {str(e)}") from e
            
            # TODO: handle other error codes properly here
            
            raise ServiceCallException(f"OpenAI service error: {str(e)}") from e

        # check for errors
        if response.error is not None:
            raise ServiceCallException(f"OpenAI service error {response.error.code}: {response.error.message}")

        # check status. Statuses are: completed, failed, in_progress, cancelled, queued, or incomplete.
        if response.status != "completed":
            raise ServiceCallException(f"OpenAI service error, finished with status: {response.status}")

        # get the output
        output = response.output_text            

        # update the tokens
        tokens = ModelTokens(
            input_tokens=response.usage.input_tokens,
            cached_input_tokens=response.usage.input_tokens_details.cached_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_output_tokens=response.usage.output_tokens_details.reasoning_tokens,
            total_tokens=response.usage.total_tokens,
            # Store current price rates for historical accuracy
            input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
            cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
            output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
            reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
        )

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
