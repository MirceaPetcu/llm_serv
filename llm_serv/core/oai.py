"""
TODO: Fix code, https://github.com/openai/openai-python/issues/874
"""
import os
import time
import asyncio

from openai import AsyncOpenAI
from pydantic import Field

from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role
from llm_serv.core.exceptions import CredentialsException, ServiceCallException, ServiceCallThrottlingException
from llm_serv.core.base import (LLMRequest, LLMResponseFormat, LLMService,
                                     LLMTokens)
from llm_serv.api import Model
from llm_serv.structured_response.model import StructuredResponse


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

class OpenAILLMService(LLMService):
    def __init__(self, model: Model):
        super().__init__(model)        
        
        # The OpenAI client is already async-compatible
        self._client = AsyncOpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT")
        )

    async def cleanup(self):
        """Clean up any resources used by the client"""
        # OpenAI client doesn't require explicit cleanup, but we include
        # this method for consistency with other providers
        self._client = None

    def __del__(self):
        """Non-async warning about proper cleanup"""
        if self._client is not None:
            import warnings
            warnings.warn(f"OpenAILLMService instance {id(self)} was not properly cleaned up. "
                         "Call 'await provider.cleanup()' when finished, or use \n'''\nasync with provider:\n\tresponse = await provider(request).\n'''")

    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    def _convert(self, request: LLMRequest) -> tuple[list, dict, dict]:
        """
        Ref here: https://platform.openai.com/docs/api-reference/chat/object
        https://platform.openai.com/docs/guides/vision#multiple-image-inputs
        returns (messages, system, config)

        Example how to send multiple images as urls:
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4o-mini",
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

        config = {
            "max_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "response_format": (
                {"type": "json_object"} if request.response_format == LLMResponseFormat.JSON else {"type": "text"}
            ),
        }

        return messages, None, config

    # Tenacity doesn't work well with async functions, so we implement our own retry
    @staticmethod
    async def async_retry(func, max_attempts=5, initial_backoff=1, backoff_multiplier=2):
        """Retry an async function with exponential backoff."""
        attempt = 0
        last_exception = None
        start_time = time.time()
        
        while attempt < max_attempts:
            try:
                # The function itself should be awaitable, not its result
                return await func()
            except Exception as e:
                attempt += 1
                last_exception = e
                
                # Handle throttling specifically
                if hasattr(e, "status_code") and e.status_code == 429:
                    if attempt >= max_attempts:
                        elapsed = time.time() - start_time
                        raise ServiceCallThrottlingException(
                            f"OpenAI service is throttling requests after {attempt} attempts over {elapsed:.1f} sec."
                        ) from e
                
                # Exponential backoff
                wait_time = min(60, initial_backoff * (backoff_multiplier ** (attempt - 1)))
                await asyncio.sleep(wait_time)
        
        # If we got here, we exhausted our retries
        raise ServiceCallException(f"Service call failed after {max_attempts} attempts: {str(last_exception)}")

    async def _service_call(
        self,
        messages: list[dict],
        system: dict | None,
        config: dict,
    ) -> tuple[str | None, LLMTokens, Exception | None]:
        output = None
        tokens = LLMTokens()
        exception = None

        async def _make_api_call():
            # OpenAI's new client is already async-aware, no need to await
            return await self._client.chat.completions.create(
                model=self.model.id,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                response_format=config["response_format"],
            )
        
        try:
            api_response = await self.async_retry(_make_api_call)
            
            output = api_response.choices[0].message.content
            tokens = LLMTokens(
                input_tokens=api_response.usage.prompt_tokens,
                completion_tokens=api_response.usage.completion_tokens,
                total_tokens=api_response.usage.total_tokens,
            )

        except Exception as e:
            if hasattr(e, "status_code"):
                if e.status_code == 400:
                    raise ServiceCallException(f"Bad request: {str(e)}")
                # Other error codes...
            
            raise ServiceCallException(f"OpenAI service error: {str(e)}")

        return output, tokens, exception


if __name__ == "__main__":
    import asyncio    
    from llm_serv.api import LLMService
    from llm_serv.api import REGISTRY
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse
    from pydantic import Field

    async def test_openai():
        model:Model = LLMService.get_model("OPENAI/gpt-4o-mini")        
        llm: OpenAILLMService = LLMService.get_provider(model)

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

        my_class = MyClass()

        conversation = Conversation.from_prompt("Please fill in the following class respecting the following instructions.")
        conversation.add_text_message(role=Role.USER, content=MyClass.to_text())

        request = LLMRequest(conversation=conversation, response_class=MyClass, response_format=LLMResponseFormat.XML)

        # Use the provider as an async context manager
        async with llm:
            try:
                response = await llm(request)
                print(response)
                assert isinstance(response.output, MyClass)
            except Exception as e:
                print(f"Error during test: {e}")

    # Run the test function with asyncio
    asyncio.run(test_openai())
