import os
import time
import random

from pydantic import Field

from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import LLMTokens
from llm_serv.core.exceptions import CredentialsException, InternalConversionException, ServiceCallException, ServiceCallThrottlingException

from llm_serv.api import Model
from llm_serv.structured_response.model import StructuredResponse

def check_credentials() -> None:
    # this provider does not require credentials
    pass

class MockLLMProvider(LLMProvider):
    def __init__(self, model: Model):
        super().__init__(model)

    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, LLMTokens]:
        print(f"Inside _llm_service_call for task: {request.conversation.messages[-1].text}.")
        random_number = random.randint(5, 10)
        await asyncio.sleep(random_number)

        message = request.conversation.messages[-1].text + f" (message took {random_number} seconds to generate)."
        print(f"Finished _llm_service_call for task: {message}.")
        return message, LLMTokens()


if __name__ == "__main__":
    import asyncio
    from llm_serv.api import ModelProvider

    async def test_mock():
        model = Model(
            provider=ModelProvider(
                name="mock",
                config={}
            ),
            name = "mock",
            id = "mock",
            max_tokens = 10,
            max_output_tokens = 10            
        )

        provider = MockLLMProvider(model)

        tasks = []
        for i in range(100):            
            print(f"Starting task {i}.")
            tasks.append(
                provider(LLMRequest(conversation=Conversation.from_prompt(f"Message {i}")))
            )
            

        await asyncio.gather(*tasks)          

    asyncio.run(test_mock())