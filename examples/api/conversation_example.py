"""
    This example demonstrates how to use the LLM Service direct API to interact with models.
    Unlike the client example that requires a running server, this approach allows direct 
    interaction with the LLM providers.

"""

import asyncio

from llm_serv import LLMService
from llm_serv.conversation import Conversation, Role
from llm_serv.core.base import LLMRequest


async def main():
    # Select a model and create service
    model = LLMService.get_model("OPENAI/gpt-4.1-mini")
    llm_service = LLMService.get_provider(model)

    # Create conversation and request
    conversation = Conversation(system="Let's play a game. I say a number, then you add 1 to it. Respond only with the number.")
    conversation.add_text_message(role=Role.USER, content="I start, 3.")

    # Run request and get a response
    response = await llm_service(LLMRequest(conversation=conversation))

    # Add the response to the conversation
    conversation.add_text_message(role=Role.ASSISTANT, content=response.output)

    # New user message
    conversation.add_text_message(role=Role.USER, content="8")

    # Run request and get a response
    response = await llm_service(LLMRequest(conversation=conversation))

    # Add the new response to the conversation
    conversation.add_text_message(role=Role.ASSISTANT, content=response.output)

    response.rprint()


if __name__ == "__main__":
    asyncio.run(main())

