from llm_serv.client import LLMServiceClient
from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.request import LLMRequest
import asyncio

async def main():
    # Initialize the client
    client = LLMServiceClient(host="localhost", port=9999)

    # List available providers and models
    providers = await client.list_providers()
    all_models = await client.list_models()

    # Set the model to use
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")

    # Create and send a chat request
    conversation = Conversation.from_prompt("What's 1+1/21 + 3**2?")

    request = LLMRequest(conversation=conversation)

    response = await client.chat(request)
    
    print("Response:", response.output)

if __name__ == "__main__":
    asyncio.run(main()) 