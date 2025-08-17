import asyncio

from rich import print as rprint

from llm_serv import Conversation, LLMRequest, LLMServiceClient
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role


async def main():
    # 1. Initialize the client
    client = LLMServiceClient(host="localhost", port=9999, timeout=15)

    # 2. Set the model to use    
    #client.set_model("OPENROUTER/llama-4-maverick-free")    
    client.set_model("OPENAI/gpt-5-mini")    
    print("Model set to:", client.model_id)

    # let's load an image
    image_url = "https://www.gstatic.com/webp/gallery/1.jpg"
    image = Image.from_url(image_url)

    # 3. Create and send a chat request
    message = Message(role=Role.USER, text="What is this image about?", images=[image])
    conversation = Conversation()
    conversation.add(message)

    request = LLMRequest(conversation=conversation)
    response = await client.chat(request)

    rprint("Full Response:", response)
    rprint("Token Usage:", response.tokens)


if __name__ == "__main__":
    asyncio.run(main())
