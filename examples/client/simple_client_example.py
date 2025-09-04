"""
    This example demonstrates how to use the LLM Service API to interact with a model.
    It uses the LLMServiceClient to interact with the API and the LLMRequest to send a request to the API.
    It requires a server instance to be running.

    You can run the backend FastAPI server by running the python command from the root of the repository:

    ```bash
    python -m llm_serv.server
    ``` 

    Ensure you have the proper credential set up, depending on the provider(s) you are using.

    
    Alternatively, you can run the server in a docker container.   
    
    First, build the docker image by running the following command:

    ```bash
    docker build -t llm-service .
    ```

    Then you can run the API by running the following command:

    ```bash
    docker run -d \
        -p 9999:9999 \
        -e AWS_PROFILE=your-aws-profile-name \
        -e AWS_DEFAULT_REGION=your-aws-region-name \
        -e AWS_ACCESS_KEY_ID=your-aws-access-key-id \
        -e AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key \
        llm-service
    ```

    This will start the API and the API will be available at http://localhost:9999.
"""

import asyncio

from rich import print as rprint

from llm_serv import Conversation, LLMRequest, LLMServiceClient


async def main():
    # 1. Initialize the client
    client = LLMServiceClient(host="localhost", port=9999, timeout=10)

    # 2 List available providers
    # Returns a list of provider names like ["AWS", "AZURE", "OPENAI"]
    providers = await client.list_providers()
    print("Available providers:", providers)

    # 3. List available models
    # Returns all models across all providers
    all_models = await client.list_models()
    print("All available models:", all_models)

    # 4. List models for a specific provider
    aws_models = await client.list_models(provider="TOGETHER")
    print("TOGETHER models:", aws_models)

    # 5. Set the model to use
    # Updated to use the new API with model_id in format "provider/name"
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")    
    #client.set_model("TOGETHER/gpt-oss-120b-fp4-medium-reasoning")
    print("Model set to:", client.model_id)

    # 6. Create and send a chat request
    conversation = Conversation.from_prompt("If A implies B, and B implies C, does A imply C?")
    request = LLMRequest(conversation=conversation)
    response = await client.chat(request)

    rprint("Full Response:", response)

    rprint("Output type:", type(response.output))

    rprint("Output:", response.output)

    rprint("Token Usage:", response.tokens)


if __name__ == "__main__":
    asyncio.run(main())
