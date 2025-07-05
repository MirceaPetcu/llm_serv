"""
    This example demonstrates how to use the LLM Service direct API to interact with models.
    Unlike the client example that requires a running server, this approach allows direct 
    interaction with the LLM providers.

    Key Components:
    - LLMService: Singleton service providing access to models and providers
    - LLMRequest: Request object containing conversation and optional parameters
    - Conversation: Object managing the chat history and messages

    Prerequisites:
    Ensure you have the proper credentials set up for the provider(s) you want to use:

    For AWS:
    - AWS_PROFILE
    - AWS_DEFAULT_REGION 
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY

    For Azure:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPEN_AI_API_VERSION
    - AZURE_OPENAI_DEPLOYMENT_NAME

    Basic Usage:
    1. List available providers and models
    2. Select a model using LLMService.get_model()
    3. Create an LLM provider instance using LLMService.get_provider()
    4. Create a conversation and request
    5. Send the request and get response
"""

import asyncio
import os

from rich import print as rprint

from llm_serv import LLMService
from llm_serv.conversation import Conversation
from llm_serv.core.base import LLMRequest
from llm_serv.core.exceptions import ServiceCallException, TimeoutException


async def main():
    # 1. List available providers
    print("\nAvailable Providers:")
    providers = LLMService.list_providers()
    for provider in providers:
        rprint(f"- {provider}")

    # 2. List available models
    print("\nAvailable Models:")
    models = LLMService.list_models()
    for model in models:
        rprint(model)

    # Check if we have the necessary credentials
    provider = "OPENAI"
    model_name = "gpt-4.1-mini"
    model_id = f"{provider}/{model_name}"
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to run this example with OpenAI models.")
        return

    # 3. Select a model and create service
    try:
        model = LLMService.get_model(model_id)
        print(f"\nSelected model: {model_id}")
        
        # Create LLM provider
        llm_service = LLMService.get_provider(model)
        print("Service created successfully")
        
        # 4. Create conversation and request
        conversation = Conversation.from_prompt("What's 1+1?")
        request = LLMRequest(conversation=conversation)
        print("Request created")
        
        # 5. Get response with a timeout
        print("Sending request to API...")
        try:
            response = await asyncio.wait_for(llm_service(request), timeout=30.0)
            
            print("\nResponse:")
            print(response.output)
            
            print("\nToken Usage:")
            print(f"Input tokens: {response.tokens.input_tokens}")
            print(f"Output tokens: {response.tokens.completion_tokens}")
            print(f"Total tokens: {response.tokens.total_tokens}")
            
        except asyncio.TimeoutError:
            print("Request timed out after 30 seconds")
        except ServiceCallException as e:
            print(f"Service call error: {e}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
