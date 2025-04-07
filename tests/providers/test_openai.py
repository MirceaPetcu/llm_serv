import asyncio
import pytest

from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.image import Image
from llm_serv.conversation.message import Message
from llm_serv.conversation.role import Role
from llm_serv.providers.oai import OpenAILLMService, check_credentials
from llm_serv.providers.base import LLMRequest
from llm_serv.registry import REGISTRY


@pytest.fixture
def openai_models():
    """Get all OpenAI models from the registry."""
    return [model for model in REGISTRY.models if model.provider.name.upper() == "OPENAI"]


@pytest.mark.asyncio
async def test_openai_models_api(openai_models):
    """Test that all OpenAI models can be called via the OpenAI API."""
    # Ensure we have OpenAI models to test
    assert len(openai_models) > 0, "No OpenAI models found in the registry"
    
    # Print the models we'll be testing
    model_names = [f"{model.name} (ID: {model.id})" for model in openai_models]
    print(f"\nTesting {len(openai_models)} OpenAI models: {', '.join(model_names)}")
        
    try:
        # Check if OpenAI credentials are available
        check_credentials()
    except Exception as e:
        pytest.skip(f"Skipping OpenAI test due to missing credentials: {str(e)}")
    
    # Create a conversation with a simple prompt
    conversation = Conversation()
    conversation.system = "You are a helpful AI assistant."
    conversation.add_text_message(
        role=Role.USER, 
        content="What is the capital of France? Respond with a single word."
    )
    
    # Track results
    successful_models = []
    failed_models = []
    
    # Test each model
    for model in openai_models:
        print(f"\nTesting model: {model.name} (ID: {model.id})")
        
        # Create service and request
        service = OpenAILLMService(model=model)
        request = LLMRequest(
            conversation=conversation,
            max_completion_tokens=100,
            temperature=0.0,
        )
        
        try:
            # Make API call with timeout of 30 seconds
            task = service(request)
            response = await asyncio.wait_for(task, timeout=30.0)
            
            # Assertions
            assert response is not None, f"Response for model {model.name} should not be None"
            assert response.output is not None, f"Response output for model {model.name} should not be None"
            assert isinstance(response.output, str), f"Response output for model {model.name} should be a string"
            assert "Paris" in response.output, f"Response for model {model.name} should contain 'Paris'"
            assert response.tokens is not None, f"Response for model {model.name} should include token usage"
            assert response.tokens.input_tokens > 0, f"Input tokens for model {model.name} should be greater than 0"
            assert response.tokens.completion_tokens > 0, f"Completion tokens for model {model.name} should be greater than 0"
            
            # Print response details
            print(f"  Response: {response.output}")
            print(f"  Input tokens: {response.tokens.input_tokens}")
            print(f"  Completion tokens: {response.tokens.completion_tokens}")
            print(f"  Total tokens: {response.tokens.total_tokens}")
            print(f"  Total time: {response.total_time:.2f}s")
            print(f"  ✅ Success: Model {model.name} works properly")
            
            # Record success
            successful_models.append(model.name)
            
        except asyncio.TimeoutError:
            error_message = "Request timed out after 30 seconds"
            print(f"  ❌ Timeout with model {model.name}: {error_message}")
            failed_models.append((model.name, error_message))
            continue
            
        except Exception as e:
            error_message = str(e)
            print(f"  ❌ Error with model {model.name}: {error_message}")
            # Record failure
            failed_models.append((model.name, error_message))
            # Continue testing other models even if one fails
            continue
    
    # Print summary
    print("\n=== OpenAI Models Test Summary ===")
    print(f"Total models tested: {len(openai_models)}")
    print(f"Successful models: {len(successful_models)} - {', '.join(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    for name, error in failed_models:
        print(f"  - {name}: {error}")
    
    # Fail the test only if all models failed
    if len(failed_models) == len(openai_models):
        pytest.fail("All OpenAI models failed to respond")

async def test_image_message():
    """ Test that we can send an image message to the OpenAI API."""
    # Load image
    image = Image.load("../resources/test_image.png")    

    # Ensure we have OpenAI models to test
    assert len(openai_models) > 0, "No OpenAI models found in the registry"
    
    # Print the models we'll be testing
    model_names = [f"{model.name} (ID: {model.id})" for model in openai_models]
    print(f"\nTesting {len(openai_models)} OpenAI models: {', '.join(model_names)}")
        
    try:
        # Check if OpenAI credentials are available
        check_credentials()
    except Exception as e:
        pytest.skip(f"Skipping OpenAI test due to missing credentials: {str(e)}")
    
    # Create a conversation with a simple prompt
    conversation = Conversation()
    conversation.system = "You are a helpful AI assistant."
    message = Message(
        role=Role.USER,
        content="Describe the image with one word.",
        images=[image],
    )
    conversation.add(message)
    
    # Track results
    successful_models = []
    failed_models = []
    
    # Test each model
    for model in openai_models:
        print(f"\nTesting model: {model.name} (ID: {model.id})")
        
        # Create service and request
        service = OpenAILLMService(model=model)
        request = LLMRequest(
            conversation=conversation,
            max_completion_tokens=5,
            temperature=0.0,
        )
        
        try:
            # Make API call with timeout of 30 seconds
            task = service(request)
            response = await asyncio.wait_for(task, timeout=30.0)
            
            # Assertions
            assert response is not None, f"Response for model {model.name} should not be None"
            assert response.output is not None, f"Response output for model {model.name} should not be None"
            assert isinstance(response.output, str), f"Response output for model {model.name} should be a string"
            assert response.tokens is not None, f"Response for model {model.name} should include token usage"
            assert response.tokens.input_tokens > 0, f"Input tokens for model {model.name} should be greater than 0"
            assert response.tokens.completion_tokens > 0, f"Completion tokens for model {model.name} should be greater than 0"
            
            # Print response details
            print(f"  Response: {response.output}")
            print(f"  Input tokens: {response.tokens.input_tokens}")
            print(f"  Completion tokens: {response.tokens.completion_tokens}")
            print(f"  Total tokens: {response.tokens.total_tokens}")
            print(f"  Total time: {response.total_time:.2f}s")
            print(f"  ✅ Success: Model {model.name} works properly")
            
            # Record success
            successful_models.append(model.name)
            
        except asyncio.TimeoutError:
            error_message = "Request timed out after 30 seconds"
            print(f"  ❌ Timeout with model {model.name}: {error_message}")
            failed_models.append((model.name, error_message))
            continue
            
        except Exception as e:
            error_message = str(e)
            print(f"  ❌ Error with model {model.name}: {error_message}")
            # Record failure
            failed_models.append((model.name, error_message))
            # Continue testing other models even if one fails
            continue
    
    # Print summary
    print("\n=== OpenAI Models Test Summary ===")
    print(f"Total models tested: {len(openai_models)}")
    print(f"Successful models: {len(successful_models)} - {', '.join(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    for name, error in failed_models:
        print(f"  - {name}: {error}")
    
    # Fail the test only if all models failed
    if len(failed_models) == len(openai_models):
        pytest.fail("All OpenAI models failed to respond")
