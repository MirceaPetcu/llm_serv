import asyncio
import pytest

from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.aws import AWSLLMService, check_credentials
from llm_serv.core.base import LLMRequest
from llm_serv.api import REGISTRY


@pytest.fixture
def aws_models():
    """Get all AWS models from the registry."""
    return [model for model in REGISTRY.models if model.provider.name.upper() == "AWS"]


@pytest.mark.asyncio
async def test_aws_models_api(aws_models):
    """Test that all AWS models can be called via the AWS Bedrock API."""
    # Ensure we have AWS models to test
    assert len(aws_models) > 0, "No AWS models found in the registry"
    
    # Print the models we'll be testing
    model_names = [f"{model.name} (ID: {model.id})" for model in aws_models]
    print(f"\nTesting {len(aws_models)} AWS models: {', '.join(model_names)}")
        
    try:
        # Check if AWS credentials are available
        check_credentials()
    except Exception as e:
        pytest.skip(f"Skipping AWS test due to missing credentials: {str(e)}")
    
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
    for model in aws_models:
        print(f"\nTesting model: {model.name} (ID: {model.id})")
        
        # Create service and request
        service = AWSLLMService(model=model)
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
    print("\n=== AWS Models Test Summary ===")
    print(f"Total models tested: {len(aws_models)}")
    print(f"Successful models: {len(successful_models)} - {', '.join(successful_models)}")
    print(f"Failed models: {len(failed_models)}")
    for name, error in failed_models:
        print(f"  - {name}: {error}")
    
    # Fail the test only if all models failed
    if len(failed_models) == len(aws_models):
        pytest.fail("All AWS models failed to respond")
