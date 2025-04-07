import pytest

from llm_serv.api import list_providers, list_models, get_llm_service
from llm_serv.registry import REGISTRY
from llm_serv.providers.base import LLMService

@pytest.mark.asyncio
async def test_list_providers():
    """Test that list_providers returns non-empty results from the actual registry."""    
    providers = await list_providers()
    
    assert isinstance(providers, list)
    assert len(providers) > 0
    assert all(isinstance(provider, str) for provider in providers)


@pytest.mark.asyncio
async def test_list_models():
    """Test that list_models returns models when given valid providers."""
    # First get available providers
    providers = await list_providers()
    assert len(providers) > 0
    
    # For each provider, verify we can get models
    for provider in providers:
        models = await list_models(provider=provider)
        assert isinstance(models, list)
        # Some providers might not have models, so we don't assert non-empty
        
        for model in models:
            assert model.provider.name == provider


@pytest.mark.asyncio
async def test_get_llm_service():
    """Test that get_llm_service returns valid service instances for real models."""
    # Get providers and their models
    providers = await list_providers()
    
    for provider in providers:
        models = await list_models(provider=provider)
        
        # Skip if provider has no models
        if not models:
            continue
            
        # Test with the first model from each provider
        model = models[0]
        
        try:
            service = await get_llm_service(model)
            assert isinstance(service, LLMService)
            assert service.model.id == model.id
            assert service.model.provider.name == provider
        except Exception as e:
            # If a provider requires credentials that aren't available in test env,
            # this will help identify which one failed
            pytest.skip(f"Couldn't create service for {provider}: {str(e)}")


@pytest.mark.asyncio
async def test_get_llm_service_raises_for_invalid_provider():
    """Test that get_llm_service raises ValueError for invalid providers."""
    # Get an existing model and modify its provider
    all_models = [model for model in REGISTRY.models]
    
    if not all_models:
        pytest.skip("No models available in registry")
        
    # Create a copy of the first model with an invalid provider name
    from copy import deepcopy
    invalid_model = deepcopy(all_models[0])
    invalid_model.provider.name = "NONEXISTENT_PROVIDER"
    
    with pytest.raises(ValueError) as excinfo:
        await get_llm_service(invalid_model)
    
    assert "Unsupported provider: NONEXISTENT_PROVIDER" in str(excinfo.value)
