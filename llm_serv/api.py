from llm_serv.providers.base import LLMService
from llm_serv.registry import REGISTRY, Model


async def list_providers() -> list[str]:
    """
    List all available providers.
    """
    return [provider.name for provider in REGISTRY.providers]

async def list_models(provider: str | None = None) -> list[Model]:
    """
    List all available models.
    """
    if provider is None:
        return REGISTRY.models  
    else:
        return [model for model in REGISTRY.models if model.provider.name == provider]

async def get_llm_service(model: Model) -> LLMService:
    """
    Factory function to create an LLM service instance based on the provider.

    Args:
        model: Model configuration from the registry

    Returns:
        LLMService: An instance of the appropriate LLM service

    Raises:
        ValueError: If the provider is not supported
    """
    provider_name = model.provider.name.upper()

    match provider_name:
        case "AWS":
            # Check credentials
            from llm_serv.providers.aws import check_credentials

            check_credentials()

            # Create LLM service
            from llm_serv.providers.aws import AWSLLMService

            return AWSLLMService(model)
        case "AZURE":
            # Check credentials
            from llm_serv.providers.azure import check_credentials

            check_credentials()

            # Create LLM service
            from llm_serv.providers.azure import AzureOpenAILLMService

            return AzureOpenAILLMService(model)
        case "OPENAI":
            # Check credentials
            from llm_serv.providers.oai import check_credentials

            check_credentials()

            # Create LLM service
            from llm_serv.providers.oai import OpenAILLMService

            return OpenAILLMService(model)
        case _:
            raise ValueError(f"Unsupported provider: {provider_name}.")
