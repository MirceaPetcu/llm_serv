import logging
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

from llm_serv.api import get_llm_service
from llm_serv.exceptions import (
    InternalConversionException,
    ServiceCallException,
    ServiceCallThrottlingException,
    StructuredResponseException,
    CredentialsException,
)
from llm_serv.providers.base import LLMRequest, LLMResponse
from llm_serv.registry import REGISTRY, Model

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        # Initialize the registry first
        logger.info("Initializing LLM Registry...")
        _ = REGISTRY.models
        logger.info(f"Registry initialized with {len(REGISTRY.models)} models")
    except Exception as e:
        logger.error(f"Failed to initialize registry: {str(e)}")
        raise

    app = FastAPI(title="LLMService", version="1.0", docs_url="/docs", redoc_url="/redoc")

    # Store startup time and initialize metrics
    app.state.start_time = time.time()
    app.state.chat_request_count = 0
    app.state.model_usage = {}  # tracks detailed usage per model
    app.state.total_tokens = {"input": 0, "completion": 0, "total": 0}

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add compression middleware - compress responses > 1KB
    app.add_middleware(GZipMiddleware, minimum_size=1000)  # 1KB

    # Add error handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()


@app.get("/list_models")
async def list_models() -> list[Model]:
    try:
        logger.info("Listing models...")
        models = REGISTRY.models
        logger.info(f"Found {len(models)} models")
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"error": "registry_error", "message": f"Failed to retrieve model list: {str(e)}"}
        ) from e


@app.get("/list_providers")
async def list_providers() -> list[str]:
    try:
        logger.info("Listing providers...")
        providers = list({model.provider.name for model in REGISTRY.models})
        logger.info(f"Found {len(providers)} providers: {providers}")
        return providers
    except Exception as e:
        logger.error(f"Failed to list providers: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "registry_error", "message": f"Failed to retrieve provider list: {str(e)}"},
        ) from e


@app.post("/chat/{model_provider}/{model_name}")
async def chat(model_provider: str, model_name: str, request: LLMRequest) -> LLMResponse:
    try:
        logger.warning(f"Chatting with model {model_provider}/{model_name}")
        logger.info(f"Request: {request}")

        # First of all, check if the model is available
        try:
            model = REGISTRY.get_model(provider=model_provider, name=model_name)
        except ValueError as e:
            logger.error(f"Model not found: {model_provider}/{model_name}")
            raise HTTPException(
                status_code=404,
                detail={"error": "model_not_found", "message": f"Model {model_provider}/{model_name} not found"},
            ) from e

        # Increment chat request counters
        app.state.chat_request_count += 1

        # Update model-specific usage counter with detailed metrics
        model_key = f"{model_provider}.{model_name}"
        app.state.model_usage[model_key] = app.state.model_usage.get(model_key, 0) + 1

        # Get the LLM service for this model
        llm_service = await get_llm_service(model)

        try:
            # This is async now, so await it
            response = await llm_service(request)
            
            logger.info(f"Response: {response}")
            return response

        except InternalConversionException as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "internal_conversion_error", "message": str(e)},
            ) from e
        except ServiceCallThrottlingException as e:
            raise HTTPException(
                status_code=429,
                detail={"error": "service_throttling", "message": str(e)},
            ) from e
        except StructuredResponseException as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "structured_response_error",
                    "message": str(e),
                    "xml": e.xml,
                    "return_class": str(e.return_class) if e.return_class else None,
                },
            ) from e
        except ServiceCallException as e:
            raise HTTPException(status_code=502, detail={"error": "service_call_error", "message": str(e)}) from e
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error": "internal_server_error", "message": f"Error processing chat request: {str(e)}"},
            ) from e

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_server_error", "message": f"Unexpected error: {str(e)}"},
        ) from e


@app.get("/health")
async def health_check(request: Request):
    try:
        uptime_seconds = time.time() - request.app.state.start_time

        # Calculate days, hours, minutes, seconds
        days, remainder = divmod(int(uptime_seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format uptime string
        uptime_parts = []
        if days > 0:
            uptime_parts.append(f"{days}d")
        if hours > 0:
            uptime_parts.append(f"{hours}h")
        if minutes > 0:
            uptime_parts.append(f"{minutes}m")
        uptime_parts.append(f"{seconds}s")

        health_data = {
            "status": "healthy",
            "uptime": " ".join(uptime_parts),
            "chat_requests": request.app.state.chat_request_count,
            "model_usage": request.app.state.model_usage,
            "tokens": request.app.state.total_tokens,
        }
        logger.debug(f"Health check response: {health_data}")
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service health check failed: {str(e)}") from e


@app.get("/check_credentials/{model_provider}/{model_name}")
async def check_credentials(model_provider: str, model_name: str):
    """
    Checks if credentials are properly set for the specified provider and model.
    
    Args:
        model_provider: Provider name (e.g., "AWS", "AZURE")
        model_name: Model name (e.g., "claude-3-haiku", "gpt-4")
        
    Returns:
        JSON response indicating the status of credentials
        
    Raises:
        HTTPException: When credentials are not properly set or model is not found
    """
    try:
        logger.info(f"Checking credentials for {model_provider}/{model_name}")
        
        # First check if the model is available
        try:
            model = REGISTRY.get_model(provider=model_provider, name=model_name)
        except ValueError as e:
            logger.error(f"Model not found: {model_provider}/{model_name}")
            raise HTTPException(
                status_code=404,
                detail={"error": "model_not_found", "message": f"Model {model_provider}/{model_name} not found"},
            ) from e
            
        # Check credentials based on provider
        provider_name = model.provider.name.upper()
        
        try:
            match provider_name:
                case "AWS":
                    from llm_serv.providers.aws import check_credentials
                    check_credentials()
                case "AZURE":
                    from llm_serv.providers.azure import check_credentials
                    check_credentials()
                case "OPENAI":
                    from llm_serv.providers.oai import check_credentials
                    check_credentials()
                case _:
                    logger.warning(f"No credential check implemented for provider: {provider_name}")
                    return {"status": "unknown", "message": f"No credential check implemented for provider: {provider_name}"}
                    
            return {"status": "success", "message": f"Credentials for {model_provider}/{model_name} are properly set"}
            
        except CredentialsException as e:
            logger.error(f"Credentials not set for {model_provider}/{model_name}: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail={"error": "credentials_not_set", "message": str(e)},
            ) from e
            
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.error(f"Unexpected error checking credentials: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error": "internal_server_error", "message": f"Unexpected error checking credentials: {str(e)}"},
            ) from e
        raise e


def main():
    try:
        port = int(os.getenv("API_PORT", "9999"))
        workers = int(os.getenv("API_WORKERS", "10"))
        logger.info(f"Starting server on port {port} with {workers} workers")
        uvicorn.run(
            "llm_serv.server:app",  # Import string instead of app object
            host="0.0.0.0", 
            port=port, 
            log_level="info",
            workers=workers,
            loop="auto"
        )
    except ValueError as e:
        logger.error(f"Invalid port configuration: {str(e)}", exc_info=True)
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
