import os
import time
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from llm_serv import __version__
from llm_serv.api import LLMService
from llm_serv.core.exceptions import (
    InternalConversionException,
    ServiceCallException,
    ServiceCallThrottlingException,
    StructuredResponseException,
    CredentialsException,
)
from llm_serv.core.base import LLMProvider, LLMRequest, LLMResponse
from llm_serv.api import Model, ModelProvider
from llm_serv.logger import logger
from llm_serv.metrics.log_manager import LogManager
from llm_serv.metrics.metrics import ModelMetrics


class GetStatsRequest(BaseModel):
    """Request model for getting model statistics."""
    model_key: str = Field(..., description="Model key in format 'provider/model'")
    start_time: float | None = Field(None, description="Start time filter (unix timestamp)")
    end_time: float | None = Field(None, description="End time filter (unix timestamp)")
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=1000)


class ModelMetricsResponse(BaseModel):
    """Pydantic model for ModelMetrics API response."""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    call_start_time: float = 0.0
    call_end_time: float = 0.0
    call_duration: float = 0.0
    tokens_per_second: float = 0.0
    status_code: int | None = None
    error_message: str = ""
    internal_retries: int = 0

    @classmethod
    def from_model_metrics(cls, metrics: ModelMetrics) -> "ModelMetricsResponse":
        """Convert ModelMetrics msgspec.Struct to Pydantic model."""
        return cls(
            input_tokens=metrics.input_tokens,
            cached_input_tokens=metrics.cached_input_tokens,
            output_tokens=metrics.output_tokens,
            reasoning_output_tokens=metrics.reasoning_output_tokens,
            total_tokens=metrics.total_tokens,
            call_start_time=metrics.call_start_time,
            call_end_time=metrics.call_end_time,
            call_duration=metrics.call_duration,
            tokens_per_second=metrics.tokens_per_second,
            status_code=metrics.status_code,
            error_message=metrics.error_message,
            internal_retries=metrics.internal_retries
        )


class GetStatsResponse(BaseModel):
    """Response model for model statistics."""
    model_key: str
    stats: dict
    logs: list[ModelMetricsResponse]
    total_returned: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown events."""
    # Startup
    await app.state.log_manager.initialize()
    logger.info("LogManager initialized")
    yield
    # Shutdown - persist logs before exit
    logger.info("Server shutting down, persisting logs...")
    await app.state.log_manager.shutdown()
    logger.info("LogManager shutdown completed")


def create_app() -> FastAPI:    
    # Initialize the FastAPI app with lifespan events
    app = FastAPI(
        title="LLMService", 
        version=__version__, 
        docs_url="/docs", 
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Set up the LLM Providers
    try:
        app.state.providers = {}
        app.state.log_manager = LogManager()
        
        models:list[Model] = LLMService.list_models()
        for model in models:
            if model.provider.name not in app.state.providers:
                app.state.providers[model.provider.name] = {}
            assert model.name not in app.state.providers[model.provider.name], (
                f"Model {model.name} already exists in provider {model.provider.name}!"
            )
            try:
                app.state.providers[model.provider.name][model.name] = LLMService.get_provider(model)                
            except CredentialsException as e:
                logger.error(f"Failed to set up LLM Provider for {model.provider.name}/{model.name}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Failed to set up LLM Providers: {str(e)}")
        raise    

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
        return JSONResponse(status_code=500, content={"detail": "Internal server error: " + str(exc)})

    return app

app = create_app()


async def _collect_metrics(log_manager: LogManager, model_key: str, response: LLMResponse, status_code: int):
    """Fire-and-forget metrics collection for successful responses."""
    try:
        # Calculate tokens per second
        tokens_per_second = 0.0
        if response.total_duration and response.total_duration > 0 and response.tokens.total_tokens > 0:
            tokens_per_second = response.tokens.total_tokens / response.total_duration
        
        metrics = ModelMetrics(
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.completion_tokens,
            total_tokens=response.tokens.total_tokens,
            call_start_time=response.start_time or 0.0,
            call_end_time=response.end_time or 0.0,
            call_duration=response.total_duration or 0.0,
            tokens_per_second=tokens_per_second,
            status_code=status_code,
            error_message="",
            internal_retries=0  # TODO: Extract from response when available
        )
        
        await log_manager.add_log(model_key, metrics)
    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)


async def _collect_error_metrics(log_manager: LogManager, model_key: str, status_code: int, error_message: str):
    """Fire-and-forget metrics collection for error responses."""
    try:
        metrics = ModelMetrics(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            call_start_time=time.time(),
            call_end_time=time.time(),
            call_duration=0.0,
            tokens_per_second=0.0,
            status_code=status_code,
            error_message=error_message,
            internal_retries=0
        )
        
        await log_manager.add_log(model_key, metrics)
    except Exception as e:
        logger.error(f"Error collecting error metrics: {str(e)}", exc_info=True)


@app.post("/list_models")
async def list_models(provider: str | None = None) -> list[Model]:
    try:
        logger.info("Listing models...")
        models: list[Model] = LLMService.list_models(provider)
        logger.info(f"Found {len(models)} models")
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"error": "registry_error", "message": f"Failed to retrieve model list: {str(e)}"}
        ) from e


@app.get("/model_info")
async def model_info(model_id: str) -> Model:
    try:
        logger.info(f"Getting model info for {model_id}...")
        model: Model = LLMService.get_model(model_id)
        return model
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)  
        raise HTTPException(
            status_code=500, detail={"error": "registry_error", "message": f"Failed to retrieve model info: {str(e)}"}
        ) from e

@app.get("/list_providers")
async def list_providers() -> list[ModelProvider]:
    try:
        logger.info("Listing providers...")
        providers:list[ModelProvider] = LLMService.list_providers()
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
        logger.info(f"Request to {model_provider}/{model_name}: {request.model_dump(exclude={'conversation'})}")

        # First of all, check if the model and providers are available
        try:
            assert model_provider in app.state.providers, f"Provider {model_provider} not found"
            assert model_name in app.state.providers[model_provider], (
                f"Model {model_name} not found in provider {model_provider}"
            )
        except ValueError as e:
            logger.error(f"Model not found: {model_provider}/{model_name}")
            raise HTTPException(
                status_code=404,
                detail={"error": "model_not_found", "message": f"Model {model_provider}/{model_name} not found"},
            ) from e

        # Increment chat request counters
        app.state.chat_request_count += 1

        # Update model-specific usage counter with detailed metrics
        model_key = f"{model_provider}/{model_name}"
        app.state.model_usage[model_key] = app.state.model_usage.get(model_key, 0) + 1

        # Get the LLM service for this model
        llm_service:LLMProvider = app.state.providers[model_provider][model_name]

        try:
            # This is async now, so await it
            response: LLMResponse = await llm_service(request=request)            
           
            logger.info(f"Response: {response.model_dump(exclude={'request': {'conversation'}})}")
            
            # Fire-and-forget metrics collection
            asyncio.create_task(
                _collect_metrics(app.state.log_manager, model_key, response, 200)
            )
            
            return response

        except InternalConversionException as e:
            logger.warning(f"Internal conversion exception: {str(e)}")
            asyncio.create_task(
                _collect_error_metrics(app.state.log_manager, model_key, 400, str(e))
            )
            raise HTTPException(
                status_code=400,
                detail={"error": "internal_conversion_exception", "message": str(e)},
            ) from e
        except ServiceCallThrottlingException as e:
            logger.warning(f"Service call throttling exception: {str(e)}")
            asyncio.create_task(
                _collect_error_metrics(app.state.log_manager, model_key, 429, str(e))
            )
            raise HTTPException(
                status_code=429,
                detail={"error": "service_throttling_exception", "message": str(e)},
            ) from e
        except StructuredResponseException as e:
            logger.warning(f"Structured response exception: {str(e)}")
            asyncio.create_task(
                _collect_error_metrics(app.state.log_manager, model_key, 422, str(e))
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "structured_response_exception",
                    "message": str(e),
                    "xml": e.xml,
                    "return_class": str(e.return_class) if e.return_class else None,
                },
            ) from e
        except ServiceCallException as e:
            logger.warning(f"Service call exception: {str(e)}")
            asyncio.create_task(
                _collect_error_metrics(app.state.log_manager, model_key, 502, str(e))
            )
            raise HTTPException(status_code=502, detail={"error": "service_call_exception", "message": str(e)}) from e
        except Exception as e:            
            logger.error(f"LLM service error: {str(e)}", exc_info=True)
            asyncio.create_task(
                _collect_error_metrics(app.state.log_manager, model_key, 500, str(e))
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "llm_service_exception", "message": f"Error processing chat request: {str(e)}"},
            ) from e

    except HTTPException:
        # Re-raise HTTPExceptions that were already properly handled by inner exception blocks
        raise
    except Exception as e:
        logger.error(f"Unexpected exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "llm_service_exception", "message": f"Unexpected error: {str(e)}"},
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


@app.post("/get_stats", response_model=GetStatsResponse)
async def get_stats(request: GetStatsRequest) -> GetStatsResponse:
    """Get statistics and logs for a specific model."""
    try:
        stats, logs = await app.state.log_manager.get_logs(
            request.model_key, 
            request.start_time, 
            request.end_time, 
            request.limit
        )
        
        # Convert ModelMetrics to ModelMetricsResponse for proper JSON serialization
        response_logs = [ModelMetricsResponse.from_model_metrics(log) for log in logs]
        
        return GetStatsResponse(
            model_key=request.model_key,
            stats=stats,
            logs=response_logs,
            total_returned=len(logs)
        )
    except Exception as e:
        logger.error(f"Error retrieving stats for {request.model_key}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "stats_retrieval_error", "message": f"Failed to retrieve stats: {str(e)}"}
        ) from e


def main():
    try:
        port = int(os.getenv("API_PORT", "9999"))        
        logger.info(f"Starting server version '{__version__}' on port '{port}'")
        
        # Pass the app instance directly instead of a string reference
        uvicorn.run(
            app,  # Pass the app instance directly
            host="0.0.0.0", 
            port=port,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            loop="auto",
            log_config=None
        )
    except ValueError as e:
        logger.error(f"Invalid port configuration: {str(e)}", exc_info=True)
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
