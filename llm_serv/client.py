import logging
from urllib.parse import quote

import httpx

from llm_serv.core.base import LLMRequest, LLMResponse
from llm_serv.core.exceptions import (CredentialsException,
                                      InternalConversionException,
                                      ModelNotFoundException,
                                      ServiceCallException,
                                      ServiceCallThrottlingException,
                                      StructuredResponseException,
                                      TimeoutException)
from llm_serv.api import LLMService, Model, ModelProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMServiceClient:
    def __init__(self, host: str, port: int, model_id: str | None = None, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = self._validate_timeout(timeout)
        self.model_id: str | None = None
        self.model_provider: str | None = None
        self.model_name: str | None = None
        self._client: httpx.AsyncClient | None = None # Initialize client as None initially
        self.logger = logger # Use the module-level logger or create a specific instance logger

        self.llm_service = LLMService()

        if model_id:
            self._set_model_id(model_id)
                
    async def __aenter__(self):
        await self._ensure_client_initialized()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def _ensure_client_initialized(self):
        """Initializes the httpx client if it hasn't been already."""
        if self._client is None:
            self.logger.info("Initializing httpx.AsyncClient")
            self._client = httpx.AsyncClient(
                base_url=self.base_url, # Set base_url here
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                headers={
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json"
                },
                event_hooks={'request': [self._log_request], 'response': [self._log_response]} # Optional: Add logging hooks
            )

    # Optional: Logging hooks for requests and responses
    async def _log_request(self, request: httpx.Request):
        self.logger.debug(f"Request: {request.method} {request.url} - Headers: {request.headers}")

    async def _log_response(self, response: httpx.Response):
        await response.aread() # Ensure response body is available for logging if needed
        self.logger.debug(f"Response: {response.status_code} - {response.url} - Body: {response.text[:100]}...") # Log truncated body

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            self.logger.info("Closing httpx.AsyncClient")
            await self._client.aclose()
            self._client = None

    def _validate_timeout(self, timeout: float) -> httpx.Timeout:
        """
        Ensures timeout is positive, defaulting to 5.0s if not.
        
        Args:
            timeout: The timeout value in seconds to validate
            
        Returns:
            httpx.Timeout: The validated timeout object
        """        
        effective_timeout = max(5.0, timeout)
        if effective_timeout != timeout:
            self.logger.warning(f"Provided timeout {timeout}s is too low, using minimum {effective_timeout}s.")
        return httpx.Timeout(effective_timeout)
    
    def _set_model_id(self, model_id: str):
        """Sets the model ID and derives provider and name."""
        if "/" not in model_id:
            raise ValueError("Invalid model ID format. Must be 'provider/name'.")            
        self.model_id = model_id
        self.model_provider, self.model_name = model_id.split("/", 1)
        self.logger.info(f"Client model set to: {self.model_id}")

    async def list_models(self, provider: str | None = None) -> list[str]:
        """
        Lists all available models from the server, as model_id strings.
        Model_ids are in the format 'provider/name'.
        
        Args:
            provider: Optional provider name to filter models
        
        Returns:
            list[str]: List of models as model_id strings.
            Example:
            [
                "AZURE_OPENAI/gpt-5-mini",
                "OPENAI/gpt-5-mini",
                "AWS/claude-3-haiku",
            ]

        Raises:
            ServiceCallException: When there is an error retrieving the model list
        """
        await self._ensure_client_initialized()
        try:
            # Use relative URL now that base_url is set on the client
            response = await self._client.post("/list_models", json={"provider": provider}) 
                
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                self.logger.error(f"Failed to list models (status {response.status_code}): {error_msg}")
                raise ServiceCallException(f"Failed to list models: {error_msg}")
                
            models_data = response.json()
            models = [Model(**model) for model in models_data]
            model_ids = [model.id for model in models]
            self.logger.info(f"Successfully listed {len(model_ids)} models.")
            return sorted(model_ids)
        
        except httpx.RequestError as e:
            self.logger.error(f"Failed to connect to server for list_models: {str(e)}", exc_info=True)
            raise ServiceCallException(f"Failed to connect to server: {str(e)}") from e

    async def get_model_info(self, model_id: str) -> Model:
        """
        Gets information about a specific model.
        """
        await self._ensure_client_initialized()
        try:    
            # URL encode the model_id to handle special characters like / and \
            encoded_model_id = quote(model_id, safe='')
            response = await self._client.get(f"/model_info/{encoded_model_id}")
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                self.logger.error(f"Failed to get model info (status {response.status_code}): {error_msg}")
                raise ServiceCallException(f"Failed to get model info: {error_msg}")
            return Model.model_validate(response.json())
        except httpx.RequestError as e:
            self.logger.error(f"Failed to connect to server for get_model_info: {str(e)}", exc_info=True)
            raise ServiceCallException(f"Failed to connect to server: {str(e)}") from e

    async def list_providers(self) -> list[str]:
        """
        Lists all available providers from the server.

        Returns:
            list[str]: List of provider names

        Raises:
            ServiceCallException: When there is an error retrieving the provider list
        """
        await self._ensure_client_initialized()
        try:
            # Use relative URL
            response = await self._client.get("/list_providers")
                
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", str(error_data))
                self.logger.error(f"Failed to list providers (status {response.status_code}): {error_msg}")
                raise ServiceCallException(f"Failed to list providers: {error_msg}")
                
            providers_data = response.json()
            providers = [ModelProvider(**provider) for provider in providers_data]
            provider_names = [provider.name for provider in providers]
            self.logger.info(f"Successfully listed {len(provider_names)} providers.")
            return sorted(provider_names)
        except httpx.RequestError as e:
            self.logger.error(f"Failed to connect to server for list_providers: {str(e)}", exc_info=True)
            raise ServiceCallException(f"Failed to connect to server: {str(e)}") from e
    
    def set_model(self, model_id: str):
        """
        Sets the model to use for subsequent chat requests.
        """
        self._set_model_id(model_id) # This now updates model_id, provider, and name

    def has_fixed_temperature(self) -> bool:
        """
        Checks if the model has a fixed temperature.
        """
        try:
            return self.llm_service.get_model(self.model_id).fixed_temperature
        except Exception as e:
            self.logger.warning(f"Failed to get model {self.model_id}: {str(e)}, could I have stale info?", exc_info=True)
            return False        

    async def chat(self, request: LLMRequest, timeout: float | None = None) -> LLMResponse:
        """
        Sends a chat request to the server using the currently set model.

        Args:
            request: LLMRequest object containing the conversation and parameters
            timeout: Optional timeout override for this specific request (in seconds)

        Returns:
            LLMResponse: Server response containing the model output

        Raises:            
            ValueError: When model is not set before calling chat
            ModelNotFoundException: When the model is not found on the backend
            CredentialsException: When credentials are not set
            InternalConversionException: When the internal conversion to the particular provider fails
            ServiceCallException: When the service call fails for other reasons
            ServiceCallThrottlingException: When the service call is throttled and retries are exhausted
            StructuredResponseException: When the structured response parsing fails
            TimeoutException: When the request times out
        """
        await self._ensure_client_initialized()
        if not self.model_id:
            # Check model_id directly
            raise ValueError("Model ID is not set. Please set it using client.set_model('provider/name') before calling chat.")
        
        # Handle request-specific timeout if provided
        request_timeout = self._validate_timeout(timeout) if timeout is not None else self.timeout
    
        # Construct URL using the set provider and name
        url = f"/chat/{self.model_provider}/{self.model_name}" 
        self.logger.info(f"Sending chat request to {url} with model {self.model_id}")

        try:
            response = await self._client.post(
                url,
                json=request.model_dump(mode="json"),
                timeout=request_timeout # Pass request-specific timeout here
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    
                    # Handle different error response formats
                    if isinstance(error_data.get("detail"), list):
                        # FastAPI validation errors format
                        error_msg = f"Validation errors: {error_data['detail']}"
                        error_type = "validation_error"
                    elif isinstance(error_data.get("detail"), dict):
                        # Custom error format
                        error_detail = error_data["detail"]
                        error_type = error_detail.get("error", "unknown_error")
                        error_msg = error_detail.get("message", str(error_data))
                    else:
                        # Fallback format
                        error_type = "unknown_error"
                        error_msg = str(error_data)
                    
                    self.logger.error(f"Chat request failed (status {response.status_code}, type {error_type}): {error_msg}")

                    if response.status_code == 404 and error_type == "model_not_found":
                        raise ModelNotFoundException(error_msg)
                    elif response.status_code == 400 and error_type == "internal_conversion_exception":
                        raise InternalConversionException(error_msg)
                    elif response.status_code == 429 and error_type == "service_throttling_exception":
                        raise ServiceCallThrottlingException(error_msg)
                    elif response.status_code == 422:
                        if error_type == "structured_response_exception":
                            error_detail = error_data.get("detail", {})
                            raise StructuredResponseException(
                                error_msg,
                                xml=error_detail.get("xml", ""),
                                return_class=error_detail.get("return_class")
                            )
                        else:
                            # Validation error or other 422 error
                            raise ServiceCallException(f"Validation error: {error_msg}")
                    elif response.status_code == 401 and error_type == "credentials_not_set":
                        raise CredentialsException(error_msg)
                    elif response.status_code == 502 and error_type == "service_call_error":
                        raise ServiceCallException(error_msg)
                    else:
                        # General service call exception for other errors
                        raise ServiceCallException(f"Chat request failed with status {response.status_code}: {error_msg}")
                except ValueError: # Handle cases where response is not valid JSON
                    error_msg = f"Chat request failed with status {response.status_code} and non-JSON response: {response.text}"
                    self.logger.error(error_msg)
                    raise ServiceCallException(error_msg) from None


            llm_response_as_json = response.json()
            llm_response = LLMResponse.model_validate(llm_response_as_json)
            self.logger.info(f"Chat request successful for model {self.model_id}")

            return llm_response

        except httpx.TimeoutException as e:
            # Use the actual timeout value used for the request
            timeout_seconds = request_timeout.read if hasattr(request_timeout, 'read') else request_timeout 
            error_msg = f"Request timed out after {timeout_seconds:.1f} seconds for {url}"
            self.logger.error(error_msg, exc_info=True)
            raise TimeoutException(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Request error connecting to server for {url}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            if isinstance(e, httpx.ReadTimeout):
                timeout_seconds = request_timeout.read if hasattr(request_timeout, 'read') else request_timeout
                raise TimeoutException(f"Read timeout after {timeout_seconds:.1f} seconds for {url}") from e
            raise ServiceCallException(f"Failed to connect to server: {str(e)}") from e
        
