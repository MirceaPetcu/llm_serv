import abc
import asyncio
import time
from functools import partial
from typing import Any, Callable, Coroutine

from llm_serv.logger import logger
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.response import LLMResponse
from llm_serv.core.components.tokens import TokenTracker
from llm_serv.core.components.types import LLMRequestType
from llm_serv.core.exceptions import (InternalConversionException,
                                      ServiceCallException,
                                      ServiceCallThrottlingException,
                                      StructuredResponseException)
from llm_serv.api import Model

# Create a module-specific logger
module_logger = logger.getChild("core.base")


class LLMProvider(abc.ABC):
    def __init__(self, model: Model):
        self.model = model
        self.logger = module_logger
        self.logger.info(f"Initializing LLM provider for model: \033[94m{model.id}\033[0m [\033[93m{model.internal_model_id}\033[0m]")

    async def start(self):
        """
        Initialize the provider's internal async client.
        """
        pass

    async def stop(self):
        """
        Clean up the provider's internal async client.
        """
        pass

    @abc.abstractmethod
    async def _llm_service_call(self, request: LLMRequest) -> tuple[str, TokenTracker]:
        """
        This method calls the underlying provider directly, and handles failure cases like throttling with retries internally
        Returns a tuple of (output_text, tokens_info).        
        """
        raise NotImplementedError()

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        """
        This method is the main entry point for the LLMProvider.
        It validates the request and delegates to the appropriate handler.
        Automatically handles the provider's async client initialization.
        """
        await self.start()

        # TODO proper validation of request        
        match request.request_type:
            case LLMRequestType.LLM:
                return await self.__llm_handler(request)
            case LLMRequestType.OCR:
                pass
            case LLMRequestType.IMAGE:
                pass

    async def __retry_wrapper(
        self,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        max_retries: int = 10,
    ) -> Any:
        """Wraps a coroutine function with exponential backoff retry logic, specifically for ServiceCallThrottlingException."""
        retries = 0
        last_exception = None
        first_attempt_time = time.time()

        while retries <= max_retries:
            try:
                return await coro_func()
            except ServiceCallThrottlingException as e:
                # Use the module-specific logger here
                self.logger.debug(f"Service throttled after {retries} retries over {time.time() - first_attempt_time:.2f} seconds.")
                last_exception = e
                retries += 1
                if retries > max_retries:
                    total_retry_duration = time.time() - first_attempt_time
                    self.logger.debug(f"Maximum retries ({max_retries}) reached after {total_retry_duration:.2f} seconds")
                    # Raise the specific throttling exception indicating exhaustion of retries
                    raise ServiceCallThrottlingException(
                        f"Service throttled after {max_retries} retries over {total_retry_duration:.2f} seconds."
                    ) from e
                # Calculate delay using exponential backoff (1, 2, 4, 8, ...)
                delay = 2 ** (retries - 1)
                self.logger.debug(f"Retrying after {delay}s delay (attempt {retries+1}/{max_retries+1})")
                await asyncio.sleep(delay)
            # Any other exception will propagate immediately and exit the loop

        # This part should technically be unreachable if max_retries >= 0,
        # because the loop either returns a result or raises ServiceCallThrottlingException.
        # Included as a safeguard against unexpected loop termination.
        # If last_exception is None here, it means the loop finished without success or a caught exception.
        raise ServiceCallException(
            f"Retry wrapper finished unexpectedly after {retries} attempts. Last known exception: {last_exception}"
        ) if last_exception else ServiceCallException(
            "Retry wrapper finished unexpectedly without result or error."
        )


    async def __llm_handler(self, request: LLMRequest) -> LLMResponse:
        first_attempt_time = time.time() # Record start time before any attempt

        try:
            response: LLMResponse = LLMResponse.from_request(request)
            response.start_time = first_attempt_time # Use the initial attempt time
            response.llm_model = self.model

            # Prepare the coroutine function call using partial to include the request argument
            service_call_coro = partial(self._llm_service_call, request=request)

            # Execute the service call through the retry wrapper
            # Note: Only ServiceCallThrottlingException will be retried internally by the wrapper
            output, model_tokens = await self.__retry_wrapper(coro_func=service_call_coro)

            # Check if the wrapper returned None unexpectedly (should raise instead)
            if output is None:
                raise ServiceCallException(
                    "LLM service call failed to return output after retries, without raising a specific exception."
                )

            response.output = output  # assign initial string output

            """
            If the response format is specified and the response class is not a string,
            attempt to convert the text output to the desired StructuredResponse class.
            Raises StructuredResponseException if the conversion fails.
            """
            if request.response_model is not None:
                try:
                    response.output = request.response_model.from_text(output)
                except Exception as conversion_error:
                    # Wrap potential conversion errors in a specific exception type
                    raise StructuredResponseException(f"Failed to convert LLM output to structured format: {conversion_error}") from conversion_error

            response.tokens.add(self.model.id, model_tokens)

            response.end_time = time.time()
            # Total time reflects the duration from the first attempt including any backoff delays managed by the wrapper
            response.total_duration = response.end_time - response.start_time

            return response

        except (InternalConversionException, StructuredResponseException, ServiceCallThrottlingException) as e:
            # Re-raise specific exceptions that are handled or expected (including the final one from the wrapper)
            raise
        except Exception as e:
            # Wrap any other unexpected exception as a ServiceCallException
            # This includes potential errors from response processing or other parts of the handler.
            raise ServiceCallException(f"Unexpected error during LLM handling: {str(e)}") from e
