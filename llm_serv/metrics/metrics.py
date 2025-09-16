import msgspec
from typing import Optional


class ModelMetrics(msgspec.Struct):
    """Model metrics for LLM serving performance tracking."""
    
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    call_start_time: float = 0.0
    call_end_time: float = 0.0
    call_duration: float = 0.0

    tokens_per_second: float = 0.0  # Total tokens per total duration
    
    status_code: Optional[int] = None
    error_message: str = ""

    internal_retries: int = 0