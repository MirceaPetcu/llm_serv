from enum import Enum
from typing import Annotated, Type
import uuid

from pydantic import BaseModel, ConfigDict, PlainSerializer, Field, field_validator

from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.types import LLMRequestType
from llm_serv.structured_response.model import StructuredResponse


class LLMRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_type: LLMRequestType = LLMRequestType.LLM
    conversation: Conversation    
    response_model: Annotated[Type[StructuredResponse] | None, PlainSerializer(lambda obj: obj.__name__)] = Field(
        default=None, exclude=True
    )
    force_native_structured_response: bool = False
    max_completion_tokens: int | None = None
    temperature: float = 1.
    max_retries: int = 5
    top_p: float | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """
    # TODO: check for valid conversations and messages
    @classmethod
    @field_validator("prompt", "messages")
    def check_prompt_or_messages(cls, v, info):
        prompt = info.data.get("prompt")
        messages = info.data.get("messages")
        if prompt is None and messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided and not None")
        if prompt is not None and messages is not None:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided")
        return v
    """
