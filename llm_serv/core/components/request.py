import uuid
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.types import LLMRequestType
from llm_serv.structured_response.model import StructuredResponse


class LLMRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_type: LLMRequestType = LLMRequestType.LLM
    conversation: Conversation    
    response_model: StructuredResponse | None = None
    force_native_structured_response: bool = False
    max_completion_tokens: int | None = None
    temperature: float = 1.
    max_retries: int = 5
    top_p: float | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_serializer('response_model')
    def serialize_response_model(self, value: StructuredResponse | None) -> dict[str, Any] | None:
        """Serialize StructuredResponse using its serialize method."""
        if value is None:
            return None
        # serialize() returns a JSON string, so we parse it back to dict for Pydantic
        json_string = value.serialize()
        return json.loads(json_string)
    
    @field_validator('response_model', mode='before')
    @classmethod
    def deserialize_response_model(cls, value: Any) -> StructuredResponse | None:
        """Deserialize StructuredResponse using its deserialize method."""
        if value is None:
            return None
        if isinstance(value, StructuredResponse):
            return value
        if isinstance(value, dict):
            # Convert dict to JSON string for deserialize function
            json_string = json.dumps(value)
            from llm_serv.structured_response.converters.deserialize import deserialize
            return deserialize(json_string)
        if isinstance(value, str):
            # Handle JSON string input
            from llm_serv.structured_response.converters.deserialize import deserialize
            return deserialize(value)
        raise ValueError(f"Cannot deserialize response_model from type {type(value)}")

