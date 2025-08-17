import uuid

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

    @field_validator("response_model", mode="before")
    @classmethod
    def _deserialize_response_model(cls, response_model):
        """
        Accepts response_model as a StructuredResponse instance, a JSON string
        produced by StructuredResponse.serialize(), or a plain dict with keys
        matching the StructuredResponse dataclass, and converts it to a
        StructuredResponse instance for internal usage.
        """
        from llm_serv.structured_response.model import StructuredResponse as SR

        if response_model is None:
            return None
        if isinstance(response_model, SR):
            return response_model
        if isinstance(response_model, str):
            try:
                # Use the deserialize method to convert JSON string back to StructuredResponse
                return SR.deserialize(response_model)
            except Exception:
                return response_model
        if isinstance(response_model, dict):
            try:
                return SR(
                    class_name=response_model.get("class_name", "StructuredResponse"),
                    definition=response_model.get("definition"),
                    instance=response_model.get("instance"),
                )
            except Exception:
                return response_model
        return response_model

    @field_serializer("response_model", when_used="json")
    def _serialize_response_model(self, response_model):
        """
        Serialize the response_model for JSON output.
        - If it's a StructuredResponse instance, use its serialize() method
        - If it's a class (Pydantic BaseModel or a plain class derived from StructuredResponse),
          build a StructuredResponse via from_basemodel() and serialize it
        - If it's a Pydantic BaseModel instance, convert via from_basemodel() and serialize
        - If it's already a string (assumed pre-serialized), pass it through
        - Otherwise, return None
        """
        from llm_serv.structured_response.model import StructuredResponse as SR

        if response_model is None:
            return None

        # Already a StructuredResponse instance - serialize to JSON string
        if isinstance(response_model, SR):
            return response_model.serialize()

        # If a class/type (e.g., Pydantic BaseModel subclass or plain class inheriting SR)
        if isinstance(response_model, type):
            try:
                sr_obj = SR.from_basemodel(response_model)
                return sr_obj.serialize()
            except Exception as e:
                raise ValueError(f"Response model {response_model} is not serializable: {e}") from e

        # If it's a Pydantic BaseModel instance
        if isinstance(response_model, BaseModel) or hasattr(response_model, "model_dump"):
            try:
                sr_obj = SR.from_basemodel(response_model)
                return sr_obj.serialize()
            except Exception as e:
                raise ValueError(f"Response model {response_model} is not serializable: {e}") from e

        # If it's already a string, assume it's a pre-serialized representation
        if isinstance(response_model, str):
            return response_model

        # Fallback: not serializable
        raise ValueError(f"Response model {response_model} is not serializable")
