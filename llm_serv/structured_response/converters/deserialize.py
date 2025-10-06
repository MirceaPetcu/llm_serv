import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_serv.structured_response.model import StructuredResponse


def deserialize(json_string: str | dict) -> "StructuredResponse":
    """Deserialize a JSON string to StructuredResponse."""
    from llm_serv.structured_response.model import StructuredResponse
    
    data = json.loads(json_string)
    sr = StructuredResponse(
        class_name=data.get("class_name", "StructuredResponse"),
        definition=data.get("definition") or {},
        instance=data.get("instance") or {},
        native=data.get("native", False)
    )
    return sr