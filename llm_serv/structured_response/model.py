from __future__ import annotations

from typing import Any

from llm_serv.structured_response.converters.deserialize import deserialize
from llm_serv.structured_response.converters.from_prompt import from_prompt
from llm_serv.structured_response.converters.manual import add_node
from llm_serv.structured_response.converters.serialize import serialize
from llm_serv.structured_response.converters.to_prompt import to_prompt
from llm_serv.structured_response.converters.to_string import to_string


class StructuredResponse:
    def __init__(
        self, 
        class_name: str = "StructuredResponse", 
        definition: dict[str, Any] | None = None, 
        instance: dict[str, Any] | None = None,
        native: bool = False
    ):
        self.class_name = class_name
        self.definition: dict[str, Any] = definition or {}
        self.instance: dict[str, Any] = instance or {}        
        self.native = native
    to_prompt = to_prompt
    from_prompt = from_prompt
    add_node = add_node
    serialize = serialize
    deserialize = deserialize
    to_string = to_string
    __str__ = to_string
    
    @staticmethod
    def from_basemodel(model):
        """Import and call from_basemodel to avoid circular imports."""
        from llm_serv.structured_response.converters.from_basemodel import from_basemodel
        return from_basemodel(model)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Minimal Pydantic compatibility - treat as any type."""
        from pydantic_core import core_schema
        return core_schema.any_schema()