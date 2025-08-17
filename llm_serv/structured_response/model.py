from __future__ import annotations

import json
from typing import Any

from llm_serv.structured_response.converters.from_prompt import from_prompt
from llm_serv.structured_response.converters.manual import add_node
from llm_serv.structured_response.converters.serialize import serialize
from llm_serv.structured_response.converters.deserialize import deserialize
from llm_serv.structured_response.converters.to_prompt import to_prompt
from llm_serv.structured_response.utils import camel_to_snake


class StructuredResponse:
    def __init__(self, class_name: str = "StructuredResponse", definition: dict[str, Any] | None = None, instance: dict[str, Any] | None = None):
        self.class_name = class_name
        self.definition: dict[str, Any] = definition or {}
        self.instance: dict[str, Any] = instance or {}        

    to_prompt = to_prompt
    from_prompt = from_prompt
    add_node = add_node
    serialize = serialize
    deserialize = deserialize
    
    @staticmethod
    def from_basemodel(model):
        """Import and call from_basemodel to avoid circular imports."""
        from llm_serv.structured_response.converters.from_basemodel import from_basemodel
        return from_basemodel(model)

    def __str__(self) -> str:
        """
        Render the current instance data as simple XML (no type/constraint attributes,
        and no index attributes on list items), matching the README example.
        """
        root_tag = camel_to_snake(self.class_name)

        def coerce_primitive_to_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            return str(value)

        def render_simple_field(field_name: str, value: Any, indent_level: int) -> list[str]:
            pad = "    " * indent_level
            text = coerce_primitive_to_text(value)
            return [f"{pad}<{field_name}>{text}</{field_name}>"]

        def render_list_field(
            field_name: str,
            items_value: Any,
            element_schema: Any | None,
            indent_level: int,
        ) -> list[str]:
            pad = "    " * indent_level
            lines: list[str] = [f"{pad}<{field_name}>"]
            # Normalize to list
            items: list[Any] = items_value or []
            for item in items:
                if isinstance(element_schema, dict):
                    # Complex list items (dict-shaped according to schema)
                    lines.append(f"{pad}    <li>")
                    # Fall back to keys in item when schema not exhaustive
                    for child_name, child_schema in element_schema.items():
                        child_value = None
                        if isinstance(item, dict):
                            child_value = item.get(child_name)
                        # Delegate to object renderer for consistency
                        lines.extend(
                            render_field(child_name, child_schema, child_value, indent_level + 2)
                        )
                    lines.append(f"{pad}    </li>")
                else:
                    # Simple list items on one line
                    text = coerce_primitive_to_text(item)
                    lines.append(f"{pad}    <li>{text}</li>")
            lines.append(f"{pad}</{field_name}>")
            return lines

        def render_object_field(
            field_name: str,
            object_schema: dict[str, Any],
            object_value: Any,
            indent_level: int,
        ) -> list[str]:
            pad = "    " * indent_level
            lines: list[str] = [f"{pad}<{field_name}>"]
            value_dict = object_value or {}
            for child_name, child_schema in object_schema.items():
                child_value = None
                if isinstance(value_dict, dict):
                    child_value = value_dict.get(child_name)
                lines.extend(
                    render_field(child_name, child_schema, child_value, indent_level + 1)
                )
            lines.append(f"{pad}</{field_name}>")
            return lines

        def render_field(
            field_name: str,
            field_schema: Any,
            field_value: Any,
            indent_level: int,
        ) -> list[str]:
            # List case (schema with explicit type == list)
            if isinstance(field_schema, dict) and field_schema.get("type") == "list":
                return render_list_field(
                    field_name,
                    field_value,
                    field_schema.get("elements"),
                    indent_level,
                )

            # Dict case (schema with explicit type == dict)
            if isinstance(field_schema, dict) and field_schema.get("type") == "dict":
                return render_object_field(field_name, field_schema.get("elements", {}), field_value, indent_level)

            # Nested object (schema dict without explicit type)
            if isinstance(field_schema, dict) and "type" not in field_schema:
                return render_object_field(field_name, field_schema, field_value, indent_level)

            # Simple field (including enum)
            return render_simple_field(field_name, field_value, indent_level)

        # If no instance, render an empty root element
        if not self.instance:
            return f"<{root_tag}>\n</{root_tag}>"

        lines: list[str] = [f"<{root_tag}>"]
        # Prefer schema order if available; otherwise iterate instance keys
        field_names: list[str]
        if isinstance(self.definition, dict):
            field_names = list(self.definition.keys())
        else:
            field_names = list(self.instance.keys())

        for field_name in field_names:
            schema_for_field: Any = (
                self.definition.get(field_name) if isinstance(self.definition, dict) else None
            )
            value_for_field = self.instance.get(field_name)

            # If no schema, infer rendering based on value shape
            if schema_for_field is None:
                if isinstance(value_for_field, list):
                    # Infer simple vs complex by inspecting first element
                    element_schema: Any | None
                    if value_for_field and isinstance(value_for_field[0], dict):
                        # Build a shallow schema from keys of first element
                        element_schema = {k: {"type": "str"} for k in value_for_field[0].keys()}
                    else:
                        element_schema = None
                    lines.extend(
                        render_list_field(field_name, value_for_field, element_schema, 1)
                    )
                elif isinstance(value_for_field, dict):
                    # Infer nested object schema from keys
                    inferred_schema = {k: {"type": "str"} for k in value_for_field.keys()}
                    lines.extend(
                        render_object_field(field_name, inferred_schema, value_for_field, 1)
                    )
                else:
                    lines.extend(render_simple_field(field_name, value_for_field, 1))
                continue

            # Render with available schema
            lines.extend(
                render_field(field_name, schema_for_field, value_for_field, indent_level=1)
            )

        lines.append(f"</{root_tag}>")
        return "\n".join(lines)

    def _extract_instance_from_model(self, model: Any) -> dict[str, Any]:
        """Extract instance data from a BaseModel instance - moved from from_basemodel.py"""
        from enum import Enum
        from typing import get_origin, get_args
        
        data: dict[str, Any] = {}
        
        for field_name, field_info in model.__class__.model_fields.items():
            value = getattr(model, field_name)
            annotation = field_info.annotation
            unwrapped_annotation = self._unwrap_optional_local(annotation)
            
            if value is None:
                data[field_name] = None
                continue
            
            if self._is_list_type_local(unwrapped_annotation):
                element_type = get_args(unwrapped_annotation)[0] if get_args(unwrapped_annotation) else Any
                unwrapped_element_type = self._unwrap_optional_local(element_type)
                
                if isinstance(unwrapped_element_type, type) and issubclass(unwrapped_element_type, BaseModel):
                    data[field_name] = [self._extract_instance_from_model(item) for item in value]
                elif isinstance(unwrapped_element_type, type) and issubclass(unwrapped_element_type, Enum):
                    data[field_name] = [item.value for item in value]
                else:
                    data[field_name] = list(value)
            elif isinstance(unwrapped_annotation, type) and issubclass(unwrapped_annotation, BaseModel):
                data[field_name] = self._extract_instance_from_model(value)
            elif isinstance(unwrapped_annotation, type) and issubclass(unwrapped_annotation, Enum):
                data[field_name] = value.value
            else:
                data[field_name] = value
        
        return data
    
    def _unwrap_optional_local(self, annotation: Any) -> Any:
        """Local helper to unwrap Optional types."""
        from typing import Union, get_origin, get_args
        origin = get_origin(annotation)
        if origin is Union:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                return args[0]
        return annotation
    
    def _is_list_type_local(self, annotation: Any) -> bool:
        """Local helper to check if annotation is a list type."""
        from typing import get_origin
        origin = get_origin(annotation)
        return origin in (list, tuple, set) or origin is list
