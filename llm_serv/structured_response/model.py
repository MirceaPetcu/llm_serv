from __future__ import annotations

import json
from typing import Any
from xml.etree import ElementTree as ET

from llm_serv.structured_response.converters.from_prompt import from_prompt
from llm_serv.structured_response.converters.manual import add_node
from llm_serv.structured_response.converters.to_prompt import to_prompt
from llm_serv.structured_response.utils import camel_to_snake


class StructuredResponse:
    def __init__(self):
        self.class_name = "StructuredResponse"
        self.definition: dict[str, Any] = {}
        self.instance: dict[str, Any] = {}        

    to_prompt = to_prompt
    from_prompt = from_prompt
    add_node = add_node

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

    def serialize(self) -> str:
        data = {
            "class_name": self.class_name,
            "definition": self.definition,
            "instance": self.instance,
        }
        return json.dumps(data)

    @staticmethod
    def deserialize(json_string: str) -> "StructuredResponse":
        data = json.loads(json_string)
        sr = StructuredResponse(
            class_name=data.get("class_name", "StructuredResponse"),
            definition=data.get("definition"),
            instance=data.get("instance"),
        )
        return sr