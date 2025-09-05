import json
from typing import Any

from llm_serv.structured_response.utils import camel_to_snake

PREAMBLE = """Respond ONLY with valid XML as shown above, with the following requirements:
- Notice that "..." represents multiple <li> items.
- Do not include any attributes in the output! The 'description' attribute is for you to understand the problem and how to respond; the 'type' is for you to understand the type of the response item, etc.
- Mind the non-string values! If it's a numeric value, only write write the number and no other text; same for enums or booleans.
- Output only VALID XML, while keeping in mind the objective at all times.
"""

def to_prompt(self) -> str:
    """
    Create the XML-like prompt according to README rules:
    - Root element is the class name (snake_case)
    - Each element has the type attribute
    - For non-list items, description is placed between [ and ], with "- as <type>" appended
    - Lists have elements attribute, with complex types as "dict"
    - Lists have <li> items with index attribute starting from 0
    - Dicts have type='dict' and description, then contain their elements as children
    - Enums have choices attribute with JSON stringified list
    - Simple list items have description as [value here - as a <type>]
    """
    if not self.definition:
        raise ValueError("Definition not initialized. Call from_basemodel first.")

    root_tag = camel_to_snake(self.class_name)

    def build_attributes(schema: dict[str, Any], base_attrs: dict[str, Any] = None) -> dict[str, Any]:
        """Build XML attributes from schema, including constraints."""
        attrs = base_attrs.copy() if base_attrs else {}
        
        # Add constraint attributes
        constraint_mappings = {
            "ge": "greater_or_equal",
            "gt": "greater_than", 
            "le": "less_or_equal",
            "lt": "less_than",
            "multiple_of": "multiple_of",
            "min_length": "min_length",
            "max_length": "max_length"
        }
        
        for src_key, dst_key in constraint_mappings.items():
            if src_key in schema:
                attrs[dst_key] = schema[src_key]
                
        return attrs

    def attrs_to_str(attrs: dict[str, Any]) -> str:
        """Convert attributes dict to XML attribute string."""
        return " ".join(f"{k}='{v}'" for k, v in attrs.items() if v is not None and v != "")

    def render_field(name: str, schema: dict[str, Any], indent: int = 1) -> list[str]:
        """Render a single field according to README rules."""
        pad = "    " * indent
        lines: list[str] = []
        
        field_type = schema.get("type")
        description = schema.get("description", "")
        
        if field_type == "list":
            # List handling
            elements = schema.get("elements")
            elements_type = "dict" if isinstance(elements, dict) else str(elements)
            
            attrs = build_attributes(schema, {
                "type": "list",
                "elements": elements_type,
                "description": description
            })
            
            lines.append(f"{pad}<{name} {attrs_to_str(attrs)}>")
            lines.append(f"{pad}    <li>")
            
            if isinstance(elements, dict):
                # Complex list elements (dict)
                for elem_name, elem_schema in elements.items():
                    lines.extend(render_field(elem_name, elem_schema, indent + 2))
            else:
                # Simple list elements (elements is a string like 'int', 'str', etc.)
                lines.append(f"{pad}        [value here - as an {elements}]")
                
            lines.append(f"{pad}    </li>")
            lines.append(f"{pad}    ...")
            lines.append(f"{pad}</{name}>")
            
        elif field_type == "dict":
            # Dict handling - has type='dict' and description, then elements as children
            attrs = build_attributes(schema, {
                "type": "dict",
                "description": description
            })
            
            lines.append(f"{pad}<{name} {attrs_to_str(attrs)}>")
            
            # Render dict elements as children
            elements = schema.get("elements", {})
            if isinstance(elements, dict):
                for elem_name, elem_schema in elements.items():
                    lines.extend(render_field(elem_name, elem_schema, indent + 1))
                    
            lines.append(f"{pad}</{name}>")
            
        elif field_type == "enum":
            # Enum handling
            choices = schema.get("choices", [])
            attrs = build_attributes(schema, {
                "type": "enum",
                "choices": json.dumps(choices)
            })
            
            content = f"[{description} - as an enum]" if description else "[value here - as an enum]"
            lines.append(f"{pad}<{name} {attrs_to_str(attrs)}>{content}</{name}>")
            
        else:
            # Simple types (str, int, float, bool)
            attrs = build_attributes(schema, {"type": field_type})
            
            # Map type names for description
            type_name_map = {
                "str": "string",
                "int": "int", 
                "float": "float",
                "bool": "bool"
            }
            display_type = type_name_map.get(field_type, field_type)
            
            content = f"[{description} - as a {display_type}]" if description else f"[value here - as a {display_type}]"
            lines.append(f"{pad}<{name} {attrs_to_str(attrs)}>{content}</{name}>")
            
        return lines

    # Build the complete XML
    lines: list[str] = [f"<{root_tag}>"]
    for field_name, field_schema in self.definition.items():
        lines.extend(render_field(field_name, field_schema, indent=1))
    lines.append(f"</{root_tag}>")
    
    return "\n".join(lines + [PREAMBLE])