import json
from typing import Any
from llm_serv.structured_response.utils import camel_to_snake


def to_prompt(self) -> str:
        """
        Create the XML-like prompt according to README rules.
        """
        if not self.definition:
            raise ValueError("Definition not initialized. Call from_basemodel first.")

        root_tag = camel_to_snake(self.class_name)

        def attrs_to_str(attrs: dict[str, Any]) -> str:
            return " ".join(
                f"{k}='{str(v)}'" for k, v in attrs.items() if v is not None and v != ""
            )

        def render_field(name: str, schema: Any, indent: int = 1) -> list[str]:
            pad = "    " * indent
            lines: list[str] = []
            # Complex list
            if isinstance(schema, dict) and schema.get("type") == "list":
                elem_schema = schema.get("elements")
                elements_attr = "dict" if isinstance(elem_schema, dict) else str(elem_schema)
                attr = {
                    "type": "list",
                    "elements": elements_attr,
                    "description": schema.get("description", "") or None,
                }
                # Add numeric/text constraints if available
                for key_src, key_dst in (
                    ("ge", "greater_or_equal"),
                    ("gt", "greater_than"),
                    ("le", "less_or_equal"),
                    ("lt", "less_than"),
                    ("multiple_of", "multiple_of"),
                    ("min_length", "min_length"),
                    ("max_length", "max_length"),
                ):
                    if key_src in schema:
                        attr[key_dst] = schema[key_src]

                lines.append(f"{pad}<{name} {attrs_to_str(attr)}>")
                # Example one element
                lines.append(f"{pad}    <li index='0'>")
                if isinstance(elem_schema, dict):
                    for sub_name, sub_schema in elem_schema.items():
                        lines.extend(render_field(sub_name, sub_schema, indent + 2))
                else:
                    lines.append(
                        f"{pad}        [value here - as an {str(elem_schema)}]"
                    )
                lines.append(f"{pad}    </li>")
                lines.append(f"{pad}    ...")
                lines.append(f"{pad}</{name}>")
                return lines

            # Nested object (no explicit type key): render children
            if isinstance(schema, dict) and "type" not in schema:
                lines.append(f"{pad}<{name} type='dict'>")
                for sub_name, sub_schema in schema.items():
                    lines.extend(render_field(sub_name, sub_schema, indent + 1))
                lines.append(f"{pad}</{name}>")
                return lines

            # Simple field
            assert isinstance(schema, dict)
            field_type = schema.get("type", "str")
            attr: dict[str, Any] = {"type": field_type}
            if field_type == "enum":
                choices = schema.get("choices", [])
                attr["choices"] = json.dumps(choices)

            # Constraints
            for key_src, key_dst in (
                ("ge", "greater_or_equal"),
                ("gt", "greater_than"),
                ("le", "less_or_equal"),
                ("lt", "less_than"),
                ("multiple_of", "multiple_of"),
                ("min_length", "min_length"),
                ("max_length", "max_length"),
            ):
                if key_src in schema:
                    attr[key_dst] = schema[key_src]

            content_desc = schema.get("description", "")
            if field_type == "list":
                # Should not reach here; list handled above
                pass
            # For non-list: description in brackets with "- as <type>"
            inner = f"[{content_desc} - as a {field_type}]" if content_desc else f"[value here - as a {field_type}]"
            # Emit line
            line = f"{pad}<{name} {attrs_to_str(attr)}>{inner}</{name}>"
            lines.append(line)
            return lines

        lines: list[str] = [f"<{root_tag}>"]
        for field_name, field_schema in self.definition.items():
            lines.extend(render_field(field_name, field_schema, indent=1))
        lines.append(f"</{root_tag}>")
        return "\n".join(lines)