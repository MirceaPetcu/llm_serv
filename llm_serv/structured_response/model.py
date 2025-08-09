from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union, get_args, get_origin
from xml.etree import ElementTree as ET
import json
import re

from pydantic import BaseModel, fields


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_list(annotation: Any) -> bool:
    return get_origin(annotation) in (list, list.__class__, tuple, set) or get_origin(annotation) is list


def _list_arg(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_args(annotation) else Any


def _camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _python_type_name(annotation: Any) -> str:
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "enum"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "dict"
    if _is_list(annotation):
        return "list"
    return getattr(annotation, "__name__", str(annotation))


def _extract_constraints(field_info: Any) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if not field_info:
        return constraints
    # Direct attributes on FieldInfo (works for non-BaseModel usage too)
    for attr in ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length"):
        value = getattr(field_info, attr, None)
        if value is not None:
            constraints[attr] = value
    # Pydantic BaseModel field_info may keep constraints in metadata
    for constraint in getattr(field_info, "metadata", []) or []:
        for attr in ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length"):
            if hasattr(constraint, attr):
                value = getattr(constraint, attr)
                if value is not None and attr not in constraints:
                    constraints[attr] = value
    return constraints


def _iter_schema_fields(cls: type) -> list[tuple[str, Any, Any]]:
    """Return (name, annotation, field_info) for a class that may be a BaseModel subclass
    or a plain class using Field() descriptors."""
    results: list[tuple[str, Any, Any]] = []
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        for name, finfo in cls.model_fields.items():
            results.append((name, finfo.annotation, finfo))
        return results
    # Plain class path: use annotations and attribute value as FieldInfo if present
    annotations = getattr(cls, "__annotations__", {}) or {}
    for name, ann in annotations.items():
        finfo = getattr(cls, name, None)
        # Only treat FieldInfo-like or None
        if isinstance(finfo, fields.FieldInfo) or finfo is None:
            results.append((name, ann, finfo))
        else:
            results.append((name, ann, None))
    return results


def _coerce_text_to_type(text: str, type_name: str) -> Any:
    if text is None:
        return None
    text = text.strip()
    if type_name == "int":
        return int(text)
    if type_name == "float":
        return float(text)
    if type_name == "bool":
        lowered = text.lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        return bool(text)
    # enum and str fall back to plain string
    return text


@dataclass
class StructuredResponse:
    class_name: str = "StructuredResponse"
    definition: Optional[dict] = None
    instance: Optional[dict] = None

    @staticmethod
    def from_basemodel(
        objects: type[BaseModel] | list[type[BaseModel]] | BaseModel | list[BaseModel]
    ) -> "StructuredResponse":
        """
        Build a StructuredResponse definition (and optionally instance data) from one or more
        Pydantic BaseModel types or instances. If instances are provided, instance data is
        initialized as well; otherwise only the schema definition is built.
        The first element defines the root class.
        """

        # Normalize input to a list
        if not isinstance(objects, list):
            objects_list: list[type[BaseModel] | BaseModel] = [objects]
        else:
            objects_list = objects

        if len(objects_list) == 0:
            raise ValueError("from_basemodel requires at least one BaseModel type or instance")

        root_obj = objects_list[0]
        is_instance_input = isinstance(root_obj, BaseModel)
        root_type: type = root_obj.__class__ if is_instance_input else root_obj  # type: ignore[assignment]

        def build_field_definition(field_annotation: Any, field_info) -> dict | str:
            """Return a definition dict for complex fields or a type string for simple fields."""
            ann = _unwrap_optional(field_annotation)

            # List handling
            if _is_list(ann):
                elem_ann = _unwrap_optional(_list_arg(ann))
                if isinstance(elem_ann, type) and (issubclass(elem_ann, BaseModel) or hasattr(elem_ann, "__annotations__")):
                    elem_def = {}
                    for sub_name, sub_ann, sub_info in _iter_schema_fields(elem_ann):
                        elem_def[sub_name] = build_field_definition(sub_ann, sub_info)
                    result: dict[str, Any] = {
                        "type": "list",
                        "description": (getattr(field_info, "description", "") if field_info else ""),
                        "elements_type": elem_def,
                    }
                else:
                    result = {
                        "type": "list",
                        "description": (getattr(field_info, "description", "") if field_info else ""),
                        "elements_type": _python_type_name(elem_ann),
                    }
                # Constraints
                for k, v in _extract_constraints(field_info).items():
                    result[k] = v
                return result

            # Enum
            if isinstance(ann, type) and issubclass(ann, Enum):
                return {
                    "type": "enum",
                    "choices": [e.value for e in ann],
                    "description": (getattr(field_info, "description", "") if field_info else ""),
                }

            # Nested BaseModel
            if isinstance(ann, type) and (issubclass(ann, BaseModel) or hasattr(ann, "__annotations__")):
                nested: dict[str, Any] = {}
                for sub_name, sub_ann, sub_info in _iter_schema_fields(ann):
                    nested[sub_name] = build_field_definition(sub_ann, sub_info)
                return nested

            # Primitive
            field_def: dict[str, Any] = {
                "type": _python_type_name(ann),
                "description": (getattr(field_info, "description", "") if field_info else ""),
            }

            # Constraints
            for k, v in _extract_constraints(field_info).items():
                field_def[k] = v
            return field_def

        definition: dict[str, Any] = {}
        for field_name, field_ann, field_info in _iter_schema_fields(root_type):
            definition[field_name] = build_field_definition(field_ann, field_info)

        resp = StructuredResponse(
            class_name=root_type.__name__,
            definition=definition,
            instance=None,
        )

        # If we got an instance, prefill instance dict from it
        if is_instance_input:
            resp.instance = resp._extract_instance_from_model(root_obj)  # type: ignore[arg-type]

        return resp

    def _extract_instance_from_model(self, model: BaseModel) -> dict:
        data: dict[str, Any] = {}
        for field_name, field_info in model.__class__.model_fields.items():
            value = getattr(model, field_name)
            ann = _unwrap_optional(field_info.annotation)
            if value is None:
                data[field_name] = None
                continue

            if _is_list(ann):
                elem_ann = _unwrap_optional(_list_arg(ann))
                if isinstance(elem_ann, type) and issubclass(elem_ann, BaseModel):
                    data[field_name] = [self._extract_instance_from_model(item) for item in value]
                else:
                    data[field_name] = list(value)
            elif isinstance(ann, type) and issubclass(ann, BaseModel):
                data[field_name] = self._extract_instance_from_model(value)
            elif isinstance(ann, type) and issubclass(ann, Enum):
                data[field_name] = value.value
            else:
                data[field_name] = value
        return data

    def from_prompt(self, xml_string: str) -> None:
        """
        Parse an LLM answer string, extract the XML section for the root element, and
        populate the instance dictionary according to the definition schema.
        """
        if not self.definition:
            raise ValueError("Definition not initialized. Call from_basemodel first.")

        root_tag = _camel_to_snake(self.class_name)
        # Extract substring between first <root_tag and last </root_tag>
        start_idx = xml_string.find(f"<{root_tag}")
        end_idx = xml_string.rfind(f"</{root_tag}>")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Root XML tags not found in LLM output")
        # Move end to include closing tag
        end_idx += len(f"</{root_tag}>")
        xml_sub = xml_string[start_idx:end_idx]

        # Ensure the opening tag is well-formed (strip attributes if any are present)
        try:
            root_element = ET.fromstring(xml_sub)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML content: {exc}") from exc

        def parse_element(element: ET.Element, schema: Any) -> Any:
            # schema can be a dict (complex) or a type descriptor for simple
            if isinstance(schema, dict) and "type" in schema and schema["type"] == "list":
                # List parsing from <li> children
                items: list[Any] = []
                for li in element.findall("li"):
                    elem_schema = schema["elements_type"]
                    if isinstance(elem_schema, dict):
                        # complex item: children tags represent fields
                        item: dict[str, Any] = {}
                        for child in li:
                            if child.tag == "li":
                                continue
                            if child.tag is ET.Comment:
                                continue
                            field_schema = elem_schema.get(child.tag)
                            if field_schema is None:
                                continue
                            item[child.tag] = parse_element(child, field_schema)
                        items.append(item)
                    else:
                        # simple item: li text
                        items.append(_coerce_text_to_type(li.text or "", str(elem_schema)))
                return items

            if isinstance(schema, dict) and "type" not in schema:
                # Nested object: parse each child according to schema
                obj: dict[str, Any] = {}
                for field_name, field_schema in schema.items():
                    child = element.find(field_name)
                    if child is None:
                        obj[field_name] = None
                        continue
                    obj[field_name] = parse_element(child, field_schema)
                return obj

            # Simple field
            assert isinstance(schema, dict)
            type_name = schema.get("type", "str")
            # For enum treat as str
            if type_name == "enum":
                return (element.text or "").strip()
            return _coerce_text_to_type(element.text or "", type_name)

        self.instance = {}
        for field_name, schema in self.definition.items():
            child = root_element.find(field_name)
            if child is None:
                self.instance[field_name] = None
                continue
            self.instance[field_name] = parse_element(child, schema)
        assert self.instance is not None

    def to_prompt(self) -> str:
        """
        Create the XML-like prompt according to README rules.
        """
        if not self.definition:
            raise ValueError("Definition not initialized. Call from_basemodel first.")

        root_tag = _camel_to_snake(self.class_name)

        def attrs_to_str(attrs: dict[str, Any]) -> str:
            return " ".join(
                f"{k}='{str(v)}'" for k, v in attrs.items() if v is not None and v != ""
            )

        def render_field(name: str, schema: Any, indent: int = 1) -> list[str]:
            pad = "    " * indent
            lines: list[str] = []
            # Complex list
            if isinstance(schema, dict) and schema.get("type") == "list":
                elem_schema = schema.get("elements_type")
                elements_type_attr = "dict" if isinstance(elem_schema, dict) else str(elem_schema)
                attr = {
                    "type": "list",
                    "elements_type": elements_type_attr,
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

    def __str__(self) -> str:
        return self.to_prompt()

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