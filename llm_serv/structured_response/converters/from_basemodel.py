from enum import Enum
from typing import Any, Union, get_args, get_origin
from pydantic import BaseModel, fields
from llm_serv.structured_response.model import StructuredResponse


def from_basemodel(model: BaseModel | type[BaseModel]) -> StructuredResponse:
    """
    Build a StructuredResponse definition from a Pydantic BaseModel.
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
                    "elements": elem_def,
                }
            else:
                result = {
                    "type": "list",
                    "description": (getattr(field_info, "description", "") if field_info else ""),
                    "elements": _python_type_name(elem_ann),
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

