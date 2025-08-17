import re
from typing import Any, Union, get_origin, get_args


def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def coerce_text_to_type(text: str, type_name: str) -> Any:
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


def unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional[T] to T, handling Union[T, None] patterns."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def is_list_type(annotation: Any) -> bool:
    """Check if annotation represents a list type."""
    origin = get_origin(annotation)
    return origin in (list, tuple, set) or origin is list


def extract_instance_from_model(model: Any) -> dict[str, Any]:
    """Extract instance data from a BaseModel instance."""
    from enum import Enum
    from pydantic import BaseModel
    
    data: dict[str, Any] = {}
    
    for field_name, field_info in model.__class__.model_fields.items():
        value = getattr(model, field_name)
        annotation = field_info.annotation
        unwrapped_annotation = unwrap_optional(annotation)
        
        if value is None:
            data[field_name] = None
            continue
        
        if is_list_type(unwrapped_annotation):
            element_type = get_args(unwrapped_annotation)[0] if get_args(unwrapped_annotation) else Any
            unwrapped_element_type = unwrap_optional(element_type)
            
            if isinstance(unwrapped_element_type, type) and issubclass(unwrapped_element_type, BaseModel):
                data[field_name] = [extract_instance_from_model(item) for item in value]
            elif isinstance(unwrapped_element_type, type) and issubclass(unwrapped_element_type, Enum):
                data[field_name] = [item.value for item in value]
            else:
                data[field_name] = list(value)
        elif isinstance(unwrapped_annotation, type) and issubclass(unwrapped_annotation, BaseModel):
            data[field_name] = extract_instance_from_model(value)
        elif isinstance(unwrapped_annotation, type) and issubclass(unwrapped_annotation, Enum):
            data[field_name] = value.value
        else:
            data[field_name] = value
    
    return data


def extract_constraints(field_info: Any) -> dict[str, Any]:
    """Extract validation constraints from Pydantic field info."""
    constraints: dict[str, Any] = {}
    if not field_info:
        return constraints
    
    # Direct attributes on FieldInfo
    constraint_attrs = (
        "ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length"
    )
    for attr in constraint_attrs:
        value = getattr(field_info, attr, None)
        if value is not None:
            constraints[attr] = value
    
    # Pydantic 2.x stores constraints in metadata
    metadata = getattr(field_info, "metadata", []) or []
    for constraint in metadata:
        for attr in constraint_attrs:
            if hasattr(constraint, attr):
                value = getattr(constraint, attr)
                if value is not None and attr not in constraints:
                    constraints[attr] = value
    
    return constraints