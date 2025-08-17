from __future__ import annotations

import enum
from typing import Any, Union, get_args, get_origin
from pydantic import BaseModel
from llm_serv.structured_response.model import StructuredResponse


def from_basemodel(model: BaseModel | type[BaseModel]) -> StructuredResponse:
    """
    Build a StructuredResponse definition from a Pydantic BaseModel type or instance.
    
    Args:
        model: Either a BaseModel class or an instance of a BaseModel
        
    Returns:
        StructuredResponse: A structured response with definition populated from the BaseModel
        and instance data populated if an instance was provided
    """
    # Determine if we have a type or instance
    is_instance_input = isinstance(model, BaseModel)
    model_type: type[BaseModel] = model.__class__ if is_instance_input else model
    
    # Create the StructuredResponse
    response = StructuredResponse()
    response.class_name = model_type.__name__
    
    # Build the definition using add_node method
    _build_definition_recursive(response, "", model_type)
    
    # If we got an instance, extract the instance data
    if is_instance_input:
        response.instance = _extract_instance_from_model(model)
    
    return response


def _build_definition_recursive(response: StructuredResponse, path_prefix: str, model_type: type[BaseModel]) -> None:
    """
    Recursively build the definition using the add_node method.
    
    Args:
        response: The StructuredResponse to populate
        path_prefix: The current path prefix for nested fields
        model_type: The BaseModel type to process
    """
    for field_name, field_info in model_type.model_fields.items():
        current_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
        annotation = field_info.annotation
        description = field_info.description or ""
        
        # Extract constraints from field_info
        constraints = _extract_constraints(field_info)
        
        # Unwrap Optional types
        unwrapped_annotation = _unwrap_optional(annotation)
        
        # Handle different field types
        if _is_list_type(unwrapped_annotation):
            _handle_list_field(response, current_path, unwrapped_annotation, description, constraints)
        elif _is_enum_type(unwrapped_annotation):
            _handle_enum_field(response, current_path, unwrapped_annotation, description, constraints)
        elif _is_basemodel_type(unwrapped_annotation):
            _handle_basemodel_field(response, current_path, unwrapped_annotation, description, constraints)
        else:
            _handle_primitive_field(response, current_path, unwrapped_annotation, description, constraints)


def _handle_list_field(response: StructuredResponse, path: str, annotation: Any, description: str, constraints: dict[str, Any]) -> None:
    """Handle list type fields."""
    element_type = _get_list_element_type(annotation)
    unwrapped_element_type = _unwrap_optional(element_type)
    
    if _is_basemodel_type(unwrapped_element_type):
        # List of BaseModel objects
        response.add_node(path, list, elements=dict, description=description, **constraints)
        _build_definition_recursive(response, path, unwrapped_element_type)
    elif _is_enum_type(unwrapped_element_type):
        # List of enums
        response.add_node(path, list, elements=enum, description=description, **constraints)
    else:
        # List of primitives
        python_type = _annotation_to_python_type(unwrapped_element_type)
        response.add_node(path, list, elements=python_type, description=description, **constraints)


def _handle_enum_field(response: StructuredResponse, path: str, annotation: Any, description: str, constraints: dict[str, Any]) -> None:
    """Handle enum type fields."""
    response.add_node(path, enum, description=description, choices=annotation, **constraints)


def _handle_basemodel_field(response: StructuredResponse, path: str, annotation: Any, description: str, constraints: dict[str, Any]) -> None:
    """Handle nested BaseModel fields."""
    response.add_node(path, dict, description=description, **constraints)
    _build_definition_recursive(response, path, annotation)


def _handle_primitive_field(response: StructuredResponse, path: str, annotation: Any, description: str, constraints: dict[str, Any]) -> None:
    """Handle primitive type fields (str, int, float, bool)."""
    python_type = _annotation_to_python_type(annotation)
    response.add_node(path, python_type, description=description, **constraints)


def _extract_instance_from_model(model: BaseModel) -> dict[str, Any]:
    """
    Extract instance data from a BaseModel instance.
    
    Args:
        model: The BaseModel instance to extract data from
        
    Returns:
        dict: The instance data as a dictionary
    """
    data: dict[str, Any] = {}
    
    for field_name, field_info in model.__class__.model_fields.items():
        value = getattr(model, field_name)
        annotation = field_info.annotation
        unwrapped_annotation = _unwrap_optional(annotation)
        
        if value is None:
            data[field_name] = None
            continue

        if _is_list_type(unwrapped_annotation):
            element_type = _get_list_element_type(unwrapped_annotation)
            unwrapped_element_type = _unwrap_optional(element_type)
            
            if _is_basemodel_type(unwrapped_element_type):
                data[field_name] = [_extract_instance_from_model(item) for item in value]
            elif _is_enum_type(unwrapped_element_type):
                data[field_name] = [item.value for item in value]
            else:
                data[field_name] = list(value)
        elif _is_basemodel_type(unwrapped_annotation):
            data[field_name] = _extract_instance_from_model(value)
        elif _is_enum_type(unwrapped_annotation):
            data[field_name] = value.value
        else:
            data[field_name] = value
    
    return data


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional[T] to T, handling Union[T, None] patterns."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_list_type(annotation: Any) -> bool:
    """Check if annotation represents a list type."""
    origin = get_origin(annotation)
    return origin in (list, tuple, set) or origin is list


def _is_enum_type(annotation: Any) -> bool:
    """Check if annotation represents an enum type."""
    return isinstance(annotation, type) and issubclass(annotation, enum.Enum)


def _is_basemodel_type(annotation: Any) -> bool:
    """Check if annotation represents a BaseModel type."""
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _get_list_element_type(annotation: Any) -> Any:
    """Get the element type from a list annotation."""
    args = get_args(annotation)
    return args[0] if args else Any


def _annotation_to_python_type(annotation: Any) -> type:
    """Convert type annotation to Python built-in type."""
    if annotation is int:
        return int
    elif annotation is float:
        return float
    elif annotation is str:
        return str
    elif annotation is bool:
        return bool
    else:
        # Fallback for unknown types
        return str


def _extract_constraints(field_info: Any) -> dict[str, Any]:
    """
    Extract validation constraints from Pydantic field info.
    
    Args:
        field_info: Pydantic FieldInfo object
        
    Returns:
        dict: Dictionary of constraint name -> value pairs
    """
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
    for constraint in getattr(field_info, "metadata", []) or []:
        for attr in constraint_attrs:
            if hasattr(constraint, attr):
                value = getattr(constraint, attr)
                if value is not None and attr not in constraints:
                    constraints[attr] = value
    
    return constraints