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
        return extract_int(text)
    if type_name == "float":
        return extract_float(text)
    if type_name == "bool":
        return extract_bool(text)
    # enum and str fall back to plain string
    return text


def coerce_primitive_to_text(value: Any) -> str:
    """Convert a primitive value to its text representation for XML output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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


def extract_int(text: str) -> int:
    """Extract an integer from a string using comprehensive regex patterns.
    
    Handles various formats:
    - Empty strings (raises ValueError)
    - Text with embedded integers: 'The answer is 2.' -> 2
    - HTML/XML tags: '2<ref id="23"/>' -> 2
    - Quoted numbers: '"-1"' -> -1
    - Multiple numbers (returns first found)
    - Negative numbers in various formats
    
    Args:
        text: Input string that may contain an integer
        
    Returns:
        int: The first integer found in the string
        
    Raises:
        ValueError: If no valid integer is found in the string
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract integer from empty string")
    
    # Step 1: Remove HTML/XML tags using regex
    # Pattern explanation: < followed by any chars except >, then >
    html_cleaned = re.sub(r'<[^>]*>', '', text)
    
    # Step 2: Remove comma separators (1,000 -> 1000)
    comma_cleaned = re.sub(r'(\d),(?=\d)', r'\1', html_cleaned)
    
    # Step 3: Remove common quote patterns around the entire string
    # Handle cases like '"42"', "'42'", etc.
    quote_cleaned = re.sub(r'^[\'\"]+|[\'\"]+$', '', comma_cleaned.strip())
    
    # Step 4: Try to find integers using multiple strategies
    
    # Strategy 1: Look for standalone integers (avoid decimals)
    # Pattern: not preceded by digit/dot, optional minus, digits, not followed by dot then digit
    standalone_pattern = r'(?<![.\d])-?\d+(?!\.\d)'
    matches = re.findall(standalone_pattern, quote_cleaned)
    if matches:
        return int(matches[0])
    
    # Strategy 2: Look for integers with word boundaries (avoid decimals)
    # Pattern: word boundary, optional minus, digits, not followed by decimal point and digit
    word_boundary_pattern = r'\b-?\d+(?!\.\d)\b'
    matches = re.findall(word_boundary_pattern, quote_cleaned)
    if matches:
        return int(matches[0])
    
    # Strategy 3: Look for any sequence of digits with optional minus
    # This catches cases where integers are surrounded by special chars
    loose_pattern = r'-?\d+'
    matches = re.findall(loose_pattern, quote_cleaned)
    if matches:
        # Filter out matches that are just minus signs
        valid_matches = [m for m in matches if m != '-' and len(m) > 0]
        if valid_matches:
            return int(valid_matches[0])
    
    # Strategy 4: Very aggressive - extract digits and try to form number
    # Remove everything except digits and minus signs
    digits_only = re.sub(r'[^\d\-]', '', text)
    if digits_only:
        # Handle multiple minus signs by taking only the first one if at start
        if digits_only.startswith('-'):
            # Keep only first minus and all digits
            cleaned = '-' + re.sub(r'[^\d]', '', digits_only[1:])
        else:
            # Remove all minus signs and keep only digits
            cleaned = re.sub(r'[^\d]', '', digits_only)
        
        if cleaned and cleaned != '-':
            try:
                return int(cleaned)
            except ValueError:
                pass
    
    # If all strategies fail, raise ValueError
    raise ValueError(f"No valid integer found in string: '{text}'")

def extract_float(text: str) -> float:
    """Extract a float from a string using comprehensive regex patterns.
    
    Handles various formats:
    - Empty strings (raises ValueError)
    - Text with embedded floats: 'The value is 3.14.' -> 3.14
    - HTML/XML tags: '2.5<ref id="23"/>' -> 2.5
    - Quoted numbers: '"-1.23"' -> -1.23
    - Scientific notation: '1.5e-3' -> 0.0015
    - Multiple numbers (returns first found)
    
    Args:
        text: Input string that may contain a float
        
    Returns:
        float: The first float found in the string
        
    Raises:
        ValueError: If no valid float is found in the string
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract float from empty string")
    
    # Step 1: Remove HTML/XML tags using regex
    html_cleaned = re.sub(r'<[^>]*>', '', text)
    
    # Step 2: Remove comma separators (1,234.56 -> 1234.56)
    comma_cleaned = re.sub(r'(\d),(?=\d)', r'\1', html_cleaned)
    
    # Step 3: Try to find floats using multiple strategies
    
    # Strategy 1: Look for floats with scientific notation
    scientific_pattern = r'-?\d+\.?\d*[eE][+-]?\d+'
    matches = re.findall(scientific_pattern, comma_cleaned)
    if matches:
        return float(matches[0])
    
    # Strategy 2: Look for quoted numbers (handle quotes around negative numbers)
    quoted_pattern = r'[\'\"]\s*(-?\d*\.?\d+(?:\.\d+)?)\s*[\'\"]+|[\'\"]+\s*(-?\d*\.?\d+(?:\.\d+)?)\s*[\'\"]*'
    matches = re.findall(quoted_pattern, comma_cleaned)
    for match_tuple in matches:
        for match in match_tuple:
            if match:  # Not empty string
                return float(match)
    
    # Strategy 3: Look for numbers starting with decimal point (.5 -> 0.5)
    decimal_start_pattern = r'(?<!\d)-?\.\d+(?!\d)'
    matches = re.findall(decimal_start_pattern, comma_cleaned)
    if matches:
        return float(matches[0])
    
    # Strategy 4: Look for regular decimal numbers with flexible boundaries
    decimal_pattern = r'(?<![.\d])-?\d+\.\d+(?![.\d])'
    matches = re.findall(decimal_pattern, comma_cleaned)
    if matches:
        return float(matches[0])
    
    # Strategy 5: Look for integers with flexible boundaries  
    integer_pattern = r'(?<![.\d])-?\d+(?![.\d])'
    matches = re.findall(integer_pattern, comma_cleaned)
    if matches:
        return float(matches[0])
    
    # Strategy 6: Very loose pattern - any number sequence
    loose_pattern = r'-?\d*\.?\d+'
    matches = re.findall(loose_pattern, comma_cleaned)
    if matches:
        valid_matches = [m for m in matches if m != '-' and len(m) > 0 and m != '.']
        if valid_matches:
            return float(valid_matches[0])
    
    # If all strategies fail, raise ValueError
    raise ValueError(f"No valid float found in string: '{text}'")

def extract_bool(text: str) -> bool:
    """Extract a boolean from a string using comprehensive patterns.
    
    Handles various formats:
    - Standard: 'true', 'false', '1', '0' (case insensitive)
    - Alternative words: 'yes', 'no', 'on', 'off', 'enable', 'disable'
    - Whitespace handling: strips whitespace before evaluation
    - Empty strings: return False
    
    Args:
        text: Input string that may represent a boolean
        
    Returns:
        bool: The boolean value represented by the string
    """
    if not text:
        return False
    
    # Strip whitespace and convert to lowercase
    cleaned = text.strip().lower()
    
    if not cleaned:
        return False
    
    # True values
    true_values = {
        "true", "1", "yes", "y", "on", "enable", "enabled", 
        "active", "ok", "okay", "positive", "affirm", "affirmative"
    }
    
    # False values  
    false_values = {
        "false", "0", "no", "n", "off", "disable", "disabled",
        "inactive", "negative", "deny", "none", "null", "nil"
    }
    
    if cleaned in true_values:
        return True
    if cleaned in false_values:
        return False
        
    # Fallback to Python's bool() function for other cases
    return bool(text)
