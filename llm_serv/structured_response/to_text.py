from datetime import date, datetime, time
from enum import Enum
from typing import Type, Union, get_args, get_origin

from pydantic import BaseModel


def response_to_xml(object: Type[BaseModel], exclude_fields: list[str] = []) -> str:
    """
    Converts a StructuredResponse class to XML format.

    Assuming the classes:

    class AnEnum(Enumeration):
        TYPE1 = "type1"
        TYPE2 = "type2"

    class SubClass(StructuredResponse):
        value: int = Field(description="A sub integer field")

    class TestStructuredResponse(StructuredResponse):
        a_string: str = Field(default="", description="A string field")
        a_string_none: Optional[str] = Field(default=None, description="An optional string field")
        a_int: int = Field(default=5, ge=0, le=10, description="An integer field with values between 0 and 10, default is 5")
        a_int_list: List[int] = Field(default=[1, 2, 3], description="A list of integers")
        a_enum: AnEnum = Field(default=AnEnum., description="An enum field with a custom description")
        a_float: float = Field(default=2.5, ge=0.0, le=5.0, description="A float field with values between 0.0 and 5.0, default is 2.5")
        a_float_list_optional: Optional[List[float]] = Field(default=None, description="An optional list of floats")
        a_optional_subclass: Optional[SubClass] = Field(default=None, description="An optional sub class field")
        a_date: date = Field(description="A date field including month from 2023")
        a_datetime: datetime = Field(description="A full date time field from 2023")
        a_time: time = Field(description="A time field from today")
        a_optional_list_of_subclass: Optional[List[SubClass]] = Field(default=None, description="An optional list of sub class fields")

    calling this will generate the following examples:

    <structured_response>
        <a_string type="string">[string]</a_string>
        <a_string_none type="string">[string]</a_string_none> <!-- if null or not applicable leave this element empty -->
        <a_int type="integer">[integer]</a_int>
        <a_int_list type="list"><!-- if null or not applicable leave this element empty -->
            <int_element type="integer">[integer]</int_element>
            ...
        </a_int_list>
        <a_enum type="enum">[One of: type1, type2]</a_enum>
        <a_float type="float">[float]</a_float>
        <a_float_list_optional type="list"><!-- if null or not applicable leave this element empty -->
            <a_float_list_optional_element type="float">[float]</a_float_list_optional_element>
            ...
        </a_float_list_optional>
        <a_optional_subclass type="class"> <!-- if null or not applicable leave this element empty -->
            <subclass>
                <value>[integer]</value>
            </subclass>
        </a_optional_subclass>
        <a_date type="date">[date]</a_date>
        <a_datetime type="datetime">[datetime]</a_datetime>
        <a_time type="time">[time]</a_time>
        <a_optional_list_of_subclass type="list"><!-- if null or not applicable leave this element empty -->
            <a_optional_list_of_subclass_element type="class">
                <subclass>
                    <value>[integer]</value>
                </subclass>
            </a_optional_list_of_subclass_element>
            ...
        </a_optional_list_of_subclass>
    </structured_response>

    Rules for XML generation:

    1. Basic Structure:
       - Each field gets its own XML tag using the field name: <field_name type="data type">...</field_name>
       - Indent each level with 4 spaces
       - Root class uses the class title as tag name, no type required as it's always a class, nested classes use their lowercase class name

    2. Optional Fields:
       - Add comment after opening tag: <!-- if null or not applicable leave this element empty -->
       - Comment applies to both simple fields and complex structures

    3. Basic Types:
       - Format as [type_name] between tags
       - Use 'integer' for int, 'string' for str, 'float' for float
       - Special handling for date/time (types):
         - date: [date]
         - time: [time]
         - datetime: [datetime]

    4. Enums:
       - Format as: [One of: value1, value2, ...]
       - Type is "enum"
       - Values are comma-separated
       - Use actual enum values, not names

    5. Lists:
       - Create wrapper tag using field name
       - Type in main list element is "list"
       - Each element uses <field_name_element> tag (append the "_element" suffix)
       - Add "..." after elements to indicate repetition
       - For basic types or list of lists:
           <field_name type="list">
               <field_name_element type="data type">[data type]</field_name_element>
               ...
           </field_name>
        - For class types:
            <field_name type="list">
               <field_name_element type="class">
                   <subclass>
                       [nested fields...]
                   </subclass>
               </field_name_element>
               ...
           </field_name>

    6. Nested StructuredResponse Classes:
       - Include full structure of nested class
       - Maintain proper indentation for nested elements
       - For single objects:
           <field_name type="class">
               <class_name>
                   [nested fields...]
               </class_name>
           </field_name>
       - For lists of objects:
           <field_name type="list">
               <field_name_element type="class">
                   <class_name>
                       [nested fields...]
                   </class_name>
               </field_name_element>
               ...
           </field_name>

    7. Type Combinations:
       - Optional Lists: Combine rules 2 and 5
       - Optional Nested Classes: Combine rules 2 and 6
       - Lists of Optional Items: Apply rule 2 to each element

    8. Tag Naming:
       - Use exact field names for main tags
       - For list elements append the "_element" suffix

    9. Comments and Whitespace:
       - Optional field comments go on same line as opening tag
       - Preserve empty lines between major sections
       - No extra whitespace within tags for basic types

    Other considerations:

    - the response_to_xml function is called recursively, so the indent_level is increased for each nested class
    - the exclude_fields parameter is used to exclude fields from the XML output
    - the field descriptions take into account the nested classes, so they are described first (each one once), with the main class description last.

    Format of the output:

    - we start with the formatting instructions on top (predefined)
    - we continue with the example xml section (recursive)
    - we end with the field descriptions section

    """

    def generate_instructions() -> list[str]:
        return [
            f"\nRespond without any other explanations or comments, prepended or appended to the <{object._title}> opening and closing tags. Pay attention that all fields are attended to, and properly enclosed within their own tags.\n Here is an example of the output format:\n"
        ]

    def generate_example_xml(
        object: Type[BaseModel], indent_level: int = 0, exclude_fields: list[str] = []
    ) -> list[str]:
        lines = []
        indent = "    " * indent_level
        
        # Use class title for root level, class name for nested levels
        if indent_level == 0:
            # Get the class title, fallback to class name if not set
            title = getattr(object, '_title', object.__name__.lower())
            if hasattr(title, 'default') and not isinstance(title, str):
                title = title.default
            # If title is the default "Structured Response", use class name instead
            if title == "Structured Response":
                title = object.__name__
            # Normalize the title using the same method as StructuredResponse
            from llm_serv.structured_response.model import StructuredResponse
            tag_name = StructuredResponse._convert_identifier_to_python_identifier(title)
        else:
            tag_name = object.__name__.lower()

        # Root response tag doesn't need type attribute
        lines.append(f"{indent}<{tag_name}>")

        for field_name, field_info in object.model_fields.items():
            if field_name in exclude_fields:
                continue

            field_type = field_info.annotation

            is_optional = get_origin(field_type) is Union and type(None) in get_args(field_type)
            if is_optional:
                field_type = next(arg for arg in get_args(field_type) if arg is not type(None))

            # Rule 5: Lists handling
            if get_origin(field_type) is list:
                field_start = f'{indent}    <{field_name} type="list">'
                if is_optional:
                    field_start += "<!-- if null or not applicable leave this element empty -->"
                lines.append(field_start)

                element_type = get_args(field_type)[0]
                if isinstance(element_type, type) and issubclass(element_type, BaseModel):
                    lines.append(f'{indent}        <{field_name}_element type="class">')
                    lines.extend(generate_example_xml(element_type, indent_level + 3, exclude_fields))
                    lines.append(f"{indent}        </{field_name}_element>")
                    lines.append(f"{indent}        ...")
                else:
                    type_name = "integer" if element_type is int else element_type.__name__.lower()
                    lines.append(
                        f'{indent}        <{field_name}_element type="{type_name}">[{type_name}]</{field_name}_element>'
                    )
                    lines.append(f"{indent}        ...")
                lines.append(f"{indent}    </{field_name}>")

            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_start = f'{indent}    <{field_name} type="class">'
                if is_optional:
                    field_start += "<!-- if null or not applicable leave this element empty -->"
                lines.append(field_start)
                lines.extend(generate_example_xml(field_type, indent_level + 2, exclude_fields))
                lines.append(f"{indent}    </{field_name}>")

            else:
                # Rule 3: Basic types on single line
                if field_type in (date, datetime, time):
                    type_name = field_type.__name__.lower()
                elif isinstance(field_type, type) and issubclass(field_type, Enum):
                    type_name = "enum"
                    values = [str(e.value) for e in field_type]
                    value_text = f"One of: {', '.join(values)}"
                else:
                    type_name = (
                        "integer"
                        if field_type is int
                        else (
                            "string"
                            if field_type is str
                            else "float" if field_type is float else (field_type.__name__.lower() if field_type else "unknown")
                        )
                    )
                    value_text = type_name

                line = f'{indent}    <{field_name} type="{type_name}">'
                line += f"[{value_text}]</{field_name}>"
                if is_optional:
                    line += "<!-- if null or not applicable leave this element empty -->"
                lines.append(line)

        lines.append(f"{indent}</{tag_name}>")
        return lines

    def generate_field_descriptions(object: Type[BaseModel], exclude_fields: list[str] = []) -> list[str]:
        all_descriptions = []
        described_classes = set()  # Track described classes to avoid duplicates

        def collect_nested_descriptions(cls):
            # Skip if already described (handles circular references)
            if cls.__name__.lower() in described_classes:
                return

            described_classes.add(cls.__name__.lower())

            # Add descriptions for this class's fields
            all_descriptions.append(
                f"\nHere is the description for each field for the <{cls.__name__.lower()}> element:"
            )
            for field_name, field_info in cls.model_fields.items():
                if field_name not in exclude_fields:
                    all_descriptions.append(_get_field_description(field_name, field_info))

            # Process nested classes (Rule 7: Type Combinations)
            for field_name, field_info in cls.model_fields.items():
                if field_name in exclude_fields:
                    continue

                field_type = field_info.annotation

                # Handle Optional types
                if get_origin(field_type) is Union and type(None) in get_args(field_type):
                    field_type = next(arg for arg in get_args(field_type) if arg is not type(None))

                # Handle Lists of nested classes
                if get_origin(field_type) is list:
                    element_type = get_args(field_type)[0]
                    if isinstance(element_type, type) and issubclass(element_type, BaseModel):
                        collect_nested_descriptions(element_type)

                # Handle direct nested classes
                elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                    collect_nested_descriptions(field_type)

        # First describe nested classes (maintains proper order)
        for field_name, field_info in object.model_fields.items():
            if field_name not in exclude_fields:
                current_field_type = field_info.annotation

                # Unwrap Optional if present
                if get_origin(current_field_type) is Union and type(None) in get_args(current_field_type):
                    current_field_type = next(arg for arg in get_args(current_field_type) if arg is not type(None))

                # Check if the field is a list of BaseModel subclasses
                if get_origin(current_field_type) is list:
                    list_element_type = get_args(current_field_type)[0]
                    if isinstance(list_element_type, type) and issubclass(list_element_type, BaseModel):
                        collect_nested_descriptions(list_element_type)
                # Check if the field is a direct BaseModel subclass
                elif isinstance(current_field_type, type) and issubclass(current_field_type, BaseModel):
                    collect_nested_descriptions(current_field_type)

        # Then describe root class
        root_tag_name = getattr(object, '_title', object.__name__.lower())
        all_descriptions.append(f"\nHere is the description for each field for the <{root_tag_name}> main element:")
        for field_name, field_info in object.model_fields.items():
            if field_name not in exclude_fields:
                all_descriptions.append(_get_field_description(field_name, field_info))

        return all_descriptions

    def _get_field_description(field_name: str, field_info) -> str:
        field_type = field_info.annotation
        field_instr = f"\n{field_name}:"

        # Check if field is Optional
        is_optional = False
        if get_origin(field_type) is Union and type(None) in get_args(field_type):
            is_optional = True
            # Get the actual type from Optional
            field_type = next(arg for arg in get_args(field_type) if arg != type(None))

        # Handle date/time types with special formatting instructions
        if field_type in (date, datetime, time):
            field_instr += f"\n  - Type: {field_type.__name__}"
            if field_type is date:
                field_instr += "\n  - Format: Use clear date format (e.g., 'YYYY-MM-DD', 'January 1, 2023', etc.) if possible, but only from the available data, without inferring missing years, months or days."
            elif field_type is time:
                field_instr += "\n  - Format: Use clear time format (e.g., 'HH:MM:SS', '3:45 PM', etc.) if possible, but only from the available data, without inferring missing hours, minutes or seconds."
            elif field_type is datetime:
                field_instr += "\n  - Format: Use clear datetime format (e.g., 'YYYY-MM-DD HH:MM:SS', 'January 1, 2023 3:45 PM', etc.) if possible, but only from the available data, without inferring missing years, months, days, hours, minutes or seconds."
        # Handle Union types
        elif get_origin(field_type) is Union:
            args = get_args(field_type)
            types = []
            for arg in args:
                if arg is type(None):
                    types.append("null")
                elif isinstance(arg, type) and issubclass(arg, BaseModel):
                    types.append("a group containing further sub fields")
                else:
                    types.append(getattr(arg, "__name__", str(arg)))
            field_instr += f"\n  - Type: Union of {' OR '.join(types)}"
        # Handle nested StructuredResponse
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            field_instr += f"\n  - Type: a group containing further sub fields"
        else:
            field_instr += f"\n  - Type: {getattr(field_type, '__name__', str(field_type))}"

        field_instr += f"\n  - {'Optional' if is_optional else 'Required'} field"

        if field_info.description:
            field_instr += f"\n  - Description: {field_info.description}"

        # Add constraints checking
        if hasattr(field_info, "metadata"):
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    field_instr += f"\n  - Minimum Value (inclusive): {constraint.ge}"
                elif hasattr(constraint, "gt"):
                    field_instr += f"\n  - Minimum Value (exclusive): {constraint.gt}"
                elif hasattr(constraint, "le"):
                    field_instr += f"\n  - Maximum Value (inclusive): {constraint.le}"
                elif hasattr(constraint, "lt"):
                    field_instr += f"\n  - Maximum Value (exclusive): {constraint.lt}"
                elif hasattr(constraint, "multiple_of"):
                    field_instr += f"\n  - Must be a multiple of: {constraint.multiple_of}"
                elif hasattr(constraint, "max_digits"):
                    field_instr += f"\n  - Maximum digits: {constraint.max_digits}"
                elif hasattr(constraint, "decimal_places"):
                    field_instr += f"\n  - Decimal places: {constraint.decimal_places}"
                elif hasattr(constraint, "min_length"):
                    field_instr += f"\n  - Minimum length: {constraint.min_length}"
                elif hasattr(constraint, "max_length"):
                    field_instr += f"\n  - Maximum length: {constraint.max_length}"
                elif hasattr(constraint, "regex"):
                    field_instr += f"\n  - Must match regex: {constraint.regex.pattern}"

        # Handle Enums
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            values = [str(e.value) for e in field_type]
            field_instr += f"\n  - Type: {field_type.__name__}"
            field_instr += f"\n  - Allowed values: {', '.join(values)}"
            # field_instr += f"\n  - It is always enclosed between <{field_name}> open and </{field_name}> closing tags."
            return field_instr

        # field_instr += f"\n  - It is always enclosed between <{field_name}> open and </{field_name}> closing tags."

        return field_instr

    instructions = []

    instructions.extend(generate_instructions())
    instructions.extend(generate_example_xml(object=object, exclude_fields=exclude_fields))
    instructions.extend(generate_field_descriptions(object=object, exclude_fields=exclude_fields))

    return "\n".join(instructions)


def instance_to_xml(instance: BaseModel, exclude_none: bool = False, exclude: set[str] | None = None, indent_level: int = 0) -> str:
    """
    Converts a StructuredResponse instance to XML format with actual values.
    
    Args:
        instance: The StructuredResponse instance to convert
        exclude_none: Whether to exclude fields with None values
        exclude: Set of field names to exclude from the output
        indent_level: Current indentation level for nested structures
        
    Returns:
        XML string representation of the instance
    """
    if exclude is None:
        exclude = set()
        
    lines = []
    indent = "    " * indent_level
    
    # Use class title for root level, class name for nested levels
    if indent_level == 0:
        title = getattr(instance.__class__, '_title', instance.__class__.__name__.lower())
        if hasattr(title, 'default') and not isinstance(title, str):
            title = title.default
        # If title is the default "Structured Response", use class name instead
        if title == "Structured Response":
            title = instance.__class__.__name__
        # Normalize the title using the same method as StructuredResponse
        from llm_serv.structured_response.model import StructuredResponse
        tag_name = StructuredResponse._convert_identifier_to_python_identifier(title)
    else:
        tag_name = instance.__class__.__name__.lower()
    
    lines.append(f"{indent}<{tag_name}>")
    
    # Iterate over model fields to preserve original types
    for field_name, field_info in instance.__class__.model_fields.items():
        # Skip excluded fields
        if field_name in exclude:
            continue
            
        # Get the actual field value from the instance
        field_value = getattr(instance, field_name, None)
        
        if field_value is None and exclude_none:
            continue
            
        field_type = field_info.annotation
        
        # Handle Optional types
        is_optional = get_origin(field_type) is Union and type(None) in get_args(field_type)
        if is_optional:
            field_type = next(arg for arg in get_args(field_type) if arg is not type(None))
        
        # Convert field value to XML
        field_xml = _convert_field_to_xml(field_name, field_value, field_type, indent_level + 1, exclude_none, exclude)
        lines.extend(field_xml)
    
    lines.append(f"{indent}</{tag_name}>")
    return "\n".join(lines)


def _convert_field_to_xml(field_name: str, field_value, field_type, indent_level: int, exclude_none: bool, exclude: set[str]) -> list[str]:
    """Helper function to convert a single field to XML format."""
    lines = []
    indent = "    " * indent_level
    
    if field_value is None:
        if not exclude_none:
            lines.append(f"{indent}<{field_name}></{field_name}>")
        return lines
    
    # Handle lists
    if get_origin(field_type) is list:
        lines.append(f'{indent}<{field_name} type="list">')
        if isinstance(field_value, list):
            element_type = get_args(field_type)[0]
            for item in field_value:
                if isinstance(item, BaseModel):
                    lines.append(f'{indent}    <{field_name}_element type="class">')
                    lines.append(instance_to_xml(item, exclude_none, exclude, indent_level + 2))
                    lines.append(f'{indent}    </{field_name}_element>')
                else:
                    type_name = _get_xml_type_name(type(item))
                    lines.append(f'{indent}    <{field_name}_element type="{type_name}">{_format_basic_value(item)}</{field_name}_element>')
        lines.append(f"{indent}</{field_name}>")
    
    # Handle nested BaseModel instances
    elif isinstance(field_value, BaseModel):
        lines.append(f'{indent}<{field_name} type="class">')
        lines.append(instance_to_xml(field_value, exclude_none, exclude, indent_level + 1))
        lines.append(f"{indent}</{field_name}>")
    
    # Handle basic types
    else:
        type_name = _get_xml_type_name(type(field_value))
        formatted_value = _format_basic_value(field_value)
        lines.append(f'{indent}<{field_name} type="{type_name}">{formatted_value}</{field_name}>')
    
    return lines


def _get_xml_type_name(python_type) -> str:
    """Convert Python type to XML type name."""
    if python_type is int:
        return "integer"
    elif python_type is str:
        return "string"
    elif python_type is float:
        return "float"
    elif python_type is bool:
        return "boolean"
    elif python_type in (date, datetime, time):
        return python_type.__name__.lower()
    elif issubclass(python_type, Enum):
        return "enum"
    else:
        return python_type.__name__.lower()


def _format_basic_value(value) -> str:
    """Format a basic value for XML output."""
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (date, datetime, time)):
        return str(value)
    elif isinstance(value, Enum):
        # For enums, show both possible values and selected value
        enum_class = value.__class__
        possible_values = [str(e.value) for e in enum_class]
        return f"[One of: {', '.join(possible_values)}] - Selected: {value.value}"
    else:
        return str(value)
