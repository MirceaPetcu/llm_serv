#!/usr/bin/env python3
"""
Comprehensive tests for to_prompt() method covering all edge cases and type combinations.
Tests are designed to ensure full compliance with README specification.
"""

import sys
import os
import enum

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from llm_serv.structured_response.model import StructuredResponse


class TestEnums:
    """Test enum classes for various scenarios."""
    
    class SimpleEnum(str, enum.Enum):
        OPTION_A = "option_a"
        OPTION_B = "option_b"
        OPTION_C = "option_c"
    
    class NumberEnum(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    class StatusEnum(str, enum.Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"


def test_basic_types():
    """Test all basic types with various constraints."""
    print("=== TESTING BASIC TYPES ===")
    
    sr = StructuredResponse()
    sr.class_name = "BasicTypesTest"
    
    # String types with constraints
    sr.add_node("simple_string", str, description="A simple string field")
    sr.add_node("constrained_string", str, description="A string with length constraints", min_length=5, max_length=50)
    sr.add_node("empty_desc_string", str)
    
    # Numeric types with constraints
    sr.add_node("simple_int", int, description="A simple integer field")
    sr.add_node("constrained_int", int, description="An integer with range constraints", ge=0, le=100)
    sr.add_node("negative_range_int", int, description="An integer with negative range", ge=-50, lt=0)
    
    sr.add_node("simple_float", float, description="A simple float field")
    sr.add_node("constrained_float", float, description="A float with range constraints", gt=0.0, le=1.0)
    sr.add_node("temperature_float", float, description="Temperature in Celsius", ge=-273.15, le=1000.0)
    
    # Boolean type
    sr.add_node("simple_bool", bool, description="A simple boolean field")
    sr.add_node("empty_desc_bool", bool)
    
    # Enum type
    sr.add_node("simple_enum", enum, description="A simple enum field", choices=TestEnums.SimpleEnum)
    sr.add_node("status_enum", enum, description="Status selection", choices=TestEnums.StatusEnum)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify key aspects
    assert "<basic_types_test>" in result
    assert "type='str'" in result
    assert "type='int'" in result  
    assert "type='float'" in result
    assert "type='bool'" in result
    assert "type='enum'" in result
    assert "choices='[\"option_a\", \"option_b\", \"option_c\"]'" in result
    assert "greater_or_equal='0'" in result
    assert "less_or_equal='100'" in result
    assert "min_length='5'" in result
    assert "max_length='50'" in result
    assert "[A simple string field - as a string]" in result
    assert "[A simple integer field - as a int]" in result
    assert "[A simple float field - as a float]" in result
    assert "[A simple boolean field - as a bool]" in result
    assert "[A simple enum field - as an enum]" in result
    
    print("✓ Basic types test passed")


def test_simple_lists():
    """Test lists with simple element types."""
    print("=== TESTING SIMPLE LISTS ===")
    
    sr = StructuredResponse()
    sr.class_name = "SimpleListsTest"
    
    # Lists with basic types
    sr.add_node("string_list", list, description="List of strings", elements=str)
    sr.add_node("int_list", list, description="List of integers", elements=int)
    sr.add_node("float_list", list, description="List of floats", elements=float)
    sr.add_node("bool_list", list, description="List of booleans", elements=bool)
    sr.add_node("enum_list", list, description="List of enums", elements=enum)
    
    # Lists with constraints
    sr.add_node("constrained_list", list, description="List with constraints", elements=int, min_length=1, max_length=10)
    
    # Empty description list
    sr.add_node("no_desc_list", list, elements=str)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify list structure
    assert "type='list'" in result
    assert "elements='str'" in result
    assert "elements='int'" in result
    assert "elements='float'" in result
    assert "elements='bool'" in result
    assert "elements='enum'" in result
    assert "<li index='0'>" in result
    assert "[value here - as an str]" in result
    assert "[value here - as an int]" in result
    assert "[value here - as an float]" in result
    assert "[value here - as an bool]" in result
    assert "[value here - as an enum]" in result
    assert "min_length='1'" in result
    assert "max_length='10'" in result
    
    print("✓ Simple lists test passed")


def test_dict_types():
    """Test dict types with nested structures."""
    print("=== TESTING DICT TYPES ===")
    
    sr = StructuredResponse()
    sr.class_name = "DictTypesTest"
    
    # Simple dict
    sr.add_node("simple_dict", dict, description="A simple dictionary")
    sr.add_node("simple_dict.name", str, description="Name field")
    sr.add_node("simple_dict.value", int, description="Value field")
    
    # Dict with enum
    sr.add_node("dict_with_enum", dict, description="Dict containing enum")
    sr.add_node("dict_with_enum.status", enum, description="Status field", choices=TestEnums.StatusEnum)
    sr.add_node("dict_with_enum.count", int, description="Count field", ge=0)
    
    # Nested dicts
    sr.add_node("nested_dict", dict, description="Dict with nested dict")
    sr.add_node("nested_dict.outer_field", str, description="Outer field")
    sr.add_node("nested_dict.inner_dict", dict, description="Inner dictionary")
    sr.add_node("nested_dict.inner_dict.inner_field", str, description="Inner field")
    sr.add_node("nested_dict.inner_dict.inner_number", float, description="Inner number", gt=0.0)
    
    # Empty description dict
    sr.add_node("no_desc_dict", dict)
    sr.add_node("no_desc_dict.field", str, description="Field in no-desc dict")
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify dict structure
    assert "type='dict'" in result
    assert "description='A simple dictionary'" in result
    assert "description='Dict containing enum'" in result
    assert "description='Dict with nested dict'" in result
    assert "<simple_dict type='dict'" in result
    assert "<nested_dict type='dict'" in result
    assert "<inner_dict type='dict'" in result
    assert "choices='[\"active\", \"inactive\"]'" in result
    assert "greater_than='0.0'" in result
    
    print("✓ Dict types test passed")


def test_complex_lists():
    """Test lists with complex (dict) elements."""
    print("=== TESTING COMPLEX LISTS ===")
    
    sr = StructuredResponse()
    sr.class_name = "ComplexListsTest"
    
    # List of dicts
    sr.add_node("people_list", list, description="List of people", elements=dict)
    sr.add_node("people_list.name", str, description="Person name")
    sr.add_node("people_list.age", int, description="Person age", ge=0, le=150)
    sr.add_node("people_list.status", enum, description="Person status", choices=TestEnums.StatusEnum)
    
    # List with nested dicts
    sr.add_node("complex_list", list, description="List with nested structures", elements=dict)
    sr.add_node("complex_list.id", int, description="Item ID")
    sr.add_node("complex_list.details", dict, description="Item details")
    sr.add_node("complex_list.details.name", str, description="Detail name")
    sr.add_node("complex_list.details.metadata", dict, description="Metadata object")
    sr.add_node("complex_list.details.metadata.created", str, description="Creation timestamp")
    sr.add_node("complex_list.details.metadata.version", float, description="Version number")
    
    # List with constraints
    sr.add_node("constrained_complex_list", list, 
                description="Constrained list", elements=dict, min_length=1)
    sr.add_node("constrained_complex_list.required_field", str, description="Required field", min_length=1)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify complex list structure
    assert "elements='dict'" in result
    assert "<li index='0'>" in result
    assert "description='List of people'" in result
    assert "description='List with nested structures'" in result
    assert "[Person name - as a string]" in result
    assert "[Person age - as a int]" in result
    assert "[Person status - as an enum]" in result
    assert "greater_or_equal='0'" in result
    assert "less_or_equal='150'" in result
    assert "choices='[\"active\", \"inactive\"]'" in result
    
    print("✓ Complex lists test passed")


def test_nested_combinations():
    """Test deeply nested combinations of lists and dicts."""
    print("=== TESTING NESTED COMBINATIONS ===")
    
    sr = StructuredResponse()
    sr.class_name = "NestedCombinationsTest"
    
    # Dict containing lists
    sr.add_node("dict_with_lists", dict, description="Dict containing various lists")
    sr.add_node("dict_with_lists.simple_numbers", list, description="Simple number list", elements=int)
    sr.add_node("dict_with_lists.complex_items", list, description="Complex item list", elements=dict)
    sr.add_node("dict_with_lists.complex_items.item_name", str, description="Item name")
    sr.add_node("dict_with_lists.complex_items.item_value", float, description="Item value")
    
    # List containing dicts with lists
    sr.add_node("list_of_dict_with_lists", list, description="List of dicts containing lists", elements=dict)
    sr.add_node("list_of_dict_with_lists.category", str, description="Category name")
    sr.add_node("list_of_dict_with_lists.tags", list, description="Tag list", elements=str)
    sr.add_node("list_of_dict_with_lists.properties", dict, description="Properties object")
    sr.add_node("list_of_dict_with_lists.properties.enabled", bool, description="Is enabled")
    sr.add_node("list_of_dict_with_lists.properties.settings", list, description="Settings list", elements=dict)
    sr.add_node("list_of_dict_with_lists.properties.settings.key", str, description="Setting key")
    sr.add_node("list_of_dict_with_lists.properties.settings.value", str, description="Setting value")
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify nested structure
    assert "dict_with_lists" in result
    assert "list_of_dict_with_lists" in result
    assert "elements='int'" in result
    assert "elements='dict'" in result
    assert "elements='str'" in result
    
    print("✓ Nested combinations test passed")


def test_all_constraints():
    """Test all supported constraint types."""
    print("=== TESTING ALL CONSTRAINTS ===")
    
    sr = StructuredResponse()
    sr.class_name = "ConstraintsTest"
    
    # Numeric constraints
    sr.add_node("ge_field", int, description="Greater or equal field", ge=10)
    sr.add_node("gt_field", float, description="Greater than field", gt=0.0)
    sr.add_node("le_field", int, description="Less or equal field", le=100)
    sr.add_node("lt_field", float, description="Less than field", lt=1.0)
    sr.add_node("multiple_of_field", int, description="Multiple of field", multiple_of=5)
    sr.add_node("range_field", float, description="Range field", ge=-10.5, le=10.5)
    
    # String constraints
    sr.add_node("min_length_field", str, description="Min length field", min_length=3)
    sr.add_node("max_length_field", str, description="Max length field", max_length=20)
    sr.add_node("length_range_field", str, description="Length range field", min_length=5, max_length=15)
    
    # List constraints
    sr.add_node("constrained_list", list, description="Constrained list", elements=str, min_length=1, max_length=5)
    
    # Combined constraints
    sr.add_node("complex_constraints", float, description="Complex constraints", ge=0.0, lt=100.0, multiple_of=0.1)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify all constraint attributes
    assert "greater_or_equal='10'" in result
    assert "greater_than='0.0'" in result
    assert "less_or_equal='100'" in result
    assert "less_than='1.0'" in result
    assert "multiple_of='5'" in result
    assert "min_length='3'" in result
    assert "max_length='20'" in result
    assert "min_length='5'" in result and "max_length='15'" in result
    assert "greater_or_equal='0.0'" in result and "less_than='100.0'" in result
    assert "multiple_of='0.1'" in result
    
    print("✓ All constraints test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=== TESTING EDGE CASES ===")
    
    # Empty definition
    sr_empty = StructuredResponse()
    sr_empty.class_name = "EmptyTest"
    try:
        sr_empty.to_prompt()
        raise AssertionError("Should raise ValueError for empty definition")
    except ValueError as e:
        assert "Definition not initialized" in str(e)
        print("✓ Empty definition correctly raises ValueError")
    
    # Single field
    sr_single = StructuredResponse()
    sr_single.class_name = "SingleFieldTest"
    sr_single.add_node("only_field", str, description="The only field")
    result = sr_single.to_prompt()
    assert "<single_field_test>" in result
    assert "[The only field - as a string]" in result
    print("✓ Single field test passed")
    
    # Class name conversion
    sr_camel = StructuredResponse()
    sr_camel.class_name = "CamelCaseClassName"
    sr_camel.add_node("test_field", str, description="Test field")
    result = sr_camel.to_prompt()
    assert "<camel_case_class_name>" in result
    print("✓ CamelCase to snake_case conversion test passed")
    
    # No description fields
    sr_no_desc = StructuredResponse()
    sr_no_desc.class_name = "NoDescTest"
    sr_no_desc.add_node("no_desc_str", str)
    sr_no_desc.add_node("no_desc_int", int)
    sr_no_desc.add_node("no_desc_bool", bool)
    sr_no_desc.add_node("no_desc_enum", enum, choices=TestEnums.SimpleEnum)
    result = sr_no_desc.to_prompt()
    assert "[value here - as a string]" in result
    assert "[value here - as a int]" in result
    assert "[value here - as a bool]" in result
    assert "[value here - as an enum]" in result
    print("✓ No description test passed")
    
    print()


def test_readme_weather_example():
    """Test the exact weather example from README to ensure compliance."""
    print("=== TESTING README WEATHER EXAMPLE ===")
    
    sr = StructuredResponse()
    sr.class_name = "WeatherPrognosis"
    
    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    # Build the exact structure from README
    sr.add_node("location", str, description="The location of the weather forecast")
    sr.add_node("current_temperature", float, description="The current temperature in degrees Celsius")
    
    # overall_rain_prob dict
    sr.add_node("overall_rain_prob", dict, description="The day's rain chance")
    sr.add_node("overall_rain_prob.chance", enum, 
                description="The chance of rain, where low is less than 25% and high is more than 75%",
                choices=ChanceScale)
    sr.add_node("overall_rain_prob.when", str, 
                description="The time of day when the rain is or is not expected")
    
    # rain_probability_timebound list
    sr.add_node("rain_probability_timebound", list,
                description="List of chances of rain, where low is less than 25% and high is more than 75%",
                elements=dict)
    sr.add_node("rain_probability_timebound.chance", enum,
                description="The chance of rain, where low is less than 25% and high is more than 75%",
                choices=ChanceScale)
    sr.add_node("rain_probability_timebound.when", str,
                description="The time of day when the rain is or is not expected")
    
    # Simple list
    sr.add_node("hourly_index", list, description="List of hourly UV index in the range of 1-10", elements=int)
    
    # More fields
    sr.add_node("wind_speed", float, description="The wind speed in km/h")
    sr.add_node("high", float, description="The high temperature in degrees Celsius", ge=-20, le=60)
    sr.add_node("low", float, description="The low temperature in degrees Celsius")
    sr.add_node("storm_tonight", bool, description="Whether there will be a storm tonight")
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify against README specification
    expected_patterns = [
        "<weather_prognosis>",
        "<location type='str'>[The location of the weather forecast - as a string]</location>",
        "<current_temperature type='float'>[The current temperature in degrees Celsius - as a float]</current_temperature>",
        "<overall_rain_prob type='dict' description='The day's rain chance'>",
        ("<chance type='enum' choices='[\"low\", \"medium\", \"high\"]'>"
         "[The chance of rain, where low is less than 25% and high is more than 75% - as an enum]</chance>"),
        "<when type='str'>[The time of day when the rain is or is not expected - as a string]</when>",
        "</overall_rain_prob>",
        ("<rain_probability_timebound type='list' elements='dict' "
         "description='List of chances of rain, where low is less than 25% and high is more than 75%'>"),
        "<li index='0'>",
        "<hourly_index type='list' elements='int' description='List of hourly UV index in the range of 1-10'>",
        "[value here - as an int]",
        "<wind_speed type='float'>[The wind speed in km/h - as a float]</wind_speed>",
        "<high type='float' greater_or_equal='-20' less_or_equal='60'>[The high temperature in degrees Celsius - as a float]</high>",
        "<storm_tonight type='bool'>[Whether there will be a storm tonight - as a bool]</storm_tonight>",
        "</weather_prognosis>"
    ]
    
    for pattern in expected_patterns:
        assert pattern in result, f"Missing expected pattern: {pattern}"
    
    print("✓ README weather example test passed")


def test_extreme_nesting():
    """Test extreme nesting scenarios."""
    print("=== TESTING EXTREME NESTING ===")
    
    sr = StructuredResponse()
    sr.class_name = "ExtremeNestingTest"
    
    # List of dicts containing lists of dicts
    sr.add_node("deep_structure", list, description="Deeply nested structure", elements=dict)
    sr.add_node("deep_structure.level1_field", str, description="Level 1 field")
    sr.add_node("deep_structure.level1_dict", dict, description="Level 1 dict")
    sr.add_node("deep_structure.level1_dict.level2_field", str, description="Level 2 field")
    sr.add_node("deep_structure.level1_dict.level2_list", list, description="Level 2 list", elements=dict)
    sr.add_node("deep_structure.level1_dict.level2_list.level3_field", str, description="Level 3 field")
    sr.add_node("deep_structure.level1_dict.level2_list.level3_enum", enum, description="Level 3 enum", choices=TestEnums.NumberEnum)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify deep nesting structure
    assert "deep_structure" in result
    assert "level1_dict" in result
    assert "level2_list" in result
    assert "level3_field" in result
    assert "level3_enum" in result
    assert "choices='[\"low\", \"medium\", \"high\"]'" in result
    
    print("✓ Extreme nesting test passed")


def test_special_characters_and_formatting():
    """Test handling of special characters and formatting edge cases."""
    print("=== TESTING SPECIAL CHARACTERS AND FORMATTING ===")
    
    sr = StructuredResponse()
    sr.class_name = "SpecialCharsTest"
    
    # Descriptions with special characters
    sr.add_node("quotes_field", str, description="Field with 'single' and \"double\" quotes")
    sr.add_node("ampersand_field", str, description="Field with & ampersand")
    sr.add_node("brackets_field", str, description="Field with [brackets] and <tags>")
    sr.add_node("unicode_field", str, description="Field with unicode: café, naïve, résumé")
    
    # Long descriptions
    sr.add_node("long_desc_field", str, 
                description="This is a very long description that spans multiple concepts and includes "
                           "various details about the field purpose, usage, constraints, and expected "
                           "behavior in different scenarios")
    
    # Empty and whitespace descriptions
    sr.add_node("empty_desc_field", str, description="")
    sr.add_node("whitespace_desc_field", str, description="   ")
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify special character handling
    assert "Field with 'single' and \"double\" quotes" in result
    assert "Field with & ampersand" in result
    assert "Field with [brackets] and <tags>" in result
    assert "café, naïve, résumé" in result
    
    print("✓ Special characters and formatting test passed")


def test_type_name_mappings():
    """Test correct type name mappings in descriptions."""
    print("=== TESTING TYPE NAME MAPPINGS ===")
    
    sr = StructuredResponse()
    sr.class_name = "TypeMappingsTest"
    
    sr.add_node("str_field", str, description="String field")
    sr.add_node("int_field", int, description="Integer field") 
    sr.add_node("float_field", float, description="Float field")
    sr.add_node("bool_field", bool, description="Boolean field")
    sr.add_node("enum_field", enum, description="Enum field", choices=TestEnums.SimpleEnum)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify type name mappings in descriptions
    assert "[String field - as a string]" in result  # str -> string
    assert "[Integer field - as a int]" in result    # int -> int
    assert "[Float field - as a float]" in result    # float -> float  
    assert "[Boolean field - as a bool]" in result   # bool -> bool
    assert "[Enum field - as an enum]" in result     # enum -> enum
    
    print("✓ Type name mappings test passed")


def test_complex_enum_scenarios():
    """Test complex enum scenarios and edge cases."""
    print("=== TESTING COMPLEX ENUM SCENARIOS ===")
    
    class SingleValueEnum(str, enum.Enum):
        ONLY_OPTION = "only_option"
    
    class LongNameEnum(str, enum.Enum):
        VERY_LONG_OPTION_NAME_WITH_UNDERSCORES = "very_long_option_name_with_underscores"
        ANOTHER_EXTREMELY_LONG_OPTION = "another_extremely_long_option"
    
    class SpecialCharsEnum(str, enum.Enum):
        OPTION_WITH_SPACES = "option with spaces"
        OPTION_WITH_QUOTES = "option's \"quoted\" value"
        OPTION_WITH_SYMBOLS = "option@#$%^&*()"
    
    sr = StructuredResponse()
    sr.class_name = "ComplexEnumTest"
    
    sr.add_node("single_enum", enum, description="Enum with single value", choices=SingleValueEnum)
    sr.add_node("long_name_enum", enum, description="Enum with long names", choices=LongNameEnum)
    sr.add_node("special_chars_enum", enum, description="Enum with special characters", choices=SpecialCharsEnum)
    
    # Enum in list
    sr.add_node("enum_list", list, description="List of special enums", elements=enum)
    
    # Enum in dict
    sr.add_node("dict_with_enum", dict, description="Dict with enum field")
    sr.add_node("dict_with_enum.special_enum", enum, description="Special enum field", choices=SpecialCharsEnum)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify enum handling
    assert "choices='[\"only_option\"]'" in result
    assert "very_long_option_name_with_underscores" in result
    assert "option with spaces" in result
    assert "option's \\\"quoted\\\" value" in result or "option's \"quoted\" value" in result
    
    print("✓ Complex enum scenarios test passed")


def test_empty_and_minimal_structures():
    """Test minimal and empty structure scenarios."""
    print("=== TESTING EMPTY AND MINIMAL STRUCTURES ===")
    
    # Minimal single field
    sr_minimal = StructuredResponse()
    sr_minimal.class_name = "MinimalTest"
    sr_minimal.add_node("field", str)
    result = sr_minimal.to_prompt()
    assert "<minimal_test>" in result
    assert "[value here - as a string]" in result
    print("✓ Minimal structure test passed")
    
    # Empty list
    sr_empty_list = StructuredResponse()
    sr_empty_list.class_name = "EmptyListTest"
    sr_empty_list.add_node("empty_list", list, elements=str)
    result = sr_empty_list.to_prompt()
    assert "elements='str'" in result
    assert "<li index='0'>" in result
    print("✓ Empty list test passed")
    
    # Empty dict
    sr_empty_dict = StructuredResponse()
    sr_empty_dict.class_name = "EmptyDictTest"
    sr_empty_dict.add_node("empty_dict", dict)
    result = sr_empty_dict.to_prompt()
    assert "type='dict'" in result
    print("✓ Empty dict test passed")
    
    print()


def test_class_name_variations():
    """Test various class name formats and their snake_case conversion."""
    print("=== TESTING CLASS NAME VARIATIONS ===")
    
    test_cases = [
        ("SimpleClass", "simple_class"),
        ("XMLHttpRequest", "xml_http_request"),
        ("HTMLParser", "html_parser"),
        ("APIResponse", "api_response"),
        ("WeatherPrognosis", "weather_prognosis"),
        ("IOError", "io_error"),
        ("HTTPSConnection", "https_connection"),
        ("single", "single"),
        ("ALLCAPS", "allcaps"),
        ("mixedCASEExample", "mixed_case_example")
    ]
    
    for class_name, expected_tag in test_cases:
        sr = StructuredResponse()
        sr.class_name = class_name
        sr.add_node("test_field", str, description="Test field")
        result = sr.to_prompt()
        assert f"<{expected_tag}>" in result, (
            f"Expected <{expected_tag}> for class {class_name}, "
            f"got result: {result[:100]}..."
        )
        assert f"</{expected_tag}>" in result
    
    print("✓ Class name variations test passed")


def test_attribute_ordering_and_formatting():
    """Test attribute ordering and formatting consistency."""
    print("=== TESTING ATTRIBUTE ORDERING AND FORMATTING ===")
    
    sr = StructuredResponse()
    sr.class_name = "AttributeOrderTest"
    
    # Field with multiple attributes to test ordering
    sr.add_node("multi_attr_field", str, 
                description="Field with multiple attributes",
                min_length=5, max_length=100)
    
    sr.add_node("numeric_multi_attr", float,
                description="Numeric field with constraints", 
                ge=-100.0, le=100.0, multiple_of=0.5)
    
    sr.add_node("enum_with_desc", enum,
                description="Enum with description",
                choices=TestEnums.NumberEnum)
    
    sr.add_node("list_multi_attr", list,
                description="List with multiple attributes",
                elements=int, min_length=1, max_length=10)
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify attribute presence and formatting
    assert "type='str'" in result
    assert "[Field with multiple attributes - as a string]" in result  # Description in content for simple types
    assert "min_length='5'" in result
    assert "max_length='100'" in result
    assert "greater_or_equal='-100.0'" in result
    assert "less_or_equal='100.0'" in result
    assert "multiple_of='0.5'" in result
    assert "description='List with multiple attributes'" in result  # Description as attribute for lists
    
    print("✓ Attribute ordering and formatting test passed")


def test_deeply_nested_edge_cases():
    """Test deeply nested structures with edge cases."""
    print("=== TESTING DEEPLY NESTED EDGE CASES ===")
    
    sr = StructuredResponse()
    sr.class_name = "DeepNestingEdgeCases"
    
    # 4-level nesting: list -> dict -> list -> dict
    sr.add_node("level1_list", list, description="Level 1 list", elements=dict)
    sr.add_node("level1_list.level2_dict", dict, description="Level 2 dict")
    sr.add_node("level1_list.level2_dict.level3_list", list, description="Level 3 list", elements=dict)
    sr.add_node("level1_list.level2_dict.level3_list.level4_field", str, description="Level 4 field")
    sr.add_node("level1_list.level2_dict.level3_list.level4_enum", enum, 
                description="Level 4 enum", choices=TestEnums.SimpleEnum)
    
    # Mixed types at same level
    sr.add_node("level1_list.mixed_str", str, description="Mixed string")
    sr.add_node("level1_list.mixed_int", int, description="Mixed integer", ge=0)
    sr.add_node("level1_list.mixed_bool", bool, description="Mixed boolean")
    
    result = sr.to_prompt()
    print(result)
    print()
    
    # Verify deep nesting structure
    assert "level1_list" in result
    assert "level2_dict" in result  
    assert "level3_list" in result
    assert "level4_field" in result
    assert "level4_enum" in result
    assert "mixed_str" in result
    assert "mixed_int" in result
    assert "mixed_bool" in result
    
    print("✓ Deeply nested edge cases test passed")


def test_readme_compliance_verification():
    """Verify exact compliance with README examples and rules."""
    print("=== TESTING README COMPLIANCE VERIFICATION ===")
    
    # Test exact README weather example structure
    sr = StructuredResponse()
    sr.class_name = "WeatherPrognosis"
    
    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    # Build exact README structure
    sr.add_node("location", str, description="The location of the weather forecast")
    sr.add_node("current_temperature", float, description="The current temperature in degrees Celsius")
    sr.add_node("overall_rain_prob", dict, description="The day's rain chance")
    sr.add_node("overall_rain_prob.chance", enum,
                description="The chance of rain, where low is less than 25% and high is more than 75%",
                choices=ChanceScale)
    sr.add_node("overall_rain_prob.when", str,
                description="The time of day when the rain is or is not expected")
    
    result = sr.to_prompt()
    
    # Verify exact README patterns
    readme_patterns = [
        "<weather_prognosis>",
        "<location type='str'>[The location of the weather forecast - as a string]</location>",
        ("<current_temperature type='float'>[The current temperature in degrees Celsius - "
         "as a float]</current_temperature>"),
        "<overall_rain_prob type='dict' description='The day's rain chance'>",
        ("<chance type='enum' choices='[\"low\", \"medium\", \"high\"]'>"
         "[The chance of rain, where low is less than 25% and high is more than 75% - as an enum]</chance>"),
        "<when type='str'>[The time of day when the rain is or is not expected - as a string]</when>",
        "</overall_rain_prob>",
        "</weather_prognosis>"
    ]
    
    for pattern in readme_patterns:
        assert pattern in result, f"Missing README pattern: {pattern}"
    
    # Test README rules compliance
    # Rule: "for non-list items, the description is placed between [ and ], with an additional '- as <type>' string appended"
    assert "[The location of the weather forecast - as a string]" in result
    assert "[The current temperature in degrees Celsius - as a float]" in result
    
    # Rule: "each element has the type tag"
    assert "type='str'" in result
    assert "type='float'" in result
    assert "type='dict'" in result
    assert "type='enum'" in result
    
    # Rule: "dicts take the name tag from their class"
    assert "<overall_rain_prob type='dict'" in result
    
    # Rule: "enums have a 'choices' tag, containing a python-like stringification"
    assert "choices='[\"low\", \"medium\", \"high\"]'" in result
    
    print("✓ README compliance verification test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running comprehensive to_prompt() tests...\n")
    
    test_functions = [
        test_basic_types,
        test_simple_lists,
        test_dict_types,
        test_complex_lists,
        test_nested_combinations,
        test_all_constraints,
        test_edge_cases,
        test_special_characters_and_formatting,
        test_type_name_mappings,
        test_complex_enum_scenarios,
        test_empty_and_minimal_structures,
        test_class_name_variations,
        test_attribute_ordering_and_formatting,
        test_deeply_nested_edge_cases,
        test_readme_compliance_verification,
        test_readme_weather_example
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== TEST SUMMARY ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! to_prompt() implementation is fully compliant with README specification.")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed. Please review and fix issues.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
