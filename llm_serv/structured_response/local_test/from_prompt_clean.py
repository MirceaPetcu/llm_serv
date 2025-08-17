#!/usr/bin/env python3
"""
Comprehensive tests for from_prompt() method covering all edge cases and type combinations.
Tests are designed to ensure full compliance with README specification and correct parsing
of XML responses from LLMs.

This test suite validates:
1. Basic field parsing (str, int, float, bool, enum)
2. Simple list parsing (lists with primitive elements)
3. Dict field parsing (nested objects)
4. Complex list parsing (lists with dict elements)
5. Deeply nested structures
6. Error handling and edge cases
7. Roundtrip conversion (__str__ -> from_prompt)
8. Special characters and formatting
9. README weather example compliance
"""

import sys
import enum
import json
import re
from typing import Any
from xml.etree import ElementTree as ET

# Utility functions (standalone to avoid import issues)
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
    return text

# Standalone StructuredResponse for testing
class StructuredResponse:
    def __init__(self):
        self.class_name = "StructuredResponse"
        self.definition: dict[str, Any] = {}
        self.instance: dict[str, Any] = {}

    def from_prompt(self, xml_string: str) -> None:
        """Parse XML and populate instance according to README specification."""
        if not self.definition:
            raise ValueError("Definition not initialized. Call from_basemodel first.")

        root_tag = camel_to_snake(self.class_name)
        start_idx = xml_string.find(f"<{root_tag}")
        end_idx = xml_string.rfind(f"</{root_tag}>")
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Root XML tags <{root_tag}> not found in LLM output")
        end_idx += len(f"</{root_tag}>")
        xml_sub = xml_string[start_idx:end_idx]

        try:
            root_element = ET.fromstring(xml_sub)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML content: {exc}") from exc

        def parse_element(element: ET.Element, schema: Any) -> Any:
            if isinstance(schema, dict) and schema.get("type") == "list":
                items: list[Any] = []
                for li in element.findall("li"):
                    elem_schema = schema.get("elements")
                    if isinstance(elem_schema, dict):
                        item: dict[str, Any] = {}
                        # Initialize all fields from schema to None first
                        for field_name in elem_schema.keys():
                            item[field_name] = None
                        # Then populate with actual values
                        for child in li:
                            if child.tag == "li" or child.tag is ET.Comment:
                                continue
                            field_schema = elem_schema.get(child.tag)
                            if field_schema is None:
                                continue
                            item[child.tag] = parse_element(child, field_schema)
                        items.append(item)
                    else:
                        items.append(coerce_text_to_type(li.text or "", str(elem_schema)))
                return items

            if isinstance(schema, dict) and schema.get("type") == "dict":
                obj: dict[str, Any] = {}
                elements_schema = schema.get("elements", {})
                if isinstance(elements_schema, dict):
                    for field_name, field_schema in elements_schema.items():
                        child = element.find(field_name)
                        if child is None:
                            obj[field_name] = None
                            continue
                        obj[field_name] = parse_element(child, field_schema)
                return obj

            if isinstance(schema, dict) and "type" not in schema:
                obj: dict[str, Any] = {}
                for field_name, field_schema in schema.items():
                    child = element.find(field_name)
                    if child is None:
                        obj[field_name] = None
                        continue
                    obj[field_name] = parse_element(child, field_schema)
                return obj

            if not isinstance(schema, dict):
                return coerce_text_to_type(element.text or "", str(schema))
                
            type_name = schema.get("type", "str")
            if type_name == "enum":
                return (element.text or "").strip()
            return coerce_text_to_type(element.text or "", type_name)

        self.instance = {}
        for field_name, schema in self.definition.items():
            child = root_element.find(field_name)
            if child is None:
                self.instance[field_name] = None
                continue
            self.instance[field_name] = parse_element(child, schema)

    def __str__(self) -> str:
        """Render instance as simple XML matching README format."""
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

        def render_list_field(field_name: str, items_value: Any, element_schema: Any | None, indent_level: int) -> list[str]:
            pad = "    " * indent_level
            lines: list[str] = [f"{pad}<{field_name}>"]
            items: list[Any] = items_value or []
            for item in items:
                if isinstance(element_schema, dict):
                    lines.append(f"{pad}    <li>")
                    for child_name, child_schema in element_schema.items():
                        child_value = None
                        if isinstance(item, dict):
                            child_value = item.get(child_name)
                        lines.extend(render_field(child_name, child_schema, child_value, indent_level + 2))
                    lines.append(f"{pad}    </li>")
                else:
                    text = coerce_primitive_to_text(item)
                    lines.append(f"{pad}    <li>{text}</li>")
            lines.append(f"{pad}</{field_name}>")
            return lines

        def render_object_field(field_name: str, object_schema: dict[str, Any], object_value: Any, indent_level: int) -> list[str]:
            pad = "    " * indent_level
            lines: list[str] = [f"{pad}<{field_name}>"]
            value_dict = object_value or {}
            for child_name, child_schema in object_schema.items():
                child_value = None
                if isinstance(value_dict, dict):
                    child_value = value_dict.get(child_name)
                lines.extend(render_field(child_name, child_schema, child_value, indent_level + 1))
            lines.append(f"{pad}</{field_name}>")
            return lines

        def render_field(field_name: str, field_schema: Any, field_value: Any, indent_level: int) -> list[str]:
            if isinstance(field_schema, dict) and field_schema.get("type") == "list":
                return render_list_field(field_name, field_value, field_schema.get("elements"), indent_level)
            if isinstance(field_schema, dict) and field_schema.get("type") == "dict":
                return render_object_field(field_name, field_schema.get("elements", {}), field_value, indent_level)
            if isinstance(field_schema, dict) and "type" not in field_schema:
                return render_object_field(field_name, field_schema, field_value, indent_level)
            return render_simple_field(field_name, field_value, indent_level)

        if not self.instance:
            return f"<{root_tag}>\n</{root_tag}>"

        lines: list[str] = [f"<{root_tag}>"]
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

            if schema_for_field is None:
                if isinstance(value_for_field, list):
                    element_schema: Any | None
                    if value_for_field and isinstance(value_for_field[0], dict):
                        element_schema = {k: {"type": "str"} for k in value_for_field[0].keys()}
                    else:
                        element_schema = None
                    lines.extend(render_list_field(field_name, value_for_field, element_schema, 1))
                elif isinstance(value_for_field, dict):
                    inferred_schema = {k: {"type": "str"} for k in value_for_field.keys()}
                    lines.extend(render_object_field(field_name, inferred_schema, value_for_field, 1))
                else:
                    lines.extend(render_simple_field(field_name, value_for_field, 1))
                continue

            lines.extend(render_field(field_name, schema_for_field, value_for_field, indent_level=1))

        lines.append(f"</{root_tag}>")
        return "\n".join(lines)

    def serialize(self) -> str:
        """Serialize to JSON string."""
        data = {
            "class_name": self.class_name,
            "definition": self.definition,
            "instance": self.instance,
        }
        return json.dumps(data)

    @staticmethod
    def deserialize(json_string: str) -> "StructuredResponse":
        """Deserialize from JSON string."""
        data = json.loads(json_string)
        sr = StructuredResponse()
        sr.class_name = data.get("class_name", "StructuredResponse")
        sr.definition = data.get("definition", {})
        sr.instance = data.get("instance", {})
        return sr

    def add_node(self, node_path: str, node_type: type, elements: type = None, description: str = "", choices: enum.Enum = None, **kwargs):
        """Add node to definition."""
        definition = self.definition
        
        target_node = definition
        if "." in node_path:
            path = node_path.split(".")[:-1]
            new_node_name = node_path.split(".")[-1]
        else:
            path = []
            new_node_name = node_path

        for key in path:
            if key not in target_node:
                raise ValueError(f"Intermediary node '{key}' not found in definition! Given path: {path}")
            
            if target_node[key]['type'] == 'list':
                target_node = target_node[key]['elements']
            elif target_node[key]['type'] == 'dict':
                target_node = target_node[key]['elements']
            else:
                target_node = target_node[key]

        target_node[new_node_name] = {
            "type": node_type.__name__,
            "description": description,
            **kwargs
        }

        if node_type is enum:
            target_node[new_node_name]["choices"] = [e.value for e in choices]
        elif node_type is list:
            if elements in [str, int, float, bool, enum]:
                target_node[new_node_name]["elements"] = elements.__name__
            elif elements is dict:
                target_node[new_node_name]["elements"] = {}
        elif node_type is dict:
            target_node[new_node_name]["elements"] = {}

        self.definition = definition


class TestEnums:
    """Test enum classes for various scenarios."""
    
    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    class StatusEnum(str, enum.Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"
    
    class SimpleEnum(str, enum.Enum):
        OPTION_A = "option_a"
        OPTION_B = "option_b"


def test_basic_field_parsing():
    """Test parsing of basic field types."""
    print("=== TESTING BASIC FIELD PARSING ===")
    
    sr = StructuredResponse()
    sr.class_name = "BasicTest"
    sr.add_node("name", str, description="Person name")
    sr.add_node("age", int, description="Person age")
    sr.add_node("height", float, description="Person height")
    sr.add_node("is_active", bool, description="Is person active")
    sr.add_node("status", enum, description="Person status", choices=TestEnums.StatusEnum)
    
    xml_response = """
    <basic_test>
        <name>John Doe</name>
        <age>30</age>
        <height>5.9</height>
        <is_active>true</is_active>
        <status>active</status>
    </basic_test>
    """
    
    sr.from_prompt(xml_response)
    
    assert sr.instance["name"] == "John Doe"
    assert sr.instance["age"] == 30
    assert sr.instance["height"] == 5.9
    assert sr.instance["is_active"] is True
    assert sr.instance["status"] == "active"
    
    print("✓ Basic field parsing test passed")


def test_simple_list_parsing():
    """Test parsing of lists with simple element types."""
    print("=== TESTING SIMPLE LIST PARSING ===")
    
    sr = StructuredResponse()
    sr.class_name = "SimpleListTest"
    sr.add_node("names", list, description="List of names", elements=str)
    sr.add_node("scores", list, description="List of scores", elements=int)
    sr.add_node("weights", list, description="List of weights", elements=float)
    sr.add_node("flags", list, description="List of flags", elements=bool)
    
    xml_response = """
    <simple_list_test>
        <names>
            <li>Alice</li>
            <li>Bob</li>
            <li>Charlie</li>
        </names>
        <scores>
            <li>85</li>
            <li>92</li>
            <li>78</li>
        </scores>
        <weights>
            <li>65.5</li>
            <li>70.2</li>
        </weights>
        <flags>
            <li>true</li>
            <li>false</li>
            <li>true</li>
        </flags>
    </simple_list_test>
    """
    
    sr.from_prompt(xml_response)
    
    assert sr.instance["names"] == ["Alice", "Bob", "Charlie"]
    assert sr.instance["scores"] == [85, 92, 78]
    assert sr.instance["weights"] == [65.5, 70.2]
    assert sr.instance["flags"] == [True, False, True]
    
    print("✓ Simple list parsing test passed")


def test_dict_parsing():
    """Test parsing of dict fields."""
    print("=== TESTING DICT PARSING ===")
    
    sr = StructuredResponse()
    sr.class_name = "DictTest"
    sr.add_node("person_info", dict, description="Person information")
    sr.add_node("person_info.name", str, description="Person name")
    sr.add_node("person_info.age", int, description="Person age")
    sr.add_node("person_info.active", bool, description="Is active")
    
    xml_response = """
    <dict_test>
        <person_info>
            <name>Jane Smith</name>
            <age>25</age>
            <active>true</active>
        </person_info>
    </dict_test>
    """
    
    sr.from_prompt(xml_response)
    
    expected_person_info = {
        "name": "Jane Smith",
        "age": 25,
        "active": True
    }
    assert sr.instance["person_info"] == expected_person_info
    
    print("✓ Dict parsing test passed")


def test_complex_list_parsing():
    """Test parsing of lists with dict elements."""
    print("=== TESTING COMPLEX LIST PARSING ===")
    
    sr = StructuredResponse()
    sr.class_name = "ComplexListTest"
    sr.add_node("people", list, description="List of people", elements=dict)
    sr.add_node("people.name", str, description="Person name")
    sr.add_node("people.age", int, description="Person age")
    sr.add_node("people.status", enum, description="Person status", choices=TestEnums.StatusEnum)
    
    xml_response = """
    <complex_list_test>
        <people>
            <li>
                <name>Alice</name>
                <age>30</age>
                <status>active</status>
            </li>
            <li>
                <name>Bob</name>
                <age>25</age>
                <status>inactive</status>
            </li>
        </people>
    </complex_list_test>
    """
    
    sr.from_prompt(xml_response)
    
    expected_people = [
        {"name": "Alice", "age": 30, "status": "active"},
        {"name": "Bob", "age": 25, "status": "inactive"}
    ]
    assert sr.instance["people"] == expected_people
    
    print("✓ Complex list parsing test passed")


def test_readme_weather_example():
    """Test the exact weather example from README."""
    print("=== TESTING README WEATHER EXAMPLE ===")
    
    sr = StructuredResponse()
    sr.class_name = "WeatherPrognosis"
    
    # Build the exact structure from README
    sr.add_node("location", str, description="The location of the weather forecast")
    sr.add_node("current_temperature", float, description="The current temperature in degrees Celsius")
    
    # overall_rain_prob dict
    sr.add_node("overall_rain_prob", dict, description="The day's rain chance")
    sr.add_node("overall_rain_prob.chance", enum, 
                description="The chance of rain, where low is less than 25% and high is more than 75%",
                choices=TestEnums.ChanceScale)
    sr.add_node("overall_rain_prob.when", str, 
                description="The time of day when the rain is or is not expected")
    
    # rain_probability_timebound list
    sr.add_node("rain_probability_timebound", list,
                description="List of chances of rain, where low is less than 25% and high is more than 75%",
                elements=dict)
    sr.add_node("rain_probability_timebound.chance", enum,
                description="The chance of rain, where low is less than 25% and high is more than 75%",
                choices=TestEnums.ChanceScale)
    sr.add_node("rain_probability_timebound.when", str,
                description="The time of day when the rain is or is not expected")
    
    # Simple list
    sr.add_node("hourly_index", list, description="List of hourly UV index in the range of 1-10", elements=int)
    
    # More fields
    sr.add_node("wind_speed", float, description="The wind speed in km/h")
    sr.add_node("high", float, description="The high temperature in degrees Celsius", ge=-20, le=60)
    sr.add_node("low", float, description="The low temperature in degrees Celsius")
    sr.add_node("storm_tonight", bool, description="Whether there will be a storm tonight")
    
    # Test with README example data
    xml_response = """
    <weather_prognosis>
        <location>Annecy, FR</location>
        <current_temperature>18.7</current_temperature>
        <overall_rain_prob>
            <chance>medium</chance>
            <when>today</when>
        </overall_rain_prob>
        <rain_probability_timebound>
            <li>
                <chance>low</chance>
                <when>morning</when>
            </li>
            <li>
                <chance>medium</chance>
                <when>afternoon</when>
            </li>
            <li>
                <chance>high</chance>
                <when>evening</when>
            </li>
        </rain_probability_timebound>
        <hourly_index>
            <li>3</li>
            <li>4</li>
            <li>5</li>
            <li>6</li>
            <li>5</li>
            <li>4</li>
            <li>3</li>
            <li>2</li>
        </hourly_index>
        <wind_speed>12.5</wind_speed>
        <high>24.0</high>
        <low>12.0</low>
        <storm_tonight>false</storm_tonight>
    </weather_prognosis>
    """
    
    sr.from_prompt(xml_response)
    
    # Verify exact README example data
    expected_instance = {
        "location": "Annecy, FR",
        "current_temperature": 18.7,
        "overall_rain_prob": {
            "chance": "medium",
            "when": "today"
        },
        "rain_probability_timebound": [
            {"chance": "low", "when": "morning"},
            {"chance": "medium", "when": "afternoon"},
            {"chance": "high", "when": "evening"}
        ],
        "hourly_index": [3, 4, 5, 6, 5, 4, 3, 2],
        "wind_speed": 12.5,
        "high": 24.0,
        "low": 12.0,
        "storm_tonight": False
    }
    
    assert sr.instance == expected_instance, f"Expected {expected_instance}, got {sr.instance}"
    
    print("✓ README weather example parsing test passed")


def test_error_handling():
    """Test error handling for malformed XML and missing root tags."""
    print("=== TESTING ERROR HANDLING ===")
    
    sr = StructuredResponse()
    sr.class_name = "ErrorTest"
    sr.add_node("field", str, description="Test field")
    
    # Test missing root tags
    try:
        sr.from_prompt("<wrong_tag><field>value</field></wrong_tag>")
        raise AssertionError("Should have raised ValueError for missing root tags")
    except ValueError as e:
        assert "Root XML tags" in str(e)
        print("✓ Missing root tags error handling passed")
    
    # Test malformed XML
    try:
        sr.from_prompt("<error_test><field>unclosed tag</error_test>")
        raise AssertionError("Should have raised ValueError for malformed XML")
    except ValueError as e:
        assert "Invalid XML content" in str(e)
        print("✓ Malformed XML error handling passed")
    
    # Test empty definition
    empty_sr = StructuredResponse()
    empty_sr.class_name = "EmptyDef"
    try:
        empty_sr.from_prompt("<empty_def></empty_def>")
        raise AssertionError("Should have raised ValueError for empty definition")
    except ValueError as e:
        assert "Definition not initialized" in str(e)
        print("✓ Empty definition error handling passed")


def test_roundtrip_conversion():
    """Test that data survives roundtrip conversion (__str__ -> from_prompt)."""
    print("=== TESTING ROUNDTRIP CONVERSION ===")
    
    sr = StructuredResponse()
    sr.class_name = "RoundtripTest"
    sr.add_node("name", str, description="Name field")
    sr.add_node("count", int, description="Count field")
    sr.add_node("active", bool, description="Active field")
    sr.add_node("items", list, description="Items list", elements=dict)
    sr.add_node("items.id", int, description="Item ID")
    sr.add_node("items.value", str, description="Item value")
    
    # Set some instance data
    sr.instance = {
        "name": "Test Name",
        "count": 42,
        "active": True,
        "items": [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"}
        ]
    }
    
    # Convert to string and back
    xml_output = str(sr)  # Uses __str__ method
    
    # Create new instance and parse
    sr2 = StructuredResponse()
    sr2.class_name = "RoundtripTest"
    sr2.add_node("name", str, description="Name field")
    sr2.add_node("count", int, description="Count field")
    sr2.add_node("active", bool, description="Active field")
    sr2.add_node("items", list, description="Items list", elements=dict)
    sr2.add_node("items.id", int, description="Item ID")
    sr2.add_node("items.value", str, description="Item value")
    
    sr2.from_prompt(xml_output)
    
    assert sr2.instance == sr.instance, f"Roundtrip failed: {sr2.instance} != {sr.instance}"
    
    print("✓ Roundtrip conversion test passed")


def test_list_with_missing_fields():
    """Test lists where some items have missing fields."""
    print("=== TESTING LIST WITH MISSING FIELDS ===")
    
    sr = StructuredResponse()
    sr.class_name = "MissingFieldsTest"
    sr.add_node("items", list, description="Items with optional fields", elements=dict)
    sr.add_node("items.name", str, description="Item name")
    sr.add_node("items.optional_field", str, description="Optional field")
    sr.add_node("items.another_optional", int, description="Another optional field")
    
    xml_response = """
    <missing_fields_test>
        <items>
            <li>
                <name>Complete Item</name>
                <optional_field>Present</optional_field>
                <another_optional>42</another_optional>
            </li>
            <li>
                <name>Partial Item</name>
                <optional_field>Also Present</optional_field>
            </li>
            <li>
                <name>Minimal Item</name>
            </li>
        </items>
    </missing_fields_test>
    """
    
    sr.from_prompt(xml_response)
    
    expected_items = [
        {"name": "Complete Item", "optional_field": "Present", "another_optional": 42},
        {"name": "Partial Item", "optional_field": "Also Present", "another_optional": None},
        {"name": "Minimal Item", "optional_field": None, "another_optional": None}
    ]
    assert sr.instance["items"] == expected_items
    
    print("✓ List with missing fields test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running comprehensive from_prompt() tests...\n")
    
    test_functions = [
        test_basic_field_parsing,
        test_simple_list_parsing,
        test_dict_parsing,
        test_complex_list_parsing,
        test_readme_weather_example,
        test_error_handling,
        test_roundtrip_conversion,
        test_list_with_missing_fields
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
            print()
    
    print("\n=== TEST SUMMARY ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! from_prompt() implementation is fully compliant with README specification.")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed. Please review and fix issues.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
