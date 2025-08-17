#!/usr/bin/env python3
"""
Comprehensive tests for serialize() and deserialize() methods covering all edge cases.
Tests ensure proper serialization and deserialization of StructuredResponse objects.
"""

import sys
import os
import json
import enum
from typing import Optional, List, Dict, Any
from enum import Enum

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pydantic import BaseModel, Field
from llm_serv.structured_response.model import StructuredResponse
from llm_serv.structured_response.converters.from_basemodel import from_basemodel


class TestEnums:
    """Test enum classes for serialization scenarios."""
    
    class Priority(str, enum.Enum):
        URGENT = "urgent"
        HIGH = "high"
        NORMAL = "normal"
        LOW = "low"
    
    class Status(str, enum.Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"


def test_empty_structured_response_serialization():
    """Test serialization of empty StructuredResponse."""
    print("=== TESTING EMPTY STRUCTURED RESPONSE SERIALIZATION ===")
    
    # Empty StructuredResponse
    sr = StructuredResponse()
    serialized = sr.serialize()
    
    # Verify JSON structure
    data = json.loads(serialized)
    assert data["class_name"] == "StructuredResponse"
    assert data["definition"] == {}
    assert data["instance"] == {}
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == "StructuredResponse"
    assert sr_restored.definition == {}
    assert sr_restored.instance == {}
    
    print("✓ Empty StructuredResponse serialization test passed")


def test_manual_structured_response_serialization():
    """Test serialization of manually created StructuredResponse."""
    print("=== TESTING MANUAL STRUCTURED RESPONSE SERIALIZATION ===")
    
    # Create manually using add_node
    sr = StructuredResponse()
    sr.class_name = "ManualTest"
    sr.add_node("name", str, description="Person name")
    sr.add_node("age", int, description="Person age", ge=0, le=150)
    sr.add_node("active", bool, description="Whether person is active")
    sr.add_node("priority", enum, description="Person priority", choices=TestEnums.Priority)
    sr.add_node("tags", list, elements=str, description="List of tags")
    
    # Add some instance data
    sr.instance = {
        "name": "John Doe",
        "age": 30,
        "active": True,
        "priority": "high",
        "tags": ["important", "manager"]
    }
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify structure
    assert data["class_name"] == "ManualTest"
    assert "name" in data["definition"]
    assert data["definition"]["name"]["type"] == "str"
    assert data["definition"]["age"]["ge"] == 0
    assert data["definition"]["age"]["le"] == 150
    assert data["definition"]["priority"]["type"] == "enum"
    assert data["definition"]["tags"]["type"] == "list"
    assert data["definition"]["tags"]["elements"] == "str"
    
    # Verify instance data
    assert data["instance"]["name"] == "John Doe"
    assert data["instance"]["age"] == 30
    assert data["instance"]["active"] is True
    assert data["instance"]["priority"] == "high"
    assert data["instance"]["tags"] == ["important", "manager"]
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    print("✓ Manual StructuredResponse serialization test passed")


def test_basemodel_type_serialization():
    """Test serialization of StructuredResponse created from BaseModel type."""
    print("=== TESTING BASEMODEL TYPE SERIALIZATION ===")
    
    class SimpleModel(BaseModel):
        name: str = Field(description="Name field")
        count: int = Field(description="Count field", ge=0)
        active: bool = Field(description="Active flag")
        priority: TestEnums.Priority = Field(description="Priority level")
        tags: List[str] = Field(description="List of tags")
    
    # Create from BaseModel type
    sr = from_basemodel(SimpleModel)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify structure
    assert data["class_name"] == "SimpleModel"
    assert len(data["definition"]) == 5
    assert data["definition"]["name"]["type"] == "str"
    assert data["definition"]["count"]["type"] == "int"
    assert data["definition"]["count"]["ge"] == 0
    assert data["definition"]["active"]["type"] == "bool"
    assert data["definition"]["priority"]["type"] == "enum"
    assert data["definition"]["priority"]["choices"] == ["urgent", "high", "normal", "low"]
    assert data["definition"]["tags"]["type"] == "list"
    assert data["definition"]["tags"]["elements"] == "str"
    assert data["instance"] == {}
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    # Verify that restored object can generate prompts
    prompt = sr_restored.to_prompt()
    assert "<simple_model>" in prompt
    assert "type='str'" in prompt
    assert "type='enum'" in prompt
    
    print("✓ BaseModel type serialization test passed")


def test_basemodel_instance_serialization():
    """Test serialization of StructuredResponse created from BaseModel instance."""
    print("=== TESTING BASEMODEL INSTANCE SERIALIZATION ===")
    
    class InstanceModel(BaseModel):
        name: str = Field(description="Name field")
        score: float = Field(description="Score field", ge=0.0, le=100.0)
        status: TestEnums.Status = Field(description="Status field")
        optional_field: Optional[str] = Field(description="Optional field")
    
    # Create instance
    instance = InstanceModel(
        name="Test Instance",
        score=85.5,
        status=TestEnums.Status.ACTIVE,
        optional_field="optional value"
    )
    
    # Create StructuredResponse from instance
    sr = from_basemodel(instance)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify structure and instance data
    assert data["class_name"] == "InstanceModel"
    assert data["definition"]["name"]["type"] == "str"
    assert data["definition"]["score"]["ge"] == 0.0
    assert data["definition"]["score"]["le"] == 100.0
    assert data["definition"]["status"]["type"] == "enum"
    assert data["definition"]["status"]["choices"] == ["active", "inactive", "pending"]
    
    # Verify instance data
    assert data["instance"]["name"] == "Test Instance"
    assert data["instance"]["score"] == 85.5
    assert data["instance"]["status"] == "active"
    assert data["instance"]["optional_field"] == "optional value"
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    # Verify that restored object can render properly
    xml_output = str(sr_restored)
    assert "<instance_model>" in xml_output
    assert "<name>Test Instance</name>" in xml_output
    assert "<score>85.5</score>" in xml_output
    assert "<status>active</status>" in xml_output
    
    print("✓ BaseModel instance serialization test passed")


def test_nested_structures_serialization():
    """Test serialization of complex nested structures."""
    print("=== TESTING NESTED STRUCTURES SERIALIZATION ===")
    
    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City name")
        postal_code: str = Field(description="Postal code", min_length=5)
    
    class Person(BaseModel):
        name: str = Field(description="Person name")
        age: int = Field(description="Person age", ge=0)
        address: Address = Field(description="Person address")
        backup_addresses: List[Address] = Field(description="Backup addresses")
        status: TestEnums.Status = Field(description="Person status")
    
    # Create instance with nested data
    main_address = Address(street="123 Main St", city="Anytown", postal_code="12345")
    backup_address = Address(street="456 Oak Ave", city="Otherville", postal_code="67890")
    
    person = Person(
        name="Alice Johnson",
        age=28,
        address=main_address,
        backup_addresses=[backup_address],
        status=TestEnums.Status.ACTIVE
    )
    
    # Create StructuredResponse from instance
    sr = from_basemodel(person)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify nested structure in definition
    assert data["class_name"] == "Person"
    assert data["definition"]["address"]["type"] == "dict"
    assert "elements" in data["definition"]["address"]
    assert data["definition"]["address"]["elements"]["street"]["type"] == "str"
    assert data["definition"]["address"]["elements"]["postal_code"]["min_length"] == 5
    
    # Verify list of nested structures
    assert data["definition"]["backup_addresses"]["type"] == "list"
    assert data["definition"]["backup_addresses"]["elements"]["street"]["type"] == "str"
    
    # Verify nested instance data
    assert data["instance"]["name"] == "Alice Johnson"
    assert data["instance"]["address"]["street"] == "123 Main St"
    assert data["instance"]["address"]["city"] == "Anytown"
    assert len(data["instance"]["backup_addresses"]) == 1
    assert data["instance"]["backup_addresses"][0]["street"] == "456 Oak Ave"
    assert data["instance"]["status"] == "active"
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    print("✓ Nested structures serialization test passed")


def test_complex_weather_serialization():
    """Test serialization of the complex weather example from README."""
    print("=== TESTING COMPLEX WEATHER SERIALIZATION ===")
    
    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    class RainProbability(BaseModel):
        chance: ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
        when: str = Field(description="The time of day when the rain is or is not expected")
    
    class WeatherPrognosis(BaseModel):
        location: str = Field(description="The location of the weather forecast")
        current_temperature: float = Field(description="The current temperature in degrees Celsius")
        overall_rain_prob: RainProbability = Field(description="The day's rain chance")
        rain_probability_timebound: Optional[List[RainProbability]] = Field(
            description="List of chances of rain, where low is less than 25% and high is more than 75%"
        )
        hourly_index: List[int] = Field(description="List of hourly UV index in the range of 1-10")
        wind_speed: float = Field(description="The wind speed in km/h")
        high: float = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
        low: float = Field(description="The low temperature in degrees Celsius")
        storm_tonight: bool = Field(description="Whether there will be a storm tonight")
    
    # Create instance matching README example
    overall_rain = RainProbability(chance=ChanceScale.MEDIUM, when="today")
    timebound_rain = [
        RainProbability(chance=ChanceScale.LOW, when="morning"),
        RainProbability(chance=ChanceScale.MEDIUM, when="afternoon"),
        RainProbability(chance=ChanceScale.HIGH, when="evening")
    ]
    
    weather = WeatherPrognosis(
        location="Annecy, FR",
        current_temperature=18.7,
        overall_rain_prob=overall_rain,
        rain_probability_timebound=timebound_rain,
        hourly_index=[3, 4, 5, 6, 5, 4, 3, 2],
        wind_speed=12.5,
        high=24.0,
        low=12.0,
        storm_tonight=False
    )
    
    # Create StructuredResponse
    sr = from_basemodel(weather)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify complex structure
    assert data["class_name"] == "WeatherPrognosis"
    
    # Check overall_rain_prob structure
    assert data["definition"]["overall_rain_prob"]["type"] == "dict"
    assert data["definition"]["overall_rain_prob"]["elements"]["chance"]["type"] == "enum"
    assert data["definition"]["overall_rain_prob"]["elements"]["chance"]["choices"] == ["low", "medium", "high"]
    
    # Check rain_probability_timebound structure
    assert data["definition"]["rain_probability_timebound"]["type"] == "list"
    assert data["definition"]["rain_probability_timebound"]["elements"]["chance"]["type"] == "enum"
    
    # Check constraints
    assert data["definition"]["high"]["ge"] == -20
    assert data["definition"]["high"]["le"] == 60
    
    # Verify instance data matches README example
    assert data["instance"]["location"] == "Annecy, FR"
    assert data["instance"]["current_temperature"] == 18.7
    assert data["instance"]["overall_rain_prob"]["chance"] == "medium"
    assert data["instance"]["overall_rain_prob"]["when"] == "today"
    assert len(data["instance"]["rain_probability_timebound"]) == 3
    assert data["instance"]["rain_probability_timebound"][0]["chance"] == "low"
    assert data["instance"]["rain_probability_timebound"][1]["chance"] == "medium"
    assert data["instance"]["rain_probability_timebound"][2]["chance"] == "high"
    assert data["instance"]["hourly_index"] == [3, 4, 5, 6, 5, 4, 3, 2]
    assert data["instance"]["high"] == 24.0
    assert data["instance"]["storm_tonight"] is False
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    # Verify functionality after restoration
    prompt = sr_restored.to_prompt()
    assert "<weather_prognosis>" in prompt
    
    xml_output = str(sr_restored)
    assert "<location>Annecy, FR</location>" in xml_output
    assert "<storm_tonight>false</storm_tonight>" in xml_output
    
    print("✓ Complex weather serialization test passed")


def test_deeply_nested_serialization():
    """Test serialization of deeply nested structures."""
    print("=== TESTING DEEPLY NESTED SERIALIZATION ===")
    
    class ContactInfo(BaseModel):
        email: str = Field(description="Email address")
        phone: Optional[str] = Field(description="Phone number")
    
    class Department(BaseModel):
        name: str = Field(description="Department name")
        budget: float = Field(description="Department budget", ge=0.0)
        head_contact: ContactInfo = Field(description="Department head contact")
    
    class Employee(BaseModel):
        employee_id: str = Field(description="Employee ID")
        name: str = Field(description="Employee name")
        contact: ContactInfo = Field(description="Employee contact")
        department: Department = Field(description="Employee department")
        skills: List[str] = Field(description="Employee skills")
    
    class Company(BaseModel):
        name: str = Field(description="Company name")
        employees: List[Employee] = Field(description="Company employees")
        departments: List[Department] = Field(description="Company departments")
    
    # Create complex nested instance
    head_contact = ContactInfo(email="head@company.com", phone="555-0001")
    dept = Department(name="Engineering", budget=500000.0, head_contact=head_contact)
    
    emp_contact = ContactInfo(email="john@company.com", phone="555-0002")
    employee = Employee(
        employee_id="EMP001",
        name="John Doe",
        contact=emp_contact,
        department=dept,
        skills=["Python", "JavaScript"]
    )
    
    company = Company(
        name="Tech Corp",
        employees=[employee],
        departments=[dept]
    )
    
    # Create StructuredResponse
    sr = from_basemodel(company)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify deeply nested definition structure
    assert data["class_name"] == "Company"
    employee_def = data["definition"]["employees"]["elements"]
    assert employee_def["contact"]["type"] == "dict"
    assert employee_def["contact"]["elements"]["email"]["type"] == "str"
    assert employee_def["department"]["type"] == "dict"
    dept_def = employee_def["department"]["elements"]
    assert dept_def["head_contact"]["type"] == "dict"
    assert dept_def["head_contact"]["elements"]["phone"]["type"] == "str"
    
    # Verify deeply nested instance data
    assert data["instance"]["name"] == "Tech Corp"
    assert len(data["instance"]["employees"]) == 1
    emp_data = data["instance"]["employees"][0]
    assert emp_data["name"] == "John Doe"
    assert emp_data["contact"]["email"] == "john@company.com"
    assert emp_data["department"]["name"] == "Engineering"
    assert emp_data["department"]["head_contact"]["email"] == "head@company.com"
    assert emp_data["skills"] == ["Python", "JavaScript"]
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    
    print("✓ Deeply nested serialization test passed")


def test_edge_cases_serialization():
    """Test serialization edge cases."""
    print("=== TESTING EDGE CASES SERIALIZATION ===")
    
    # Empty model
    class EmptyModel(BaseModel):
        pass
    
    sr_empty = from_basemodel(EmptyModel)
    serialized_empty = sr_empty.serialize()
    data_empty = json.loads(serialized_empty)
    assert data_empty["class_name"] == "EmptyModel"
    assert data_empty["definition"] == {}
    assert data_empty["instance"] == {}
    
    # Model with only optional fields set to None
    class OptionalModel(BaseModel):
        optional_str: Optional[str] = Field(description="Optional string")
        optional_int: Optional[int] = Field(description="Optional integer")
    
    optional_instance = OptionalModel(optional_str=None, optional_int=None)
    sr_optional = from_basemodel(optional_instance)
    serialized_optional = sr_optional.serialize()
    data_optional = json.loads(serialized_optional)
    
    assert data_optional["instance"]["optional_str"] is None
    assert data_optional["instance"]["optional_int"] is None
    
    # Model with empty lists
    class EmptyListsModel(BaseModel):
        empty_strings: List[str] = Field(description="Empty string list")
        empty_ints: List[int] = Field(description="Empty int list")
    
    empty_lists_instance = EmptyListsModel(empty_strings=[], empty_ints=[])
    sr_empty_lists = from_basemodel(empty_lists_instance)
    serialized_empty_lists = sr_empty_lists.serialize()
    data_empty_lists = json.loads(serialized_empty_lists)
    
    assert data_empty_lists["instance"]["empty_strings"] == []
    assert data_empty_lists["instance"]["empty_ints"] == []
    
    # Test roundtrips for all edge cases
    sr_empty_restored = StructuredResponse.deserialize(serialized_empty)
    assert sr_empty_restored.definition == sr_empty.definition
    
    sr_optional_restored = StructuredResponse.deserialize(serialized_optional)
    assert sr_optional_restored.instance == sr_optional.instance
    
    sr_empty_lists_restored = StructuredResponse.deserialize(serialized_empty_lists)
    assert sr_empty_lists_restored.instance == sr_empty_lists.instance
    
    print("✓ Edge cases serialization test passed")


def test_large_data_serialization():
    """Test serialization of large data structures."""
    print("=== TESTING LARGE DATA SERIALIZATION ===")
    
    class DataPoint(BaseModel):
        timestamp: str = Field(description="Data timestamp")
        value: float = Field(description="Data value")
        quality: TestEnums.Priority = Field(description="Data quality")
    
    class Dataset(BaseModel):
        name: str = Field(description="Dataset name")
        data_points: List[DataPoint] = Field(description="List of data points")
        metadata: Dict[str, Any] = Field(description="Dataset metadata", default_factory=dict)
    
    # Create large dataset
    data_points = []
    for i in range(100):  # Create 100 data points
        data_points.append(DataPoint(
            timestamp=f"2024-01-{i+1:02d}T12:00:00Z",
            value=float(i * 1.5),
            quality=TestEnums.Priority.NORMAL if i % 2 == 0 else TestEnums.Priority.HIGH
        ))
    
    dataset = Dataset(
        name="Large Test Dataset",
        data_points=data_points,
        metadata={"source": "test", "version": "1.0", "count": 100}
    )
    
    # Create StructuredResponse
    sr = from_basemodel(dataset)
    
    # Serialize
    serialized = sr.serialize()
    data = json.loads(serialized)
    
    # Verify structure
    assert data["class_name"] == "Dataset"
    assert len(data["instance"]["data_points"]) == 100
    assert data["instance"]["data_points"][0]["timestamp"] == "2024-01-01T12:00:00Z"
    assert data["instance"]["data_points"][0]["value"] == 0.0
    assert data["instance"]["data_points"][99]["value"] == 148.5
    assert data["instance"]["metadata"]["count"] == 100
    
    # Test roundtrip
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.class_name == sr.class_name
    assert sr_restored.definition == sr.definition
    assert sr_restored.instance == sr.instance
    assert len(sr_restored.instance["data_points"]) == 100
    
    print("✓ Large data serialization test passed")


def test_malformed_json_handling():
    """Test handling of malformed JSON during deserialization."""
    print("=== TESTING MALFORMED JSON HANDLING ===")
    
    # Test invalid JSON
    try:
        StructuredResponse.deserialize("invalid json")
        assert False, "Should have raised an exception"
    except json.JSONDecodeError:
        pass  # Expected
    
    # Test missing fields
    minimal_json = json.dumps({"class_name": "Test"})
    sr_minimal = StructuredResponse.deserialize(minimal_json)
    assert sr_minimal.class_name == "Test"
    assert sr_minimal.definition == {}
    assert sr_minimal.instance == {}
    
    # Test with None values
    none_json = json.dumps({
        "class_name": "Test",
        "definition": None,
        "instance": None
    })
    sr_none = StructuredResponse.deserialize(none_json)
    assert sr_none.class_name == "Test"
    assert sr_none.definition == {}
    assert sr_none.instance == {}
    
    print("✓ Malformed JSON handling test passed")


def test_serialization_consistency():
    """Test that multiple serialization/deserialization cycles are consistent."""
    print("=== TESTING SERIALIZATION CONSISTENCY ===")
    
    class ConsistencyModel(BaseModel):
        name: str = Field(description="Name field")
        values: List[int] = Field(description="List of values")
        status: TestEnums.Status = Field(description="Status field")
    
    instance = ConsistencyModel(
        name="Consistency Test",
        values=[1, 2, 3, 4, 5],
        status=TestEnums.Status.PENDING
    )
    
    sr_original = from_basemodel(instance)
    
    # Perform multiple serialization/deserialization cycles
    current_sr = sr_original
    for cycle in range(5):
        serialized = current_sr.serialize()
        current_sr = StructuredResponse.deserialize(serialized)
        
        # Verify consistency after each cycle
        assert current_sr.class_name == sr_original.class_name
        assert current_sr.definition == sr_original.definition
        assert current_sr.instance == sr_original.instance
    
    print("✓ Serialization consistency test passed")


def test_json_formatting():
    """Test that serialized JSON is properly formatted and readable."""
    print("=== TESTING JSON FORMATTING ===")
    
    class FormattingModel(BaseModel):
        name: str = Field(description="Name with special characters: àáâãäå")
        unicode_field: str = Field(description="Unicode: 🌟⭐✨")
        number: float = Field(description="Number field")
    
    instance = FormattingModel(
        name="Test with àáâãäå",
        unicode_field="Unicode test: 🌟⭐✨",
        number=123.456
    )
    
    sr = from_basemodel(instance)
    serialized = sr.serialize()
    
    # Verify it's valid JSON
    data = json.loads(serialized)
    assert data["instance"]["name"] == "Test with àáâãäå"
    assert data["instance"]["unicode_field"] == "Unicode test: 🌟⭐✨"
    assert data["instance"]["number"] == 123.456
    
    # Test roundtrip preserves special characters
    sr_restored = StructuredResponse.deserialize(serialized)
    assert sr_restored.instance["name"] == "Test with àáâãäå"
    assert sr_restored.instance["unicode_field"] == "Unicode test: 🌟⭐✨"
    
    print("✓ JSON formatting test passed")


def run_all_tests():
    """Run all serialization test functions."""
    print("Starting comprehensive serialization/deserialization tests...\n")
    
    test_functions = [
        test_empty_structured_response_serialization,
        test_manual_structured_response_serialization,
        test_basemodel_type_serialization,
        test_basemodel_instance_serialization,
        test_nested_structures_serialization,
        test_complex_weather_serialization,
        test_edge_cases_serialization,
        test_large_data_serialization,
        test_malformed_json_handling,
        test_serialization_consistency,
        test_json_formatting,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
    
    print(f"\n=== SERIALIZATION TEST SUMMARY ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All serialization tests passed!")
    else:
        print(f"⚠️  {failed} serialization test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
