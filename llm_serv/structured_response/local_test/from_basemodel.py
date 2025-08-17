#!/usr/bin/env python3
"""
Comprehensive tests for from_basemodel() method covering all edge cases and type combinations.
Tests are designed to ensure full compliance with README specification and proper use of add_node method.
"""

import sys
import os
import enum
from typing import Optional, List, Dict, Any
from enum import Enum

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pydantic import BaseModel, Field
from llm_serv.structured_response.model import StructuredResponse
from llm_serv.structured_response.converters.from_basemodel import from_basemodel


class TestEnums:
    """Test enum classes for various scenarios."""
    
    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    class Priority(str, enum.Enum):
        URGENT = "urgent"
        HIGH = "high"
        NORMAL = "normal"
        LOW = "low"
    
    class Status(str, enum.Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"


def test_basic_primitive_types():
    """Test basic primitive types with various constraints."""
    print("=== TESTING BASIC PRIMITIVE TYPES ===")
    
    class BasicTypes(BaseModel):
        simple_string: str = Field(description="A simple string field")
        constrained_string: str = Field(description="A string with length constraints", min_length=5, max_length=50)
        simple_int: int = Field(description="A simple integer field")
        constrained_int: int = Field(description="An integer with range constraints", ge=0, le=100)
        negative_range_int: int = Field(description="An integer with negative range", ge=-50, lt=0)
        simple_float: float = Field(description="A simple float field")
        constrained_float: float = Field(description="A float with range constraints", gt=0.0, le=100.0)
        simple_bool: bool = Field(description="A simple boolean field")
        optional_string: Optional[str] = Field(description="An optional string field")
        optional_int: Optional[int] = Field(description="An optional integer field")
    
    # Test with type
    sr_type = from_basemodel(BasicTypes)
    assert sr_type.class_name == "BasicTypes"
    assert "simple_string" in sr_type.definition
    assert sr_type.definition["simple_string"]["type"] == "str"
    assert sr_type.definition["constrained_string"]["min_length"] == 5
    assert sr_type.definition["constrained_string"]["max_length"] == 50
    assert sr_type.definition["constrained_int"]["ge"] == 0
    assert sr_type.definition["constrained_int"]["le"] == 100
    assert sr_type.definition["negative_range_int"]["ge"] == -50
    assert sr_type.definition["negative_range_int"]["lt"] == 0
    assert sr_type.definition["constrained_float"]["gt"] == 0.0
    assert sr_type.definition["constrained_float"]["le"] == 100.0
    assert sr_type.instance == {}
    
    # Test with instance
    instance = BasicTypes(
        simple_string="test",
        constrained_string="hello world",
        simple_int=42,
        constrained_int=50,
        negative_range_int=-25,
        simple_float=3.14,
        constrained_float=50.5,
        simple_bool=True,
        optional_string="optional value",
        optional_int=None
    )
    
    sr_instance = from_basemodel(instance)
    assert sr_instance.class_name == "BasicTypes"
    assert sr_instance.instance["simple_string"] == "test"
    assert sr_instance.instance["constrained_int"] == 50
    assert sr_instance.instance["simple_bool"] is True
    assert sr_instance.instance["optional_string"] == "optional value"
    assert sr_instance.instance["optional_int"] is None
    
    print("✓ Basic primitive types test passed")


def test_enum_types():
    """Test enum type handling."""
    print("=== TESTING ENUM TYPES ===")
    
    class EnumTest(BaseModel):
        priority: TestEnums.Priority = Field(description="Task priority level")
        status: TestEnums.Status = Field(description="Current status")
        optional_chance: Optional[TestEnums.ChanceScale] = Field(description="Optional chance scale")
    
    # Test with type
    sr_type = from_basemodel(EnumTest)
    assert sr_type.definition["priority"]["type"] == "enum"
    assert sr_type.definition["priority"]["choices"] == ["urgent", "high", "normal", "low"]
    assert sr_type.definition["status"]["type"] == "enum"
    assert sr_type.definition["status"]["choices"] == ["active", "inactive", "pending"]
    
    # Test with instance
    instance = EnumTest(
        priority=TestEnums.Priority.HIGH,
        status=TestEnums.Status.ACTIVE,
        optional_chance=TestEnums.ChanceScale.MEDIUM
    )
    
    sr_instance = from_basemodel(instance)
    assert sr_instance.instance["priority"] == "high"
    assert sr_instance.instance["status"] == "active"
    assert sr_instance.instance["optional_chance"] == "medium"
    
    print("✓ Enum types test passed")


def test_simple_list_types():
    """Test simple list types (lists of primitives)."""
    print("=== TESTING SIMPLE LIST TYPES ===")
    
    class SimpleListTest(BaseModel):
        string_list: List[str] = Field(description="List of strings")
        int_list: List[int] = Field(description="List of integers")
        float_list: List[float] = Field(description="List of floats")
        bool_list: List[bool] = Field(description="List of booleans")
        optional_list: Optional[List[str]] = Field(description="Optional list of strings")
        enum_list: List[TestEnums.ChanceScale] = Field(description="List of chance scales")
    
    # Test with type
    sr_type = from_basemodel(SimpleListTest)
    assert sr_type.definition["string_list"]["type"] == "list"
    assert sr_type.definition["string_list"]["elements"] == "str"
    assert sr_type.definition["int_list"]["elements"] == "int"
    assert sr_type.definition["float_list"]["elements"] == "float"
    assert sr_type.definition["bool_list"]["elements"] == "bool"
    assert sr_type.definition["enum_list"]["elements"] == "enum"
    
    # Test with instance
    instance = SimpleListTest(
        string_list=["hello", "world"],
        int_list=[1, 2, 3, 4],
        float_list=[1.1, 2.2, 3.3],
        bool_list=[True, False, True],
        optional_list=["optional", "values"],
        enum_list=[TestEnums.ChanceScale.LOW, TestEnums.ChanceScale.HIGH]
    )
    
    sr_instance = from_basemodel(instance)
    assert sr_instance.instance["string_list"] == ["hello", "world"]
    assert sr_instance.instance["int_list"] == [1, 2, 3, 4]
    assert sr_instance.instance["float_list"] == [1.1, 2.2, 3.3]
    assert sr_instance.instance["bool_list"] == [True, False, True]
    assert sr_instance.instance["optional_list"] == ["optional", "values"]
    assert sr_instance.instance["enum_list"] == ["low", "high"]
    
    print("✓ Simple list types test passed")


def test_nested_basemodel():
    """Test nested BaseModel structures."""
    print("=== TESTING NESTED BASEMODEL ===")
    
    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City name")
        postal_code: str = Field(description="Postal code", min_length=5, max_length=10)
        country: str = Field(description="Country name")
    
    class Person(BaseModel):
        name: str = Field(description="Person's full name")
        age: int = Field(description="Person's age", ge=0, le=150)
        address: Address = Field(description="Person's address")
        optional_address: Optional[Address] = Field(description="Optional secondary address")
    
    # Test with type
    sr_type = from_basemodel(Person)
    assert sr_type.class_name == "Person"
    assert sr_type.definition["address"]["type"] == "dict"
    assert "elements" in sr_type.definition["address"]
    assert "street" in sr_type.definition["address"]["elements"]
    assert sr_type.definition["address"]["elements"]["street"]["type"] == "str"
    assert sr_type.definition["address"]["elements"]["postal_code"]["min_length"] == 5
    assert sr_type.definition["address"]["elements"]["postal_code"]["max_length"] == 10
    
    # Test with instance
    address_instance = Address(
        street="123 Main St",
        city="Anytown",
        postal_code="12345",
        country="USA"
    )
    
    person_instance = Person(
        name="John Doe",
        age=30,
        address=address_instance,
        optional_address=None
    )
    
    sr_instance = from_basemodel(person_instance)
    assert sr_instance.instance["name"] == "John Doe"
    assert sr_instance.instance["age"] == 30
    assert sr_instance.instance["address"]["street"] == "123 Main St"
    assert sr_instance.instance["address"]["city"] == "Anytown"
    assert sr_instance.instance["optional_address"] is None
    
    print("✓ Nested BaseModel test passed")


def test_list_of_basemodels():
    """Test lists containing BaseModel objects."""
    print("=== TESTING LIST OF BASEMODELS ===")
    
    class RainProbability(BaseModel):
        chance: TestEnums.ChanceScale = Field(description="The chance of rain")
        when: str = Field(description="The time of day when rain is expected")
        confidence: float = Field(description="Confidence level", ge=0.0, le=1.0)
    
    class WeatherForecast(BaseModel):
        location: str = Field(description="Weather forecast location")
        daily_rain: List[RainProbability] = Field(description="Daily rain probabilities")
        optional_hourly: Optional[List[RainProbability]] = Field(description="Optional hourly rain data")
    
    # Test with type
    sr_type = from_basemodel(WeatherForecast)
    assert sr_type.definition["daily_rain"]["type"] == "list"
    assert sr_type.definition["daily_rain"]["elements"]["chance"]["type"] == "enum"
    assert sr_type.definition["daily_rain"]["elements"]["chance"]["choices"] == ["low", "medium", "high"]
    assert sr_type.definition["daily_rain"]["elements"]["when"]["type"] == "str"
    assert sr_type.definition["daily_rain"]["elements"]["confidence"]["ge"] == 0.0
    assert sr_type.definition["daily_rain"]["elements"]["confidence"]["le"] == 1.0
    
    # Test with instance
    rain_data = [
        RainProbability(chance=TestEnums.ChanceScale.LOW, when="morning", confidence=0.8),
        RainProbability(chance=TestEnums.ChanceScale.HIGH, when="evening", confidence=0.9)
    ]
    
    forecast_instance = WeatherForecast(
        location="Paris, France",
        daily_rain=rain_data,
        optional_hourly=None
    )
    
    sr_instance = from_basemodel(forecast_instance)
    assert sr_instance.instance["location"] == "Paris, France"
    assert len(sr_instance.instance["daily_rain"]) == 2
    assert sr_instance.instance["daily_rain"][0]["chance"] == "low"
    assert sr_instance.instance["daily_rain"][0]["when"] == "morning"
    assert sr_instance.instance["daily_rain"][0]["confidence"] == 0.8
    assert sr_instance.instance["daily_rain"][1]["chance"] == "high"
    assert sr_instance.instance["optional_hourly"] is None
    
    print("✓ List of BaseModels test passed")


def test_deeply_nested_structures():
    """Test deeply nested BaseModel structures."""
    print("=== TESTING DEEPLY NESTED STRUCTURES ===")
    
    class ContactInfo(BaseModel):
        email: str = Field(description="Email address")
        phone: Optional[str] = Field(description="Phone number")
    
    class Department(BaseModel):
        name: str = Field(description="Department name")
        budget: float = Field(description="Department budget", ge=0.0)
        head_contact: ContactInfo = Field(description="Department head contact")
    
    class Employee(BaseModel):
        employee_id: str = Field(description="Unique employee ID")
        name: str = Field(description="Employee name")
        contact: ContactInfo = Field(description="Employee contact information")
        department: Department = Field(description="Employee's department")
        skills: List[str] = Field(description="List of employee skills")
        certifications: Optional[List[str]] = Field(description="Optional certifications")
    
    class Company(BaseModel):
        company_name: str = Field(description="Company name")
        employees: List[Employee] = Field(description="List of company employees")
        departments: List[Department] = Field(description="List of company departments")
        headquarters: Optional[str] = Field(description="Headquarters location")
    
    # Test with type
    sr_type = from_basemodel(Company)
    assert sr_type.class_name == "Company"
    
    # Check deeply nested structure
    employee_def = sr_type.definition["employees"]["elements"]
    assert employee_def["contact"]["type"] == "dict"
    assert employee_def["contact"]["elements"]["email"]["type"] == "str"
    assert employee_def["contact"]["elements"]["phone"]["type"] == "str"
    assert employee_def["department"]["type"] == "dict"
    assert employee_def["department"]["elements"]["head_contact"]["type"] == "dict"
    assert employee_def["department"]["elements"]["head_contact"]["elements"]["email"]["type"] == "str"
    
    # Test with instance
    contact1 = ContactInfo(email="john@company.com", phone="123-456-7890")
    contact2 = ContactInfo(email="jane@company.com", phone=None)
    head_contact = ContactInfo(email="head@company.com", phone="098-765-4321")
    
    dept1 = Department(name="Engineering", budget=500000.0, head_contact=head_contact)
    dept2 = Department(name="Sales", budget=300000.0, head_contact=head_contact)
    
    emp1 = Employee(
        employee_id="EMP001",
        name="John Doe",
        contact=contact1,
        department=dept1,
        skills=["Python", "JavaScript"],
        certifications=["AWS", "Docker"]
    )
    
    emp2 = Employee(
        employee_id="EMP002",
        name="Jane Smith",
        contact=contact2,
        department=dept2,
        skills=["Sales", "Marketing"],
        certifications=None
    )
    
    company_instance = Company(
        company_name="Tech Corp",
        employees=[emp1, emp2],
        departments=[dept1, dept2],
        headquarters="San Francisco, CA"
    )
    
    sr_instance = from_basemodel(company_instance)
    assert sr_instance.instance["company_name"] == "Tech Corp"
    assert len(sr_instance.instance["employees"]) == 2
    assert sr_instance.instance["employees"][0]["name"] == "John Doe"
    assert sr_instance.instance["employees"][0]["contact"]["email"] == "john@company.com"
    assert sr_instance.instance["employees"][0]["contact"]["phone"] == "123-456-7890"
    assert sr_instance.instance["employees"][1]["contact"]["phone"] is None
    assert sr_instance.instance["employees"][0]["department"]["name"] == "Engineering"
    assert sr_instance.instance["employees"][0]["department"]["head_contact"]["email"] == "head@company.com"
    assert sr_instance.instance["employees"][0]["skills"] == ["Python", "JavaScript"]
    assert sr_instance.instance["employees"][0]["certifications"] == ["AWS", "Docker"]
    assert sr_instance.instance["employees"][1]["certifications"] is None
    
    print("✓ Deeply nested structures test passed")


def test_complex_weather_example():
    """Test the complex weather example from README."""
    print("=== TESTING COMPLEX WEATHER EXAMPLE ===")
    
    class RainProbability(BaseModel):
        chance: TestEnums.ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
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
    
    # Test with type
    sr_type = from_basemodel(WeatherPrognosis)
    assert sr_type.class_name == "WeatherPrognosis"
    
    # Check overall_rain_prob structure
    assert sr_type.definition["overall_rain_prob"]["type"] == "dict"
    assert sr_type.definition["overall_rain_prob"]["elements"]["chance"]["type"] == "enum"
    assert sr_type.definition["overall_rain_prob"]["elements"]["chance"]["choices"] == ["low", "medium", "high"]
    
    # Check rain_probability_timebound structure
    assert sr_type.definition["rain_probability_timebound"]["type"] == "list"
    assert sr_type.definition["rain_probability_timebound"]["elements"]["chance"]["type"] == "enum"
    assert sr_type.definition["rain_probability_timebound"]["elements"]["when"]["type"] == "str"
    
    # Check constraints
    assert sr_type.definition["high"]["ge"] == -20
    assert sr_type.definition["high"]["le"] == 60
    
    # Test with instance
    overall_rain = RainProbability(chance=TestEnums.ChanceScale.MEDIUM, when="today")
    timebound_rain = [
        RainProbability(chance=TestEnums.ChanceScale.LOW, when="morning"),
        RainProbability(chance=TestEnums.ChanceScale.HIGH, when="evening")
    ]
    
    weather_instance = WeatherPrognosis(
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
    
    sr_instance = from_basemodel(weather_instance)
    assert sr_instance.instance["location"] == "Annecy, FR"
    assert sr_instance.instance["current_temperature"] == 18.7
    assert sr_instance.instance["overall_rain_prob"]["chance"] == "medium"
    assert sr_instance.instance["overall_rain_prob"]["when"] == "today"
    assert len(sr_instance.instance["rain_probability_timebound"]) == 2
    assert sr_instance.instance["rain_probability_timebound"][0]["chance"] == "low"
    assert sr_instance.instance["rain_probability_timebound"][1]["chance"] == "high"
    assert sr_instance.instance["hourly_index"] == [3, 4, 5, 6, 5, 4, 3, 2]
    assert sr_instance.instance["high"] == 24.0
    assert sr_instance.instance["storm_tonight"] is False
    
    print("✓ Complex weather example test passed")


def test_edge_cases():
    """Test various edge cases and corner scenarios."""
    print("=== TESTING EDGE CASES ===")
    
    # Empty BaseModel
    class EmptyModel(BaseModel):
        pass
    
    sr_empty = from_basemodel(EmptyModel)
    assert sr_empty.class_name == "EmptyModel"
    assert sr_empty.definition == {}
    
    # Model with only optional fields
    class OptionalOnlyModel(BaseModel):
        optional_field1: Optional[str] = Field(description="Optional field 1")
        optional_field2: Optional[int] = Field(description="Optional field 2")
    
    sr_optional = from_basemodel(OptionalOnlyModel)
    assert sr_optional.definition["optional_field1"]["type"] == "str"
    assert sr_optional.definition["optional_field2"]["type"] == "int"
    
    # Instance with all None values
    optional_instance = OptionalOnlyModel(optional_field1=None, optional_field2=None)
    sr_optional_instance = from_basemodel(optional_instance)
    assert sr_optional_instance.instance["optional_field1"] is None
    assert sr_optional_instance.instance["optional_field2"] is None
    
    # Model with empty lists
    class EmptyListsModel(BaseModel):
        empty_string_list: List[str] = Field(description="Empty string list")
        empty_int_list: List[int] = Field(description="Empty int list")
    
    empty_lists_instance = EmptyListsModel(empty_string_list=[], empty_int_list=[])
    sr_empty_lists = from_basemodel(empty_lists_instance)
    assert sr_empty_lists.instance["empty_string_list"] == []
    assert sr_empty_lists.instance["empty_int_list"] == []
    
    print("✓ Edge cases test passed")


def test_multiple_constraint_types():
    """Test fields with multiple constraint types."""
    print("=== TESTING MULTIPLE CONSTRAINT TYPES ===")
    
    class ConstraintsModel(BaseModel):
        bounded_int: int = Field(description="Integer with multiple constraints", ge=10, le=100, multiple_of=5)
        length_string: str = Field(description="String with length constraints", min_length=3, max_length=20)
        range_float: float = Field(description="Float with range constraints", gt=0.0, lt=1000.0)
    
    # Test with type
    sr_type = from_basemodel(ConstraintsModel)
    
    bounded_int_def = sr_type.definition["bounded_int"]
    assert bounded_int_def["ge"] == 10
    assert bounded_int_def["le"] == 100
    assert bounded_int_def["multiple_of"] == 5
    
    length_string_def = sr_type.definition["length_string"]
    assert length_string_def["min_length"] == 3
    assert length_string_def["max_length"] == 20
    
    range_float_def = sr_type.definition["range_float"]
    assert range_float_def["gt"] == 0.0
    assert range_float_def["lt"] == 1000.0
    
    # Test with instance
    constraints_instance = ConstraintsModel(
        bounded_int=50,
        length_string="hello",
        range_float=123.45
    )
    
    sr_instance = from_basemodel(constraints_instance)
    assert sr_instance.instance["bounded_int"] == 50
    assert sr_instance.instance["length_string"] == "hello"
    assert sr_instance.instance["range_float"] == 123.45
    
    print("✓ Multiple constraint types test passed")


def test_circular_reference_prevention():
    """Test that we handle potential circular references gracefully."""
    print("=== TESTING CIRCULAR REFERENCE PREVENTION ===")
    
    # This is a simplified test - in practice, we'd need more sophisticated handling
    class Node(BaseModel):
        value: str = Field(description="Node value")
        metadata: Dict[str, str] = Field(description="Node metadata", default_factory=dict)
        tags: List[str] = Field(description="Node tags", default_factory=list)
    
    # Test with type
    sr_type = from_basemodel(Node)
    assert sr_type.class_name == "Node"
    assert sr_type.definition["value"]["type"] == "str"
    
    # Test with instance
    node_instance = Node(
        value="test_node",
        metadata={"key1": "value1", "key2": "value2"},
        tags=["tag1", "tag2"]
    )
    
    sr_instance = from_basemodel(node_instance)
    assert sr_instance.instance["value"] == "test_node"
    assert sr_instance.instance["metadata"] == {"key1": "value1", "key2": "value2"}
    assert sr_instance.instance["tags"] == ["tag1", "tag2"]
    
    print("✓ Circular reference prevention test passed")


def test_mixed_complex_types():
    """Test models mixing various complex types."""
    print("=== TESTING MIXED COMPLEX TYPES ===")
    
    class Task(BaseModel):
        id: str = Field(description="Task ID")
        priority: TestEnums.Priority = Field(description="Task priority")
        completed: bool = Field(description="Whether task is completed")
    
    class Project(BaseModel):
        name: str = Field(description="Project name")
        tasks: List[Task] = Field(description="Project tasks")
        metadata: Dict[str, Any] = Field(description="Project metadata", default_factory=dict)
        status: TestEnums.Status = Field(description="Project status")
        tags: List[str] = Field(description="Project tags")
        priority_counts: Dict[str, int] = Field(description="Count by priority", default_factory=dict)
    
    class Portfolio(BaseModel):
        owner: str = Field(description="Portfolio owner")
        projects: List[Project] = Field(description="List of projects")
        active_project: Optional[Project] = Field(description="Currently active project")
        summary_stats: Dict[str, float] = Field(description="Summary statistics", default_factory=dict)
    
    # Test with type
    sr_type = from_basemodel(Portfolio)
    assert sr_type.class_name == "Portfolio"
    
    # Check nested structures
    projects_def = sr_type.definition["projects"]["elements"]
    assert projects_def["tasks"]["type"] == "list"
    assert projects_def["tasks"]["elements"]["priority"]["type"] == "enum"
    assert projects_def["tasks"]["elements"]["priority"]["choices"] == ["urgent", "high", "normal", "low"]
    
    # Test with instance
    task1 = Task(id="T001", priority=TestEnums.Priority.HIGH, completed=False)
    task2 = Task(id="T002", priority=TestEnums.Priority.NORMAL, completed=True)
    
    project1 = Project(
        name="Project Alpha",
        tasks=[task1, task2],
        metadata={"created": "2024-01-01", "owner": "team1"},
        status=TestEnums.Status.ACTIVE,
        tags=["important", "frontend"],
        priority_counts={"high": 1, "normal": 1}
    )
    
    portfolio_instance = Portfolio(
        owner="John Manager",
        projects=[project1],
        active_project=project1,
        summary_stats={"completion_rate": 0.5, "avg_priority": 2.5}
    )
    
    sr_instance = from_basemodel(portfolio_instance)
    assert sr_instance.instance["owner"] == "John Manager"
    assert len(sr_instance.instance["projects"]) == 1
    assert sr_instance.instance["projects"][0]["name"] == "Project Alpha"
    assert len(sr_instance.instance["projects"][0]["tasks"]) == 2
    assert sr_instance.instance["projects"][0]["tasks"][0]["priority"] == "high"
    assert sr_instance.instance["projects"][0]["tasks"][1]["completed"] is True
    assert sr_instance.instance["active_project"]["name"] == "Project Alpha"
    
    print("✓ Mixed complex types test passed")


def test_prompt_generation():
    """Test that the generated definitions work correctly with to_prompt()."""
    print("=== TESTING PROMPT GENERATION ===")
    
    class SimpleModel(BaseModel):
        name: str = Field(description="The name")
        count: int = Field(description="The count", ge=0)
        active: bool = Field(description="Whether active")
        priority: TestEnums.Priority = Field(description="The priority level")
        tags: List[str] = Field(description="List of tags")
    
    sr = from_basemodel(SimpleModel)
    prompt = sr.to_prompt()
    
    # Basic checks that prompt contains expected elements
    assert "<simple_model>" in prompt
    assert "type='str'" in prompt
    assert "type='int'" in prompt
    assert "type='bool'" in prompt
    assert "type='enum'" in prompt
    assert "type='list'" in prompt
    assert "greater_or_equal='0'" in prompt
    assert 'choices=\'["urgent", "high", "normal", "low"]\'' in prompt
    
    print("✓ Prompt generation test passed")


def test_serialization_roundtrip():
    """Test that serialization and deserialization work correctly."""
    print("=== TESTING SERIALIZATION ROUNDTRIP ===")
    
    class TestModel(BaseModel):
        name: str = Field(description="Test name")
        value: int = Field(description="Test value", ge=0)
        active: bool = Field(description="Test active flag")
    
    # Test with instance
    instance = TestModel(name="test", value=42, active=True)
    sr_original = from_basemodel(instance)
    
    # Serialize and deserialize
    serialized = sr_original.serialize()
    sr_restored = StructuredResponse.deserialize(serialized)
    
    # Check that everything is preserved
    assert sr_restored.class_name == sr_original.class_name
    assert sr_restored.definition == sr_original.definition
    assert sr_restored.instance == sr_original.instance
    
    print("✓ Serialization roundtrip test passed")


def test_field_without_description():
    """Test fields that don't have descriptions."""
    print("=== TESTING FIELDS WITHOUT DESCRIPTION ===")
    
    class NoDescModel(BaseModel):
        field_with_desc: str = Field(description="This has a description")
        field_without_desc: str
        int_field: int
        enum_field: TestEnums.Status
    
    sr = from_basemodel(NoDescModel)
    
    assert sr.definition["field_with_desc"]["description"] == "This has a description"
    assert sr.definition["field_without_desc"]["description"] == ""
    assert sr.definition["int_field"]["description"] == ""
    assert sr.definition["enum_field"]["description"] == ""
    
    print("✓ Fields without description test passed")


def test_all_constraint_types():
    """Test all supported constraint types from Pydantic."""
    print("=== TESTING ALL CONSTRAINT TYPES ===")
    
    class AllConstraintsModel(BaseModel):
        ge_field: int = Field(description="Greater or equal", ge=10)
        gt_field: int = Field(description="Greater than", gt=0)
        le_field: int = Field(description="Less or equal", le=100)
        lt_field: int = Field(description="Less than", lt=50)
        multiple_of_field: int = Field(description="Multiple of", multiple_of=5)
        min_length_field: str = Field(description="Minimum length", min_length=3)
        max_length_field: str = Field(description="Maximum length", max_length=10)
        combined_constraints: float = Field(description="Combined constraints", gt=0.0, le=1.0)
    
    sr = from_basemodel(AllConstraintsModel)
    
    assert sr.definition["ge_field"]["ge"] == 10
    assert sr.definition["gt_field"]["gt"] == 0
    assert sr.definition["le_field"]["le"] == 100
    assert sr.definition["lt_field"]["lt"] == 50
    assert sr.definition["multiple_of_field"]["multiple_of"] == 5
    assert sr.definition["min_length_field"]["min_length"] == 3
    assert sr.definition["max_length_field"]["max_length"] == 10
    assert sr.definition["combined_constraints"]["gt"] == 0.0
    assert sr.definition["combined_constraints"]["le"] == 1.0
    
    print("✓ All constraint types test passed")


def run_all_tests():
    """Run all test functions."""
    print("Starting comprehensive from_basemodel tests...\n")
    
    test_functions = [
        test_basic_primitive_types,
        test_enum_types,
        test_simple_list_types,
        test_nested_basemodel,
        test_list_of_basemodels,
        test_deeply_nested_structures,
        test_complex_weather_example,
        test_edge_cases,
        test_multiple_constraint_types,
        test_prompt_generation,
        test_serialization_roundtrip,
        test_field_without_description,
        test_all_constraint_types,
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
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
