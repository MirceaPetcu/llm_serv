from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from pydantic import Field

from llm_serv.structured_response.model import StructuredResponse


class StatusEnum(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class PriorityEnum(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Address(StructuredResponse):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    postal_code: str = Field(description="Postal code")
    country: str = Field(default="USA", description="Country code")


class Contact(StructuredResponse):
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    address: Optional[Address] = Field(default=None, description="Physical address")


class Task(StructuredResponse):
    id: int = Field(description="Task ID")
    title: str = Field(description="Task title")
    description: Optional[str] = Field(default=None, description="Task description")
    priority: PriorityEnum = Field(default=PriorityEnum.MEDIUM, description="Task priority")
    tags: list[str] = Field(default=[], description="List of tags")
    estimated_hours: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Estimated hours")
    is_completed: bool = Field(default=False, description="Completion status")


class Project(StructuredResponse):
    name: str = Field(description="Project name")
    status: StatusEnum = Field(default=StatusEnum.PENDING, description="Project status")
    tasks: list[Task] = Field(default=[], description="List of project tasks")
    budget: Optional[float] = Field(default=None, ge=0.0, description="Project budget")
    team_members: list[str] = Field(default=[], description="List of team member names")


class ComplicatedTestClass(StructuredResponse):
    # Basic types
    basic_string: str = Field(description="A basic string field")
    basic_int: int = Field(default=42, ge=0, le=1000, description="A basic integer field")
    basic_float: float = Field(default=3.14, description="A basic float field")
    basic_bool: bool = Field(default=True, description="A basic boolean field")
    
    # Optional basic types
    optional_string: Optional[str] = Field(default=None, description="An optional string field")
    optional_int: Optional[int] = Field(default=None, description="An optional integer field")
    optional_float: Optional[float] = Field(default=None, description="An optional float field")
    
    # Date/time fields
    creation_date: date = Field(description="Creation date")
    last_modified: datetime = Field(description="Last modification timestamp")
    daily_meeting_time: time = Field(description="Daily meeting time")
    optional_deadline: Optional[datetime] = Field(default=None, description="Optional deadline")
    
    # Enums
    status: StatusEnum = Field(default=StatusEnum.ACTIVE, description="Current status")
    priority: Optional[PriorityEnum] = Field(default=None, description="Optional priority")
    
    # Lists of basic types
    tags: list[str] = Field(default=[], description="List of string tags")
    scores: list[int] = Field(default=[1, 2, 3], description="List of integer scores")
    weights: list[float] = Field(default=[1.0, 2.5, 3.7], description="List of float weights")
    flags: list[bool] = Field(default=[True, False], description="List of boolean flags")
    
    # Optional lists
    optional_categories: Optional[list[str]] = Field(default=None, description="Optional list of categories")
    optional_numbers: Optional[list[int]] = Field(default=None, description="Optional list of numbers")
    
    # Nested single objects
    primary_contact: Contact = Field(description="Primary contact information")
    optional_address: Optional[Address] = Field(default=None, description="Optional address")
    
    # Lists of nested objects
    all_contacts: list[Contact] = Field(default=[], description="List of all contacts")
    projects: list[Project] = Field(default=[], description="List of projects")
    
    # Optional lists of nested objects
    optional_backup_contacts: Optional[list[Contact]] = Field(default=None, description="Optional backup contacts")
    archived_projects: Optional[list[Project]] = Field(default=None, description="Optional archived projects")


def create_test_instance():
    """Create a fully populated test instance with complex nested data."""
    
    # Create addresses
    main_address = Address(
        street="123 Main St",
        city="New York",
        postal_code="10001",
        country="USA"
    )
    
    secondary_address = Address(
        street="456 Oak Ave",
        city="San Francisco", 
        postal_code="94102",
        country="USA"
    )
    
    # Create contacts
    primary_contact = Contact(
        email="john.doe@example.com",
        phone="+1-555-0123",
        address=main_address
    )
    
    secondary_contact = Contact(
        email="jane.smith@example.com",
        phone=None,
        address=secondary_address
    )
    
    backup_contact = Contact(
        email="backup@example.com",
        phone="+1-555-9999",
        address=None
    )
    
    # Create tasks
    task1 = Task(
        id=1,
        title="Design UI mockups",
        description="Create wireframes and mockups for the new feature",
        priority=PriorityEnum.HIGH,
        tags=["design", "ui", "mockups"],
        estimated_hours=16.5,
        is_completed=True
    )
    
    task2 = Task(
        id=2,
        title="Implement backend API",
        description=None,
        priority=PriorityEnum.CRITICAL,
        tags=["backend", "api", "development"],
        estimated_hours=40.0,
        is_completed=False
    )
    
    task3 = Task(
        id=3,
        title="Write documentation",
        priority=PriorityEnum.LOW,
        tags=["docs"],
        estimated_hours=None,
        is_completed=False
    )
    
    # Create projects
    project1 = Project(
        name="Mobile App Redesign",
        status=StatusEnum.ACTIVE,
        tasks=[task1, task2],
        budget=50000.0,
        team_members=["Alice", "Bob", "Charlie"]
    )
    
    project2 = Project(
        name="Legacy System Migration",
        status=StatusEnum.PENDING,
        tasks=[task3],
        budget=None,
        team_members=["Dave", "Eve"]
    )
    
    archived_project = Project(
        name="Old Website",
        status=StatusEnum.INACTIVE,
        tasks=[],
        budget=5000.0,
        team_members=["Former Employee"]
    )
    
    # Create the main test instance
    test_instance = ComplicatedTestClass(
        basic_string="Hello, World!",
        basic_int=100,
        basic_float=99.99,
        basic_bool=False,
        
        optional_string="Optional value",
        optional_int=None,
        optional_float=2.718,
        
        creation_date=date(2024, 1, 15),
        last_modified=datetime(2024, 1, 20, 14, 30, 45),
        daily_meeting_time=time(9, 30, 0),
        optional_deadline=datetime(2024, 3, 1, 23, 59, 59),
        
        status=StatusEnum.ACTIVE,
        priority=PriorityEnum.HIGH,
        
        tags=["important", "urgent", "client-facing"],
        scores=[85, 92, 78, 96],
        weights=[0.1, 0.3, 0.4, 0.2],
        flags=[True, True, False, True],
        
        optional_categories=["category1", "category2"],
        optional_numbers=None,
        
        primary_contact=primary_contact,
        optional_address=main_address,
        
        all_contacts=[primary_contact, secondary_contact],
        projects=[project1, project2],
        
        optional_backup_contacts=[backup_contact],
        archived_projects=[archived_project]
    )
    
    return test_instance


if __name__ == "__main__":
    # Create and test the complicated instance
    test_instance = create_test_instance()
    
    print("=== Testing __str__ method (XML output) ===")
    print(str(test_instance))
    print(test_instance.to_xml())
    print(test_instance.to_xml(exclude={"basic_string", "primary_contact"}))
    