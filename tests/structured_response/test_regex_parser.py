import pytest

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field
from llm_serv.structured_response.model import StructuredResponse


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ContactMethod(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SLACK = "slack"
    TEAMS = "teams"


class Department(BaseModel):
    name: str = Field(description="Department name")
    budget: float = Field(description="Department annual budget")
    head_count: int = Field(description="Number of employees")
    active: bool = Field(description="Whether department is active")


class Contact(BaseModel):
    name: str = Field(description="Contact person name")
    method: ContactMethod = Field(description="Preferred contact method")
    value: str = Field(description="Contact value (email, phone, etc.)")
    primary: bool = Field(description="Whether this is primary contact")


class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    postal_code: Optional[str] = Field(description="Postal code")
    coordinates: Optional[list[float]] = Field(description="GPS coordinates [lat, lng]")


class Resource(BaseModel):
    name: str = Field(description="Resource name")
    type: str = Field(description="Resource type")
    available: bool = Field(description="Whether resource is available")
    cost_per_hour: Optional[float] = Field(description="Cost per hour in USD")
    skills: list[str] = Field(description="List of skills")


class Milestone(BaseModel):
    title: str = Field(description="Milestone title")
    description: Optional[str] = Field(description="Milestone description")
    due_date: str = Field(description="Due date in ISO format")
    completed: bool = Field(description="Whether milestone is completed")
    priority: Priority = Field(description="Milestone priority")
    dependencies: list[str] = Field(description="List of dependent milestone IDs")


class Task(BaseModel):
    id: str = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: Optional[str] = Field(description="Task description")
    status: Status = Field(description="Current task status")
    priority: Priority = Field(description="Task priority")
    estimated_hours: Optional[float] = Field(description="Estimated hours to complete")
    actual_hours: Optional[float] = Field(description="Actual hours spent")
    assigned_to: Optional[str] = Field(description="Person assigned to task")
    tags: list[str] = Field(description="Task tags")
    subtasks: Optional[list[str]] = Field(description="List of subtask IDs")

class SuccessMetric(BaseModel):
    name: str = Field(description="Success metric name")
    value: Union[str, float, int] = Field(description="Success metric value")

class Project(BaseModel):
    id: str = Field(description="Unique project identifier")
    name: str = Field(description="Project name")
    description: Optional[str] = Field(description="Project description")
    status: Status = Field(description="Current project status")
    priority: Priority = Field(description="Project priority")
    start_date: str = Field(description="Project start date")
    end_date: Optional[str] = Field(description="Project end date")
    budget: float = Field(description="Project budget in USD")
    spent: Optional[float] = Field(description="Amount spent so far")
    department: Department = Field(description="Owning department")
    address: Optional[Address] = Field(description="Project location")
    contacts: list[Contact] = Field(description="Project contacts")
    tasks: list[Task] = Field(description="Project tasks")
    milestones: list[Milestone] = Field(description="Project milestones")
    resources: Optional[list[Resource]] = Field(description="Assigned resources")
    risk_factors: list[str] = Field(description="Identified risk factors")
    metadata: str = Field(description="Additional metadata")


class SuperComplexProjectManagement(BaseModel):
    """
    A super complex class for testing the regex parser with:
    - Deep nesting (6+ levels)
    - Multiple list types (simple strings, complex objects)
    - Multiple enum types
    - Optional fields at various levels
    - Mixed data types (str, int, float, bool)
    - Dictionaries with various value types
    - Edge cases like empty lists, null values
    """
    organization_name: str = Field(description="Organization name")
    fiscal_year: int = Field(description="Current fiscal year")
    total_budget: float = Field(description="Total organization budget")
    active: bool = Field(description="Whether organization is active")
    headquarters: Optional[Address] = Field(description="Main headquarters address")
    departments: Optional[list[Department]] = Field(description="List of departments")
    projects: Optional[list[Project]] = Field(description="List of active projects")
    global_contacts: Optional[list[Contact]] = Field(description="Global organization contacts")
    strategic_priorities: Optional[list[str]] = Field(description="High-level strategic priorities")
    quarterly_targets: Optional[list[str]] = Field(description="Quarterly financial targets")
    compliance_certifications: Optional[list[str]] = Field(description="ISO certifications etc.")
    board_members: Optional[list[str]] = Field(description="Board member names")
    subsidiary_companies: Optional[list[str]] = Field(description="Subsidiary company names")
    annual_reports: Optional[str] = Field(description="Links to annual reports by year")


def test_simple_parsing():
    """Test basic parsing functionality."""
    simple_xml = """
    <super_complex_project_management>
        <organization_name>TechCorp Industries</organization_name>
        <fiscal_year>2024</fiscal_year>
        <total_budget>50000000.0</total_budget>
        <active>true</active>
    </super_complex_project_management>
    """
    
    sr = StructuredResponse.from_basemodel(SuperComplexProjectManagement)
    result = sr.from_prompt(simple_xml)
    
    assert result.instance['organization_name'] == "TechCorp Industries"
    assert result.instance['fiscal_year'] == 2024
    assert result.instance['total_budget'] == 50000000.0
    assert result.instance['active'] is True
    print("✓ Simple parsing test passed")


def test_complex_nested_structure():
    """Test parsing of deeply nested structures with lists and complex objects."""
    complex_xml = """
    <super_complex_project_management>
        <organization_name>MegaCorp Solutions</organization_name>
        <fiscal_year>2024</fiscal_year>
        <total_budget>100000000.5</total_budget>
        <active>true</active>
        
        <headquarters>
            <address>
            <street>123 Innovation Drive</street>
            <city>San Francisco</city>
            <country>USA</country>
            <postal_code>94105</postal_code>
            <coordinates>
                <li>37.7749</li>
                <li>-122.4194</li>
            </coordinates>
            </address>
        </headquarters>
        
        <departments>
            <li>
                <name>Engineering</name>
                <budget>25000000.0</budget>
                <head_count>150</head_count>
                <active>true</active>
            </li>
            <li>
                <name>Marketing</name>
                <budget>5000000.0</budget>
                <head_count>45</head_count>
                <active>true</active>
            </li>
            <li>
                <name>Sales</name>
                <budget>8000000.0</budget>
                <head_count>75</head_count>
                <active>true</active>
            </li>
        </departments>
        
        <projects>
            <li>
                <id>PROJ-001</id>
                <name>AI Platform Development</name>
                <description>Building next-generation AI platform</description>
                <status>in_progress</status>
                <priority>high</priority>
                <start_date>2024-01-15</start_date>
                <end_date>2024-12-31</end_date>
                <budget>15000000.0</budget>
                <spent>8500000.0</spent>
                
                <department>
                    <name>Engineering</name>
                    <budget>25000000.0</budget>
                    <head_count>150</head_count>
                    <active>true</active>
                </department>
                
                <address>
                    <street>456 Tech Blvd</street>
                    <city>Austin</city>
                    <country>USA</country>
                    <postal_code>78701</postal_code>
                    <coordinates>
                        <li>30.2672</li>
                        <li>-97.7431</li>
                    </coordinates>
                </address>
                
                <contacts>
                    <li>
                        <name>John Smith</name>
                        <method>email</method>
                        <value>john.smith@megacorp.com</value>
                        <primary>true</primary>
                    </li>
                    <li>
                        <name>Sarah Johnson</name>
                        <method>slack</method>
                        <value>@sarah.johnson</value>
                        <primary>false</primary>
                    </li>
                    <li>
                        <name>Mike Chen</name>
                        <method>phone</method>
                        <value>+1-555-0123</value>
                        <primary>false</primary>
                    </li>
                </contacts>
                
                <tasks>
                    <li>
                        <id>TASK-001</id>
                        <title>Design AI Architecture</title>
                        <description>Create high-level system architecture</description>
                        <status>completed</status>
                        <priority>critical</priority>
                        <estimated_hours>160.0</estimated_hours>
                        <actual_hours>180.5</actual_hours>
                        <assigned_to>John Smith</assigned_to>
                        <tags>
                            <li>architecture</li>
                            <li>ai</li>
                            <li>design</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-001</li>
                            <li>SUBTASK-002</li>
                        </subtasks>
                    </li>
                    <li>
                        <id>TASK-002</id>
                        <title>Implement Core ML Pipeline</title>
                        <description>Build the machine learning pipeline infrastructure</description>
                        <status>in_progress</status>
                        <priority>high</priority>
                        <estimated_hours>240.0</estimated_hours>
                        <actual_hours>120.0</actual_hours>
                        <assigned_to>Sarah Johnson</assigned_to>
                        <tags>
                            <li>ml</li>
                            <li>pipeline</li>
                            <li>infrastructure</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-003</li>
                            <li>SUBTASK-004</li>
                            <li>SUBTASK-005</li>
                        </subtasks>
                    </li>
                </tasks>
                
                <milestones>
                    <li>
                        <title>Alpha Release</title>
                        <description>First working prototype</description>
                        <due_date>2024-06-30</due_date>
                        <completed>true</completed>
                        <priority>high</priority>
                        <dependencies>
                            <li>MILESTONE-PREP-001</li>
                        </dependencies>
                    </li>
                    <li>
                        <title>Beta Release</title>
                        <description>Feature-complete beta version</description>
                        <due_date>2024-09-30</due_date>
                        <completed>false</completed>
                        <priority>high</priority>
                        <dependencies>
                            <li>MILESTONE-ALPHA-001</li>
                            <li>MILESTONE-TESTING-001</li>
                        </dependencies>
                    </li>
                    <li>
                        <title>Production Release</title>
                        <description>Full production deployment</description>
                        <due_date>2024-12-31</due_date>
                        <completed>false</completed>
                        <priority>critical</priority>
                        <dependencies>
                            <li>MILESTONE-BETA-001</li>
                            <li>MILESTONE-SECURITY-001</li>
                        </dependencies>
                    </li>
                </milestones>
                
                <resources>
                    <li>
                        <name>Senior AI Engineer</name>
                        <type>human</type>
                        <available>true</available>
                        <cost_per_hour>150.0</cost_per_hour>
                        <skills>
                            <li>Python</li>
                            <li>TensorFlow</li>
                            <li>MLOps</li>
                        </skills>
                    </li>
                    <li>
                        <name>Cloud Infrastructure</name>
                        <type>computing</type>
                        <available>true</available>
                        <cost_per_hour>45.0</cost_per_hour>
                        <skills>
                            <li>AWS</li>
                            <li>Kubernetes</li>
                            <li>Docker</li>
                        </skills>
                    </li>
                    <li>
                        <name>ML Training Cluster</name>
                        <type>hardware</type>
                        <available>false</available>
                        <cost_per_hour>200.0</cost_per_hour>
                        <skills>
                            <li>GPU Computing</li>
                            <li>Distributed Training</li>
                            <li>High Performance</li>
                        </skills>
                    </li>
                </resources>
                
                <risk_factors>
                    <li>Technical complexity</li>
                    <li>Resource availability</li>
                    <li>Market timing</li>
                    <li>Regulatory compliance</li>
                    <li>Data privacy concerns</li>
                    <li>Scalability challenges</li>
                </risk_factors>
                
                <metadata>
                    'created_by': John Smith, 
                    'last_updated': 2024-03-15, 
                    'version': 1.2.0, 
                    'environment': production, 
                    'compliance_level': enterprise, 
                    'security_clearance': confidential, 
                </metadata>
            </li>
            <li>
                <id>PROJ-002</id>
                <name>Customer Portal Redesign</name>
                <description>Complete overhaul of customer-facing web portal</description>
                <status>pending</status>
                <priority>medium</priority>
                <start_date>2024-04-01</start_date>
                <end_date>2024-08-31</end_date>
                <budget>3000000.0</budget>
                <spent>0.0</spent>
                
                <department>
                    <name>Marketing</name>
                    <budget>5000000.0</budget>
                    <head_count>45</head_count>
                    <active>true</active>
                </department>
                
                <address>
                    <street>789 Design Ave</street>
                    <city>New York</city>
                    <country>USA</country>
                    <postal_code>10001</postal_code>
                    <coordinates>
                        <li>40.7128</li>
                        <li>-74.0060</li>
                    </coordinates>
                </address>
                
                <contacts>
                    <li>
                        <name>Emma Davis</name>
                        <method>teams</method>
                        <value>emma.davis@megacorp.com</value>
                        <primary>true</primary>
                    </li>
                </contacts>
                
                <tasks>
                    <li>
                        <id>TASK-003</id>
                        <title>User Research</title>
                        <description>Conduct comprehensive user research and usability studies</description>
                        <status>pending</status>
                        <priority>high</priority>
                        <estimated_hours>80.0</estimated_hours>
                        <actual_hours>0.0</actual_hours>
                        <assigned_to>Emma Davis</assigned_to>
                        <tags>
                            <li>research</li>
                            <li>ux</li>
                            <li>usability</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-006</li>
                        </subtasks>
                    </li>
                </tasks>
                
                <milestones>
                    <li>
                        <title>Research Complete</title>
                        <description>User research and analysis finished</description>
                        <due_date>2024-05-15</due_date>
                        <completed>false</completed>
                        <priority>medium</priority>
                        <dependencies>
                            <li>MILESTONE-RESEARCH-001</li>
                        </dependencies>
                    </li>
                </milestones>
                
                <resources>
                    <li>
                        <name>UX Designer</name>
                        <type>human</type>
                        <available>true</available>
                        <cost_per_hour>95.0</cost_per_hour>
                        <skills>
                            <li>Figma</li>
                            <li>User Research</li>
                            <li>Prototyping</li>
                        </skills>
                    </li>
                </resources>
                
                <risk_factors>
                    <li>User adoption resistance</li>
                    <li>Design complexity</li>
                </risk_factors>
            
                
                <metadata>
                    'created_by': Emma Davis, 
                    'last_updated': 2024-03-20, 
                    'version': 0.1.0, 
                    'environment': development, 
                </metadata>
            </li>
        </projects>
        
        <global_contacts>
            <li>
                <name>CEO Alice Williams</name>
                <method>email</method>
                <value>alice.williams@megacorp.com</value>
                <primary>true</primary>
            </li>
            <li>
                <name>CTO Bob Martinez</name>
                <method>phone</method>
                <value>+1-555-0100</value>
                <primary>false</primary>
            </li>
            <li>
                <name>CFO Carol Chen</name>
                <method>teams</method>
                <value>carol.chen@megacorp.com</value>
                <primary>false</primary>
            </li>
        </global_contacts>
        
        <strategic_priorities>
            <li>AI Innovation</li>
            <li>Market Expansion</li>
            <li>Sustainability</li>
            <li>Digital Transformation</li>
            <li>Customer Experience</li>
            <li>Operational Excellence</li>
        </strategic_priorities>
        
        <quarterly_targets>
            <q1>25000000.0</q1>
            <q2>30000000.0</q2>
            <q3>35000000.0</q3>
            <q4>40000000.0</q4>
        </quarterly_targets>
        
        <compliance_certifications>
            <li>ISO 27001</li>
            <li>SOC 2 Type II</li>
            <li>GDPR Compliant</li>
            <li>HIPAA Compliant</li>
            <li>PCI DSS Level 1</li>
            <li>ISO 9001</li>
        </compliance_certifications>
        
        <board_members>
            <li>Dr. Richard Thompson</li>
            <li>Prof. Maria Rodriguez</li>
            <li>James Wilson III</li>
            <li>Lisa Kim</li>
            <li>Michael O'Brien</li>
        </board_members>
        
        <subsidiary_companies>
            <li>MegaCorp Europe Ltd</li>
            <li>MegaCorp Asia Pacific</li>
            <li>MegaCorp Canada Inc</li>
            <li>Innovation Labs LLC</li>
            <li>SecureData Solutions</li>
        </subsidiary_companies>
        
        <annual_reports>
            <year_2023>https://megacorp.com/reports/2023</year_2023>
            <year_2022>https://megacorp.com/reports/2022</year_2022>
            <year_2021>https://megacorp.com/reports/2021</year_2021>
            <year_2020>https://megacorp.com/reports/2020</year_2020>
        </annual_reports>
    </super_complex_project_management>
    """
    
    sr = StructuredResponse.from_basemodel(SuperComplexProjectManagement)
    result = sr.from_prompt(complex_xml)
    
    # Test basic fields
    assert result.instance['organization_name'] == "MegaCorp Solutions"
    assert result.instance['fiscal_year'] == 2024
    assert result.instance['active'] is True
    
    # Test nested object
    assert result.instance['headquarters']['city'] == "San Francisco"
    assert result.instance['headquarters']['coordinates'] == [37.7749, -122.4194]
    
    # Test list of objects
    assert len(result.instance['departments']) == 3
    assert result.instance['departments'][0]['name'] == "Engineering"
    assert result.instance['departments'][0]['head_count'] == 150
    
    # Test deeply nested structures
    project = result.instance['projects'][0]
    assert project['name'] == "AI Platform Development"
    assert project['status'] == "in_progress"
    assert project['priority'] == "high"
    
    # Test nested objects within lists
    task = project['tasks'][0]
    assert task['title'] == "Design AI Architecture"
    assert task['status'] == "completed"
    assert len(task['tags']) == 3
    assert "architecture" in task['tags']
    
    assert len(result.instance['global_contacts']) == 3
    assert result.instance['global_contacts'][0]['name'] == "CEO Alice Williams"
    assert result.instance['global_contacts'][0]['method'] == "email"
    assert result.instance['global_contacts'][0]['value'] == "alice.williams@megacorp.com"
    assert result.instance['global_contacts'][0]['primary'] is True
    
    assert result.instance['global_contacts'][1]['name'] == "CTO Bob Martinez"
    assert result.instance['global_contacts'][1]['method'] == "phone"
    assert result.instance['global_contacts'][1]['value'] == "+1-555-0100"
    assert result.instance['global_contacts'][1]['primary'] is False
    print("✓ Complex nested structure test passed")


def test_malformed_xml_handling():
    """Test parsing of malformed XML that the regex parser should handle."""
    malformed_xml = """
    <super_complex_project_management>
        <organization_name>MegaCorp Solutions</organization_name>
        <fiscal_year>2024</fiscal_year>
        <total_budget>100000000.5</total_budget>
        <active>true</active>
        
        <headquarters>
            <address>
            <street>123 Innovation Drive</street>
            <city>San Francisco</city>
            <country>USA</country>
            <postal_code>94105</postal_code>
            <coordinates>
                <li>37.7749</li>
                <li>-122.4194</li>
            </coordinates>
            </address>
        </headquarters>
        
        <departments>
            <li>
                <name>Engineering</name>
                <budget>25000000.0</budget>
                <head_count>150</head_count>
                <active>true</active>
            </li>
            <li>
                <name>Marketing</name>
                <budget>5000000.0</budget>
                <head_count>45</head_count>
                <active>true</active>
            </li>
            <li>
                <name>Sales</name>
                <budget>8000000.0</budget>
                <head_count>75</head_count>
                <active>true</active>
            </li>
        </departments>
        
        <projects>
            <li>
                <id>PROJ-001<id>
                <name>AI Platform Development</name>
                <description>Building next-generation AI platform</description>
                <status>in_progress</status>
                <priority>high</priority>
                <start_date>2024-01-15</start_date>
                <end_date>2024-12-31</end_date>
                <budget>15000000.0</budget>
                <spent>8500000.0</spent>
                
                <department>
                    <name>Engineering</name>
                    <budget>25000000.0</budget>
                    <head_count>150</head_count>
                    <active>true</active>
                </department>
                
                <address>
                    <street>456 Tech Blvd</street>
                    <city>Austin</city>
                    <country>USA</country>
                    <postal_code>78701</postal_code>
                    <coordinates>
                        <li>30.2672</li>
                        <li>-97.7431</li>
                    </coordinates>
                </address>
                
                <contacts>
                    <li>
                        <name>John Smith</name>
                        <method>email</method>
                        <value>john.smith@megacorp.com</value>
                        <primary>true</primary>
                    </li>
                    <li>
                        <name>Sarah Johnson</name>
                        <method>slack</method>
                        <value>@sarah.johnson</value>
                        <primary>false</primary>
                    </li>
                    <li>
                        <name>Mike Chen</name>
                        <method>phone</method>
                        <value>+1-555-0123</value>
                        <primary>false</primary>
                    </li>
                </contacts>
                
                <tasks>
                    <li>
                        <id>TASK-001</id>
                        <title>Design AI Architecture</title>
                        <description>Create high-level system architecture</description>
                        <status>completed</status>
                        <priority>critical</priority>
                        <estimated_hours>160.0</estimated_hours>
                        <actual_hours>180.5</actual_hours>
                        <assigned_to>John Smith</assigned_to>
                        <tags>
                            <li>architecture</li>
                            <li>ai</li>
                            <li>design</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-001</li>
                            <li>SUBTASK-002</li>
                        </subtasks>
                    </li>
                    <li>
                        <id>TASK-002</id>
                        <title>Implement Core ML Pipeline</title>
                        <description>Build the machine learning pipeline infrastructure</description>
                        <status>in_progress</status>
                        <priority>high</priority>
                        <estimated_hours>240.0</estimated_hours>
                        <actual_hours>120.0</actual_hours>
                        <assigned_to>Sarah Johnson</assigned_to>
                        <tags>
                            <li>ml</li>
                            <li>pipeline</li>
                            <li>infrastructure</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-003</li>
                            <li>SUBTASK-004</li>
                            <li>SUBTASK-005</li>
                        </subtasks>
                    </li>
                </tasks>
                
                <milestones>
                    <li>
                        <title>Alpha Release</title>
                        <description>First working prototype</description>
                        <due_date>2024-06-30</due_date>
                        <completed>true</completed>
                        <priority>high</priority>
                        <dependencies>
                            <li>MILESTONE-PREP-001</li>
                        </dependencies>
                    </li>
                    <li>
                        <title>Beta Release</tiksjaktlesaljhdjsahdjsa>
                        <description>Feature-complete beta version</description>
                        <due_date>2024-09-30</due_date>
                        <completed>false</completed>
                        <priority>high</priority>
                        <dependencies>
                            <li>MILESTONE-ALPHA-001</li>
                            <li>MILESTONE-TESTING-001</li>
                        </dependencies>
                    </li>
                    <li>
                        <title>Production Release</title>
                        <description>Full production deployment</description>
                        <due_date>2024-12-31</due_date>
                        <completed>false</completed>
                        <priority>critical</priority>
                        <dependencies>
                            <li>MILESTONE-BETA-001</li>
                            <li>MILESTONE-SECURITY-001</li>
                        </dependencies>
                    </li>
                </milestones>
                
                <resources>
                    <li>
                        <name>Senior AI Engineer</name>
                        <type>human
                        <available>true</available>
                        <cost_per_hour>150.0</cost_per_hour>
                        <skills>
                            <li>Python</li>
                            <li>TensorFlow</li>
                            <li>MLOps</li>
                        </skills>
                    </li>
                    <li>
                        <name>Cloud Infrastructure</name>
                        <type>computing</type>
                        <available>true
                        <cost_per_hour>45.0<cost_per_hour>
                        <skills>
                            <li>AWS <li> cloud infrastructure</li></li>
                            <li>Kubernetes</li>
                            <li>Docker</li>
                        </skills>
                    </li>
                    <li>
                        <name>ML Training Cluster</name>
                        <type>hardware</type>
                        <available>false</available>
                        <cost_per_hour>200.0</cost_per_hour>
                        <skills>
                            <li>GPU Computing</li>
                            <li>Distributed Training</li>
                            <li>High Performance</li>
                        </skills>
                    </li>
                </resources>
                
                <risk_factors>
                    <li>Technical complexity</li>
                    <li>Resource availability</li>
                    <li>Market timing</li>
                    <li>Regulatory compliance</li>
                    <li>Data privacy concerns</li>
                    <li>Scalability challenges</li>
                </risk_factors>
                
                <metadata>
                    'created_by': John Smith, 
                    'last_updated': 2024-03-15, 
                    'version': 1.2.0, 
                    'environment': production, 
                    'compliance_level': enterprise, 
                    'security_clearance': confidential, 
                </metadata>
            </li>
            <li>
                <id>PROJ-002</id>
                <name>Customer Portal Redesign</name>
                <description>Complete overhaul of customer-facing web portal</description>
                <status>pending</status>
                <priority>medium</priority>
                <start_date>2024-04-01</start_date>
                <end_date>2024-08-31</end_date>
                <budget>3000000.0</budget>
                <spent>0.0</spent>
                
                <department>
                    <name>Marketing</name>
                    <budget>5000000.0</budget>
                    <head_count>45</head_count>
                    <active>true</active>
                </department>
                
                <address>
                    <street>789 Design Ave</street>
                    <city>New York</city>
                    <country>USA</country>
                    <postal_code>10001</postal_code>
                    <coordinates>
                        <li>40.7128</li>
                        <li>-74.0060</li>
                    </coordinates>
                </address>
                
                <contacts>
                    <li>
                        <name>Emma Davis</name>
                        <method>teams</method>
                        <value>emma.davis@megacorp.com</value>
                        <primary>true</primary>
                    </li>
                
                <tasks, desc='sa nu cumva sa iei tagul asta <ref/> frumos'>
                    <li>
                        <id>TASK-003</id>
                        <title>User Research</title>
                        <description>Conduct comprehensive user research and usability studies</description>
                        <status>pending</status>
                        <priority>high</priority>
                        <estimated_hours>80.0</estimated_hours>
                        <actual_hours>0.0</actual_hours>
                        <assigned_to>Emma Davis</assigned_to>
                        <tags>
                            <li>research</li>
                            <li>ux</li>
                            <li>usability</li>
                        </tags>
                        <subtasks>
                            <li>SUBTASK-006</li>
                        </subtasks>
                    </li>
                </tasks>
                
                <milestones, description='it should be good'>
                    <li>
                        <title>Research Complete</title>
                        <description>User research and analysis finished</description>
                        <due_date>2024-05-15</due_date>
                        <completed>false</completed>
                        <priority>medium</priority>
                        <dependencies>
                            <li>MILESTONE-RESEARCH-001</li>
                        </dependencies>
                    </li>
                </milestones>
                
                <resources>
                    <li>
                        <name id=1>UX Designer</name>
                        <type id='3'3>human</type>
                        <available>true</available>
                        <cost_per_hour>95.0</cost_per_hour>
                        <skills>
                            <li>Figma</li>
                            <li>User Research</li>
                            <li>Prototyping</li>
                        </skills>
                    </li>
                
                <risk_factors id='5>
                    <li>User adoption resistance</li>
                    <li>Design complexity</li>
                </risk_factors>
            
                
                <metadata>
                    'created_by': Emma Davis, 
                    'last_updated': 2024-03-20, 
                    'version': 0.1.0, 
                    'environment': development, 
                </metadata>
            </li>
        </projects>
        
        <global_contacts>
            <li>
                <name>CEO Alice Williams</name>
                <method>email</method>
                <value>alice.williams@megacorp.com</value>
                <primary>true</primary>
            </li>
            <li>
                <name>CTO Bob Martinez</name>
                <method>phone</method>
                <value>+1-555-0100</value>
                <primary>false</primary>
            </li>
            <li>
                <name>CFO Carol Chen</name>
                <method>teams</method>
                <value>carol.chen@megacorp.com</value>
                <primary>false</primary>
            </li>
        </global_contacts>
        
        <strategic_priorities>
            <li>AI Innovation</li>
            <li>Market Expansion</li>
            <li>Sustainability</li>
            <li>Digital Transformation</li>
            <li>Customer Experience</li>
            <li>Operational Excellence</li>
        </strategic_priorities>
        
        <quarterly_uauau_targets>
            <q1>25000000.0</q1>
            <q2>30000000.0</q2>
            <q3>35000000.0</q3>
            <q4>40000000.0</q4>
        </quarterly_targets>
        
        <compliance_certifications>
            <mark_test>
            <li>ISO 27001</li>
            <li>SOC 2 Type II</li>
            <li>GDPR Compliant</li>
            <li>HIPAA Compliant</li>
            <li>PCI DSS Level 1</li>
            <li>ISO 9001</li>
        </compliance_certifications>
        
        <board_members>
            <li>Dr. Richard Thompson</li>
            <li>Prof. Maria Rodriguez</li>
            <li>James Wilson III</li>
            <li>Lisa Kim</li>
            <li>Michael O'Brien</li>
        <///board_members>
        
        <subsidiary_companies>
            <li>MegaCorp Europe Ltd</li>
            <li>MegaCorp Asia Pacific</li>
            <li>MegaCorp Canada Inc</li>
            <li>Innovation Labs LLC</li>
            <li>SecureData Solutions</li>
        </subsidiary_companies>
        
    </super_complex_project_management>
    """
    
    sr = StructuredResponse.from_basemodel(SuperComplexProjectManagement)
    result = sr.from_prompt(malformed_xml)
    
    # Test basic fields
    assert result.instance['organization_name'] == "MegaCorp Solutions"
    assert result.instance['fiscal_year'] == 2024
    assert result.instance['active'] is True
    
    # Test nested object
    assert result.instance['headquarters']['city'] == "San Francisco"
    assert result.instance['headquarters']['coordinates'] == [37.7749, -122.4194]
    assert result.instance['quarterly_targets'] is None
    # Test list of objects
    assert len(result.instance['departments']) == 3
    assert result.instance['departments'][0]['name'] == "Engineering"
    assert result.instance['departments'][0]['head_count'] == 150
    assert result.instance['compliance_certifications'][0] == "ISO 27001"
    assert result.instance['compliance_certifications'][1] == "SOC 2 Type II"
    assert result.instance['compliance_certifications'][2] == "GDPR Compliant"
    assert result.instance['compliance_certifications'][3] == "HIPAA Compliant"
    assert result.instance['compliance_certifications'][4] == "PCI DSS Level 1"
    assert result.instance['compliance_certifications'][5] == "ISO 9001"
    
    # Test deeply nested structures
    project = result.instance['projects'][0]
    assert project['id'] == "PROJ-001"
    assert project['name'] == "AI Platform Development"
    assert project['status'] == "in_progress"
    assert project['priority'] == "high"
    assert project['start_date'] == "2024-01-15"
    assert project['end_date'] == "2024-12-31"
    assert project['budget'] == 15000000.0
    assert project['spent'] == 8500000.0
    assert project['department']['name'] == "Engineering"
    assert project['department']['head_count'] == 150
    assert project['department']['active'] is True
    assert project['address']['city'] == "Austin"
    assert project['address']['coordinates'] == [30.2672, -97.7431]
    assert project['contacts'][0]['name'] == "John Smith"
    assert project['contacts'][0]['method'] == "email"
    assert project['contacts'][0]['value'] == "john.smith@megacorp.com"
    assert project['contacts'][0]['primary'] is True
    assert project['tasks'][0]['id'] == "TASK-001"
    assert project['tasks'][0]['title'] == "Design AI Architecture"
    assert project['tasks'][0]['status'] == "completed"
    assert project['tasks'][0]['priority'] == "critical"
    assert project['tasks'][0]['estimated_hours'] == 160.0
    assert project['tasks'][0]['actual_hours'] == 180.5
    assert project['tasks'][0]['assigned_to'] == "John Smith"
    assert project['tasks'][0]['tags'] == ["architecture", "ai", "design"]
    assert project['tasks'][0]['subtasks'] == ["SUBTASK-001", "SUBTASK-002"]
    assert project['milestones'][0]['title'] == "Alpha Release"
    assert project['milestones'][0]['description'] == "First working prototype"
    assert project['milestones'][0]['due_date'] == "2024-06-30"
    assert project['milestones'][0]['completed'] is True
    assert project['milestones'][0]['priority'] == "high"
    assert project['milestones'][0]['dependencies'] == ["MILESTONE-PREP-001"]
    assert project['resources'][0]['name'] == "Senior AI Engineer"
    assert project['resources'][0]['type'] == "human"
    assert project['resources'][0]['available'] is True
    assert project['resources'][0]['cost_per_hour'] == 150.0
    assert project['resources'][0]['skills'] == ["Python", "TensorFlow", "MLOps"]
    assert project['resources'][1]['skills'] == ["AWS <li> cloud infrastructure</li>", "Kubernetes", "Docker"]
    assert project['risk_factors'][0] == "Technical complexity"
    assert project['risk_factors'][1] == "Resource availability"
    assert project['risk_factors'][2] == "Market timing"
    assert project['risk_factors'][3] == "Regulatory compliance"
    assert project['risk_factors'][4] == "Data privacy concerns"
    assert project['risk_factors'][5] == "Scalability challenges"
    assert result.instance['board_members'] == ["Dr. Richard Thompson", "Prof. Maria Rodriguez", "James Wilson III", "Lisa Kim", "Michael O'Brien"]

    task = project['tasks'][0]
    assert task['title'] == "Design AI Architecture"
    assert task['status'] == "completed"
    assert len(task['tags']) == 3
    assert "architecture" in task['tags']
    project2 = result.instance['projects'][1]
    assert project2['id'] == "PROJ-002"
    assert project2['name'] == "Customer Portal Redesign"
    assert project2['status'] == "pending"
    assert project2['priority'] == "medium"
    assert project2['start_date'] == "2024-04-01"
    assert project2['end_date'] == "2024-08-31"
    assert project2['budget'] == 3000000.0
    assert project2['spent'] == 0.0
    assert project2['department']['name'] == "Marketing"
    assert project2['department']['head_count'] == 45
    assert project2['department']['active'] is True
    assert project2['address']['city'] == "New York"
    assert project2['address']['coordinates'] == [40.7128, -74.0060]
    assert project2['contacts'][0]['name'] == "Emma Davis"
    assert project2['contacts'][0]['method'] == "teams"
    assert project2['contacts'][0]['value'] == "emma.davis@megacorp.com"
    assert project2['contacts'][0]['primary'] is True
    assert project2['tasks'][0]['id'] == "TASK-003"
    assert project2['tasks'][0]['title'] == "User Research"
    assert project2['tasks'][0]['status'] == "pending"
    assert project2['tasks'][0]['priority'] == "high"
    assert project2['tasks'][0]['estimated_hours'] == 80.0
    assert project2['tasks'][0]['actual_hours'] == 0.0
    assert project2['tasks'][0]['assigned_to'] == "Emma Davis"
    assert project2['tasks'][0]['tags'] == ["research", "ux", "usability"]
    assert project2['tasks'][0]['subtasks'] == ["SUBTASK-006"]
    assert project2['milestones'][0]['title'] == "Research Complete"
    assert project2['milestones'][0]['description'] == "User research and analysis finished"
    assert project2['milestones'][0]['due_date'] == "2024-05-15"
    assert project2['milestones'][0]['completed'] is False
    assert project2['milestones'][0]['priority'] == "medium"
    assert project2['milestones'][0]['dependencies'] == ["MILESTONE-RESEARCH-001"]
    assert project2['resources'][0]['name'] == "UX Designer"
    assert project2['resources'][0]['type'] == "human"
    assert project2['resources'][0]['available'] is True
    assert project2['resources'][0]['cost_per_hour'] == 95.0
    assert project2['resources'][0]['skills'] == ["Figma", "User Research", "Prototyping"]
    assert project2['risk_factors'][0] == "User adoption resistance"
    assert project2['risk_factors'][1] == "Design complexity"
    
    assert len(result.instance['global_contacts']) == 3
    assert result.instance['global_contacts'][0]['name'] == "CEO Alice Williams"
    assert result.instance['global_contacts'][0]['method'] == "email"
    assert result.instance['global_contacts'][0]['value'] == "alice.williams@megacorp.com"
    assert result.instance['global_contacts'][0]['primary'] is True
    
    assert result.instance['global_contacts'][1]['name'] == "CTO Bob Martinez"
    assert result.instance['global_contacts'][1]['method'] == "phone"
    assert result.instance['global_contacts'][1]['value'] == "+1-555-0100"
    assert result.instance['global_contacts'][1]['primary'] is False
    print("✓ Malformed XML handling test passed")


def test_edge_cases_and_empty_values():
    """Test parsing of edge cases like empty lists, missing optional fields."""
    edge_case_xml = """
    <super_complex_project_management>
        <organization_name></organization_name>
        <fiscal_year>2024</fiscal_year>
        <total_budget>0.0</total_budget>
        <active>false</active>
        
        <headquarters>
            <street></street>
            <city>Empty City</city>
            <country>Nowhere</country>
        </headquarters>
        
        <departments></departments>
        <projects></projects>
        <global_contacts></global_contacts>
        <strategic_priorities></strategic_priorities>
        
        <quarterly_targets></quarterly_targets>
        <annual_reports></annual_reports>
    </super_complex_project_management>
    """
    
    sr = StructuredResponse.from_basemodel(SuperComplexProjectManagement)
    result = sr.from_prompt(edge_case_xml)
    
    assert result.instance['organization_name'] == ""
    assert result.instance['total_budget'] == 0.0
    assert result.instance['active'] is False
    assert result.instance['headquarters']['city'] == "Empty City"
    assert result.instance['headquarters']['country'] == "Nowhere"
    assert result.instance['departments'] == []
    assert result.instance['projects'] == []
    assert result.instance['global_contacts'] == []
    assert result.instance['strategic_priorities'] == []
    print("✓ Edge cases test passed")


def test_special_characters_and_encoding():
    """Test parsing with special characters, unicode, and encoding issues."""
    special_xml = """
    <super_complex_project_management>
        <organization_name>Spéciâl Çhàracters & More™ <>&"'</organization_name>
        <fiscal_year>2024</fiscal_year>
        <total_budget>1000000.0</total_budget>
        <active>true</active>
        
        <headquarters>
            <street>123 Üñîcödé Street</street>
            <city>Montréal</city>
            <country>Canada</country>
            <postal_code>H1A 1A1</postal_code>
        </headquarters>
        
        <strategic_priorities>
            <li>Iñtërnâtiônàlizàtiôn</li>
            <li>Security & Compliance</li>
            <li>AI/ML & Data Science</li>
        </strategic_priorities>
    </super_complex_project_management>
    """
    
    sr = StructuredResponse.from_basemodel(SuperComplexProjectManagement)
    result = sr.from_prompt(special_xml)
    
    assert "Spéciâl Çhàracters" in result.instance['organization_name']
    assert result.instance['headquarters']['city'] == "Montréal"
    assert result.instance['headquarters']['country'] == "Canada"
    assert result.instance['strategic_priorities'][0] == "Iñtërnâtiônàlizàtiôn"
    assert result.instance['strategic_priorities'][1] == "Security & Compliance"
    assert result.instance['strategic_priorities'][2] == "AI/ML & Data Science"
    print("✓ Special characters test passed")


class ChanceScale(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RainProbability(BaseModel):
    chance: ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
    when: str = Field(description="The time of day when the rain is or is not expected")

class UVIndex(BaseModel):
    index: int = Field(description="The UV index, where 0 is no UV and 11+ is dangerous")
    dangerous: bool = Field(description="Whether the UV index is dangerous")

class WeatherPrognosis(BaseModel):
    location: str = Field(description="The location of the weather forecast")
    current_temperature: float = Field(description="The current temperature in degrees Celsius")
    rain_probability: Optional[list[RainProbability]] = Field(
        description="The chance of rain, where low is less than 25% and high is more than 75%"
    )
    wind_speed: Optional[float] = Field(description="The wind speed in km/h")
    uv_index: UVIndex = Field(description="The UV index, where 0 is no UV and 11+ is dangerous")
    high: Optional[float] = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
    low: Optional[float] = Field(description="The low temperature in degrees Celsius")
    storm_tonight: bool = Field(description="Whether there will be a storm tonight")
    windspeed: list[float] = Field(description="The wind speed in km/h, per hour")


def test_weather_prognosis():
    s = """
            <weather_prognosis>
        <location>Annecy</location>
        <current_temperature>10.0
        <rain_probability>
            <li>
            <chance>high</chance>
            <when>morning
            </li>
            <li>
            <chance>low
            <when>afternoon</when>
            </li>
        </rain_probability>
        <wind_speed>5.0</wind_speed>
        <uv_index>
            <index>4</index>
            <dangerous, description= 'e foarte rau <<<<<<id/>>>>><<><><'>true
        
        <high>15.0</high>
        <low>5.0</low>
        <storm_tonight>false</storm_tonight>
        <windspeed>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
            <li>5.0</li>
        </windspeed>
        </weather_prognosis>
            """
    sr: StructuredResponse = StructuredResponse.from_basemodel(WeatherPrognosis)

    sr = sr.from_prompt(s)

    assert sr.instance['rain_probability'][0]['chance'] == "high"
    assert sr.instance['rain_probability'][0]['when'] == "morning"
    assert sr.instance['rain_probability'][1]['chance'] == "low"
    assert sr.instance['rain_probability'][1]['when'] == "afternoon"
    assert sr.instance['wind_speed'] == 5.0
    assert sr.instance['uv_index']['index'] == 4
    assert sr.instance['uv_index']['dangerous'] is True
    assert sr.instance['high'] == 15.0
    assert sr.instance['low'] == 5.0
    assert sr.instance['storm_tonight'] is False
    assert sr.instance['windspeed'] == [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    print("✓ Weather prognosis test passed")


if __name__ == "__main__":
    pytest.main([__file__])

