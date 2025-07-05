"""
This test file is for testing the StructuredResponse class with all complex types.
"""

from datetime import date, datetime, time
from enum import Enum
from rich import print as rprint
from typing import Dict, List, Optional
from pydantic import Field

from llm_serv.structured_response.model import StructuredResponse


class AnEnum(Enum):
    TYPE1 = "type1"
    TYPE2 = "type2"

class SubClassType1(StructuredResponse):
    sub_string: str = Field(default="", description="A sub string field")


class SubClassType2(StructuredResponse):
    sub_list: Optional[List[str]] = Field(default=None, description="A sub list of strings field")

class SubClassType3(StructuredResponse):
    element1: SubClassType1 = Field(default=SubClassType1(), description="An element 1 field")
    element_sublist: List[SubClassType2] = Field(default=[], description="An element 2 list of sub class type 2 fields")

class TestStructuredResponse(StructuredResponse):
    example_string: str = Field(default="", description="A string field")
    example_string_none: Optional[str] = Field(default=None, description="An optional string field")    
    example_int: int = Field(default=5, ge=0, le=10, description="An integer field with values between 0 and 10, default is 5")
    example_int_list: List[int] = Field(default=[1, 2, 3], description="A list of integers")
    example_enum: AnEnum = Field(default=AnEnum.TYPE1, description="An enum field with a custom description")
    example_float: float = Field(default=2.5, ge=0.0, le=5.0, description="A float field with values between 0.0 and 5.0, default is 2.5")
    example_list: List[SubClassType1] = Field(default=[SubClassType1()], description="A list of sub class type 1 fields")
    example_float_list_optional: Optional[List[float]] = Field(default=None, description="An optional list of floats")        
    example_optional_subclasstype1: Optional[SubClassType1] = Field(default=None, description="An optional sub class type 1 field")
    example_nested_subclasstype3: SubClassType3 = Field(default=SubClassType3(), description="A nested sub class type 3 field"),
    example_date: date = Field(description="A date field including month from 2023")
    example_datetime: datetime = Field(description="A full date time field from 2023")
    example_time: time = Field(description="A time field from today")
    example_optional_list_of_subclasstype1: Optional[List[SubClassType1]] = Field(default=None, description="An optional list of sub class type 1 fields")

if __name__ == "__main__":
    print("Testing StructuredResponse with all complex types")
    class recognized_materiality_risks_element(StructuredResponse):
        risk_opportunity_name: str = Field(description = "Name of the risk or opportunity.")
        score: int = Field(description = "Score 1 to 5 as presented in the scoring instructions.")
        score_reasoning: str = Field(description = "Clear explanation of the score with grounded citations (references in the <ref id=""/> format) to facts.")
        what_to_improve: str = Field(description = "What can be done to improve the score?")

    class recognized_materiality_risks(StructuredResponse):
        risks: list[recognized_materiality_risks_element] = Field(description = "List of risks with their scores and reasoning grounded in facts.")

    xml_text = recognized_materiality_risks.to_text()
    
    print(xml_text)

