import re
from typing import List

from pydantic import BaseModel, ConfigDict

from llm_serv.structured_response.from_text import response_from_xml
from llm_serv.structured_response.to_text import instance_to_xml, response_to_xml


class StructuredResponse(BaseModel):
    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True
    )
    
    # Class-level attributes for title and description
    _title: str = "Structured Response"
    _description: str = ""

    def get_title(self) -> str:        
        return self.__class__._title
    
    def get_description(self) -> str:
        return self.__class__._description

    def set_title(self, title: str) -> None:
        self.__class__._title = title
    
    def set_description(self, description: str) -> None:
        self.__class__._description = description

    @classmethod
    def get_class_title(cls) -> str:
        return cls._title
    
    @classmethod
    def get_class_description(cls) -> str:
        return cls._description

    @classmethod
    def set_class_title(cls, title: str) -> None:
        cls._title = title
    
    @classmethod
    def set_class_description(cls, description: str) -> None:
        cls._description = description

    @classmethod
    def from_text(cls, xml: str, exclude_fields: List[str] = []) -> 'StructuredResponse':
        """
        This method is used to convert an XML string into a StructuredResponse object.
        """
        return response_from_xml(xml, return_class=cls, is_root=True, exclude_fields=exclude_fields)

    @classmethod
    def to_text(cls, exclude_fields: List[str] = []) -> str:
        """
        This method is used to convert a StructuredResponse object into a prompt-ready text string.
        """
        return response_to_xml(object=cls, exclude_fields=exclude_fields)
    
    def __str__(self):
        """
        Returns an XML string representation of the StructuredResponse object,
        excluding fields with None values.
        """
        return instance_to_xml(self, exclude_none=True, exclude=set())
    
    def to_xml(self, exclude_none: bool = True, exclude: set[str] | None = None) -> str:
        """
        Returns an XML string representation of the StructuredResponse object.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            exclude: Set of field names to exclude from the output
        """
        return instance_to_xml(self, exclude_none=exclude_none, exclude=exclude or set())
        
    @staticmethod        
    def _convert_identifier_to_python_identifier(identifier: str) -> str:
        python_identifier = identifier.strip().lower()
        python_identifier = python_identifier.replace(' ', '_').replace('/', '_').replace('-', '_')                
        python_identifier = re.sub(r'[^a-zA-Z0-9_]', '_', python_identifier)        
        python_identifier = re.sub(r'_+', '_', python_identifier)
        return python_identifier