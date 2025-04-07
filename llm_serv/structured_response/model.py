from typing import List
from pydantic import BaseModel, ConfigDict
from llm_serv.structured_response.from_text import response_from_xml
from llm_serv.structured_response.to_text import response_to_xml


class StructuredResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=False, arbitrary_types_allowed=True)

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
        Returns a JSON string representation of the StructuredResponse object,
        excluding fields with None values.
        """
        return self.model_dump_json(indent=2, exclude_none=True)
        
