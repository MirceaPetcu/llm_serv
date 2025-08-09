from pydantic import BaseModel


class StructuredResponse():
    class_name: str = "StructuredResponse"
   
    definition: dict | None = None
    instance: dict | None = None

    @staticmethod
    def from_basemodel(objects: type[BaseModel] | list[type[BaseModel]] | BaseModel | list[BaseModel]) -> None:
        """
        Converts one or more basemodel object definitions to the internal representation.
        The first basemodel is the main class, whereas the rest are internal classes used in the main class. 
        If the objects are types, we'll only fill the definition; else we fill the instance as well.
        """

        return StructuredResponse

    """
    @staticmethod
    def from_dict(data: dict) -> None: # won't implement this for now
    """

    def from_prompt(xml_string: str) -> None:
        """
        Load from the LLM prompt
        """


    def to_prompt(self) -> str:
        """
        Write the prompt for the llm.
        """

    def __str__(self) -> str:
        """
        Return the string representation of the StructuredResponse object
        """
        return self.to_prompt()

    def serialize(self) -> str:
        """
        Serialize to a str
        """

    @staticmethod
    def deserialize(json_string: str) -> 'StructuredResponse':
        """
        Deserialize from a string and return a StructuredResponse object
        """