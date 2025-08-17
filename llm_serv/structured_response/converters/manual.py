import enum
from typing import Any

FORBIDDEN_KEYS = ["type", "description", "elements", "choices", "int", "float", "bool", "dict", "enum", "list", "item"]
BASE_TYPES = ['str', 'int', 'float', 'bool', 'enum']

def add_node(self,
             node_path: str, 
             node_type: type, 
             elements: type | None = None,
             description: str = "",
             choices: enum.Enum | None = None,
             **kwargs: Any) -> None:  # noqa: E501
    """
    Add a node to the StructuredResponse definition.
    """
    definition = self.definition

    # step 0, check parameters
    assert node_type in [str, int, float, bool, list, dict, enum], f"Invalid node type: {node_type}, valid types are: str, int, float, bool, dict, enum, list"  # noqa: E501
    assert node_type not in [list] or elements is not None, "Elements type is required for list nodes"
    if elements is not None:
        assert elements in [str, int, float, bool, dict, enum], f"Invalid elements type: {elements}, valid types are: str, int, float, bool, dict, enum"  # noqa: E501
    if node_type == enum:
        assert choices is not None, "Choices are required for enum nodes"
        assert issubclass(choices, enum.Enum), "Choices must be an enum class"

    # step 1, get the target node by path
    target_node = definition
    if "." in node_path:
        path = node_path.split(".")[:-1]
        new_node_name = node_path.split(".")[-1]
    else:
        path = []
        new_node_name = node_path
    
    for key in kwargs.keys():
        assert key not in FORBIDDEN_KEYS, f"Key name '{key}' is forbidden!"

    for key in path:  # walk the path to the target node
        print(f"Navigating to node-name={key}")
        if key not in target_node:
            raise ValueError(f"Intermediary node '{key}' not found in definition! Given path: {path}")
        
        # is the key a list?
        if target_node[key]['type'] == 'list':
            if target_node[key]['elements'] in BASE_TYPES:
                raise ValueError(f"Key '{key}' is a list, but it is a base type! We cannot further add nodes to it. Given path: {path}")
            assert isinstance(target_node[key]['elements'], dict), f"Key '{key}' is a list, target is a complex type but its elements is not a dict! Given path: {path}"  # noqa: E501
            target_node = target_node[key]['elements']
        elif target_node[key]['type'] == 'dict':
            assert isinstance(target_node[key]['elements'], dict), f"Key '{key}' is a dict, target is a complex type but its elements is not a dict! Given path: {path}"  # noqa: E501            
            target_node = target_node[key]['elements']
        else:                
            target_node = target_node[key]

    # step 2, add the new node    
    target_node[new_node_name] = {
        "type": node_type.__name__,
        "description": description,
        **kwargs
    }

    if node_type == enum:
        target_node[new_node_name]["choices"] = [e.value for e in choices]
    elif isinstance(node_type, type) and issubclass(node_type, list):  
        if elements in [str, int, float, bool, enum]:  # simple types
            target_node[new_node_name]["elements"] = elements.__name__            
        elif isinstance(elements, type) and issubclass(elements, dict):
            target_node[new_node_name]["elements"] = {}            
        else:
            raise ValueError(f"Invalid elements type: {elements}, valid types are: str, int, float, bool, dict, enum")
    elif isinstance(node_type, type) and issubclass(node_type, dict):        
        target_node[new_node_name]["elements"] = {}            

    self.definition = definition


if __name__ == "__main__":    
    from llm_serv.structured_response.model import StructuredResponse

    class Clan(str, enum.Enum):
        CLAN_1 = "clan_1"
        CLAN_2 = "clan_2"
        CLAN_3 = "clan_3"

    sr = StructuredResponse()

    sr.class_name = "PersonDetails"
    sr.add_node(node_path="name", node_type=str, description="The name of the person", min_length=1, max_length=100)
    #sr.add_node(node_path="age", node_type=int, description="The age of the person", ge=-1, le=120)
    #sr.add_node(node_path="children_clans_set", node_type=list, description="The clans of the children of the person", elements=enum)
    #sr.add_node(node_path="self_clan", node_type=enum, description="The clan of the person", choices=Clan)
    sr.add_node(node_path="children_details", node_type=list, description="The details of the children of the person", elements=dict)
    sr.add_node(node_path="children_details.clan", node_type=enum, description="The clan of the child", choices=Clan)
    sr.add_node(node_path="children_details.age", node_type=int, description="The age of the child", tag='ignore')
    sr.add_node(node_path="children_details.details", node_type=dict, description="The full name and age details of the child group")
    sr.add_node(node_path="children_details.details.name", node_type=str, description="The first name of the child", lowercase=True)
    sr.add_node(node_path="children_details.details.age", node_type=int, description="The age of the child")
    
    
    from rich import print as rprint    
    rprint(sr.definition)

    rprint(sr.to_prompt())

    sr_weather = StructuredResponse()
    sr_weather.class_name = "WeatherPrognosis"

    class ChanceScale(str, enum.Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    sr_weather.add_node(node_path="location", node_type=str, description="The location of the weather forecast")
    sr_weather.add_node(node_path="current_temperature", node_type=float, description="The current temperature in degrees Celsius")
    
    # overall_rain_prob
    sr_weather.add_node(node_path="overall_rain_prob", node_type=dict, description="The day's rain chance")
    sr_weather.add_node(node_path="overall_rain_prob.chance", node_type=enum, \
                        description="The chance of rain, where low is less than 25% and high is more than 75%", \
                        choices=ChanceScale)
    sr_weather.add_node(node_path="overall_rain_prob.when", node_type=str, \
                        description="The time of day when the rain is or is not expected")

    # rain_probability_timebound
    sr_weather.add_node(node_path="rain_probability_timebound", node_type=list, \
                        description="List of chances of rain, where low is less than 25% and high is more than 75%", \
                        elements=dict)
    sr_weather.add_node(node_path="rain_probability_timebound.chance", node_type=enum, \
                        description="The chance of rain, where low is less than 25% and high is more than 75%", \
                        choices=ChanceScale)
    sr_weather.add_node(node_path="rain_probability_timebound.when", node_type=str, \
                        description="The time of day when the rain is or is not expected")

    sr_weather.add_node(node_path="hourly_index", node_type=list, description="List of hourly UV index in the range of 1-10", elements=int)
    sr_weather.add_node(node_path="wind_speed", node_type=float, description="The wind speed in km/h")
    sr_weather.add_node(node_path="high", node_type=float, description="The high temperature in degrees Celsius", ge=-20, le=60)
    sr_weather.add_node(node_path="low", node_type=float, description="The low temperature in degrees Celsius")
    sr_weather.add_node(node_path="storm_tonight", node_type=bool, description="Whether there will be a storm tonight")

    rprint(sr_weather.definition)
    from pprint import pprint
    print(sr_weather.to_prompt())
