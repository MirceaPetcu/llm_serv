# How StructuredResponses work

Internally the StructuredResponse class has a class_name (str), a definition field (dict) and an instance data field (Box() as python-box).

The purpose of this class is to allow structured responses communication to-and-from an LLM, as well as seamless serialization and deserialization.

## How this class looks like

Let's start with a predefined set of BaseModels we want to pass to an LLM to obtain a structured response:

```python
class ChanceScale(Enum):
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
    rain_probability_timebound: Optional[list[RainProbability]] = Field(
        description="List of chances of rain, where low is less than 25% and high is more than 75%"
    )
    hourly_index: list[int] = Field(description="List of hourly UV index in the range of 1-10")
    wind_speed: float = Field(description="The wind speed in km/h")
    high: float = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
    low: float = Field(description="The low temperature in degrees Celsius")
    storm_tonight: bool = Field(description="Whether there will be a storm tonight")    
```

First we'd want to create a structured response definition from this class type. We'll use the StructuredResponse.from_basemodel() method like this:

response = StructuredResponse.from_basemodel(WeatherPrognosis)

This method will create a StructuredResponse instance with the class_name as "WeatherPrognosis" and an internal definition dict that looks like this:

```python
{
    "location": {
        "type": "str",
        "description": "The location of the weather forecast"
    },
    "current_temperature": {
        "type": "float",
        "description": "The current temperature in degrees Celsius"
    },
    "overall_rain_prob": {
        "type": "dict",
        "description": "The day's rain chance",
        "elements": {            
            "chance": {
                "type": "enum",
                "choices": ["low", "medium", "high"],
                "description": "The chance of rain, where low is less than 25% and high is more than 75%"
            },
            "when": {
                "type": "str",
                "description": "The time of day when the rain is or is not expected"
            }
        }
    }
    "rain_probability_timebound": {
        "type": "list",
        "description": "List of chances of rain, where low is less than 25% and high is more than 75%",
        "elements": {
            "chance": {
                "type": "enum",
                "choices": ["low", "medium", "high"],
                "description": "The chance of rain, where low is less than 25% and high is more than 75%"
            },
            "when": {
                "type": "str",
                "description": "The time of day when the rain is or is not expected"
            }
        }
    },
    "hourly_index": {
        "type": "list",
        "elements": "int",
        "description": "List of hourly UV index in the range of 1-10"
    },
    "wind_speed": {
        "type": "float",
        "description": "The wind speed in km/h"
    },
    "high": {
        "type": "float",
        "description": "The high temperature in degrees Celsius",
        "ge": -20,
        "le": 60
    },
    "low": {
        "type": "float",
        "description": "The low temperature in degrees Celsius"
    },
    "storm_tonight": {
        "type": "bool",
        "description": "Whether there will be a storm tonight"
    }
}
```

Then, we'll pass this class to a LLM like this:

```python
llm_prompt = f"Some prompt here, respond as instructed {response.to_prompt()}"
```

The to_prompt() method will create an xml-like representation, suitable for an LLM, where possible BaseModels are dicts:

```xml
<weather_prognosis>
    <location type='str'>[The location of the weather forecast - as a string]</location>
    <current_temperature type='float'>[The current temperature in degrees Celsius - as a float]</current_temperature>
    <overall_rain_prob type='dict' description="The day's rain chance">
        <chance type='enum' choices='["low", "medium", "high"]'>[The chance of rain, where low is less than 25% and high is more than 75% - as an enum]</chance>
        <when type='str'>[The time of day when the rain is or is not expected - as a string]</when>
    </overall_rain_prob>
    <rain_probability_timebound type='list' elements='dict' description='List of chances of rain, where low is less than 25% and high is more than 75%'>
        <li index='0'>
            <chance type='enum' choices='["low", "medium", "high"]'>[The chance of rain, where low is less than 25% and high is more than 75% - as an enum]</chance>
            <when type='str'>[The time of day when the rain is or is not expected - as a string]</when>
        </li>
        ...
    </rain_probability_timebound>
    <hourly_index type='list' elements='int' description='List of hourly UV index in the range of 1-10'>
        <li index='0'>
            [value here - as an int]
        </li>
        ...
    </hourly_index>
    <wind_speed type='float'>[The wind speed in km/h - as a float]</wind_speed>
    <high type='float' greater_or_equal='-20' less_or_equal='60'>[The high temperature in degrees Celsius - as a float]</high>
    <low type='float'>[The low temperature in degrees Celsius - as a float]</low>
    <storm_tonight type='bool'>[Whether there will be a storm tonight - as a bool]</storm_tonight>
</weather_prognosis>
```

Rules by which the prompt text is created:
- the root element is the class name
- for non-list items, the description is placed between [ and ], with an additional "- as <type>" string appended
- each element has the type tag
- the Optional or | None is ignored from BaseModels
- lists have an elements tag, with complex types or subclasses as "dict"
- lists have one or more <li> items, each li having an index tag, starting from 0
- dicts take the name tag from their class, whereas in list elements we dump the values directly without the extra class tag like in dicts.
- enums have a 'choices' tag, containing a python-like stringification of the list of possible values it can take
- simple list items (int, str, float, bool) have their description directly as [value here - as a <type>]

Let's assume the prompt is passed then through the LLM and we get a reply. We parse the reply and extract the instance data with response.from_prompt(llm_response_as_string).

The internal dict with instance data will be filled like:

```python
{
    "location": "Annecy, FR",
    "current_temperature": 18.7,
    "overall_rain_prob": {
        {"chance": "medium", "when": "today"},
    }
    "rain_probability_timebound": [
        {"chance": "low", "when": "morning"},
        {"chance": "medium", "when": "afternoon"},
        {"chance": "high", "when": "evening"}
    ],
    "hourly_index": [3, 4, 5, 6, 5, 4, 3, 2],
    "wind_speed": 12.5,
    "high": 24.0,
    "low": 12.0,
    "storm_tonight": false
}
```

The from_prompt method searches the llm response string to find the start and end tags (left and right find), then parses the valid XML to extract all instance data an populate the internal dictionary. 

Finally, we'll use the serialize and deserialize methods to serialize and deserialize instances to and from strings. As the internal components are dicts, the serialization and deserialization is straight-forward. 


Calling str(response) will invoke the __str__ method which will output the instance data as simple XML:

```xml
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
    </rain_probability>
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
```
