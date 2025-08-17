from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

from model import StructuredResponse
from llm_serv.structured_response.converters.from_basemodel import from_basemodel
from box import Box

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
    rain_probability: Optional[list[RainProbability]] = Field(
        description="List of chances of rain, where low is less than 25% and high is more than 75%"
    )
    hourly_index: list[int] = Field(description="List of hourly UV index in the range of 1-10")
    wind_speed: float = Field(description="The wind speed in km/h")
    high: float = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
    low: float = Field(description="The low temperature in degrees Celsius")
    storm_tonight: bool = Field(description="Whether there will be a storm tonight") 

# test 1, build from a BaseModel
#sr = from_basemodel(WeatherPrognosis)
#print(sr)


data = {
    "a": "b",
    "c": [
        {
            "d": "1",
            "f": "2"
        },
        {
            "d": "3",
            "f": "4"
        }
    ],
    "e": 5    
}

sr = StructuredResponse()
sr.instance = Box(data)

print(len(sr.instance.c))
print(type(sr.instance.c))
