from enum import Enum
from typing import Optional

from pydantic import Field

from llm_serv.structured_response.model import StructuredResponse


class ChanceScale(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RainProbability(StructuredResponse):
    chance: ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
    when: str = Field(description="The time of day when the rain is or is not expected")


class WeatherPrognosis(StructuredResponse):
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


def test_plain_class_definition_and_prompt():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    assert resp.class_name == "WeatherPrognosis"
    assert "location" in resp.definition
    # list of dict for nested class
    rp = resp.definition["rain_probability"]
    assert rp["type"] == "list" and isinstance(rp["elements_type"], dict)

    prompt = resp.to_prompt()
    assert "<weather_prognosis>" in prompt
    assert "<location type='str'>[The location of the weather forecast - as a str]</location>" in prompt
    assert "<chance type='enum" in prompt
    assert "<hourly_index type='list' elements_type='int'" in prompt


