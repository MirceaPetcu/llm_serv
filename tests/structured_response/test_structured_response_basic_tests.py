import json
import pytest
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from llm_serv.structured_response.model import StructuredResponse


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


def test_from_basemodel_builds_definition():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    assert isinstance(resp, StructuredResponse)
    assert resp.class_name == "WeatherPrognosis"
    assert isinstance(resp.definition, dict) and resp.instance is None

    d = resp.definition
    assert set(d.keys()) == {
        "location",
        "current_temperature",
        "rain_probability",
        "hourly_index",
        "wind_speed",
        "high",
        "low",
        "storm_tonight",
    }

    assert d["location"]["type"] == "str"
    assert d["current_temperature"]["type"] == "float"
    assert d["wind_speed"]["type"] == "float"
    assert d["storm_tonight"]["type"] == "bool"

    # Constraints
    assert d["high"]["type"] == "float"
    assert d["high"]["ge"] == -20
    assert d["high"]["le"] == 60

    # List of complex objects
    rp = d["rain_probability"]
    assert rp["type"] == "list"
    assert isinstance(rp["elements"], dict)
    assert set(rp["elements"].keys()) == {"chance", "when"}
    assert rp["elements"]["chance"]["type"] == "enum"
    assert rp["elements"]["chance"]["choices"] == ["low", "medium", "high"]
    assert rp["elements"]["when"]["type"] == "str"

    # List of simple
    hi = d["hourly_index"]
    assert hi["type"] == "list"
    assert hi["elements"] == "int"


def test_to_prompt_contains_expected_sections():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    prompt = resp.to_prompt()

    # Root
    assert "<weather_prognosis>" in prompt
    assert "</weather_prognosis>" in prompt

    # Simple field contains type and bracketed description
    assert "<location type='str'>[The location of the weather forecast - as a str]</location>" in prompt

    # Enum in list of dicts
    assert "<rain_probability type='list'" in prompt
    assert "elements='dict'" in prompt
    assert "<li index='0'>" in prompt
    assert "<chance type='enum' choices='[\"low\", \"medium\", \"high\"]'>" in prompt
    assert "<when type='str'>[The time of day when the rain is or is not expected - as a str]</when>" in prompt

    # Simple list
    assert "<hourly_index type='list' elements='int'" in prompt
    assert "...</hourly_index>" not in prompt  # ensure proper closing only once
    assert "<wind_speed type='float'>[The wind speed in km/h - as a float]</wind_speed>" in prompt

    # Constraints rendered
    assert "<high type='float' greater_or_equal='-20' less_or_equal='60'>" in prompt


def test_from_prompt_parses_llm_xml_into_instance():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    xml = (
        "Some preface...\n"
        "<weather_prognosis>\n"
        "    <location type='str'>Annecy, FR</location>\n"
        "    <current_temperature type='float'>18.7</current_temperature>\n"
        "    <rain_probability type='list' elements='dict'>\n"
        "        <li index='0'>\n"
        "            <chance type='enum' choices='[\"low\", \"medium\", \"high\"]'>low</chance>\n"
        "            <when type='str'>morning</when>\n"
        "        </li>\n"
        "        <li index='1'>\n"
        "            <chance type='enum' choices='[\"low\", \"medium\", \"high\"]'>medium</chance>\n"
        "            <when type='str'>afternoon</when>\n"
        "        </li>\n"
        "    </rain_probability>\n"
        "    <hourly_index type='list' elements='int'>\n"
        "        <li index='0'>3</li>\n"
        "        <li index='1'>4</li>\n"
        "    </hourly_index>\n"
        "    <wind_speed type='float'>12.5</wind_speed>\n"
        "    <high type='float'>24.0</high>\n"
        "    <low type='float'>12.0</low>\n"
        "    <storm_tonight type='bool'>false</storm_tonight>\n"
        "</weather_prognosis>\n"
        "... trailing"
    )
    resp.from_prompt(xml)

    assert resp.instance == {
        "location": "Annecy, FR",
        "current_temperature": pytest.approx(18.7),
        "rain_probability": [
            {"chance": "low", "when": "morning"},
            {"chance": "medium", "when": "afternoon"},
        ],
        "hourly_index": [3, 4],
        "wind_speed": pytest.approx(12.5),
        "high": pytest.approx(24.0),
        "low": pytest.approx(12.0),
        "storm_tonight": False,
    }


def test_serialize_deserialize_roundtrip():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    # attach a small instance
    resp.instance = {"location": "X", "hourly_index": [1, 2]}
    s = resp.serialize()
    loaded = StructuredResponse.deserialize(s)
    assert isinstance(loaded, StructuredResponse)
    assert loaded.class_name == resp.class_name
    assert loaded.definition == resp.definition
    assert loaded.instance == resp.instance


def test_str_calls_to_prompt():
    resp = StructuredResponse.from_basemodel(WeatherPrognosis)
    assert str(resp) == resp.to_prompt()


def test_from_basemodel_with_instance_prefills_instance():
    inst = WeatherPrognosis(
        location="Annecy, FR",
        current_temperature=18.0,
        rain_probability=[
            RainProbability(chance=ChanceScale.LOW, when="morning"),
        ],
        hourly_index=[3, 4],
        wind_speed=12.5,
        high=24.0,
        low=12.0,
        storm_tonight=False,
    )
    resp = StructuredResponse.from_basemodel(inst)
    assert resp.instance == {
        "location": "Annecy, FR",
        "current_temperature": 18.0,
        "rain_probability": [{"chance": "low", "when": "morning"}],
        "hourly_index": [3, 4],
        "wind_speed": 12.5,
        "high": 24.0,
        "low": 12.0,
        "storm_tonight": False,
    }


