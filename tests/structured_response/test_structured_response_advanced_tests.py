import pytest
from enum import Enum
from typing import Optional, Annotated

from pydantic import BaseModel, Field

from llm_serv.structured_response.model import StructuredResponse


class Level(Enum):
    A = "A"
    B = "B"


class Sub(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=5)] = Field(description="Short name")
    value: Annotated[int, Field(ge=0, le=10)] = Field(description="Score")


class Complex(BaseModel):
    flag: bool = Field(description="A boolean flag")
    ratio: float = Field(ge=0.0, le=1.0, description="A ratio in [0,1]")
    level: Level = Field(description="A categorical level")
    tags: list[str] = Field(description="List of tags")
    subs: Optional[list[Sub]] = Field(description="Optional list of subs")
    sub: Optional[Sub] = Field(description="Optional sub")


def test_definition_contains_constraints_and_enums():
    resp = StructuredResponse.from_basemodel(Complex)
    d = resp.definition

    assert d["ratio"]["ge"] == 0.0
    assert d["ratio"]["le"] == 1.0
    assert d["level"]["type"] == "enum" and d["level"]["choices"] == ["A", "B"]
    assert d["tags"]["type"] == "list" and d["tags"]["elements_type"] == "str"
    # String length constraints
    assert d["sub"]["name"]["min_length"] == 2
    assert d["sub"]["name"]["max_length"] == 5
    assert d["sub"]["value"]["ge"] == 0
    assert d["sub"]["value"]["le"] == 10


def test_prompt_and_parse_roundtrip_with_values():
    resp = StructuredResponse.from_basemodel(Complex)
    prompt = resp.to_prompt()
    assert "<complex>" in prompt and "</complex>" in prompt

    xml = (
        "<complex>\n"
        "  <flag type='bool'>true</flag>\n"
        "  <ratio type='float'>0.5</ratio>\n"
        "  <level type='enum' choices='[\"A\", \"B\"]'>A</level>\n"
        "  <tags type='list' elements_type='str'>\n"
        "    <li index='0'>alpha</li>\n"
        "    <li index='1'>beta</li>\n"
        "  </tags>\n"
        "  <subs type='list' elements_type='dict'>\n"
        "    <li index='0'>\n"
        "      <name type='str'>ab</name>\n"
        "      <value type='int'>7</value>\n"
        "    </li>\n"
        "  </subs>\n"
        "  <sub type='dict'>\n"
        "    <name type='str'>xy</name>\n"
        "    <value type='int'>3</value>\n"
        "  </sub>\n"
        "</complex>"
    )
    resp.from_prompt(xml)
    assert resp.instance == {
        "flag": True,
        "ratio": pytest.approx(0.5),
        "level": "A",
        "tags": ["alpha", "beta"],
        "subs": [{"name": "ab", "value": 7}],
        "sub": {"name": "xy", "value": 3},
    }


def test_missing_elements_become_none():
    resp = StructuredResponse.from_basemodel(Complex)
    xml = "<complex><flag>true</flag></complex>"
    resp.from_prompt(xml)
    assert resp.instance["flag"] is True
    assert resp.instance["ratio"] is None
    assert resp.instance["level"] is None
    assert resp.instance["tags"] is None
    assert resp.instance["subs"] is None
    assert resp.instance["sub"] is None


def test_invalid_xml_raises():
    resp = StructuredResponse.from_basemodel(Complex)
    with pytest.raises(ValueError):
        resp.from_prompt("<complex><flag>true</flag>")


