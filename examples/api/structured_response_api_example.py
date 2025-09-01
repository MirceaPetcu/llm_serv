import asyncio
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from rich import print as rprint

from llm_serv import LLMService
from llm_serv.conversation import Conversation
from llm_serv.core.base import LLMRequest
from llm_serv.structured_response.model import StructuredResponse


class ChanceScale(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RainProbability(BaseModel):
    chance: ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
    when: str = Field(description="The time of day when the rain is or is not expected")

class UVIndex(BaseModel):
    index: int = Field(description="The UV index, where 0 is no UV and 11+ is dangerous")

class WeatherPrognosis(BaseModel):
    location: str = Field(description="The location of the weather forecast")
    current_temperature: float = Field(description="The current temperature in degrees Celsius")
    rain_probability: Optional[list[RainProbability]] = Field(
        description="The chance of rain, where low is less than 25% and high is more than 75%"
    )
    wind_speed: Optional[float] = Field(description="The wind speed in km/h")
    uv_index: UVIndex = Field(description="The UV index, where 0 is no UV and 11+ is dangerous")
    high: Optional[float] = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
    low: Optional[float] = Field(description="The low temperature in degrees Celsius")
    storm_tonight: bool = Field(description="Whether there will be a storm tonight")
    windspeed: list[float] = Field(description="The wind speed in km/h, per hour")


async def main():
    model = LLMService.get_model("GOOGLE/gemini-2.5-flash")    
    llm_service = LLMService.get_provider(model)

    response_model = StructuredResponse.from_basemodel(WeatherPrognosis)

    input_text = """
    The temperature today in Annecy is 10°C. There is a 80% chance of rain in the morning and 20% chance of rain in the afternoon. Winds will be from the south at 5 km/h.
    We expect a high of 15°C and a low of 5°C. The UV index is moderate.
    """  # noqa: E501

    prompt = f"""
    You are a weather expert. You are given a weather forecast for a specific location.

    Here is the weather forecast:
    {input_text}

    Output format:    
    {response_model.to_prompt()}
    """

    print(prompt)

    conversation = Conversation.from_prompt(prompt)
    request = LLMRequest(
        conversation=conversation,
        response_model=response_model
    )

    # Use await for async service call
    response = await llm_service(request)

    print("\nResponse type:")
    print(type(response.output))

    print("\nResponse:")
    rprint(response.output)

    print("\nToken Usage:")
    print(f"Input tokens: {response.tokens.input_tokens}")
    print(f"Output tokens: {response.tokens.completion_tokens}")
    print(f"Total tokens: {response.tokens.total_tokens}")

    print("\nRich print:")
    response.rprint()

    print("\nXML print:")
    print(f"{response.output}")

    await llm_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
