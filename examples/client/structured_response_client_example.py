from enum import Enum
from typing import Optional

from pydantic import Field
from rich import print as rprint

from llm_serv import LLMServiceClient, Conversation, LLMRequest
from llm_serv.structured_response.model import StructuredResponse


class ChanceScale(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RainProbability(StructuredResponse):
    chance: ChanceScale = Field(description="The chance of rain, where low is less than 25% and high is more than 75%")
    when: str = Field(description="The time of day when the rain is or is not expected")
    exact_mm_ammount: list[str] = Field(description="The exact amount of rain in mm for a 3-hour period estimation, estimation per hour, should have exactly 3 items")  # noqa: E501

class WeatherPrognosis(StructuredResponse):
    location: str = Field(description="The location of the weather forecast")
    current_temperature: float = Field(description="The current temperature in degrees Celsius")
    rain_probability: Optional[list[RainProbability]] = Field(
        description="The chance of rain, where low is less than 25% and high is more than 75%"
    )
    wind_speed: Optional[float] = Field(description="The wind speed in km/h")
    high: Optional[float] = Field(ge=-20, le=60, description="The high temperature in degrees Celsius")
    low: Optional[float] = Field(description="The low temperature in degrees Celsius")
    storm_tonight: bool = Field(description="Whether there will be a storm tonight")


async def main():
    input_text = """
    The temperature today in Annecy is 10°C. There is a 80% chance of rain in the morning and 20% chance of rain in the afternoon. Winds will be from the south at 5 km/h.
    We expect a high of 15°C and a low of 5°C.
    """  # noqa: E501

    # Initialize the client
    client = LLMServiceClient(host="localhost", port=9999, timeout=20.0)

    # Create the response model
    response_model = StructuredResponse.from_basemodel(WeatherPrognosis)

    # Set the model to use
    client.set_model("OPENAI/gpt-5-mini")

    prompt = f"""
    You are a weather expert. You are given a weather forecast for a specific location.

    Here is the weather forecast:
    {input_text}

    Here is the structured response:
    {response_model.to_prompt()}
    """

    print(prompt)

    conversation = Conversation.from_prompt(prompt)
    request = LLMRequest(
        conversation=conversation,
        response_model=response_model,
        max_completion_tokens=4000,
    )

    response = await client.chat(request)

    print("\nResponse:")
    rprint(response.output)

    print(f"Output type: {type(response.output)}")
    assert isinstance(response.output, StructuredResponse)

    print("\nXML print:")
    print(str(response.output))

    print("\nInstance dict:")
    rprint(response.output.instance)

    rprint("Token Usage:", response.tokens)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
