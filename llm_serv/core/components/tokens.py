from pydantic import BaseModel, field_validator, computed_field

class LLMTokens(BaseModel):
    input_tokens: int = 0
    completion_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.completion_tokens

    @field_validator("input_tokens", "completion_tokens")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("Token counts must be non-negative")
        return v

    def __add__(self, other: "LLMTokens") -> "LLMTokens":
        return LLMTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __iadd__(self, other: "LLMTokens") -> "LLMTokens":
        self.input_tokens += other.input_tokens
        self.completion_tokens += other.completion_tokens
        return self
