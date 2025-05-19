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


class ModelTokens(BaseModel):
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

    def __add__(self, other: "ModelTokens") -> "ModelTokens":
        return ModelTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __iadd__(self, other: "ModelTokens") -> "ModelTokens":
        self.input_tokens += other.input_tokens
        self.completion_tokens += other.completion_tokens
        return self

class TokenTracker(BaseModel):
    """
    A class to track the number of tokens used in an LLM call, by model name.
    """    
    stats: dict[str, ModelTokens] = {}

    def __add__(self, other: "TokenTracker") -> "TokenTracker":
        for model_name, tokens in other.stats.items():
            if model_name not in self.stats:
                self.stats[model_name] = ModelTokens()
            self.stats[model_name] += tokens
        return self

    def __iadd__(self, other: "TokenTracker") -> "TokenTracker":
        for model_name, tokens in other.stats.items():
            if model_name not in self.stats:
                self.stats[model_name] = ModelTokens()
            self.stats[model_name] += tokens
        return self

    def add(self, model_name: str, tokens: ModelTokens) -> None:
        if model_name not in self.stats:
            self.stats[model_name] = ModelTokens()
        self.stats[model_name] += tokens

if __name__ == "__main__":
    tracker = TokenTracker()
    tracker += TokenTracker(stats={"gpt-4o": ModelTokens(input_tokens=100, completion_tokens=200)})
    tracker2 = TokenTracker(stats={"gpt-4.1": ModelTokens(input_tokens=100, completion_tokens=200)})
    tracker3 = TokenTracker(stats={"gpt-4o": ModelTokens(input_tokens=400, completion_tokens=300)})
    
    tracker_sum = tracker + tracker2 + tracker3
    print(tracker_sum)