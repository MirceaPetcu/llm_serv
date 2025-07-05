from pydantic import BaseModel, field_validator, computed_field

class ModelTokens(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @field_validator("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("Token counts must be non-negative")
        return v

    def __add__(self, other: "ModelTokens") -> "ModelTokens":
        return ModelTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __iadd__(self, other: "ModelTokens") -> "ModelTokens":
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.total_tokens += other.total_tokens
        return self

class TokenTracker(BaseModel):
    """
    A class to track the number of tokens used in an LLM call, by model name.
    """    
    stats: dict[str, ModelTokens] = {}

    @property
    def input_tokens(self) -> int:
        if len(self.stats) == 0:
            return 0
        return sum(model_tokens.input_tokens for model_tokens in self.stats.values())

    @property
    def completion_tokens(self) -> int:
        if len(self.stats) == 0:
            return 0
        return sum(model_tokens.output_tokens for model_tokens in self.stats.values())

    @property
    def total_tokens(self) -> int:
        if len(self.stats) == 0:
            return 0
        return sum(model_tokens.total_tokens for model_tokens in self.stats.values())

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
    tracker += TokenTracker(stats={"gpt-4o": ModelTokens(input_tokens=100, output_tokens=200)})
    tracker2 = TokenTracker(stats={"gpt-4.1": ModelTokens(input_tokens=100, output_tokens=200)})
    tracker3 = TokenTracker(stats={"gpt-4o": ModelTokens(input_tokens=400, output_tokens=300)})
    
    tracker_sum = tracker + tracker2 + tracker3
    print(tracker_sum)