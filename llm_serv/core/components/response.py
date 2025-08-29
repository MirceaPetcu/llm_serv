import uuid
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import TokenTracker
from llm_serv.core.exceptions import StructuredResponseException
from llm_serv.structured_response.model import StructuredResponse


class LLMResponse(BaseModel):    
    # Input parameters
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_model: StructuredResponse | None = None    

    # Output parameters    
    raw_output: str | None = None    
    
    # Meta parameters
    native_response_format_used: bool | None = None
    conversation: Conversation = Field(default_factory=lambda: Conversation())
    llm_model: Model | None = None
    tokens: TokenTracker = Field(default_factory=TokenTracker)    
    start_time: float | None = None  # time.time() as fractions of a second
    end_time: float | None = None  # time.time() as fractions of a second
    total_duration: float | None = None  # time in seconds of the entire request, including retries (fractions included)    
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_serializer('response_model')
    def serialize_response_model(self, value: StructuredResponse | None) -> dict[str, Any] | None:
        """Serialize StructuredResponse using its serialize method."""
        if value is None:
            return None
        # serialize() returns a JSON string, so we parse it back to dict for Pydantic
        json_string = value.serialize()
        return json.loads(json_string)
    
    @field_validator('response_model', mode='before')
    @classmethod
    def deserialize_response_model(cls, value: Any) -> StructuredResponse | None:
        """Deserialize StructuredResponse using its deserialize method."""
        if value is None:
            return None
        if isinstance(value, StructuredResponse):
            return value
        if isinstance(value, dict):
            # Convert dict to JSON string for deserialize function
            json_string = json.dumps(value)
            from llm_serv.structured_response.converters.deserialize import deserialize
            return deserialize(json_string)
        if isinstance(value, str):
            # Handle JSON string input
            from llm_serv.structured_response.converters.deserialize import deserialize
            return deserialize(value)
        raise ValueError(f"Cannot deserialize response_model from type {type(value)}")

    @property
    def output(self) -> StructuredResponse | str | None:
        if self.raw_output is None:
            return None
        
        if self.response_model is None:
            return self.raw_output

        assert isinstance(self.response_model, StructuredResponse), f"Response model must be a StructuredResponse instance, got {type(self.response_model)}"  # noqa: E501
        try:
            return self.response_model.from_prompt(self.raw_output)               
        except Exception as e:
            raise StructuredResponseException(
                message=f"Failed to convert LLM output to structured format: {e}",
                xml=self.raw_output,
                return_class=str(type(self.response_model))
            ) from e

    @classmethod
    def from_request(cls, request: LLMRequest) -> "LLMResponse":
        response = LLMResponse(
            id = request.id,
            response_model = request.response_model,
            conversation = request.conversation
        )
        return response

    def rprint(self, subtitle: str | None = None):
        try:
            import json
            from enum import Enum

            from rich import print as rprint
            from llm_serv.conversation.role import Role
            from rich.console import Console
            from rich.json import JSON
            from rich.panel import Panel

            console = Console()

            # Custom JSON encoder to handle Enums and other non-serializable types
            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Enum):
                        return obj.value
                    try:
                        # Try to convert to dict if it has a model_dump method
                        if hasattr(obj, "model_dump"):
                            return obj.model_dump(exclude_none=True)
                        # Try to convert to dict if it has a dict method
                        if hasattr(obj, "__dict__"):
                            return obj.__dict__
                    except:  # noqa: E722
                        pass
                    # Let the base class handle it or raise TypeError
                    return super().default(obj)

            # Prepare panel content
            content_parts = []
            
            # Add system message if present
            if self.conversation.system:
                content_parts.append(f"[bold dark_magenta][SYSTEM][/bold dark_magenta] [dark_magenta]{self.conversation.system}[/dark_magenta]")  # noqa: E501
            
            # Process conversation messages
            for message in self.conversation.messages:
                if message.role == Role.USER:
                    content_parts.append(f"[bold dark_blue][USER][/bold dark_blue] [dark_blue]{message.text}[/dark_blue]")
                elif message.role == Role.ASSISTANT:
                    content_parts.append(f"[bold dark_green][ASSISTANT][/bold dark_green] [dark_green]{message.text}[/dark_green]")

            # Add the final output
            content_parts.append("[bold bright_green][ASSISTANT - OUTPUT][/bold bright_green]")
            if isinstance(self.output, str):
                content_parts.append(f"[bright_green]{self.output}[/bright_green]")
            else:
                try:
                    # First convert the data to a JSON-serializable format using our custom encoder
                    data = str(self.output)
                        
                    # Convert to JSON string with our custom encoder that handles Enums
                    json_str = json.dumps(data, indent=2, cls=EnhancedJSONEncoder)
                    
                    # Use rich's console to directly print the formatted JSON
                    content_parts.append("[bright_green]")
                    
                    # Create a temporary console that outputs to a string
                    str_console = Console(width=100, file=None)
                    with str_console.capture() as capture:
                        str_console.print(JSON.from_data(json.loads(json_str)))
                    
                    # Add the captured output to our content
                    content_parts.append(capture.get())
                    content_parts.append("[/bright_green]")
                except Exception as exc:
                    content_parts.append(f"[bright_red]Error serializing output: {str(exc)}[/bright_red]")
                    content_parts.append(f"[bright_red]Output type: {type(self.output)}[/bright_red]")

            # Create panel title (stats line)
            title = ""
            if self.tokens:
                model_str = f"LLMRequest: {self.llm_model.provider.name}/{self.llm_model.name}"
                title = f"{model_str} | Time: {self.total_duration:.2f}s | Input/Output tokens: {self.tokens.input_tokens}/{self.tokens.completion_tokens} | Total tokens: {self.tokens.total_tokens}"  # noqa: E501

            # Print single panel with all content
            console.print(
                Panel(
                    "\n".join(content_parts),
                    title=title,
                    title_align="right",
                    border_style="magenta",
                    subtitle=subtitle,
                    subtitle_align="left",
                )
            )
        except Exception as e:
            # Fallback to basic printing if rich formatting fails
            try:
                from rich import print as rprint
                rprint(f"[bold red]Error in rprint method: {str(e)}[/bold red]")
                rprint("[yellow]Falling back to basic output:[/yellow]")

                # Print basic conversation info
                if (
                    hasattr(self, "request")
                    and self.request
                    and hasattr(self.request, "conversation")
                ):
                    if (
                        hasattr(self.conversation, "system")
                        and self.conversation.system
                    ):
                        rprint(
                            f"[dark_magenta]System: {self.conversation.system}[/dark_magenta]"
                        )

                    if hasattr(self.conversation, "messages"):
                        for msg in self.conversation.messages:
                            role = getattr(msg, "role", "unknown")
                            text = getattr(msg, "text", "no text")
                            rprint(f"[blue]{role}: {text}[/blue]")

                # Print output
                if hasattr(self, "output"):
                    rprint(f"[green]Output: {self.output}[/green]")

                # Print token info
                if hasattr(self, "tokens") and self.tokens:
                    rprint(
                        f"[cyan]Tokens: {self.tokens.total_tokens} (Input: {self.tokens.input_tokens}, Output: {self.tokens.completion_tokens})[/cyan]"  # noqa: E501
                    )
            except Exception as inner_e:
                # Last resort: plain print without any formatting
                print(f"Error in rprint fallback: {str(inner_e)}")
                print(f"Original error: {str(e)}")
                print("Output:", self.output)

