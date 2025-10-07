"""
TODO: Fix code, https://github.com/openai/openai-python/issues/874
Error codes: https://platform.openai.com/docs/guides/error-codes
"""
import asyncio
import os

from openai import AsyncOpenAI, RateLimitError
from pydantic import Field, BaseModel

from llm_serv.logger import logger
from llm_serv.api import Model
from llm_serv.conversation.conversation import Conversation
from llm_serv.conversation.role import Role
from llm_serv.core.base import LLMProvider
from llm_serv.core.components.request import LLMRequest
from llm_serv.core.components.tokens import ModelTokens
from llm_serv.core.exceptions import CredentialsException, InternalConversionException, ServiceCallException, ServiceCallThrottlingException
from llm_serv.structured_response.model import StructuredResponse


class OpenAILLMProvider(LLMProvider):
    @staticmethod
    def check_credentials() -> None:
        required_variables = ["OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"]
            
        missing_vars = []
        for var in required_variables:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise CredentialsException(
                f"Missing required environment variables for OpenAI: {', '.join(missing_vars)}"
            )

    def __init__(self, model: Model):
        super().__init__(model)        
        
        OpenAILLMProvider.check_credentials()

        # The OpenAI client is already async-compatible
        self._client = AsyncOpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT")
        )

    def _resolve_ref(self, *, root: dict[str, object], ref: str) -> object:
        if not ref.startswith("#/"):
            raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

        path = ref[2:].split("/")
        resolved = root
        for key in path:
            value = resolved[key]
            assert isinstance(value, dict), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
            resolved = value

        return resolved

    def _ensure_strict_json_schema(
        self,
        json_schema: object,
        *,
        path: tuple[str, ...],
        root: dict[str, object],
    ) -> dict[str, any]:
        """Mutates the given JSON schema to ensure it conforms to the `strict` standard
        that the API expects.
        """
        if not isinstance(json_schema, dict):
            raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

        defs = json_schema.get("$defs")
        if isinstance(defs, dict):
            for def_name, def_schema in defs.items():
                self._ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

        definitions = json_schema.get("definitions")
        if isinstance(definitions, dict):
            for definition_name, definition_schema in definitions.items():
                self._ensure_strict_json_schema(definition_schema, path=(*path, "definitions", definition_name), root=root)

        typ = json_schema.get("type")
        if typ == "object" and "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False

        # object types
        # { 'type': 'object', 'properties': { 'a':  {...} } }
        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            json_schema["required"] = [prop for prop in properties.keys()]
            json_schema["properties"] = {
                key: self._ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
                for key, prop_schema in properties.items()
            }

        # arrays
        # { 'type': 'array', 'items': {...} }
        items = json_schema.get("items")
        if isinstance(items, dict):
            json_schema["items"] = self._ensure_strict_json_schema(items, path=(*path, "items"), root=root)

        # unions
        any_of = json_schema.get("anyOf")
        if isinstance(any_of, list):
            json_schema["anyOf"] = [
                self._ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
                for i, variant in enumerate(any_of)
            ]

        # intersections
        all_of = json_schema.get("allOf")
        if isinstance(all_of, list):
            if len(all_of) == 1:
                json_schema.update(self._ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
                json_schema.pop("allOf")
            else:
                json_schema["allOf"] = [
                    self._ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                    for i, entry in enumerate(all_of)
                ]

        # strip `None` defaults as there's no meaningful distinction here
        # the schema will still be `nullable` and the model will default
        # to using `None` anyway
        if "default" in json_schema and json_schema.get("default", None) is None:
            json_schema.pop("default")

        # we can't use `$ref`s if there are also other properties defined, e.g.
        # `{"$ref": "...", "description": "my description"}`
        #
        # so we unravel the ref
        # `{"type": "string", "description": "my description"}`
        ref = json_schema.get("$ref")
        if ref and len(json_schema.keys()) > 1:
            assert isinstance(ref, str), f"Received non-string $ref - {ref}"

            resolved = self._resolve_ref(root=root, ref=ref)
            if not isinstance(resolved, dict):
                raise ValueError(f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}")

            # properties from the json schema take priority over the ones on the `$ref`
            json_schema.update({**resolved, **json_schema})
            json_schema.pop("$ref")
            # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied,
            # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid.
            return self._ensure_strict_json_schema(json_schema, path=path, root=root)

        return json_schema

    async def _convert(self, request: LLMRequest) -> dict:
        """
        Ref here: https://platform.openai.com/docs/api-reference/responses/create
        https://platform.openai.com/docs/guides/vision#multiple-image-inputs
        returns (input, config)
        """
        
        input_messages = []
        instructions = None
        # Handle system message if present
        if request.conversation.system is not None and len(request.conversation.system) > 0:
            instructions = request.conversation.system

        # Process each message
        for message in request.conversation.messages:
            content = []

            # Add text content if present
            if message.text:
                content.append({"type": "input_text", "text": message.text})

            # Add images if present
            for image in message.images:
                content.append(
                    {
                        "type": "input_image",
                        "image_url":  f"data:image/{image.format or 'jpeg'};base64,{image.export_as_base64(image.image)}",
                    }
                )

            input_messages.append({"role": message.role.value, "content": content})

        """
        TODO: strict format handling
        "response_format": (
            {"type": "json_object"} if request.response_format == LLMResponseFormat.JSON else {"type": "text"}
        ),
        """
        
        config = {
            "max_output_tokens": request.max_completion_tokens if request.max_completion_tokens is not None else self.model.max_output_tokens,  # noqa: E501
            "temperature": request.temperature,
            "top_p": request.top_p
        }

        return {
            "instructions": instructions,
            "input": input_messages,            
            "config": config
        }
    
    async def _llm_service_call(
        self,
        request: LLMRequest,
    ) -> tuple[str, ModelTokens]:
        # prepare request
        try:
            processed = await self._convert(request)
            config = processed["config"]
            input_messages = processed["input"]
            instructions = processed["instructions"]
        except Exception as e:
            raise InternalConversionException(f"Failed to convert request: {str(e)}") from e

        # Prepare parameters for the API call, excluding top_p initially.
        request_params = {
            "model": self.model.internal_model_id,
            "input": input_messages,
            "max_output_tokens": config["max_output_tokens"],
            "temperature": config["temperature"],            
        }

        if instructions is not None:
            request_params["instructions"] = instructions

        if config["top_p"] is not None:
            request_params["top_p"] = config["top_p"]
            
        if request.response_model.native:
            assert isinstance(request.response_model, StructuredResponse), f"Response model must be a StructuredResponse instance, got {type(request.response_model)}"  # noqa: E501
            # Handle different types of response models
            ensured_schema = self._ensure_strict_json_schema(
                request.response_model.definition, path=(), root=request.response_model.definition
            )
            request_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_model.class_name, 
                    "strict": True,
                    "schema": ensured_schema,
                }
            }
        else:
            request_params["text"] = { "format": { "type": "text" } }

        
        # call the LLM provider using responses API, no need to retry, it is handled in the base class                   
        try: 
            response = await self._client.responses.create(**request_params)

            # update the tokens
            tokens = ModelTokens(
                input_tokens=response.usage.input_tokens - response.usage.input_tokens_details.cached_tokens,
                cached_input_tokens=response.usage.input_tokens_details.cached_tokens,
                output_tokens=response.usage.output_tokens - response.usage.output_tokens_details.reasoning_tokens,
                reasoning_output_tokens=response.usage.output_tokens_details.reasoning_tokens,
                total_tokens=response.usage.total_tokens,
                # Store current price rates for historical accuracy
                input_price_per_1m_tokens=self.model.input_price_per_1m_tokens,
                cached_input_price_per_1m_tokens=self.model.cached_input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.model.output_price_per_1m_tokens,
                reasoning_output_price_per_1m_tokens=self.model.reasoning_output_price_per_1m_tokens,
            )

        except Exception as e:
            if isinstance(e, RateLimitError):  # package specific exception into our own for base class processing
                raise ServiceCallThrottlingException(f"OpenAI service is throttling requests: {str(e)}") from e
            
            # TODO: handle other error codes properly here
            
            raise ServiceCallException(f"OpenAI service error: {str(e)}") from e

        logger.info(f"'{response.model}' status response '{response.status}', output:\n{response.output_text}")           

        # check for errors
        if response.error is not None:
            raise ServiceCallException(f"OpenAI service error {response.error.code}: {response.error.message}")

        # check status. Statuses are: completed, failed, in_progress, cancelled, queued, or incomplete.
        if response.status != "completed":
            raise ServiceCallException(f"OpenAI service error, finished with status: {response.status}")

        # check that we actually have an output        
        output = str(response.output_text).strip()

        if len(output) == 0:
            raise ServiceCallException(f"OpenAI service error, call finished with 'completed' status, max_output_tokens={config['max_output_tokens']}, output_tokens={tokens.output_tokens} out of which reasoning={tokens.reasoning_output_tokens}, total_tokens={tokens.total_tokens}, but got an empty output!")  # noqa: E501
        
        return output, tokens


if __name__ == "__main__":
    import asyncio

    from pydantic import Field

    from llm_serv import LLMService
    from llm_serv.conversation.role import Role
    from llm_serv.structured_response.model import StructuredResponse

    async def test_openai():
        model = LLMService.get_model("OPENAI/o4-mini")
        llm = OpenAILLMProvider(model)

        class MyClass(StructuredResponse):
            example_string: str = Field(
                default="", description="A string field that should be filled with a random person name in Elven language"
            )
            example_int: int = Field(
                default=0, ge=0, le=10, description="An integer field with a random value, greater than 5."
            )
            example_float: float = Field(
                default=0, ge=0.0, le=10.0, description="A float field with a value exactly half of the integer value"
            )
       
        response_model = StructuredResponse.from_basemodel(MyClass)
        conversation = Conversation.from_prompt("Please fill in the following class respecting the following instructions.")
        conversation.add_text_message(role=Role.USER, content=response_model.to_prompt())

        request = LLMRequest(conversation=conversation, response_model=response_model)

        response = await llm(request)
        
        print(response)
        assert isinstance(response.output, StructuredResponse)
    
        await llm.stop()    

    # Run the test function with asyncio
    asyncio.run(test_openai())
