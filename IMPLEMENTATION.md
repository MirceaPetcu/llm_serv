# Architecture

We should have a LLMProvider base class that wraps up all providers (OpenAI, etc).
The base class offers:

- ``__init__(model: Model)``

Exceptions thrown:
-

----

The server instantiates LLMProviders which internally manage their own clients for each service. This means that each provider singleton class is responsible to manage any number of async calls. 

Each provider should handle internally client complexities like non-async, queues, etc.



This means that code like this should work without problems:

```python
service: AWSLLMProvider = llm_serv.get_provider(provider="AWS")
request: LLMRequest = ...

tasks = [service(request) for i in range(100000)]

results = asyncio.gather(*tasks, return_exceptions=True)

```

A Client should work like:


```python
models = LLMClient.list_models(provider:str|None = None)

client: LLMClient = LLMClient(Model, host, port, timeout)

client.test() # raises an exception if something is wrong

client.chat() # this is blocking until we get a respone or an exception

chats = [client.async_chat(request) for i in range(1000)]

results = asyncio.gather(*tasks, return_exceptions=True)
```

---



How to list providers and models:

```python
from llm_serv import LLMService

models: list[Model] = LLMService.list_models()  # all models, or pass provider="str" to filter a specific provider

for model in models:
    print("Model ID: {model.id}")

providers: list[ModelProvider] = LLMService.list_providers()
```
 
How to get a model directly:

```python
model: Model = LLMService.get_model(model_id="OPENAI/gpt-4o-mini")  # throws exceptions/ModelNotFoundException if there is no such model
```

How to get the llm provider object:

```python
llm: LLMProvider = LLMService.get_provider(model="AWS/claude-3-5-sonnet")
llm: LLMProvider = LLMService.get_provider(model=model)  # use a Model from the list_models() or get_model()
```
