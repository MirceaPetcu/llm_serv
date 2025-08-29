from pydantic import BaseModel, Field
from llm_serv.client import LLMServiceClient
from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.request import LLMRequest
import asyncio

from llm_serv.structured_response.model import StructuredResponse

async def simple_case():
    prompt = """
    {'id': '49b4d1b4-720d-4cc2-887a-141beb256d5e', 'request': {'id': 'b3067f3f-c094-4a16-94a5-a56385701b37', 'request_type': 'LLM', 'conversation': {...}, 'response_model': '{"class_name": "company_description", "definition": {"poem": {"type": "str", "description": "Write a LONG poem (100 verses minimum) about the company\'s energy generation distribution percents, types of energy, object of activity and future plans. I need to know the company\'s all aspects of business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point within each and every affirmation made using the format <ref id=\'1\'/> or <ref id=\'0,4,5\'/> for multiple citations referencing given statements/facts/premises/etc by their ids."}, "poem_citations": {"type": "list", "elements": "int", "description": "Citation aggregator list for \'poem\'. ALL citations (ids-integers) referenced in the \'poem\' field must be added to this list as well. All the ref ids MUST exist in both \'poem\' (as ref tags) and \'poem_citations\' (as an integer list) fields."}}, "instance": {"poem": "A titan stands, with power vast and grand,", "poem_citations": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 1]}}', 'force_native_structured_response': False, 'max_completion_tokens': None, 'temperature': 0.5, 'max_retries': 5, 'top_p': None}, 'output': '{"class_name": "company_description", "definition": {"poem": {"type": "str", "description": "Write a LONG poem (100 verses minimum) about the company\'s energy generation distribution percents, types of energy, object of activity and future plans. I need to know the company\'s all aspects of business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point within each and every affirmation made using the format <ref id=\'1\'/> or <ref id=\'0,4,5\'/> for multiple citations referencing given statements/facts/premises/etc by their ids."}, "poem_citations": {"type": "list", "elements": "int", "description": "Citation aggregator list for \'poem\'. ALL citations (ids-integers) referenced in the \'poem\' field must be added to this list as well. All the ref ids MUST exist in both \'poem\' (as ref tags) and \'poem_citations\' (as an integer list) fields."}}, "instance": {"poem": "A titan stands, with power vast and grand,", "poem_citations": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 1]}}', 'native_response_format_used': None, 'tokens': {'stats': {...}}, 'llm_model': {'id': 'TOGETHER/DeepSeek-V3.1-thinking', 'internal_model_id': 'deepseek-ai/DeepSeek-V3.1', 'provider': {...}, 'max_tokens': 128000, 'max_output_tokens': 4096, 'fixed_temperature': False, 'capabilities': {...}, 'price': {...}, 'config': {}}, 'start_time': 1756467180.7632828, 'end_time': 1756467200.0955017, 'total_duration': 19.332218885421753}
    """

    client = LLMServiceClient(host="localhost", port=9999)
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")
    conversation = Conversation.from_prompt(prompt)

    request = LLMRequest(conversation=conversation)

    response = await client.chat(request)

    print("Response:", response.output)


async def structured_case():    
    class Response(BaseModel):        
        explanation: str = Field(description="The detailed explanation of the answer")

    sr: StructuredResponse = StructuredResponse.from_basemodel(Response)
    
    prompt = f"""
    What's 1+1/21 + 3**2?    
    {sr.to_prompt()}
    """

    client = LLMServiceClient(host="localhost", port=9999)
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")

    request = LLMRequest(conversation=Conversation.from_prompt(prompt), response_model=sr)

    response = await client.chat(request)

    print("Response:", response.output.instance)

async def main():
    # Initialize the client
    await simple_case()
    print("-===========================-")
    await structured_case()

if __name__ == "__main__":
    asyncio.run(main()) 