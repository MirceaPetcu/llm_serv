from pydantic import BaseModel, Field
from llm_serv.client import LLMServiceClient
from llm_serv.conversation.conversation import Conversation
from llm_serv.core.components.request import LLMRequest
import asyncio

from llm_serv.structured_response.model import StructuredResponse

async def simple_case():
    prompt = """
You are an expert analyst tasked to think in-depth and compose a solution to the problem given,
 based only on the given data. You always explain your reasoning and absolutely always CITE the used fact(s). 
 ## Available Data  
 <problem_statement>  
 Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of 
 energy, object of activity and future plans. I need to know the company's all aspects of business.  
 </problem_statement>  
 <problem_understanding>  
 None  
 </problem_understanding> 
 <broader_context>  
 None  
 </broader_context> 
 <reasoning_tips>  
 Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of 
 energy, object of activity and future plans. I need to know the company's all aspects of business.  
 </reasoning_tips>  
 <scoring_instructions>  
 Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of 
 energy, object of activity and future plans. I need to know the company's all aspects of business.  
 </scoring_instructions>  
 <style_guide>  
 None  
 </style_guide>  
 <facts>  
 <fact id='8'>  
  <content>The company's total net capacity is 40,657 MW.</content> 
  <explanation>Statement 1 explicitly states the total net capacity across all facilities.</explanation> 
 </fact>  
 <fact id='9'>  
  <content>Natural gas accounts for 59% of the company's total net capacity.</content> 
  <explanation>Statement 0 provides the percentage share of natural gas in the total net 
 capacity.</explanation>  
 </fact>  
 <fact id='10'>  
  <content>The net capacity from natural gas is 24,120 MW.</content> 
  <explanation>Statement 0 provides the megawatt value for the natural gas capacity share.</explanation> 
 </fact>  
 <fact id='11'>  
  <content>Coal accounts for 21% of the company's total net capacity.</content> 
  <explanation>Statement 0 provides the percentage share of coal-based generation in the total net 
 capacity.</explanation>  
 </fact>  
 <fact id='12'>  
  <content>The net capacity from coal is 8,428 MW.</content>  
  <explanation>Statement 0 provides the megawatt value for the coal capacity share. Statement 4 confirms this 
 total capacity for the coal/lignite fleet.</explanation>  
 </fact>  
 <fact id='13'>  
  <content>The coal/lignite-fueled generation fleet is comprised of seven generation facilities.</content> 
  <explanation>Statement 4 provides the count of generation facilities for the coal/lignite 
 fleet.</explanation>  
 </fact>  
 <fact id='14'>  
  <content>Nuclear power accounts for 16% of the company's total net capacity.</content> 
  <explanation>Statement 0 provides the percentage share of nuclear power in the total net  
 capacity.</explanation>  
 </fact>  
 <fact id='15'>  
  <content>The net capacity from nuclear power is 6,448 MW.</content>  
  <explanation>Statement 0 provides the megawatt value for the nuclear capacity share.</explanation> 
 </fact>  
 <fact id='16'>  
  <content>Renewable energy (solar and battery) accounts for 4% of the company's total net  
 capacity.</content>  
  <explanation>Statement 0 provides the percentage share of renewable energy in the total net capacity and 
 specifies the types included.</explanation>  
 </fact>  
 <fact id='17'>  
  <content>The net capacity from renewable energy (solar and battery) is 1,474 MW.</content> 
  <explanation>Statement 0 provides the megawatt value for the renewable energy capacity share.</explanation> 
 </fact>  
 <fact id='18'>  
  <content>The East Segment has a net capacity of 19,746 MW.</content> 
  <explanation>Statement 1 provides the megawatt value for the East Segment's capacity.</explanation> 
 </fact>  
 <fact id='19'>  
  <content>The West Segment has a net capacity of 1,880 MW.</content>  
  <explanation>Statement 1 provides the megawatt value for the West Segment's capacity.</explanation> 
 </fact>  
 <fact id='20'>  
  <content>The predominant technology in the East Segment is Combined Cycle Gas Turbine (CCGT) fueled by 
 natural gas.</content>  
  <explanation>Statement 1 describes the primary technology and its fuel source for the East 
 Segment.</explanation>  
 </fact>  
 <fact id='21'>  
  <content>Coal-fired steam turbines (ST) are significant in the East Segment.</content> 
  <explanation>Statement 1 notes the significance of coal-fired steam turbines in the East  
 Segment.</explanation>  
 </fact>  
 <fact id='22'>  
  <content>Nuclear capacity is substantial in the East Segment.</content> 
  <explanation>Statement 1 notes the substantial nature of nuclear capacity in the East  
 Segment.</explanation>  
 </fact>  
 <fact id='23'>  
  <content>Renewable technologies are present but represent a smaller share in the East Segment.</content> 
  <explanation>Statement 1 notes the presence but smaller share of renewable technologies in the East 
 Segment.</explanation>  
 </fact>  
 <fact id='24'>  
  <content>The West Segment primarily features CCGT, battery, and some fuel oil technologies.</content> 
  <explanation>Statement 1 describes the primary technologies featured in the West Segment.</explanation>  
 </fact>  
 <fact id='25'>  
  <content>The Texas segment facilities have a combined net capacity that is dominated by natural gas-fired 
 plants.</content>  
  <explanation>Statement 8 describes the dominance of natural gas-fired plants in the Texas 
 segment.</explanation>  
 </fact>  
 <fact id='26'>  
  <content>Three coal-fired steam turbine facilities are significant contributors in the Texas 
 segment.</content> 
  <explanation>Statement 8 notes the significance of three specific coal-fired facilities in the Texas  
 segment.</explanation>  
 </fact>  
 <fact id='27'>  
  <content>The Comanche Peak nuclear facility in the Texas segment has a capacity of 2,400 MW.</content> 
  <explanation>Statement 8 provides the name and capacity of a specific nuclear facility in the Texas 
 segment.</explanation>  
 </fact>  
 <fact id='28'>  
  <content>Solar and battery storage facilities constitute smaller capacity segments in the Texas 
 segment.</content> 
  <explanation>Statement 8 notes the smaller capacity contribution of solar and battery storage in the Texas  
 segment.</explanation>  
 </fact>  
 <fact id='29'>  
  <content>The company has a substantial capital allocation plan intended for investments in renewable  
 assets, including solar development projects and battery ESS.</content>  
  <explanation>Statement 3 describes the company's future investment plans for renewable 
 assets.</explanation> 
 </fact>  
 <fact id='30'>  
  <content>The company plans to continually assess potential strategic acquisitions or investments in 
 renewable assets, emerging technologies and related projects.</content>  
  <explanation>Statement 3 describes the company's ongoing strategic planning regarding acquisitions and 
 investments.</explanation>  
 </fact>  
 <fact id='31'>  
  <content>The company is committed to sustainability and setting aggressive targets.</content> 
  <explanation>Statement 7 states the company's commitment to sustainability and its approach to target 
 setting.</explanation>  
 </fact>  
 <fact id='32'>  
  <content>The company is transitioning its fleet to low-to-no carbon resources.</content>  
  <explanation>Statement 7 states the company's strategic direction of transitioning its generation  
 fleet.</explanation>  
 </fact>  
 <fact id='33'>  
  <content>Certain of the company's subsidiaries are in various stages of developing and constructing solar 
 generation facilities and battery ESS.</content> 
  <explanation>Statement 9 describes the current activities of some subsidiaries regarding renewable project  
 development.</explanation>  
 </fact>  
 <fact id='34'>  
  <content>Certain of the company's solar and battery ESS projects have signed long-term contracts or made 
 similar arrangements for the sale of electricity.</content>  
  <explanation>Statement 9 describes the commercial arrangements for some of the company's renewable 
 projects.</explanation>  
 </fact>  
 </facts>  
 <premises>  
 </premises>  
 ## Task  
 - THINK in-depth and GENERATE a solution to the <problem_statement>, based on the given <facts> and <premises>. 
 - Use <problem_understanding> and <broader_context> to get the big-picture - usually we are dealing with  
 solving a specific sub-problem (in <problem_statement>) in the given context.  
 - Use the <reasoning_tips> to guide your reasoning. 
 - Use the <scoring_instructions> to understand how to present your solution. Formatting is exemplified below. 
 - Follow the <style_guide> when writing your solution. 
 ## Instructions 
 - Look at <premises> as these provide a large amount of evidence and should be the first to be carefully  
 considered and cited and then look at <facts>, if available. 
 - Base your reasoning only on the given <facts> and <premises>, do not make up any reasoning on non-explicitly  
 stated facts.  
 - Be comprehensive in your answer, unless the Output format field description says otherwise (the field 
 description takes precedence over this instructions).  
 ## Citation guidelines  
 - ALWAYS cite (reference by id with <ref> tags) the used <facts> and/or <premises> item in your response, after 
 EACH statement you make, according to the descriptions in the output format example below. 
 - Any score, explanation or output you generate must be explained and grounded in the used <facts>/<premises>,  
 referenced by their ids. Example: <ref id='3'/>  
 - If multiple facts/premises are used to support an idea, list their ids jointly. Example: <ref id='1,23,45'/>  
 - Remember, multiple facts/premises can support an item, and multiple items can be supported by the same  
 fact/premise.  
 - As a self consistency check, make sure that all <ref> fact/premise ids used are also in the "xzy_citations" 
 fields of the item they support, if these fields are present. 
 Remember the problem statement:  
 <problem_statement>  
 Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of 
 energy, object of activity and future plans. I need to know the company's all aspects of business.  
 </problem_statement>  
 ## Output format  
 <company_description> 
  <poem type='str'>[Write a LONG poem (100 verses minimum) about the company's energy generation distribution 
 percents, types of energy, object of activity and future plans. I need to know the company's all aspects of  
 business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point 
 within each and every affirmation made using the format <ref id='1'/> or <ref id='0,4,5'/> for multiple 
 citations referencing given statements/facts/premises/etc by their ids. - as a string]</poem> 
  <poem_citations type='list' elements='int' description='Citation aggregator list for 'poem'. ALL citations  
 (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in 
 both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.'>  
 <li> 
  [int value here]
 </li>  
 ...  
  </poem_citations> 
 </company_description>  
 Respond ONLY with valid XML as shown above, with the following requirements: 
 - Notice that "..." represents multiple <li> items. 
 - Do not include any attributes in the output! The 'description' attribute is for you to understand the problem 
 and how to respond; the 'type' is for you to understand the type of the response item, etc. 
 - Output only VALID XML, while keeping in mind the objective at all times.  
 - Understand that the text here between [ and ] and in the description fields are for you to understand what to 
 respond with, they are essential instructions as well! 
 - Remember to meticulously place citation <ref/> tags where appropriate in your response.
    """

    client = LLMServiceClient(host="localhost", port=9999)
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")
    conversation = Conversation.from_prompt(prompt)

    request = LLMRequest(conversation=conversation)

    response = await client.chat(request)

    print("Response:", response.output)


async def structured_case():    
    class CompanyDescription(BaseModel):        
        poem: str = Field(description="Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of energy, object of activity and future plans. I need to know the company's all aspects of business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point within each and every affirmation made using the format <ref id='1'/> or <ref id='0,4,5'/> for multiple citations referencing given statements/facts/premises/etc by their ids. - as a string")
        poem_citations: list[int] = Field(description="Citation aggregator list for 'poem'. ALL citations (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.")

    sr: StructuredResponse = StructuredResponse.from_basemodel(CompanyDescription)
    
    prompt = f"""
    What's 1+1/21 + 3**2?    
    {sr.to_prompt()}
    """

    client = LLMServiceClient(host="localhost", port=9999)
    client.set_model("TOGETHER/DeepSeek-V3.1-thinking")

    request = LLMRequest(conversation=Conversation.from_prompt(prompt), response_model=sr, max_completion_tokens=9699)

    response = await client.chat(request)

    print("Response:", response.output.instance)

async def main():
    # Initialize the client
    await simple_case()
    print("-===========================-")
    await structured_case()

if __name__ == "__main__":
    asyncio.run(main()) 