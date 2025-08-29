from pydantic import BaseModel, Field
from llm_serv.structured_response.model import StructuredResponse


prompt = """<company_description>
  <poem type='str'>
In the world's energy & air, vast and grand,<ref id='8'/>
A company stands with power in hand.<ref id='8'/>
Forty thousand six fifty-seven MW net,<ref id=8/>
A capacity that's hard to forget.<ref id='8'>

Natural gas leads the way, so clear,<ref id='9'/>
Fifty-nine percent, holding dear.<ref id='9'/>
Twenty-four thousand one twenty MW strong,<ref id='10'/>
Fueling progress all day long.<ref id='10'/>

  </poem>
  <poem_citations type='list' elements='int' description='Citation aggregator list & refs for 'poem'. ALL citations (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.'>
    <li>8</li>
    <li>9</li></poem_citations>
</company_description>
"""

class CompanyDescription(BaseModel):        
    poem: str = Field(description="Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of energy, object of activity and future plans. I need to know the company's all aspects of business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point within each and every affirmation made using the format <ref id='1'/> or <ref id='0,4,5'/> for multiple citations referencing given statements/facts/premises/etc by their ids. - as a string")
    poem_citations: list[int] = Field(description="Citation aggregator list for 'poem'. ALL citations (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.")

sr: StructuredResponse = StructuredResponse.from_basemodel(CompanyDescription)

sr.from_prompt(prompt)

from pprint import pprint 
pprint(sr.instance)