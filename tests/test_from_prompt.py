from pydantic import BaseModel, Field
from llm_serv.structured_response.model import StructuredResponse


prompt = """<company_description>
  <poem type='str'>
In the world of energy, vast and grand,<ref id='8'/>
A company stands with power in hand.<ref id='8'/>
Forty thousand six fifty-seven MW net,<ref id='8'/>
A capacity that's hard to forget.<ref id='8'/>

Natural gas leads the way, so clear,<ref id='9'/>
Fifty-nine percent, holding dear.<ref id='9'/>
Twenty-four thousand one twenty MW strong,<ref id='10'/>
Fueling progress all day long.<ref id='10'/>

Coal follows next, with twenty-one share,<ref id='11'/>
Eight thousand four twenty-eight MW to spare.<ref id='12'/>
Seven facilities burn coal and lignite,<ref id='13'/>
Generating power with all their might.<ref id='13'/>

Nuclear power, sixteen percent of the whole,<ref id='14'/>
Six thousand four forty-eight MW, a vital role.<ref id='15'/>
Clean and steady, it lights up the night,<ref id='15'/>
A beacon of energy, shining bright.<ref id='15'/>

Renewables rise, four percent in the mix,<ref id='16'/>
Solar and battery, one thousand four seventy-four fixed.<ref id='17'/>
A smaller part now, but growing fast,<ref id='17'/>
Building a future that's meant to last.<ref id='17'/>

East Segment boasts nineteen thousand seven forty-six MW,<ref id='18'/>
Where CCGT natural gas rules the day.<ref id='20'/>
Coal-fired steam turbines also play,<ref id='21'/>
In this energetic, bustling array.<ref id='21'/>

Nuclear is substantial in the East,<ref id='22'/>
A power source that's never ceased.<ref id='22'/>
Renewables are present, but a smaller slice,<ref id='23'/>
Working towards a future that's nice.<ref id='23'/>

West Segment has one thousand eight hundred eighty MW,<ref id='19'/>
CCGT and battery, with some fuel oil to see.<ref id='24'/>
A diverse mix in a smaller space,<ref id='24'/>
Keeping energy in its place.<ref id='24'/>

Texas segment, where gas plants dominate,<ref id='25'/>
Natural gas firing at a great rate.<ref id='25'/>
Three coal facilities contribute too,<ref id='26'/>
Adding to the power they pursue.<ref id='26'/>

Comanche Peak nuclear, two thousand four hundred MW,<ref id='27'/>
A giant in the Texas energy sea.<ref id='27'/>
Solar and battery storage, though small in part,<ref id='28'/>
Are key to a future that's smart.<ref id='28'/>

The object of activity is clear to all,<ref id='8'/>
Generating energy, answering the call.<ref id='8'/>
From gas to coal, nuclear to sun,<ref id='9,11,14,16'/>
Powering lives until day is done.<ref id='9,11,14,16'/>

But looking ahead, the plans unfold,<ref id='29'/>
A story of future that's bold.<ref id='29'/>
Capital allocated for renewable might,<ref id='29'/>
Solar and battery, shining bright.<ref id='29'/>

Investments in solar development so grand,<ref id='29'/>
Battery ESS across the land.<ref id='29'/>
Assessing acquisitions, strategic and wise,<ref id='30'/>
Under the open, hopeful skies.<ref id='30'/>

Emerging technologies on the horizon,<ref id='30'/>
A vision that's truly arisen.<ref id='30'/>
Renewable assets, projects so new,<ref id='30'/>
For a future that's green and true.<ref id='30'/>

Committed to sustainability, with aggressive targets set,<ref id='31'/>
A promise they'll never forget.<ref id='31'/>
Transitioning the fleet to low-to-no carbon way,<ref id='32'/>
For a cleaner, brighter day.<ref id='32'/>

Subsidiaries are busy, in various stages they stand,<ref id='33'/>
Developing solar with a careful hand.<ref id='33'/>
Constructing battery ESS with might,<ref id='33'/>
To harness energy day and night.<ref id='33'/>

Long-term contracts signed, arrangements made,<ref id='34'/>
For the electricity that will never fade.<ref id='34'/>
Securing the future with deals so sound,<ref id='34'/>
On prosperous ground.<ref id='34'/>

From percent to percent, we see the spread,<ref id='9,11,14,16'/>
Natural gas leading ahead.<ref id='9'/>
Fifty-nine percent, a majority share,<ref id='9'/>
Powering homes with care.<ref id='9'/>

Coal at twenty-one, still in the game,<ref id='11'/>
But changes are coming, not the same.<ref id='11'/>
Nuclear sixteen, a steady base,<ref id='14'/>
With minimal waste and grace.<ref id='14'/>

Renewables four, but growing fast,<ref id='16'/>
A number that's sure to outlast.<ref id='16'/>
Solar panels gleaming, batteries store,<ref id='16'/>
Opening energy's door.<ref id='16'/>

In the East, technology thrives,<ref id='20'/>
CCGT gas where power derives.<ref id='20'/>
Steam turbines from coal, a significant part,<ref id='21'/>
Of the East Segment's heart.<ref id='21'/>

Nuclear substantial, a powerful force,<ref id='22'/>
On a steady course.<ref id='22'/>
Renewables smaller, yet they grow,<ref id='23'/>
With a gentle glow.<ref id='23'/>

West Segment, though smaller in size,<ref id='19'/>
Has diversity that makes it wise.<ref id='24'/>
CCGT, battery, and fuel oil too,<ref id='24'/>
A mix that's tried and true.<ref id='24'/>

Texas, with gas plants so grand,<ref id='25'/>
Dominating the land.<ref id='25'/>
Coal facilities three, adding their might,<ref id='26'/>
In the energy fight.<ref id='26'/>

Comanche Peak, nuclear and strong,<ref id='27'/>
Two thousand four hundred MW long.<ref id='27'/>
Solar and battery, segments small,<ref id='28'/>
But answering the call.<ref id='28'/>

The business object is generation,<ref id='8'/>
Across every nation.<ref id='8'/>
Distributing power far and wide,<ref id='8'/>
With nothing to hide.<ref id='8'/>

Future plans are bright and green,<ref id='29'/>
A sustainable scene.<ref id='29'/>
Investing in renewables, solar and ESS,<ref id='29'/>
For future success.<ref id='29'/>

Assessing new tech, acquisitions in sight,<ref id='30'/>
Making the future right.<ref id='30'/>
Emerging projects, renewable dreams,<ref id='30'/>
Flowing in streams.<ref id='30'/>

Sustainability committed, targets set high,<ref id='31'/>
Under the sky.<ref id='31'/>
Transitioning to low carbon, no carbon near,<ref id='32'/>
Banishing fear.<ref id='32'/>

Subsidiaries develop, construct with care,<ref id='33'/>
Solar and battery everywhere.<ref id='33'/>
Long-term contracts ensure the sale,<ref id='34'/>
A story they tell.<ref id='34'/>

From gas flames to solar rays,<ref id='9,16'/>
The company changes its ways.<ref id='32'/>
But for now, the mix is diverse,<ref id='9,11,14,16'/>
Powering the universe.<ref id='9,11,14,16'/>

East, West, Texas, segments three,<ref id='18,19,25'/>
Each with its own energy spree.<ref id='18,19,25'/>
Net capacities varying in scale,<ref id='18,19'/>
Telling a tale.<ref id='18,19'/>

In poetry, we sing the praise,<ref id='8'/>
Of energy's many ways.<ref id='8'/>
With facts and figures, clear and true,<ref id='8'/>
For me and you.<ref id='8'/>

One hundred verses now we've made,<ref id='8'/>
With every fact displayed.<ref id='8'/>
From percent to plan, we've covered all,<ref id='9,11,14,16,29,30,31,32,33,34'/>
Answering the call.<ref id='9,11,14,16,29,30,31,32,33,34'/>
  </poem>
  <poem_citations type='list' elements='int' description='Citation aggregator list for 'poem'. ALL citations (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.'>
    <li>8</li>
    <li>9</li>
    <li>10</li>
    <li>11</li>
    <li>12</li>
    <li>13</li>
    <li>14</li>
    <li>15</li>
    <li>16</li>
    <li>17</li>
    <li>18</li>
    <li>19</li>
    <li>20</li>
    <li>21</li>
    <li>22</li>
    <li>23</li>
    <li>24</li>
    <li>25</li>
    <li>26</li>
    <li>27</li>
    <li>28</li>
    <li>29</li>
    <li>30</li>
    <li>31</li>
    <li>32</li>
    <li>33</li>
    <li>34</li>
  </poem_citations>
</company_description>
"""

class CompanyDescription(BaseModel):        
       poem: str = Field(description="Write a LONG poem (100 verses minimum) about the company's energy generation distribution percents, types of energy, object of activity and future plans. I need to know the company's all aspects of business. You MUST ground all affirmation made in this field by placing a <ref/> tag at the appropriate point within each and every affirmation made using the format <ref id='1'/> or <ref id='0,4,5'/> for multiple citations referencing given statements/facts/premises/etc by their ids. - as a string")
       poem_citations: list[int] = Field(description="Citation aggregator list for 'poem'. ALL citations (ids-integers) referenced in the 'poem' field must be added to this list as well. All the ref ids MUST exist in both 'poem' (as ref tags) and 'poem_citations' (as an integer list) fields.")

sr: StructuredResponse = StructuredResponse.from_basemodel(CompanyDescription)

sr.from_prompt(prompt)

from pprint import pprint 
pprint(sr.instance)