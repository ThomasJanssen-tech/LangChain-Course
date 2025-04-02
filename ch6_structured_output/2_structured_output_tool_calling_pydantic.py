import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field
from typing import List, Optional



class Person(BaseModel):
    """Provide structured information about a person"""

    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")
    occupation: str = Field(description="The person's occupation")
    number_of_children: Optional[int] = Field(description="Amount of kids")
    related_persons: List[str] = Field(description="A list of related persons")


person_info = """
Benjamin Franklin (January 17, 1706 [O.S. January 6, 1705][Note 1] - April 17, 1790) was an American polymath: a writer, scientist, inventor, statesman, diplomat, printer, publisher and political philosopher.[1] Among the most influential intellectuals of his time, Franklin was one of the Founding Fathers of the United States; a drafter and signer of the Declaration of Independence; and the first postmaster general.[2]

Born in the Province of Massachusetts Bay, Franklin became a successful newspaper editor and printer in Philadelphia, the leading city in the colonies, publishing The Pennsylvania Gazette at age 23.[3] He became wealthy publishing this and Poor Richard's Almanack, which he wrote under the pseudonym "Richard Saunders".[4] After 1767, he was associated with the Pennsylvania Chronicle, a newspaper known for its revolutionary sentiments and criticisms of the policies of the British Parliament and the Crown.[5] He pioneered and was the first president of the Academy and College of Philadelphia, which opened in 1751 and later became the University of Pennsylvania. He organized and was the first secretary of the American Philosophical Society and was elected its president in 1769. He was appointed deputy postmaster-general for the British colonies in 1753,[6] which enabled him to set up the first national communications network.
"""


prompt = PromptTemplate.from_template(
"""
"Provide the name of the person (first + last name), the amount of children (in case provided),
 the occupation and a list of related persons for the following person: 
 
 {person_info}

Don't make things up, and only use the information which is provided to you
""")
                                              
llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0
)


llm_with_structured_output = llm.with_structured_output(Person)

chain = prompt | llm_with_structured_output

output = chain.invoke({"person_info":person_info})

print(output)

print(f"First name: {output.first_name}")
print(f"Last name: {output.last_name}")
print(f"Occupation: {output.occupation}")
print(f"Amount of children: {output.number_of_children}")
print(f"Related persons: {output.related_persons}")
