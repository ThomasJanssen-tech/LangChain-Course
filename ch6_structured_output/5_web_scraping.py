import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

from pydantic import BaseModel, Field
from typing import List, Optional

from langchain.chat_models import init_chat_model

class PropertyDetails(BaseModel):
    location: str = Field(description="The proprty's location")
    amount_of_bedrooms: int = Field(description="Amount of bedrooms")
    amount_of_bathrooms: str = Field(description="Amount of bathrooms")
    price_usd: int = Field(description="The proprty's price in USD")

class PropertyPage(BaseModel):
    houses: List[PropertyDetails] = Field(description="Details about property")

loader = WebBaseLoader("https://www.python-unlimited.com/webscraping/houses.php")

page = loader.load()

#print(page[0].page_content)

classification_prompt = PromptTemplate.from_template(
    """
Scrape information about every property from the source code I provide you and 
return in structured format.

Source code:
{input}
"""
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    temperature = 0
)


llm_with_structured_output = llm.with_structured_output(PropertyPage)

chain = classification_prompt | llm_with_structured_output

output = chain.invoke({"input": page[0].page_content}).model_dump()

df = pd.json_normalize(output['houses'])

df.to_excel("properties.xlsx",index=False)