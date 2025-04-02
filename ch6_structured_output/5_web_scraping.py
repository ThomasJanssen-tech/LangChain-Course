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

class BookDetails(BaseModel):
    title: str = Field(description="The title of the book")
    price: float = Field(description="The book's price")
    availability: str = Field(description="Whether the book is available")
    rating: int = Field(description="The book's rating")

class BookPage(BaseModel):
    books: List[BookDetails] = Field(description="Details about each book")

loader = WebBaseLoader("https://books.toscrape.com/",default_parser="html.parser")

page = loader.load()

print(page[0].page_content)

classification_prompt = PromptTemplate.from_template(
    """
Scrape information about every book from the source code I provide you and 
return in structured format.

Source code:
{input}
"""
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0
)


llm_with_structured_output = llm.with_structured_output(BookPage)

chain = classification_prompt | llm_with_structured_output

output = chain.invoke({"input": page[0].page_content}).model_dump()


print(output)

exit()

df = pd.json_normalize(output['books'])

df.to_excel("books.xlsx",index=False)