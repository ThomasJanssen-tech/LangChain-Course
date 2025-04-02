import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from typing import List, Optional

from langchain.chat_models import init_chat_model

class Classification(BaseModel):
    sentiment: str = Field(enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        enum=["spanish", "english", "french", "german", "italian"]
    )


review = """
My fries were cold and there was an insect in my burger. Never coming here again!!!!!
"""

classification_prompt = PromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0
)

llm_with_structured_output = llm.with_structured_output(Classification)

chain = classification_prompt | llm_with_structured_output

output = chain.invoke({"input": review})

print("===Guest review details:")

print(f"Sentiment: {output.sentiment}")
print(f"Agressiveness: {output.aggressiveness}")
print(f"Language: {output.language}")