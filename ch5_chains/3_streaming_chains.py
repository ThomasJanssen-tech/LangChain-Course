import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

example_prompt = PromptTemplate.from_template("Write a story about: {subject}")

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0
)

chain = example_prompt | llm | StrOutputParser()

for chunk in chain.stream("goldfish"):
    print(chunk, end="", flush=True)
