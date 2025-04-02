import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
#from langchain_community.llms import Ollama

llm = ChatOpenAI()

output = llm.invoke("why is the sky blue?")

print(output.content)