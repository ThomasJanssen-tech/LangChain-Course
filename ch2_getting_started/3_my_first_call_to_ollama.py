import os
from dotenv import load_dotenv

load_dotenv()

#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b"
)

output = llm.invoke("why is the sky blue?")

print(output.content)