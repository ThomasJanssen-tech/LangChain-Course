from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama.chat_models import ChatOllama

llm_openai = ChatOpenAI(model="gpt-4o-mini")

llm_anthropic = ChatAnthropic(model="claude-3-7-sonnet-latest")

llm_ollama = ChatOllama(model="llama3.2:1b")

output = llm_openai.invoke("why is the sky blue?")

print(output.content)