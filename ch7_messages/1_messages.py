import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model

messages = [
    SystemMessage(content="Always return the opposite of what the user says"),
    HumanMessage(content="I don't like Python")
]


llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    temperature = 0.5
)

output = llm.invoke(messages)

print(output.content)