import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

messages = [
    SystemMessage("Act like a cowboy")
]

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    temperature = 1
)


while True:

    human_message = input("User (type quit to exit):")

    if(human_message == "quit"):
        break

    messages.append(HumanMessage(human_message))

    output = llm.invoke(messages)

    print(f"AI says: {output.content}")

    messages.append(AIMessage(output.content))


