import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

llm = init_chat_model(os.getenv("CHAT_MODEL"), model_provider=os.getenv("MODEL_PROVIDER"))

output = llm.invoke("write a story about an astronaut")

print(output.content)