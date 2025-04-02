import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0,
    max_tokens = 1000
)

output = llm.invoke("write a story about a cowboy")

print(output.content)