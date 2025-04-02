import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0.7
)

output = llm.invoke("write a story about a cowboy. make it as long as you can")

print(output.response_metadata['model'])

usage = output.usage_metadata


print(f"Input Tokens: {usage['input_tokens']}")
print(f"Output Tokens: {usage['output_tokens']}")
print(f"Total Tokens: {usage['total_tokens']}")