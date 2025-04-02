import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

import streamlit as st

st.title("Ask questions to an LLM")

prompt = st.chat_input("Ask me anything..")

if prompt:

    llm = init_chat_model(
        os.getenv("CHAT_MODEL"), 
        model_provider = os.getenv("MODEL_PROVIDER"),
        temperature = 0.7
    )

    output = llm.invoke(f"{prompt}")


    st.write(f"{output.content}")