import os

from dotenv import load_dotenv

load_dotenv()

from languages import *

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

import streamlit as st

example_prompt = PromptTemplate.from_template("Translate the following text from English to {language}: {text_to_translate}")


llm = init_chat_model(
    os.getenv("CHAT_MODEL"), 
    model_provider = os.getenv("MODEL_PROVIDER"),
    temperature = 0
)

st.title("Language Translation from English")

source_text = st.text_area("Text to translate:")
target_language = st.selectbox("Target language:", languages)
translate = st.button("Translate Now!")

if translate:

    prompt = example_prompt.invoke({"language":target_language,"text_to_translate":source_text})

    output = llm.invoke(prompt)

    st.write(f"{output.content}")
