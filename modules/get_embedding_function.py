import streamlit as st
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

def get_embedding_function():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_APIKEY"],
        model_name="BAAI/bge-base-en-v1.5",
        task="feature-extraction" 
    )

import requests

API_KEY = st.secrets["HF_APIKEY"]
MODEL = "BAAI/bge-base-en-v1.5"

response = requests.post(
    f"https://api-inference.huggingface.co/models/{MODEL}",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"inputs": "test embedding"}
)

st.write(response.json())

