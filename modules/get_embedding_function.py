import streamlit as st
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

st.write("APINYA:", st.secrets["HF_APIKEY"])

def get_embedding_function():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_APIKEY"],
        model_name="BAAI/bge-base-en-v1.5"
    )
