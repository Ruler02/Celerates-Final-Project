from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import streamlit as st

def get_embedding_function():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_APIKEY"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_url="https://router.huggingface.co"
    )
