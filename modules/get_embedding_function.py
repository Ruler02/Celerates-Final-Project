from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


def get_embedding_function():
    hf_key = st.secrets["HF_APIKEY"]

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        api_key=hf_key,
    )
