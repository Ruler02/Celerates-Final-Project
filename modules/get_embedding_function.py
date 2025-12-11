import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def get_embedding_function():
    """
    Get embedding function dengan caching Streamlit
    Hanya load sekali per session
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

