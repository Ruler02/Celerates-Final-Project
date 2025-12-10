import os
from langchain.embeddings import HuggingFaceHubEmbeddings  # resmi

def get_embedding_function():
    """
    Ambil HuggingFace embeddings yang aman di Streamlit Cloud.
    API key diambil dari os.environ["HUGGINGFACEHUB_API_TOKEN"].
    """
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
    if hf_token is None:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN tidak ditemukan! Masukkan API key di app.py atau Streamlit Secrets.")

    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",  # model resmi HF
        huggingfacehub_api_token=hf_token,
        model_kwargs={"device": "cpu"}  # paksa CPU
    )

    return embeddings