from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

def get_embedding_function(api_key):
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    return embeddings
