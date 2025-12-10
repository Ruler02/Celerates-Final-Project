from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(api_key):
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
    )
    return embeddings
