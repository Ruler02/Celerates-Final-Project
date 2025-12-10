# modules/get_embedding_function.py
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Embedding function yang aman di CPU (tidak ada NotImplementedError)
    untuk RAG di Streamlit Cloud.
    """
    # Paksa CPU
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # Bungkus ke LangChain Embedding
    embeddings = HuggingFaceEmbeddings(model_name=None)  # dummy
    embeddings.client = lambda texts: model.encode(texts, show_progress_bar=False)

    return embeddings
