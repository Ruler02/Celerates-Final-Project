from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

def get_embedding_function():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformerEmbeddings(model)
