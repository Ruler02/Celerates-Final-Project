from langchain_huggingface import HuggingFaceBgeEmbeddings
import os

def get_embedding_function(api_key=None):
    """
    Mengembalikan embedding function menggunakan HuggingFace BGE Embedding.
    Jika model private → wajib pakai API Key.
    """
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

    # Model embedding ringan & akurat untuk RAG
    model_name = "sentence-transformers/all-mpnet-base-v2"

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    return embedding_model
