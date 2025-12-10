from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embedding_function(api_key=None):
    """
    Embedding function menggunakan HuggingFace Sentence-Transformer
    untuk integrasi dengan ChromaDB.
    """

    if api_key is not None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

    model_name = "sentence-transformers/all-mpnet-base-v2"

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embedding
