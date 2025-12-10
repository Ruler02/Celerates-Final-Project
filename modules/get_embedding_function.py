from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
