from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function(api_key=None):
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embedding
