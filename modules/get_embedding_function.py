from langchain.embeddings import HuggingFaceHubEmbeddings

def get_embedding_function(api_key: str):
    """
    Membuat HuggingFaceHubEmbeddings untuk Streamlit Cloud.
    """
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=api_key
    )
    return embeddings