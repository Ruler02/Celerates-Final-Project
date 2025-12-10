from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function(api_key):
    """
    Membuat HuggingFace embeddings yang aman untuk Streamlit Cloud.
    """
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=api_key,
    )
    return embeddings