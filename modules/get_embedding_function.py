from langchain_community.embeddings import HuggingFaceHubEmbeddings

def get_embedding_function(api_key):
    return HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=api_key
    )
