from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(api_key):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    HuggingFaceAPI=api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)
    
    return embeddings
