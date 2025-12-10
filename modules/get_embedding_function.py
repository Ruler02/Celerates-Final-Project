from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
import os

def get_embedding_function(api_key):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
