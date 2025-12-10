from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import os
from modules.get_embedding_function import get_embedding_function

def query_rag(query_text, chroma_path, diagnosis, benign_template, malignant_template, api_key):
    """
    Melakukan pencarian RAG dan generate jawaban menggunakan Hugging Face endpoint.
    """
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

    # Ambil embedding function dengan api_key
    embedding_function = get_embedding_function(api_key)
    
    # Load vectorstore Chroma
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Cari dokumen paling relevan
    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Pilih template sesuai diagnosis
    template = malignant_template if diagnosis == "M" else benign_template
    prompt_text = ChatPromptTemplate.from_template(template).format(
        context=context_text, question=query_text
    )

    # Gunakan Hugging Face endpoint
    hf_model = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",
        task="text-generation",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.03,
        huggingfacehub_api_token=api_key
    )
    chat_hf = ChatHuggingFace(llm=hf_model, verbose=False)

    messages = [
        ("system", "You are a helpful medical assistant."),
        ("human", prompt_text),
    ]

    response = chat_hf.invoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return response_text, sources
