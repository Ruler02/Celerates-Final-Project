from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from modules.get_embedding_function import get_embedding_function


def query_rag(query_text, chroma_path, diagnosis, benign_template, malignant_template, api_key):

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    template = malignant_template if diagnosis == "M" else benign_template
    prompt = ChatPromptTemplate.from_template(template).format(
        context=context_text, question=query_text
    )

    llm_model = ChatGoogleGenerativeAI(
        api_version="v1",
        model="gemini-2.0-flash-latest",
        api_key=api_key
    )

    response = llm_model.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources
