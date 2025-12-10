import streamlit as st
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from get_embedding_function import get_embedding_function

GOOGLE_API_KEY = st.secrets["AIzaSyDPvbi1VECzygOIzy6QWE3zzv9JOTK-g4A"]
CHROMA_PATH = st.secrets["ck-6DEJFeP2LR8cAQYN6raUAqQzyfC6gRo31L7BVoxJaJzx"]

st.set_page_config(page_title="Chatbot RAG Debug", layout="wide")
st.title("💬 Chatbot RAG dengan Chroma - DEBUG MODE")

# Initialize Chroma database
CHROMA_PATH = "ck-6DEJFeP2LR8cAQYN6raUAqQzyfC6gRo31L7BVoxJaJzx"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    """Query the RAG system and return response with sources"""
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # DEBUG: Check database
        total_docs = db._collection.count()
        st.write(f"**[DEBUG] Total dokumen di database: {total_docs}**")
        
        if total_docs == 0:
            st.error("❌ DATABASE KOSONG! Tidak ada dokumen di Chroma. Silakan import PDF terlebih dahulu.")
            return "Database kosong", []
        
        # DEBUG: Perform similarity search
        st.write(f"**[DEBUG] Melakukan similarity search dengan k=20...**")
        results = db.similarity_search_with_score(query_text, k=20)
        
        st.write(f"**[DEBUG] Dokumen yang diambil: {len(results)}**")
        
        if len(results) == 0:
            st.warning("⚠️ Tidak ada dokumen yang cocok dengan query!")
            return "Tidak ada dokumen yang cocok", []
        
        # DEBUG: Show retrieved documents
        with st.expander("📋 DEBUG: Dokumen yang Diambil"):
            for idx, (doc, score) in enumerate(results[:5], 1):
                st.write(f"**Doc {idx} - Score: {score:.6f}**")
                st.write(f"ID: {doc.metadata.get('id', 'N/A')}")
                st.write(f"Content: {doc.page_content[:200]}...")
                st.divider()
        
        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        st.write(f"**[DEBUG] Context length: {len(context_text)} characters**")
        
        if len(context_text) == 0:
            st.error("❌ Context kosong!")
            return "Context kosong", []
        
        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        st.write(f"**[DEBUG] Prompt length: {len(prompt)} characters**")
        
        # Show context preview
        with st.expander("📝 DEBUG: Context Preview"):
            st.text(context_text[:500] + "...")
        
        # Get response
        st.write("**[DEBUG] Mengirim ke LLM...**")
        model = ChatGoogleGenerativeAI( model="gemini-1.5-pro")
        response_text = model.invoke(prompt)
        
        # Extract sources
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        
        response_content = response_text.content if hasattr(response_text, 'content') else str(response_text)
        
        st.write("**[DEBUG] ✓ Response berhasil dihasilkan**")
        
        return response_content, sources
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("🔧 Error Details"):
            st.text(traceback.format_exc())
        return f"Error: {str(e)}", []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# Chat input
prompt = st.chat_input("Masukkan pertanyaan Anda...")

if prompt:
    # Add user message to history and display
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get response from RAG
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("⏳ Mencari informasi dan menyiapkan jawaban..."):
            response_text, sources = query_rag(prompt)
        
        st.divider()
        
        # Display response
        st.markdown(response_text)
        
        # Display sources in expander
        with st.expander("📚 Lihat Sumber Referensi"):
            unique_sources = list(set([s for s in sources if s is not None]))
            if unique_sources:
                for i, source in enumerate(unique_sources, 1):
                    st.write(f"**{i}. {source}**")
            else:
                st.info("Tidak ada sumber yang ditemukan")
    
    # Add assistant response to history
    st.session_state.messages.append(AIMessage(response_text))

# Sidebar info
with st.sidebar:
    st.title("ℹ️ Informasi")
    st.markdown("""
    ### Tentang Chatbot Ini
    - Menggunakan **Chroma** sebagai Vector Database
    - Menggunakan **Google Generative AI (Gemini 2.0 Flash)** untuk generating responses
    - Sistem **RAG (Retrieval Augmented Generation)**
    
    ### Debugging Mode
    Debug information akan ditampilkan di atas chat response
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()