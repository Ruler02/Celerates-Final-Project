import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace  # 🔄 Diganti ke HuggingFace\
from modules.services import Query_RAG
from langchain_community.vectorstores import Chroma
from modules.get_embedding_function import get_embedding_function
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import dari modules
from modules.model_loader import load_model, load_dataset
from modules.services import query_rag
from modules.ui_styles import set_background_image
from modules.prompts import PROMPT_TEMPLATE_BENIGN, PROMPT_TEMPLATE_MALIGNANT
from modules.get_embedding_function import get_embedding_function

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")  # ← ini yang benar

# Optional emoji icons
CHAT_AVATAR_USER = "👤"
CHAT_AVATAR_AI = "🤖"

st.set_page_config(page_title="Breast Cancer AI Assistant", layout="wide")

# === Load Model & Dataset ===
model = load_model()
dataset = load_dataset()

# === Sidebar Navigation ===
st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio("Pilih Menu", ["🏠 Home", "📊 Prediksi Kanker", "💬 Chatbot"])



# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":
    set_background_image("assets/bg.jpeg")
    st.title("🎗️ AI Breast Cancer Assistant")
    st.image("assets/header.jpg", use_column_width=True)

    st.markdown("### 🔑 API Key Hugging Face (optional jika model private)")

    # Pastikan session_state sudah ada
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key_input = st.text_input(
        "API Key:", type="password", value=st.session_state.api_key
    )

def validate_key(key):
    try:
        # Set token di environment sementara
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
        # Coba panggil model minimal
        test = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V3.2",
            task="text-generation",
            max_new_tokens=10,
        )
        chat = ChatHuggingFace(llm=test, verbose=False)
        _ = chat.invoke([("system", "Test"), ("human", "Halo!")])
        return True
    except Exception as e:
        print("Error saat validasi:", e)
        return False

if st.button("Simpan API Key 🔐"):
    if not api_key_input:
        st.error("API Key tidak boleh kosong!")
    else:
        if validate_key(api_key_input):
            st.session_state.api_key = api_key_input
            st.success("🎉 API Key valid & aktif digunakan!")
        else:
            st.error("❌ API Key tidak valid atau kuota habis!")


# =========================
# PREDIKSI PAGE
# =========================
elif page == "📊 Prediksi Kanker":
    st.title("📊 Prediksi Kanker Payudara")

    df_input = {
        col: st.number_input(col, value=float(dataset[col].mean()))
        for col in dataset.columns
    }

    if st.button("🔍 Prediksi"):
        prediction = model.predict(pd.DataFrame([df_input]))[0]
        st.session_state.diagnosis = prediction

        if prediction == "M":
            st.error("⚠️ Prediksi: **Malignant (Ganas)**")
        else:
            st.success("💚 Prediksi: **Benign (Jinak)**")

        st.info("Silakan lanjut ke menu Chatbot untuk bertanya lebih lanjut.")

# =========================
# CHATBOT PAGE
# =========================
elif page == "💬 Chatbot":
    set_background_image("assets/bg.jpeg")
    st.title("💬 Medical Assistant Chatbot")

    if not hasattr(st.session_state, "diagnosis") or st.session_state.diagnosis is None:
        st.warning("Silakan lakukan prediksi kanker terlebih dahulu.")
        st.stop()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant",
                             avatar=CHAT_AVATAR_USER if isinstance(msg, HumanMessage) else CHAT_AVATAR_AI):
            st.markdown(msg.content)

    prompt = st.chat_input("Tanyakan apapun mengenai hasil diagnosa Anda...")

    if prompt is not None and prompt.strip() != "":
    # Simpan ke session
        if "last_prompt" not in st.session_state or prompt != st.session_state.last_prompt:
         st.session_state.last_prompt = prompt
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user", avatar=CHAT_AVATAR_USER):
          st.markdown(prompt)
    
    model_hf = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",  # ganti sesuai model yang ingin digunakan
        task="text-generation",
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.03,
        huggingfacehub_api_token=st.session_state.api_key
    )

    chat_hf = ChatHuggingFace(llm=model_hf, verbose=True)

    # Proses query RAG + text generation
    with st.chat_message("assistant", avatar=CHAT_AVATAR_AI):
        with st.spinner("⏳ Mencari informasi & menganalisis..."):
            # Ambil jawaban RAG
            response_rag, sources = Query_RAG(
                query_text=prompt,
                chroma_path=CHROMA_PATH,
                diagnosis=st.session_state.diagnosis,
                benign_template=PROMPT_TEMPLATE_BENIGN,
                malignant_template=PROMPT_TEMPLATE_MALIGNANT,
                api_key=st.session_state.api_key
            )

            # Generate jawaban tambahan via Hugging Face
            response_hf = chat_hf.invoke([("system", "You are a helpful medical assistant."),
                                          ("human", response_rag)])

            st.markdown(response_hf.content)

            # Sumber referensi
            with st.expander("📚 Sumber Referensi"):
                unique_sources = list({x for x in sources if x})
                if unique_sources:
                    for i, src in enumerate(unique_sources, 1):
                        st.write(f"**{i}. {src}**")
                else:
                    st.info("Tidak ada sumber referensi ditemukan.")
            
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            print("Jumlah dokumen di DB:", len(db.get(include=['documents'])['documents']))

        st.session_state.messages.append(AIMessage(content=response_hf.content))

# =====================
# SIDEBAR EXTRA
# =====================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
