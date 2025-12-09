import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# Import dari modules
from modules.model_loader import load_model, load_dataset
from modules.services import query_rag
from modules.ui_styles import set_background_image
from modules.prompts import PROMPT_TEMPLATE_BENIGN, PROMPT_TEMPLATE_MALIGNANT
from modules.get_embedding_function import get_embedding_function

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "..", "data", "chroma_db")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Knowledge")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "data", "chroma_db")

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
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":
    set_background_image("assets/bg.jpeg")
    st.title("🎗️ AI Breast Cancer Assistant")
    st.image("assets/header.jpg", use_column_width=True)

    st.markdown("### 🔑 Masukkan API Key Gemini")

    api_key_input = st.text_input(
        "API Key:", type="password", value=st.session_state.api_key
    )

    def validate_key(key):
        try:
            test = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=key)
            _ = test.invoke("Halo, cek API Key!")
            return True
        except Exception:
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

    if not st.session_state.api_key:
        st.warning("Silakan isi API Key terlebih dahulu di halaman Home.")
        st.stop()

    if st.session_state.diagnosis is None:
        st.warning("Silakan lakukan prediksi kanker terlebih dahulu.")
        st.stop()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant",
                             avatar=CHAT_AVATAR_USER if isinstance(msg, HumanMessage) else CHAT_AVATAR_AI):
            st.markdown(msg.content)

    # Chat input
    prompt = st.chat_input("Tanyakan apapun mengenai hasil diagnosa Anda...")

    if prompt:
        st.session_state.messages.append(HumanMessage(prompt))

        with st.chat_message("user", avatar=CHAT_AVATAR_USER):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=CHAT_AVATAR_AI):
            with st.spinner("⏳ Mencari informasi & menganalisis..."):
                response, sources = query_rag(
                    query_text=prompt,
                    chroma_path=CHROMA_PATH,
                    diagnosis=st.session_state.diagnosis,
                    benign_template=PROMPT_TEMPLATE_BENIGN,
                    malignant_template=PROMPT_TEMPLATE_MALIGNANT,
                    api_key=st.session_state.api_key,
                )

            st.markdown(response)

            # Show document sources
            with st.expander("📚 Sumber Referensi"):
                unique_sources = list({x for x in sources if x})
                if unique_sources:
                    for i, src in enumerate(unique_sources, 1):
                        st.write(f"**{i}. {src}**")
                else:
                    st.info("Tidak ada sumber referensi ditemukan.")

        st.session_state.messages.append(AIMessage(response))


# =====================
# SIDEBAR EXTRA
# =====================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
