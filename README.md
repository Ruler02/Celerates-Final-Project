


Medical Assistant Chatbot (CantCheerBOt)
Chatbot berbasis RAG yang membantu pengguna mendapatkan informasi medis berdasarkan hasil prediksi kanker payudara.
💡 Tentang Nama
CantCheerBOt berasal dari kombinasi kata:

Can't - Tidak bisa (mencegah kanker)
Cancer - Kanker
Cheer - Memberi semangat

Sehingga makna aplikasi ini adalah memberi semangat untuk tidak terkena kanker.

📋 Fitur Utama
Klasifikasi Kanker Payudara

Input numerik untuk setiap fitur dataset (30 fitur)
Nilai awal diisi dari rata-rata dataset
Prediksi menggunakan machine learning model
Simpan hasil diagnosis ke session state
Tampilkan pesan sukses/error berdasarkan hasil (Benign/Malignant)
Arahkan user ke menu Chatbot untuk informasi lebih lanjut

Halaman Chatbot

Validasi status diagnosis user
Menampilkan riwayat percakapan
Menerima input pertanyaan dari user
Menampilkan respons chatbot secara real-time

RAG Chatbot dengan LLM

Load model HuggingFace (DeepSeek)
Cari konteks relevan dari knowledge base
Generate jawaban berdasarkan konteks
Tampilkan sumber referensi
Hapus riwayat chat dengan tombol di sidebar

🛠️ Teknologi

Streamlit - Framework
HuggingFace - LLM Model
ChromaDB - Vector Database
ScikitLearn - Classification
Python - Language

🔄 Alur Kerja

![My Image](https://github.com/Mercytopsy/RAG-MultiDoc-Chatbot/blob/main/Architectural%20Diagram.png)
