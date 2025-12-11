# 🎗️ Celerates Final Project  
**Klasifikasi Kanker Payudara & Asisten Chatbot AI**

✨ App berbasis Python yang menggabungkan machine learning untuk deteksi kanker payudara dan chatbot AI untuk interaksi pengguna._

---

## 🧠 Daftar Isi

1. [Tentang Proyek](#tentang-proyek)  
2. [Fitur](#fitur)  
3. [Teknologi](#teknologi)  
4. [Struktur Repository](#struktur-repository)  
5. [Instalasi](#instalasi)  
6. [Dataset](#dataset)  
---

## 📌 Tentang Proyek

Repository ini berisi aplikasi AI dengan dua tujuan:

💡 **1. Klasifikasi Kanker Payudara**  
Model machine learning yang dilatih untuk mengklasifikasikan data tumor payudara (misalnya jinak vs ganas) menggunakan algoritma standar industri seperti Random Forest, KNN, atau Naive Bayes. Model menghasilkan prediksi dari fitur input.

🗣️ **2. Asisten Chatbot AI**  
Asisten cerdas yang dapat menjawab pertanyaan pengguna, menjelaskan prediksi, dan memberikan panduan terkait kanker payudara. Menggunakan pencarian vektor dan pipeline ala LangChain untuk konteks dokumen.

💻 Proyek ini dapat diakses melalui antarmuka Python sederhana (`app.py`) yang mengintegrasikan prediksi ML dan UI chatbot.

---

## 🚀 Fitur

✔️ Prediksi kanker payudara berbasis ML  
✔️ Chatbot interaktif untuk penjelasan dan panduan  
✔️ Antarmuka web Python sederhana  
✔️ Kode modular untuk pengembangan mudah  
✔️ Model pretrained tersedia  

---

## 🧰 Teknologi

| Komponen        | Teknologi                           |
|-----------------|------------------------------------|
| Backend         | Python                             |
| Web App         | Streamlit / Flask (berdasarkan app.py) |
| Library ML      | Scikit-learn, TensorFlow/PyTorch   |
| Vector DB       | Chroma (folder tersedia)           |
| Model Bahasa    | OpenAI / HuggingFace (opsional)   |
| Penyimpanan Data| Folder lokal `data/`               |

---

## 📁 Struktur Repository

Celerates-Final-Project/
│
├── assets/ # Aset frontend (gambar, ikon)
├── chroma_db/ # Vector embeddings lokal
├── data/ # Dataset mentah & terproses
├── models/ # Model ML & tokenizer
├── modules/ # Modul Python (ML + logika chatbot)
├── app.py # Entry point aplikasi
├── requirements.txt # Dependensi Python
└── README.md # Dokumentasi proyek

## ⚙️ Instalasi

  1. **Clone repository**
  
     ```bash
     git clone https://github.com/Ruler02/Celerates-Final-Project.git
     cd Celerates-Final-Project
  
  2. Buat environment Python
  
     python -m venv venv
     source venv/bin/activate     # Linux / Mac
     venv\Scripts\activate        # Windows
  
  3. Instal dependensi
     pip install -r requirements.txt
   
  ##📂 Dataset
  
  Proyek ini menggunakan dataset tabular dengan fitur tumor (misal: radius, tekstur, perimeter).
  
  👉 Dataset umum: Wisconsin Breast Cancer Dataset (CSV atau sklearn built-in).
  📌 Tempatkan dataset di folder data/.

  ##🔄 Flowchart Sistem
  
  ![Flowchart sistem](assets/flowwchart Database (1).png)
