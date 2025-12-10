%%writefile app.py
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from utils import process_document

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="AI Rapot Generator", page_icon="🎓", layout="wide")
st.title("🎓 Generator Narasi Rapot Otomatis")
st.markdown("Upload Rubrik PDF -> Isi Skor -> Jadi Narasi ✨")

# --- Inisialisasi Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "criteria_list" not in st.session_state:
    st.session_state.criteria_list = []

# --- Sidebar: Pengaturan ---
with st.sidebar:
    st.header("⚙️ Pengaturan Guru")
    
    # LOGIKA PINTAR: Cek apakah API Key ada di Secrets (Cloud)
    if "GROQ_API_KEY" in st.secrets:
        st.success("✅ API Key terdeteksi dari sistem!")
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        # Jika dijalankan lokal/tanpa secrets, minta input manual
        api_key = st.text_input("Masukkan Groq API Key", type="password")
        if not api_key:
            st.warning("⚠️ Masukkan API Key untuk memulai.")

    uploaded_file = st.file_uploader("Upload PDF Rubrik", type="pdf")
    
    if st.button("Reset / Mulai Ulang"):
        st.session_state.vectorstore = None
        st.session_state.criteria_list = []
        st.rerun()

# --- Fungsi: Ekstraksi Kriteria (Otak 1) ---
def extract_criteria(vectorstore, api_key):
    """Menggunakan AI untuk membaca judul-judul aspek penilaian di PDF"""
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    retriever = vectorstore.as_retriever()
    
    # Prompt khusus untuk mencari daftar judul
    extract_prompt = ChatPromptTemplate.from_template("""
    Analisis dokumen rubrik berikut.
    Temukan daftar JUDUL ASPEK PENILAIAN utama (biasanya ada nomor 1, 2, 3 atau dicetak tebal di kolom aspek).
    Contoh aspek: "Kehadiran", "Keterlibatan", "Pemahaman Tools". Tapi jangan mengulang judul yang sudah ditemukan.
    
    <context>
    {context}
    </context>
    
    Tugasmu: Hanya keluarkan daftar judul aspek dipisahkan dengan koma (,). 
    Jangan ada teks pembuka/penutup.
    Jangan mengulang judul yang sudah ditulis. 
    Contoh Output: Kehadiran, Keterlibatan, Kreativitas
    """)
    
    # Chain sederhana
    chain = (
        {"context": retriever, "input": RunnablePassthrough()} 
        | extract_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Jalankan
    result = chain.invoke("Sebutkan semua aspek penilaian yang ada di tabel.")
    # Bersihkan hasil menjadi list python
    criteria = [x.strip() for x in result.split(',')]
    return criteria

# --- LOGIKA UTAMA ---

# 1. Proses Dokumen (Jika ada file baru)
if uploaded_file and api_key:
    if st.session_state.vectorstore is None:
        with st.spinner("Sedang membaca PDF dan mendeteksi kriteria penilaian..."):
            try:
                # Proses vectorstore (Backend)
                st.session_state.vectorstore = process_document(uploaded_file)
                
                # Proses ekstraksi kriteria (AI membaca struktur tabel)
                st.session_state.criteria_list = extract_criteria(st.session_state.vectorstore, api_key)
                
                st.success(f"Berhasil mendeteksi {len(st.session_state.criteria_list)} kriteria!")
            except Exception as e:
                st.error(f"Error: {e}")

# 2. Tampilan Form Penilaian (Dinamis)
if st.session_state.vectorstore and st.session_state.criteria_list:
    st.divider()
    st.header("2. Input Nilai Siswa")
    
    col_bio1, col_bio2 = st.columns(2)
    with col_bio1:
        nama_siswa = st.text_input("Nama Siswa", placeholder="Contoh: Audy")
    with col_bio2:
        kelas = st.text_input("Program", placeholder="Contoh: Code & Explore Modul 1")

    st.subheader("Skor Penilaian (1-4)")
    
    # --- DYNAMIC FORM GENERATION ---
    # Di sini kita membuat tombol secara otomatis berdasarkan hasil bacaan AI
    user_scores = {}
    
    # Bikin grid 2 kolom agar rapi
    cols = st.columns(2) 
    
    for i, criteria in enumerate(st.session_state.criteria_list):
        with cols[i % 2]: # Ganti-ganti kolom kiri/kanan
            # Menampilkan Radio Button Horizontal
            score = st.radio(
                label=f"**{criteria}**", 
                options=[1, 2, 3, 4], 
                horizontal=True,
                key=f"score_{i}" # Key unik agar tidak bentrok
            )
            user_scores[criteria] = score
            
    catatan_tambahan = st.text_area("Catatan Khusus (Opsional)", placeholder="Misal: Sering terlambat tapi sangat kreatif...")

    # 3. Generate Rapot (Otak 2)
    if st.button("✨ Buat Narasi Rapot", type="primary"):
        if not nama_siswa:
            st.warning("Nama siswa belum diisi!")
        else:
            # Rangkai data input menjadi teks untuk dikirim ke AI
            score_summary = ", ".join([f"{k}: {v}" for k, v in user_scores.items()])
            full_query = f"""
            Nama Siswa: {nama_siswa}
            Kelas: {kelas}
            
            Detail Skor Per Aspek:
            {score_summary}
            
            Catatan Tambahan Guru:
            {catatan_tambahan}
            """
            
            # --- RAG CHAIN ---
            llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
            
            prompt = ChatPromptTemplate.from_template("""
            Anda adalah Wali Kelas profesional. Tugas Anda adalah membuat **Narasi Deskripsi Rapot** yang personal untuk orang tua.
            
            Gunakan referensi RUBRIK berikut untuk menerjemahkan skor angka (1-4) menjadi kalimat deskriptif yang tepat:
            <rubrik>
            {context}
            </rubrik>

            Data Siswa:
            {input}

            Instruksi Penulisan:
            1. Buka dengan sapaan hormat kepada Orang Tua [Nama Siswa].
            2. Paragraf 1: Jelaskan KEKUATAN siswa (aspek dengan skor 3 atau 4). Gabungkan deskripsi dari rubrik dengan kalimat yang mengapresiasi.
            3. Paragraf 2: Jelaskan AREA PENGEMBANGAN (aspek dengan skor 1 atau 2). Gunakan bahasa yang "sandwich" (positif-korektif-positif) dan tidak menghakimi. Berikan saran konkret berdasarkan kolom "NextStep" atau "Saran" di rubrik jika ada.
            4. Tutup dengan kalimat motivasi dan harapan.
            5. Gaya bahasa: Hangat, Profesional, Bahasa Indonesia Baku tapi bukan robot.
            """)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = st.session_state.vectorstore.as_retriever()
            
            rag_chain = (
                RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
                .assign(context=lambda x: format_docs(x["context"]))
                | prompt
                | llm
                | StrOutputParser()
            )
            
            with st.spinner("AI sedang merangkai kata-kata mutiara..."):
                final_narrative = rag_chain.invoke(full_query)
                
                st.success("Selesai!")
                st.markdown("### 📝 Hasil Narasi Rapot:")
                st.text_area("Siap Copy-Paste:", value=final_narrative, height=400)

elif not api_key:
    st.info("👈 Masukkan API Key di sidebar untuk memulai.")
