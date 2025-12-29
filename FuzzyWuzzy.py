import streamlit as st
from fuzzywuzzy import process, fuzz

# Judul Aplikasi
st.title("🔍 Fuzzy String Matcher")
st.write("Aplikasi untuk mencari kecocokan kata meskipun ada typo.")

# Dataset (Bisa Anda tambah atau ganti)
data_pilihan = [
    "Apple iPhone 15", "Samsung Galaxy S23", "Google Pixel 8",
    "MacBook Pro M3", "Dell XPS 13", "Sony WH-1000XM5",
    "Asus ROG Zephyrus", "Logitech MX Master 3"
]

# Sidebar untuk melihat database
with st.sidebar:
    st.subheader("Database Produk")
    st.write(data_pilihan)

# Input Pencarian
query = st.text_input("Masukkan nama produk (coba tulis dengan typo):", "")

if query:
    # Menggunakan fuzzywuzzy untuk mencari kecocokan
    hasil_terbaik, skor = process.extractOne(query, data_pilihan, scorer=fuzz.token_sort_ratio)

    st.divider()
    
    if skor > 50:
        st.success(f"Hasil Terbaik: **{hasil_terbaik}**")
        st.metric(label="Skor Kemiripan", value=f"{skor}%")
        
        # Progres bar visual
        st.progress(skor / 100)
    else:
        st.warning(f"Tidak ada kecocokan yang kuat. (Skor tertinggi: {skor}%)")
        st.info("Coba masukkan kata kunci yang lebih spesifik.")

else:
    st.info("Silakan ketik sesuatu di kolom pencarian di atas.")
