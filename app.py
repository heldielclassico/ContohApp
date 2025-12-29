import streamlit as st
import assemblyai as aai
import time

# Konfigurasi Halaman
st.set_page_config(page_title="YouTube Summarizer AI", page_icon="🎥", layout="centered")

# --- UI Header ---
st.title("🎥 YouTube Video Summarizer")
st.markdown("Ringkas isi video YouTube dalam hitungan detik menggunakan AI.")
st.divider()

# --- Sidebar untuk Konfigurasi ---
with st.sidebar:
    st.header("Konfigurasi")
    api_key = st.text_input("Masukkan AssemblyAI API Key", type="password")
    st.info("Dapatkan API Key gratis di [assemblyai.com](https://www.assemblyai.com)")

# --- Input Utama ---
video_url = st.text_input("Tempel link YouTube di sini:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Mulai Ringkas ✨"):
    if not api_key:
        st.error("Silakan masukkan API Key terlebih dahulu di sidebar!")
    elif not video_url:
        st.warning("Silakan masukkan URL video YouTube!")
    else:
        try:
            with st.status("Sedang memproses video...", expanded=True) as status:
                st.write("Mengambil audio dari YouTube...")
                aai.settings.api_key = api_key
                transcriber = aai.Transcriber()

                st.write("Mentranskripsi suara ke teks (AI sedang mendengarkan)...")
                transcript = transcriber.transcribe(video_url)

                if transcript.status == aai.TranscriptStatus.error:
                    st.error(f"Gagal: {transcript.error}")
                else:
                    st.write("Membuat ringkasan poin-poin penting...")
                    # LeMUR Summarization
                    prompt = "Berikan ringkasan poin-poin utama dari video ini dalam Bahasa Indonesia yang mudah dipahami."
                    summary_result = transcript.lemur.summarize(context=prompt)
                    
                    status.update(label="Selesai!", state="complete", expanded=False)

            # --- Menampilkan Hasil ---
            st.success("✅ Berhasil Diringkas!")
            
            tab1, tab2 = st.tabs(["📋 Ringkasan AI", "📝 Transkrip Lengkap"])
            
            with tab1:
                st.subheader("Poin-Poin Utama")
                st.markdown(summary_result.response)
            
            with tab2:
                st.subheader("Teks Asli Video")
                st.write(transcript.text)
                st.download_button("Download Transkrip", transcript.text, file_name="transkrip.txt")

        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Dibuat dengan Python, Streamlit, dan AssemblyAI")
