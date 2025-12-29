import streamlit as st
import assemblyai as aai
import yt_dlp
import os

# Konfigurasi Halaman
st.set_page_config(page_title="YT Summarizer", page_icon="🎥")

def download_audio(url):
    """Fungsi untuk mengunduh audio dari YouTube"""
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'audio_temp.%(ext)s',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "audio_temp.m4a"

st.title("🎥 YouTube AI Summarizer")
st.markdown("Ringkas video YouTube apapun dengan teknologi AssemblyAI.")

# Input API Key & URL
api_key = st.sidebar.text_input("AssemblyAI API Key", type="password")
video_url = st.text_input("Masukkan URL Video YouTube")

if st.button("Proses Video"):
    if not api_key or not video_url:
        st.warning("Mohon isi API Key dan URL video.")
    else:
        audio_path = None
        try:
            with st.status("Sedang bekerja...", expanded=True) as status:
                # 1. Download
                st.write("Mengunduh audio...")
                audio_path = download_audio(video_url)
                
                # 2. Transcribe
                st.write("Menganalisis suara...")
                aai.settings.api_key = api_key
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(audio_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    st.error(f"Kesalahan Transkripsi: {transcript.error}")
                else:
                    # 3. Summarize
                    st.write("Membuat ringkasan...")
                    summary = transcript.lemur.summarize(
                        context="Buat ringkasan dalam Bahasa Indonesia yang informatif."
                    )
                    
                    status.update(label="Proses Selesai!", state="complete")
                    
                    # Tampilkan Hasil
                    st.subheader("📋 Ringkasan")
                    st.write(summary.response)
                    
                    with st.expander("Lihat Transkrip Lengkap"):
                        st.write(transcript.text)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
        
        finally:
            # Menghapus file audio agar tidak memenuhi penyimpanan
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                st.info("File sementara telah dibersihkan.")
