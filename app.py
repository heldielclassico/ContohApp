import streamlit as st
import assemblyai as aai
from pytube import YouTube
import os
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YouTube AI Summarizer", page_icon="🎥", layout="centered")

def clean_url(url):
    """Membersihkan spasi dan karakter tambahan pada URL"""
    return url.strip()

def validate_youtube_url(url):
    """Validasi format URL YouTube yang beragam"""
    pattern = r'^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|live/|shorts/)?([a-zA-Z0-9_-]{11})(\S+)?$'
    return re.match(pattern, url)

def download_audio(url):
    """Mengunduh audio menggunakan pytube"""
    try:
        yt = YouTube(url)
        # Ambil stream audio saja
        audio_stream = yt.streams.filter(only_audio=True).first()
        # Download dengan nama file sementara
        out_file = audio_stream.download(filename="temp_video_audio.mp4")
        # Ubah nama menjadi .m4a agar standar
        final_file = "audio_input.m4a"
        if os.path.exists(final_file):
            os.remove(final_file)
        os.rename(out_file, final_file)
        return final_file, yt.title
    except Exception as e:
        raise Exception(f"Gagal mengunduh audio: {str(e)}")

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🎥 YouTube AI Summarizer")
st.markdown("Ubah video YouTube menjadi ringkasan teks dalam Bahasa Indonesia.")

# Sidebar untuk API Key
with st.sidebar:
    st.header("Pengaturan")
    api_key = st.text_input("AssemblyAI API Key", type="password", help="Dapatkan di assemblyai.com")
    st.divider()
    st.info("Aplikasi ini menggunakan Pytube untuk ekstraksi audio dan AssemblyAI untuk kecerdasan buatan.")

# Input Utama
video_url_input = st.text_input("Tempel Link YouTube Anda:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Mulai Ringkas ✨"):
    video_url = clean_url(video_url_input)
    
    if not api_key:
        st.error("Silakan masukkan API Key di sidebar!")
    elif not video_url:
        st.warning("Silakan masukkan URL video!")
    elif not validate_youtube_url(video_url):
        st.error("Format URL tidak valid. Pastikan Anda menyalin link dengan benar.")
    else:
        audio_path = None
        try:
            with st.status("Sedang memproses...", expanded=True) as status:
                # 1. Download Audio
                st.write("📥 Mengunduh data dari YouTube...")
                audio_path, video_title = download_audio(video_url)
                st.write(f"**Judul Video:** {video_title}")
                
                # 2. Transkripsi
                st.write("🎙️ AI sedang mendengarkan video...")
                aai.settings.api_key = api_key
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(audio_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    st.error(f"Gagal Transkripsi: {transcript.error}")
                else:
                    # 3. Ringkasan LeMUR
                    st.write("📝 Meringkas poin-poin penting...")
                    prompt = "Buatlah ringkasan informatif dalam Bahasa Indonesia yang mencakup poin-poin utama dari video ini."
                    summary_result = transcript.lemur.summarize(context=prompt)
                    
                    status.update(label="Selesai!", state="complete")
                    
                    # TAMPILKAN HASIL
                    st.success("Berhasil diringkas!")
                    
                    tab1, tab2 = st.tabs(["📋 Ringkasan", "📝 Transkrip"])
                    with tab1:
                        st.subheader("Ringkasan Utama")
                        st.write(summary_result.response)
                    with tab2:
                        st.subheader("Teks Lengkap")
                        st.write(transcript.text)
                        st.download_button("Simpan Transkrip (.txt)", transcript.text, file_name="transkrip.txt")

        except Exception as e:
            st.error(f"Kesalahan Sistem: {str(e)}")
            st.info("Tips: Pastikan video bersifat publik dan tidak dibatasi umur.")
        
        finally:
            # Hapus file sementara agar server tidak penuh
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

# Footer
st.divider()
st.caption("Dibuat dengan Python & Streamlit")
