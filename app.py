import assemblyai as aai

aai.settings.api_key = "YOUR_API_KEY"
transcriber = aai.Transcriber()

# AssemblyAI bisa langsung menerima URL YouTube di beberapa versi SDK-nya
transcript = transcriber.transcribe("https://www.youtube.com/watch?v=example")
print(transcript.export_subtitles_vtt())

Buatkan requirments nya
