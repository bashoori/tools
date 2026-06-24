import streamlit as st
import openai
import tempfile
import os
import math
import subprocess
from pathlib import Path

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def split_audio(file_path: str, max_mb: int = 24) -> list:
    file_size = os.path.getsize(file_path)
    max_bytes = max_mb * 1024 * 1024
    if file_size <= max_bytes:
        return [file_path]
    suffix = Path(file_path).suffix
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", file_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    n_chunks = math.ceil(file_size / max_bytes) + 1
    chunk_duration = duration / n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_duration
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ss", str(start), "-t", str(chunk_duration), "-c", "copy", tmp.name],
            capture_output=True,
        )
        chunks.append(tmp.name)
    return chunks

st.set_page_config(page_title="Audio to Text Transcriber", page_icon=":microphone:", layout="centered")
st.title(":microphone: Audio to Text Transcriber")
st.caption("Upload any audio file — large files are auto-split and transcribed using OpenAI Whisper.")
st.divider()

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required. Get yours at platform.openai.com")
    language = st.selectbox("Audio Language (optional)", options=["Auto-detect", "English", "French", "Spanish", "German", "Portuguese", "Italian", "Japanese", "Chinese", "Arabic"])
    output_format = st.radio("Output Format", options=["Plain Text", "Timestamped (SRT-style)"])
    st.divider()
    st.caption("Built by Bita Ashoori | Powered by OpenAI Whisper")

LANG_MAP = {"Auto-detect": None, "English": "en", "French": "fr", "Spanish": "es", "German": "de", "Portuguese": "pt", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Arabic": "ar"}

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "mp4", "wav", "m4a", "ogg", "webm"], help="Any size supported — large files are automatically split into chunks.")

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"File: {uploaded_file.name} | {file_size_mb:.1f} MB")
    if file_size_mb > 25:
        st.warning(f"File is {file_size_mb:.1f} MB (over OpenAI's 25 MB limit). It will automatically be split into chunks for transcription.")

if st.button("Transcribe", type="primary", use_container_width=True, disabled=not uploaded_file):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    lang_code = LANG_MAP[language]
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    needs_split = file_size_mb > 24
    try:
        client = openai.OpenAI(api_key=api_key)
        if needs_split:
            n_chunks = math.ceil(file_size_mb / 24) + 1
            status = st.status(f"Large file ({file_size_mb:.1f} MB) — splitting into ~{n_chunks} chunks...", expanded=True)
            with status:
                chunks = split_audio(tmp_path, max_mb=24)
                st.write(f"Split into {len(chunks)} chunks. Transcribing...")
        else:
            chunks = [tmp_path]
            status = st.status("Transcribing...", expanded=False)
        all_transcripts = []
        with status:
            for i, chunk_path in enumerate(chunks):
                if needs_split:
                    st.write(f"Transcribing chunk {i + 1} of {len(chunks)}...")
                with open(chunk_path, "rb") as audio_file:
                    if output_format == "Timestamped (SRT-style)":
                        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="verbose_json", timestamp_granularities=["segment"], **({"language": lang_code} if lang_code else {}))
                        lines = []
                        for seg in response.segments:
                            start = _fmt_time(seg["start"])
                            end = _fmt_time(seg["end"])
                            lines.append(f"[{start} -> {end}]\n{seg['text'].strip()}")
                        all_transcripts.append("\n\n".join(lines))
                    else:
                        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text", **({"language": lang_code} if lang_code else {}))
                        text = response if isinstance(response, str) else response.text
                        all_transcripts.append(text)
                if chunk_path != tmp_path:
                    os.unlink(chunk_path)
            status.update(label="Transcription complete!", state="complete")
        os.unlink(tmp_path)
        separator = "\n\n--- (continued) ---\n\n" if needs_split else ""
        transcript = separator.join(all_transcripts)
    except openai.AuthenticationError:
        st.error("Invalid API key. Please check your OpenAI key in the sidebar.")
        st.stop()
    except openai.RateLimitError:
        st.error("Rate limit hit. Wait a moment and try again.")
        st.stop()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()
    st.success("Done!")
    st.subheader("Transcript")
    st.text_area("", value=transcript, height=350)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(label="Download as .txt", data=transcript, file_name=f"{Path(uploaded_file.name).stem}_transcript.txt", mime="text/plain", use_container_width=True)
    with col2:
        st.download_button(label="Download as .md", data=f"# Transcript: {uploaded_file.name}\n\n{transcript}", file_name=f"{Path(uploaded_file.name).stem}_transcript.md", mime="text/markdown", use_container_width=True)
    st.divider()
    st.caption(f"{len(transcript.split()):,} words | {len(transcript):,} characters")

if not uploaded_file:
    st.info("Upload an audio file above to get started.")
    with st.expander("How it works"):
        st.markdown("""
1. Enter your OpenAI API key in the sidebar
2. Upload an audio file (MP3, MP4, WAV, M4A, OGG, WEBM) — **any size**
3. Files over 25 MB are automatically split into chunks and transcribed in parts
4. Choose your language and output format
5. Hit Transcribe and download the result

Uses OpenAI Whisper. Cost: ~$0.006 per minute of audio.
        """)
