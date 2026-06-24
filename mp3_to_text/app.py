import streamlit as st
import openai
import tempfile
import os
from pathlib import Path

# Helper
def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


# Page config
st.set_page_config(
    page_title="MP3 to Text Transcriber",
    page_icon=":microphone:",
    layout="centered",
)

st.title(":microphone: MP3 to Text Transcriber")
st.caption("Upload an audio file and get a clean text transcript using OpenAI Whisper.")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required. Get yours at platform.openai.com",
    )
    language = st.selectbox(
        "Audio Language (optional)",
        options=["Auto-detect", "English", "French", "Spanish", "German",
                 "Portuguese", "Italian", "Japanese", "Chinese", "Arabic"],
    )
    output_format = st.radio(
        "Output Format",
        options=["Plain Text", "Timestamped (SRT-style)"],
    )
    st.divider()
    st.caption("Built by Bita Ashoori | Powered by OpenAI Whisper")

LANG_MAP = {
    "Auto-detect": None,
    "English": "en", "French": "fr", "Spanish": "es",
    "German": "de", "Portuguese": "pt", "Italian": "it",
    "Japanese": "ja", "Chinese": "zh", "Arabic": "ar",
}

# File upload
uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "mp4", "wav", "m4a", "ogg", "webm"],
    help="Supports MP3, WAV, M4A, MP4, OGG, WEBM -- max 25 MB (OpenAI limit)",
)

if uploaded_file:
    st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix.lstrip('.')}")
    st.caption(f"File: {uploaded_file.name} | {uploaded_file.size / 1024:.1f} KB")

    if uploaded_file.size > 25 * 1024 * 1024:
        st.error("File exceeds 25 MB. Please compress or trim the audio before uploading.")
        st.stop()

# Transcribe
if st.button("Transcribe", type="primary", use_container_width=True, disabled=not uploaded_file):

    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    with st.spinner("Transcribing... this may take a moment for longer files."):
        try:
            client = openai.OpenAI(api_key=api_key)

            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            lang_code = LANG_MAP[language]

            with open(tmp_path, "rb") as audio_file:
                if output_format == "Timestamped (SRT-style)":
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        **({"language": lang_code} if lang_code else {}),
                    )
                    lines = []
                    for i, seg in enumerate(response.segments, 1):
                        start = _fmt_time(seg["start"])
                        end = _fmt_time(seg["end"])
                        lines.append(f"[{start} -> {end}]\n{seg['text'].strip()}")
                    transcript = "\n\n".join(lines)
                else:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        **({"language": lang_code} if lang_code else {}),
                    )
                    transcript = response if isinstance(response, str) else response.text

            os.unlink(tmp_path)

        except openai.AuthenticationError:
            st.error("Invalid API key. Please check your OpenAI key.")
            st.stop()
        except openai.RateLimitError:
            st.error("Rate limit hit. Wait a moment and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    st.success("Transcription complete!")
    st.subheader("Transcript")
    st.text_area("", value=transcript, height=300)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as .txt",
            data=transcript,
            file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="Download as .md",
            data=f"# Transcript: {uploaded_file.name}\n\n{transcript}",
            file_name=f"{Path(uploaded_file.name).stem}_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    word_count = len(transcript.split())
    char_count = len(transcript)
    st.caption(f"{word_count:,} words | {char_count:,} characters")


# Empty state
if not uploaded_file:
    st.info("Upload an audio file above to get started.")
    with st.expander("How it works"):
        st.markdown("""
1. Enter your OpenAI API key in the sidebar
2. Upload an MP3, WAV, M4A, or other audio file (max 25 MB)
3. Choose your language and output format
4. Hit Transcribe -- results appear in seconds
5. Download the transcript as .txt or .md

Uses OpenAI Whisper -- one of the most accurate speech recognition models available.
Cost: ~$0.006 per minute of audio.
        """)
