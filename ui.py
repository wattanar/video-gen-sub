"""Streamlit UI: upload video -> download .srt.

Run:  streamlit run ui.py
"""
import io
import pathlib

import streamlit as st

from transcriber import transcribe, segments_to_srt

MODELS = ["tiny", "base", "small", "medium", "large-v3"]

st.title("Video to SRT")

with st.sidebar:
    model_size = st.selectbox("Model", MODELS, index=1)
    language = st.text_input("Language code (blank = auto-detect)", "")
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    translate = st.checkbox("Translate to English", value=True)

uploaded = st.file_uploader("Upload video",
                           type=["mp4", "mkv", "mov", "webm", "avi", "m4v"])

if uploaded is not None:
    # Drop stale results when a different video is uploaded
    if st.session_state.get("_srt_key") != uploaded.file_id:
        st.session_state.clear()
        st.session_state["_srt_key"] = uploaded.file_id
    if st.button("Generate subtitles", type="primary"):
        buf = io.BytesIO(uploaded.getvalue())
        buf.name = uploaded.name
        with st.spinner("Transcribing ..."):
            segments = list(transcribe(buf, model_size=model_size,
                                       language=language or None,
                                       device=device,
                                       translate_to_english=translate))
        st.session_state.srt = segments_to_srt(segments)

if srt := st.session_state.get("srt"):
    st.write(st.code(srt, language=None))
    stem = pathlib.Path(uploaded.name).stem if uploaded else "subtitles"
    st.download_button("Download .srt", srt.encode("utf-8"),
                       file_name=f"{stem}.srt", mime="application/x-subrip")
