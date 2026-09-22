"""Video -> SRT transcriber built on faster-whisper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class Segment:
    start: float
    end: float
    text: str


_model_cache: dict[tuple[str, str, str], object] = {}


def get_model(model_size: str = "base", device: str = "auto",
              compute_type: str = "int8"):
    key = (model_size, device, compute_type)
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        _model_cache[key] = WhisperModel(model_size, device=device,
                                         compute_type=compute_type)
    return _model_cache[key]


def transcribe(video_path: str, model_size: str = "base",
               language: Optional[str] = None,
               device: str = "auto",
               compute_type: str = "int8",
               translate_to_english: bool = True,
               ) -> Iterator[Segment]:
    """Transcribe a video file, yielding (start, end, text) segments.

    faster-whisper extracts audio internally via PyAV (ffmpeg bundled),
    so no separate extraction step is needed.

    If the source language is not English and translate_to_english is on,
    Whisper's built-in translate task emits English directly.
    """
    model = get_model(model_size, device, compute_type)
    if language is None:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(video_path, sampling_rate=16000)
        language, _prob, _all = model.detect_language(audio)
    task = "translate" if (translate_to_english and language != "en") \
        else "transcribe"
    segments, _info = model.transcribe(
        video_path,
        language=language,
        task=task,
        vad_filter=True,
    )
    for seg in segments:
        text = seg.text.strip()
        if text:
            yield Segment(seg.start, seg.end, text)


def format_timestamp(seconds: float) -> str:
    """Seconds -> SRT timestamp `HH:MM:SS,mmm`."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def video_to_srt(video_path: str, output_path: Optional[str] = None,
                 model_size: str = "base", language: Optional[str] = None,
                 device: str = "auto",
                 compute_type: str = "int8",
                 translate_to_english: bool = True,
                 ) -> str:
    """Full pipeline. Returns the SRT text; writes to output_path if given."""
    segments = list(transcribe(video_path, model_size, language, device,
                               compute_type, translate_to_english))
    srt = segments_to_srt(segments)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt)
    return srt
