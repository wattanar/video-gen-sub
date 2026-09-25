"""Video -> SRT transcriber built on whisper.cpp (CPU / AMD Vulkan).

Runs `whisper-cli` as a subprocess: PyAV decodes the video to a 16 kHz
mono WAV, whisper.cpp runs silero VAD + (translate) transcription, and the
JSON output is parsed into segments.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterator, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_RATE = 16000

_GGML_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
VAD_MODEL_NAME = "ggml-silero-v6.2.0.bin"
_VAD_MODEL_URL = f"https://huggingface.co/ggml-org/whisper-vad/resolve/main/{VAD_MODEL_NAME}"

_SEG_LINE = re.compile(r"^\[(\d+):(\d+):(\d+)\.(\d+) --> ")

_vulkan_ok: Optional[bool] = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _whisper_cli() -> str:
    cli = os.environ.get("WHISPER_CLI")
    if cli:
        return cli
    local = PROJECT_ROOT / ".whisper-cpp" / "build" / "bin" / "whisper-cli"
    if local.is_file():
        return str(local)
    found = shutil.which("whisper-cli")
    if found:
        return found
    sys.exit("error: whisper-cli not found — run `make whisper` first")


def _download(url: str, dest: pathlib.Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)


def _ensure_model(name: str) -> pathlib.Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"ggml-{name}.bin"
    if not path.exists():
        _download(_GGML_MODEL_URL.format(name=name), path)
    return path


def _ensure_vad_model() -> pathlib.Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / VAD_MODEL_NAME
    if not path.exists():
        _download(_VAD_MODEL_URL, path)
    return path


def _vulkan_available() -> bool:
    global _vulkan_ok
    if _vulkan_ok is None:
        info = shutil.which("vulkaninfo")
        _vulkan_ok = bool(info) and subprocess.run(
            [info, "--summary"], capture_output=True).returncode == 0
    return _vulkan_ok


def _resolve_device(device: str) -> Optional[int]:
    """whisper-cli gpu_device index: 0 = first GPU; None = CPU (pass --no-gpu)."""
    if device == "cpu":
        return None
    if device == "vulkan":
        if not _vulkan_available():
            sys.exit("error: --device vulkan requested but no Vulkan device found")
        return 0
    if device == "auto":
        return 0 if _vulkan_available() else None
    raise ValueError(f"unknown device: {device}")


def _decode_to_wav(video_path: "str | BinaryIO", wav_path: pathlib.Path) -> float:
    """Decode video audio to 16 kHz mono s16le WAV. Returns duration in seconds."""
    import av
    duration = 0.0
    with av.open(video_path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    wf.writeframes(resampled.to_ndarray().tobytes())
                    duration += resampled.samples / SAMPLE_RATE
    return duration


def transcribe(video_path: "str | BinaryIO", model_size: str = "base",
               language: Optional[str] = None,
               device: str = "auto",
               translate_to_english: bool = True,
               progress_callback: Optional[Callable[[float], None]] = None,
               ) -> Iterator[Segment]:
    """Transcribe a video file or stream, yielding (start, end, text) segments.

    device: 'auto' (Vulkan GPU if available, else CPU), 'cpu', or 'vulkan'.

    progress_callback, if given, receives a fraction 0.0-1.0 of audio
    covered by the last completed segment.
    """
    cli = _whisper_cli()
    model = _ensure_model(model_size)
    vad = _ensure_vad_model()
    device_id = _resolve_device(device)
    with tempfile.TemporaryDirectory() as td:
        wav = pathlib.Path(td) / "audio.wav"
        duration = _decode_to_wav(video_path, wav)
        cmd = [cli, "-m", str(model), "-f", str(wav),
               "-l", language or "auto",
               "-t", str(min(os.cpu_count() or 1, 8))]
        if device_id is None:
            cmd.append("--no-gpu")
        else:
            cmd += ["--device", str(device_id)]
        cmd += ["--vad", "-vm", str(vad),
               "-oj", "-of", str(pathlib.Path(td) / "out")]
        if translate_to_english and (language is None or language.lower() != "en"):
            cmd.append("--translate")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:  # per-segment prints give progress
            m = _SEG_LINE.match(line)
            if m and progress_callback and duration > 0:
                h, mm, s, frac = (int(x) for x in m.groups())
                pos = h * 3600 + mm * 60 + s + frac / 1000
                progress_callback(min(1.0, pos / duration))
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed (exit {proc.returncode})")
        result = json.loads((pathlib.Path(td) / "out.json").read_text(encoding="utf-8"))
    for seg in result.get("transcription", []):
        text = seg["text"].strip()
        if text:
            yield Segment(seg["offsets"]["from"] / 1000,
                          seg["offsets"]["to"] / 1000, text)


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
                 translate_to_english: bool = True,
                 progress_callback: Optional[Callable[[float], None]] = None,
                 ) -> str:
    """Full pipeline. Returns the SRT text; writes to output_path if given."""
    segments = list(transcribe(video_path, model_size, language, device,
                               translate_to_english, progress_callback))
    srt = segments_to_srt(segments)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt)
    return srt
