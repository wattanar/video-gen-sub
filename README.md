# video-gen-sub

Generate `.srt` subtitles from video files. Local transcription with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API key,
audio extraction handled internally via PyAV (no system ffmpeg dependency).

Non-English speech is automatically translated to English in the same pass
(Whisper `translate` task).

## Setup

```bash
make venv        # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Model weights download on first run (~74 MB `base`, ~460 MB `small`,
~1.5 GB `medium`). Set `HF_TOKEN` to avoid rate limits.

## CLI

```bash
make run VIDEO=video.mp4
# equivalent:
.venv/bin/python app.py video.mp4 [-o out.srt] [-m tiny|base|small|medium|large-v3]
                                  [-l th] [--device cuda] [--no-translate]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-o` | `<video>.srt` | Output path |
| `-m` | `base` | Model size (bigger = more accurate) |
| `-l` | auto-detect | Source language code, e.g. `ja`, `th`, `en` |
| `--device` | `auto` | `cpu` / `cuda` / `auto` |
| `--no-translate` | off | Keep original language instead of English output |

## Web UI

```bash
make serve       # .venv/bin/streamlit run ui.py → http://localhost:8501
```

Upload a video, pick model / language / translate, download the `.srt`.

## Output format

Standard SRT — sequential index, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps,
UTF-8 text.

## Notes

- `base` is too weak for many languages (e.g. Japanese); use `small` or
  `medium` for non-English content.
- On CPU, `medium` processes ~60 s of audio in a few minutes; use `cuda`
  for large batches.
- VAD filtering is enabled, so silent gaps produce no segments.

## Layout

- `transcriber.py` — transcription pipeline + SRT writer
- `app.py` — CLI entry point
- `ui.py` — Streamlit web UI
- `check.py` — SRT writer sanity checks (`make check`)
