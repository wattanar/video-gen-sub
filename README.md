# video-gen-sub

Generate `.srt` subtitles from video files. Local transcription with
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) — no API key. Audio
extraction is handled by PyAV (no system ffmpeg dependency). Runs on CPU or
AMD GPU via the Vulkan backend (`radv`).

Non-English speech is automatically translated to English in the same pass
(Whisper `translate` task).

## Setup

```bash
make venv      # python3 -m venv .venv
make install   # pip install -r requirements.txt
make whisper   # build whisper.cpp with -DGGML_VULKAN=ON (~few minutes, one-time)
```

`make whisper` clones `whisper.cpp` into `.whisper-cpp/` and builds
`whisper-cli` with the Vulkan backend. Rebuilds are skipped if the binary
already exists.

## CLI

```bash
make run VIDEO=video.mp4
# equivalent:
.venv/bin/python app.py video.mp4 [-o out.srt] [-m tiny|base|small|medium|large-v3]
                                  [-l th] [--device auto|cpu|vulkan] [--no-translate]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-o` | `<video>.srt` | Output path |
| `-m` | `base` | Whisper model size (bigger = more accurate) |
| `-l` | auto-detect | Source language code, e.g. `ja`, `th`, `en` |
| `--device` | `auto` | `cpu` / `vulkan` / `auto` (Vulkan GPU if available, else CPU) |
| `--no-translate` | off | Keep original language instead of English output |

Models (ggml format) download on first use into `./models/`
(~74 MB `tiny`, ~148 MB `base`, ~466 MB `small`, ~1.5 GB `medium`) plus the
~0.9 MB silero VAD model.

## Web UI

```bash
make serve     # streamlit run ui.py → http://localhost:8501
```

Upload a video, pick model / language / device / translate, download the `.srt`.

## Output format

Standard SRT — sequential index, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps,
UTF-8 text.

## Notes

- `base` is too weak for many languages (e.g. Japanese); use `small` or
  `medium` for non-English content.
- Silero VAD is enabled, so silent gaps produce no segments.
- whisper-cli reloads the model on each run (~1 s); fine for one-shot
  subtitle generation.
- Progress: the CLI prints a percentage, the web UI shows a progress bar
  (updates per completed segment).

## Layout

- `transcriber.py` — PyAV decode + whisper-cli subprocess + SRT writer
- `app.py` — CLI entry point
- `ui.py` — Streamlit web UI
- `check.py` — SRT writer sanity checks (`make check`)
- `models/` — downloaded ggml models (gitignored)
- `.whisper-cpp/` — whisper.cpp build (gitignored)
