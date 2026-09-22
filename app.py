#!/usr/bin/env python3
"""CLI: generate .srt from a video file.

Usage:
    python app.py video.mp4 [-o subtitles.srt] [-m base] [-l th]
"""
import argparse
import pathlib
import sys

from transcriber import video_to_srt

MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate .srt from a video.")
    p.add_argument("video", help="Input video file (mp4, mkv, mov, webm, ...)")
    p.add_argument("-o", "--output", help="Output .srt path (default: <video>.srt)")
    p.add_argument("-m", "--model", default="base", choices=MODELS,
                   help="Whisper model size (default: base)")
    p.add_argument("-l", "--language", default=None,
                   help="Language code e.g. 'th', 'en' (default: auto-detect)")
    p.add_argument("--no-translate", action="store_true",
                   help="Keep original language (default: translate non-English to English)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Inference device (default: auto)")
    args = p.parse_args()

    if not pathlib.Path(args.video).is_file():
        sys.exit(f"error: file not found: {args.video}")

    out = args.output or str(pathlib.Path(args.video).with_suffix(".srt"))
    print(f"Transcribing {args.video} (model={args.model}, "
          f"lang={args.language or 'auto'}, "
          f"translate={'no' if args.no_translate else 'yes'}) ...")
    video_to_srt(args.video, out, model_size=args.model,
                 language=args.language, device=args.device,
                 translate_to_english=not args.no_translate)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
