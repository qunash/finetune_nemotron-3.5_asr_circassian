#!/usr/bin/env python3
"""Transcribe a WAV with a GGUF Nemotron-3.5-ASR bundle via parakeet.cpp.

Thin wrapper over ``parakeet-cli``: it selects the language prompt, uses every CPU
thread by default, and reports RTF. Two modes:

  offline  one full-context pass over the clip — fastest for files (use this)
  stream   cache-aware chunked path — low latency, incremental output

Build the bundle + binary first with ``gguf/convert.sh``.

    gguf/transcribe.py --model gguf_out/nemotron.q8_0.gguf --input clip.wav --lang ady
    gguf/transcribe.py --model gguf_out/nemotron.q8_0.gguf --input clip.wav --lang kbd --mode stream
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import wave

# Where gguf/convert.sh drops the prebuilt binary, plus a native build fallback.
_DEFAULT_CLIS = (
    "third_party/parakeet.cpp/bin/parakeet-cli",
    "third_party/parakeet.cpp/src/build/examples/cli/parakeet-cli",
)


def audio_seconds(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def find_cli(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [shutil.which("parakeet-cli"), *_DEFAULT_CLIS]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # The prebuilt tarball may nest the binary one level down; do a shallow scan.
    root = "third_party/parakeet.cpp/bin"
    for dirpath, _, files in os.walk(root) if os.path.isdir(root) else []:
        if "parakeet-cli" in files:
            return os.path.join(dirpath, "parakeet-cli")
    sys.exit("parakeet-cli not found; run gguf/convert.sh first or pass --cli PATH")


def main() -> None:
    ap = argparse.ArgumentParser(description="GGUF Nemotron transcription via parakeet.cpp")
    ap.add_argument("--model", required=True, help="GGUF bundle, e.g. gguf_out/nemotron.q8_0.gguf")
    ap.add_argument("--input", required=True, help="16 kHz mono WAV")
    ap.add_argument("--lang", default="", help="prompt key (e.g. ady, kbd); empty = model default")
    ap.add_argument("--mode", choices=["offline", "stream"], default="offline")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timestamps", action="store_true", help="per-word start/end/confidence")
    ap.add_argument("--cli", default=None, help="path to parakeet-cli (auto-detected otherwise)")
    args = ap.parse_args()

    cli = find_cli(args.cli)
    cmd = [cli, "transcribe", "--model", args.model, "--input", args.input,
           "--threads", str(args.threads)]
    if args.lang:
        cmd += ["--lang", args.lang]
    if args.mode == "stream":
        cmd.append("--stream")
    if args.timestamps:
        cmd.append("--timestamps")

    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    wall = time.perf_counter() - t0
    if rc != 0:
        sys.exit(rc)

    secs = audio_seconds(args.input)
    rtf = wall / secs if secs else float("nan")
    print(f"\n[{args.mode} | lang={args.lang or 'default'} | {args.threads} threads] "
          f"audio={secs:.1f}s wall={wall:.1f}s RTF={rtf:.3f} "
          f"({1 / rtf:.1f}x real-time, includes model load)", file=sys.stderr)


if __name__ == "__main__":
    main()
