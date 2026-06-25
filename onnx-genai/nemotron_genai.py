#!/usr/bin/env python3
"""Streaming CPU transcription via Microsoft's onnxruntime-genai runtime.

This is the *fast path* alternative to ``nemotron_stream.py``. Same model
(NVIDIA Nemotron-3.5-ASR, cache-aware FastConformer + RNN-T), but the whole hot
loop — mel front end, encoder, RNN-T greedy decode, and cache management — runs
in onnxruntime-genai's native C++ engine instead of our Python/NumPy loop. The
encoder graph also ships with fused multi-head-attention kernels and MatMulNBits
int4/int8 weights, neither of which our plain ``torch.onnx.export`` bundle has.
Microsoft reports RTF > 6x on CPU for every quantized variant; expect a large
speedup over the hand-written path. See the deep-dive notes in the PR/commit.

    python nemotron_genai.py clip.wav                       # stock multilingual int4
    python nemotron_genai.py clip.wav --language ady        # ady rides the uk-UA slot
    python nemotron_genai.py clip.wav --model ./my_genai_bundle

IMPORTANT — model format. onnxruntime-genai needs the model in *its* layout
(encoder.onnx / decoder.onnx / joint.onnx + genai_config.json + tokenizer), NOT
the two-graph bundle ``export_quantize.py`` produces. ``--model`` therefore
accepts either a local genai bundle directory or a Hugging Face repo id (it is
downloaded on first use). The default is the official stock checkpoint, which is
ideal for an apples-to-apples *speed* comparison against ``nemotron_stream.py``
(runtime speed is architecture-bound, not weight-bound). To transcribe a
fine-tuned Circassian checkpoint here, first re-export it with ``export_genai.py``
(``python export_genai.py FT.nemo out_genai``) — the stock weights do not know ady/kbd.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

# Friendly language keys -> Nemotron prompt slot id (subset; pass --language <int>
# for anything else). ady/kbd ride uk-UA/bg-BG, mirroring nemotron_stream.py.
LANG_ID = {"auto": 101, "en": 0, "en-US": 0, "en-GB": 1, "ru": 11, "ru-RU": 11,
           "uk": 19, "uk-UA": 19, "bg": 30, "bg-BG": 30, "tr": 18, "tr-TR": 18,
           "ar": 7, "de": 9, "fr": 8, "es": 3}
LANG_ALIAS = {"ady": "uk-UA", "kbd": "bg-BG"}


def resolve_lang_id(language: str) -> int:
    """Map a friendly language key (or a raw integer string) to a prompt slot id."""
    key = LANG_ALIAS.get(language, language)
    if key in LANG_ID:
        return LANG_ID[key]
    if key.lstrip("-").isdigit():
        return int(key)
    raise ValueError(f"unknown language {language!r}; use one of {sorted(LANG_ID) + sorted(LANG_ALIAS)} "
                     f"or a raw integer prompt id")


def resolve_model(model: str) -> str:
    """Return a local genai bundle dir; download from the HF hub if given a repo id."""
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download  # only needed for the download path
    return snapshot_download(model)


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """16 kHz mono float32. Uses soundfile if present (resample/mixdown), else stdlib wave."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * sample_rate / sr))
        return audio.astype(np.float32)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"expected mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == sample_rate, f"expected {sample_rate} Hz, got {wf.getframerate()}"
        assert wf.getsampwidth() == 2, f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit"
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(model_dir: str, audio_path: str, ep: str = "cpu", language: str | None = None,
               use_vad: bool = False, strip_lang_tags: bool = True) -> str:
    """Stream a clip through onnxruntime-genai, printing deltas live; return the transcript."""
    genai_config = Path(model_dir) / "genai_config.json"
    if not genai_config.exists():
        raise SystemExit(f"{genai_config} not found. onnxruntime-genai needs its own bundle format "
                         f"(encoder/decoder/joint.onnx + genai_config.json), not the export_quantize.py "
                         f"bundle. Pass an HF repo id or a genai-format directory.")
    import onnxruntime_genai as og  # imported here so --help works without the package

    cfg = json.loads(genai_config.read_text())["model"]
    sample_rate, chunk_samples = cfg["sample_rate"], cfg["chunk_samples"]
    audio = load_audio(audio_path, sample_rate)

    config = og.Config(model_dir)
    if ep != "follow_config":  # else honour the EP baked into genai_config.json
        config.clear_providers()
        if ep != "cpu":
            config.append_provider(ep)

    model = og.Model(config)
    processor = og.StreamingProcessor(model)
    processor.set_option("use_vad", "true" if use_vad else "false")
    tokenizer = og.Tokenizer(model)
    tok_stream = tokenizer.create_stream()
    generator = og.Generator(model, og.GeneratorParams(model))
    if language is not None:
        generator.set_runtime_option("lang_id", str(resolve_lang_id(language)))

    def drain() -> str:
        out = ""
        while not generator.is_done():
            generator.generate_next_token()
            toks = generator.get_next_tokens()
            if len(toks):
                piece = tok_stream.decode(toks[0])
                if piece:
                    print(piece, end="", flush=True)
                    out += piece
        return out

    t0 = time.perf_counter()
    text = ""
    for i in range(0, len(audio), chunk_samples):
        inputs = processor.process(audio[i:i + chunk_samples].astype(np.float32))
        if inputs is not None:
            generator.set_inputs(inputs)
            text += drain()
    inputs = processor.flush()  # emit whatever the trailing partial chunk produced
    if inputs is not None:
        generator.set_inputs(inputs)
        text += drain()
    wall = time.perf_counter() - t0
    print()

    if strip_lang_tags:
        text = LANG_TAG_RE.sub("", text).strip()
    secs = len(audio) / sample_rate
    rtf = secs / wall if wall else float("nan")
    print(f"[genai | {ep} | lang={language or 'config-default'}] "
          f"audio={secs:.1f}s wall={wall:.1f}s RTF={rtf:.2f}x real-time (incl. model load)")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-transcribe a WAV via onnxruntime-genai (fast CPU path)")
    ap.add_argument("audio", help="16 kHz mono WAV file")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="genai bundle dir or HF repo id (default: stock int4 multilingual)")
    ap.add_argument("--language", "-l", default=None,
                    help="prompt key (auto/en-US/ady/kbd/...) or raw integer id; default = config")
    ap.add_argument("-e", "--execution-provider", default="cpu",
                    choices=["cpu", "cuda", "dml", "follow_config"])
    ap.add_argument("--use-vad", action="store_true", help="enable Silero VAD chunk skipping")
    ap.add_argument("--keep-lang-tags", action="store_true", help="do not strip trailing <xx-XX> tags")
    args = ap.parse_args()

    transcribe(resolve_model(args.model), args.audio, ep=args.execution_provider,
               language=args.language, use_vad=args.use_vad,
               strip_lang_tags=not args.keep_lang_tags)


if __name__ == "__main__":
    main()
