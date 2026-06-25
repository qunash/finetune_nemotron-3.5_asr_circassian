#!/usr/bin/env python3
"""Streaming CPU transcription for an INT8 Nemotron-3.5-ASR ONNX bundle.

Pure ONNX Runtime + NumPy: no torch, no NeMo at inference time. Produce the bundle with
``export_quantize.py`` first. The pipeline mirrors NeMo's cache-aware streaming exactly:

    log-mel front end  ->  INT8 FastConformer encoder (cache-aware, prompt-conditioned)
                       ->  FP32 RNN-T decoder+joint (greedy)

Language (e.g. ``auto`` / ``ady`` / ``kbd``) selects the encoder prompt slot; latency
(``ultra``/``low``/``balanced``/``medium``/``high``) selects the matching encoder graph.

    python nemotron_stream.py BUNDLE_DIR clip.wav --language ady --latency balanced
"""
from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

# NeMo appends a "<xx-XX>" tag after the final punctuation (used by auto language ID).
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

_ORT_TO_NP = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}


def log_mel(audio: np.ndarray, fb: np.ndarray, n_fft: int, hop: int, win: int,
            preemph: float, guard: float) -> np.ndarray:
    """NeMo-compatible 128-bin log-mel: preemph -> centered Hann STFT -> power -> mel -> ln(x+guard).

    Returns band-major features [n_mels, n_frames]. Matches AudioToMelSpectrogramPreprocessor
    with normalize="NA" and dither=0 (verified against NeMo at export time).
    """
    audio = np.asarray(audio, dtype=np.float64)
    pre = np.empty_like(audio)
    pre[0] = audio[0]
    pre[1:] = audio[1:] - preemph * audio[:-1]

    pad = n_fft // 2
    pre = np.pad(pre, (pad, pad), mode="reflect")  # center=True, reflect padding

    n_frames = 1 + (len(pre) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = pre[idx]

    hann = np.zeros(n_fft, dtype=np.float64)
    off = (n_fft - win) // 2
    hann[off:off + win] = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win) / (win - 1)))
    frames = frames * hann[None, :]

    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = power @ fb.T.astype(np.float64)
    return np.log(mel + guard).T.astype(np.float32)


def load_wav(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load 16-bit mono PCM WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"expected mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == sample_rate, f"expected {sample_rate} Hz, got {wf.getframerate()}"
        assert wf.getsampwidth() == 2, f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit"
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _find(names: list[str], *needles: str, exclude: str | None = None) -> str:
    for n in names:
        if all(s in n for s in needles) and (exclude is None or exclude not in n):
            return n
    raise KeyError(f"none of {names} match {needles} (exclude={exclude})")


class NemotronStreamer:
    """Cache-aware streaming RNN-T transcriber over an INT8 ONNX bundle."""

    def __init__(self, bundle_dir: str | Path, language: str = "auto", latency: str | None = None,
                 num_threads: int = 0, max_symbols: int = 10, low_memory: bool = False,
                 strip_lang_tags: bool = True, lang_tag_pattern: str | None = None):
        self.dir = Path(bundle_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.latency = latency or self.cfg["default_latency"]
        if self.latency not in self.cfg["latencies"]:
            raise ValueError(f"latency {self.latency!r} not exported; have {list(self.cfg['latencies'])}")
        self.lat = self.cfg["latencies"][self.latency]
        self.mel = self.cfg["mel"]
        self.blank = int(self.cfg["blank_id"])
        self.max_symbols = max_symbols
        self.strip = strip_lang_tags
        # ady/kbd ride the uk-UA/bg-BG prompt slots, so the model emits "<uk-UA>"/"<bg-BG>" — the
        # default NeMo pattern matches those; override if your checkpoint emits something else.
        self.tag_re = re.compile(lang_tag_pattern) if lang_tag_pattern else LANG_TAG_RE

        self.fb = np.fromfile(self.dir / "filterbank.bin", dtype=np.float32).reshape(self.mel["n_mels"], -1)

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        if low_memory:  # trade a little speed for a smaller resident set
            so.enable_cpu_mem_arena = False
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(self.dir / self.lat["encoder_file"]), so, providers=prov)
        self.dec = ort.InferenceSession(str(self.dir / self.cfg["decoder_file"]), so, providers=prov)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.dir / "tokenizer.model"))

        # Resolve graph I/O by name so the loop is robust to export ordering.
        self._enc_out = [o.name for o in self.enc.get_outputs()]
        self._enc_dtype = {i.name: _ORT_TO_NP[i.type] for i in self.enc.get_inputs()}
        din = [i.name for i in self.dec.get_inputs()]
        self._d_enc = _find(din, "encoder")
        self._d_tok = _find(din, "target", exclude="length")
        self._d_tlen = _find(din, "length")
        self._d_s1 = _find(din, "state", "1")
        self._d_s2 = _find(din, "state", "2")
        self._d_state_shape = [d if isinstance(d, int) else 1 for d in
                               next(i for i in self.dec.get_inputs() if i.name == self._d_s1).shape]
        self._d_tok_dt = _ORT_TO_NP[next(i for i in self.dec.get_inputs() if i.name == self._d_tok).type]
        self._d_tlen_dt = _ORT_TO_NP[next(i for i in self.dec.get_inputs() if i.name == self._d_tlen).type]
        dout = [o.name for o in self.dec.get_outputs()]
        self._o_logits = next(i for i, n in enumerate(dout) if "state" not in n and "len" not in n)
        self._o_s1 = next(i for i, n in enumerate(dout) if "state" in n and "1" in n)
        self._o_s2 = next(i for i, n in enumerate(dout) if "state" in n and "2" in n)

        self.set_language(language)
        self.reset()

    def set_language(self, language: str) -> None:
        pd = self.cfg["prompt_dictionary"]
        if language not in pd:
            raise ValueError(f"language {language!r} not in prompt_dictionary "
                             f"(have e.g. {sorted(pd)[:12]} ...)")
        self.language = language
        self.prompt_index = np.array([int(pd[language])], dtype=np.int64)

    def reset(self) -> None:
        """Clear encoder caches, decoder state, and the rolling pre-encode mel window."""
        cs = self.cfg["cache"]
        self.caches = {n: np.zeros(cs[n], self._enc_dtype[n]) for n in cs}
        self.pre = np.zeros((self.mel["n_mels"], self.lat["pre_encode_cache"]), np.float32)
        self.s1 = np.zeros(self._d_state_shape, np.float32)
        self.s2 = np.zeros(self._d_state_shape, np.float32)
        self.prev = self.blank  # padding_idx -> zero embedding, NeMo's RNN-T start symbol

    def _encode(self, chunk_mel: np.ndarray):
        feed = {
            "processed_signal": chunk_mel[None].astype(np.float32),
            "processed_signal_length": np.array([chunk_mel.shape[1]], np.int64),
            "prompt_index": self.prompt_index,
            **self.caches,
        }
        out = self.enc.run(self._enc_out, feed)
        named = dict(zip(self._enc_out, out))
        for n in self.caches:
            self.caches[n] = named[n + "_next"]
        return named["encoded"], int(named["encoded_len"][0])

    def _decode_frame(self, frame: np.ndarray) -> list[int]:
        """RNN-T greedy: emit up to max_symbols non-blank tokens for one encoder frame."""
        ids: list[int] = []
        for _ in range(self.max_symbols):
            out = self.dec.run(None, {
                self._d_enc: frame.astype(np.float32),
                self._d_tok: np.array([[self.prev]], self._d_tok_dt),
                self._d_tlen: np.array([1], self._d_tlen_dt),
                self._d_s1: self.s1,
                self._d_s2: self.s2,
            })
            token = int(np.argmax(out[self._o_logits].reshape(-1)))
            if token == self.blank:
                break
            ids.append(token)
            self.prev = token
            self.s1, self.s2 = out[self._o_s1], out[self._o_s2]
        return ids

    def push(self, mel_new: np.ndarray) -> list[int]:
        """Feed exactly one chunk's worth of new mel frames; return emitted token ids."""
        chunk = np.concatenate([self.pre, mel_new], axis=1)
        encoded, n = self._encode(chunk)
        # Next pre-encode cache = last `pre` frames of [pre | new] (correct even when shift < pre).
        self.pre = chunk[:, -self.pre.shape[1]:]
        ids: list[int] = []
        for t in range(n):
            ids += self._decode_frame(encoded[:, :, t:t + 1])
        return ids

    def transcribe(self, audio) -> str:
        """Transcribe a full clip (path or 16 kHz mono float32) via the streaming loop."""
        if isinstance(audio, (str, Path)):
            audio = load_wav(audio)
        audio = np.asarray(audio, np.float32)
        shift = self.lat["mel_shift"]
        audio = np.pad(audio, (0, shift * self.mel["hop_length"]))  # flush the trailing partial chunk

        self.reset()
        mel = log_mel(audio, self.fb, self.mel["n_fft"], self.mel["hop_length"],
                      self.mel["win_length"], self.mel["preemph"], self.mel["log_guard"])
        ids: list[int] = []
        cursor, frames = 0, mel.shape[1]
        while cursor + shift <= frames:
            ids += self.push(mel[:, cursor:cursor + shift])
            cursor += shift

        text = self.sp.DecodeIds(ids) if ids else ""
        return self.tag_re.sub("", text).strip() if self.strip else text


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-transcribe a WAV with an INT8 Nemotron ONNX bundle")
    ap.add_argument("bundle_dir", help="directory produced by export_quantize.py")
    ap.add_argument("audio", help="16 kHz mono WAV file")
    ap.add_argument("--language", default="auto", help="prompt key, e.g. auto / ady / kbd / en-US")
    ap.add_argument("--latency", default=None, help="ultra|low|balanced|medium|high (default: bundle default)")
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    ap.add_argument("--keep-lang-tags", action="store_true", help="do not strip trailing <xx-XX> tags")
    ap.add_argument("--lang-tag-pattern", default=None, help="override the language-tag regex")
    ap.add_argument("--low-memory", action="store_true", help="disable the CPU memory arena")
    args = ap.parse_args()

    asr = NemotronStreamer(args.bundle_dir, language=args.language, latency=args.latency,
                           num_threads=args.threads, low_memory=args.low_memory,
                           strip_lang_tags=not args.keep_lang_tags,
                           lang_tag_pattern=args.lang_tag_pattern)
    print(f"[{args.language} @ {asr.latency} = {asr.lat['chunk_ms']} ms]")
    print(asr.transcribe(args.audio))


if __name__ == "__main__":
    main()
