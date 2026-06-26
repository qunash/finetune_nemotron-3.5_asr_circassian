#!/usr/bin/env python3
"""Local web app for real-time microphone dictation with the fine-tuned Nemotron-3.5-ASR.

A FastAPI server with one WebSocket endpoint. The browser captures 16 kHz mono PCM from the
microphone, streams chunks down the socket, and the server runs them through the onnxruntime-genai
streaming RNN-T engine, pushing token deltas back up as they are decoded.

    .venv/bin/python -m webapp.server            # serve on http://127.0.0.1:8000
    .venv/bin/python -m webapp.server --model onnx-genai/out_genai --port 8000

Languages ady (id 19) and kbd (id 30) are the fine-tuned slots in prompt_dictionary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import onnxruntime_genai as og

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

# ady/kbd are the fine-tuned prompt slots in this checkpoint (see prompt_dictionary.json).
LANGUAGES = {
    "ady": {"id": 19, "label": "Adyghe", "native": "Адыгабзэ"},
    "kbd": {"id": 30, "label": "Kabardian", "native": "Адыгэбзэ"},
}

LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")  # NeMo trailing language tag


class StreamingTranscriber:
    """One onnxruntime-genai streaming session.

    Reused across chunks for a single dictation: the encoder caches and decoder state evolve
    chunk-to-chunk inside the Generator/StreamingProcessor, so we hold one Generator + tokenizer
    stream per WebSocket connection and feed it incrementally.
    """

    def __init__(self, model_dir: str, language: str):
        self.lang_id = LANGUAGES[language]["id"]
        self.language = language

        config = og.Config(model_dir)
        config.clear_providers()  # CPU execution provider
        self.model = og.Model(config)
        self.processor = og.StreamingProcessor(self.model)
        self.processor.set_option("use_vad", "false")
        self.tokenizer = og.Tokenizer(self.model)
        self.tok_stream = self.tokenizer.create_stream()
        self.generator = og.Generator(self.model, og.GeneratorParams(self.model))
        self.generator.set_runtime_option("lang_id", str(self.lang_id))

        self._text = ""

    def _drain(self) -> str:
        """Greedy-decode every ready token; return the concatenated piece string."""
        piece = ""
        while not self.generator.is_done():
            self.generator.generate_next_token()
            toks = self.generator.get_next_tokens()
            if len(toks):
                p = self.tok_stream.decode(toks[0])
                if p:
                    piece += p
        return piece

    def push(self, audio: np.ndarray) -> str:
        """Feed one chunk of 16 kHz mono float32; return the new transcript delta."""
        inputs = self.processor.process(audio.astype(np.float32))
        delta = ""
        if inputs is not None:
            self.generator.set_inputs(inputs)
            delta = self._drain()
        self._text += delta
        return delta

    def flush(self) -> str:
        """Emit whatever the trailing partial chunk produced; call when recording stops."""
        inputs = self.processor.flush()
        delta = ""
        if inputs is not None:
            self.generator.set_inputs(inputs)
            delta = self._drain()
        self._text += delta
        return delta

    def finalize(self) -> str:
        """Return the full transcript with the trailing language tag stripped."""
        return LANG_TAG_RE.sub("", self._text).strip()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """16-bit little-endian signed PCM -> float32 in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


# Load the model once at startup; per-connection transcribers all share this Model object.
MODEL_DIR: str = ""
_model: og.Model | None = None


def build_app(model_dir: str) -> FastAPI:
    global MODEL_DIR, _model
    MODEL_DIR = model_dir

    cfg = json.loads((Path(model_dir) / "genai_config.json").read_text())["model"]
    print(f"[startup] loading model from {model_dir} (sr={cfg['sample_rate']}, "
          f"chunk={cfg['chunk_samples']} samples = {cfg['chunk_samples']/cfg['sample_rate']:.2f}s)...")
    t0 = time.perf_counter()
    config = og.Config(model_dir)
    config.clear_providers()
    _model = og.Model(config)
    # Warm up the StreamingProcessor construction path so the first request isn't slow.
    print(f"[startup] model loaded in {time.perf_counter()-t0:.1f}s")

    app = FastAPI(title="Nemotron Circassian ASR")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/languages")
    async def languages():
        return {"languages": [
            {"code": k, "id": v["id"], "label": v["label"], "native": v["native"]}
            for k, v in LANGUAGES.items()
        ]}

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket):
        await ws.accept()
        tx: StreamingTranscriber | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                # Dispatch by WebSocket message type, not content sniffing: an Int16 PCM frame
                # whose low byte happens to be 0x7b ('{') would otherwise be misparsed as JSON,
                # throwing a utf-8 error that kills the socket — after which the still-recording
                # browser reconnects and streams audio with no 'start', hence the confusing
                # "send a 'start' control frame first" follow-up. Binary = audio, text = control.
                if msg.get("bytes") is not None:
                    if tx is None:
                        await ws.send_json({"type": "error",
                                            "message": "send a 'start' control frame first"})
                        continue
                    audio = pcm16_to_float(bytes(msg["bytes"]))
                    delta = await asyncio.to_thread(tx.push, audio)
                    if delta:
                        await ws.send_json({"type": "delta", "text": delta})
                elif msg.get("text") is not None:
                    payload = json.loads(msg["text"])
                    kind = payload.get("type")
                    if kind == "start":
                        lang = payload.get("language", "ady")
                        if lang not in LANGUAGES:
                            await ws.send_json({"type": "error",
                                                "message": f"unknown language {lang!r}"})
                            break
                        tx = await asyncio.to_thread(StreamingTranscriber, MODEL_DIR, lang)
                        await ws.send_json({"type": "ready", "language": lang,
                                            "lang_id": LANGUAGES[lang]["id"]})
                    elif kind == "audio":  # base64-encoded PCM (fallback path)
                        if tx is None:
                            continue
                        audio = pcm16_to_float(base64.b64decode(payload["data"]))
                        delta = await asyncio.to_thread(tx.push, audio)
                        if delta:
                            await ws.send_json({"type": "delta", "text": delta})
                    elif kind == "stop":
                        if tx is not None:
                            await asyncio.to_thread(tx.flush)
                            final = await asyncio.to_thread(tx.finalize)
                            await ws.send_json({"type": "final", "text": final})
                            tx = None
                    elif kind == "abort":
                        tx = None
                        await ws.send_json({"type": "reset"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Local Nemotron Circassian dictation web app")
    ap.add_argument("--model", default="onnx-genai/out_genai",
                    help="path to the genai bundle directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    app = build_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
