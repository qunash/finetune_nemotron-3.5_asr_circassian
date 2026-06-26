# Fine-tuning Nemotron 3.5 ASR for Circassian

Adapt [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (cache-aware FastConformer-RNNT, prompt-conditioned, 40 locales) to **Adyghe (`ady`)** and **Kabardian (`kbd`)**. The notebook fine-tunes on a Circassian speech corpus; the scripts export the resulting `.nemo` checkpoint to several CPU inference formats and include a local dictation web app.

**Language slots:** `ady` and `kbd` reuse the pretrained `uk-UA` (id 19) and `bg-BG` (id 30) prompt slots rather than cold unused indices.

## Repository layout

| Path | Purpose |
|------|---------|
| `finetune_nemotron35_asr_circassian.ipynb` | End-to-end fine-tuning on GPU (NeMo, data prep, training, eval) |
| `cpu_onnx/` | Export to a portable **INT8 ONNX** bundle; pure Python streaming inference |
| `onnx-genai/` | Export to **onnxruntime-genai** format; fastest Python CPU path |
| `gguf/` | Convert to **GGUF** for [parakeet.cpp](https://github.com/mudler/parakeet.cpp); fastest overall on CPU |
| `webapp/` | Real-time microphone dictation (FastAPI + WebSocket, uses genai bundle) |

Generated weights (`.nemo`, `.onnx`, `.gguf`, export directories) are gitignored.

## Quick start

### 1. Fine-tune (GPU)

Open `finetune_nemotron35_asr_circassian.ipynb` on a CUDA machine with the Circassian raw data mounted (see notebook config for paths). Requires NeMo 26.06+, PyTorch, and wandb. Produces a fine-tuned `.nemo` checkpoint.

### 2. Export for inference

All exporters run on the machine where you fine-tuned (needs `nemo_toolkit[asr]` + torch). Pick one path:

**onnxruntime-genai** (recommended for speed + webapp):

```bash
python onnx-genai/export_genai.py FT.nemo onnx-genai/out_genai --chunk-size 0.56
python onnx-genai/nemotron_genai.py clip.wav --model onnx-genai/out_genai --language ady
```

**INT8 ONNX** (no genai dependency; pure ORT + NumPy):

```bash
python cpu_onnx/export_quantize.py FT.nemo bundle/ --latencies balanced --device auto
python cpu_onnx/nemotron_stream.py bundle/ clip.wav --language ady
```

**GGUF / parakeet.cpp** (fastest CPU; C++ binary):

```bash
gguf/convert.sh FT.nemo gguf_out              # produces gguf_out/nemotron.q8_0.gguf
gguf/transcribe.py --model gguf_out/nemotron.q8_0.gguf --input clip.wav --lang ady
```

### 3. Dictation web app

Export a genai bundle first, then:

```bash
python -m venv .venv && .venv/bin/pip install fastapi uvicorn onnxruntime-genai numpy
./webapp/run.sh                                    # http://127.0.0.1:8000
./webapp/run.sh --model onnx-genai/out_genai       # custom bundle path
```

The browser captures 16 kHz mono PCM, streams it over WebSocket, and displays incremental transcript deltas. Supports Adyghe and Kabardian.

## Inference comparison

| Path | Runtime | Best for |
|------|---------|----------|
| `onnx-genai/nemotron_genai.py` | onnxruntime-genai (C++ engine, INT4 encoder) | Fast CLI transcription; webapp backend |
| `cpu_onnx/nemotron_stream.py` | ONNX Runtime + NumPy (INT8 encoder) | Portable Python, no genai install |
| `gguf/transcribe.py` | parakeet.cpp (block-quantized GGUF) | Maximum CPU throughput; offline or streaming |

All paths expect **16 kHz mono** audio (WAV). Language is selected via `--language ady` / `--lang kbd` (or the corresponding prompt slot ids).

## Dependencies

- **Fine-tuning / export:** `nemo_toolkit[asr]`, torch, onnx, onnxruntime (see script docstrings for optional packages)
- **onnx-genai inference:** `onnxruntime-genai` (optionally `huggingface_hub` for stock model download)
- **cpu_onnx inference:** `onnxruntime`, `numpy`, `sentencepiece`
- **gguf:** parakeet.cpp CLI (fetched or built by `convert.sh`), `gguf` Python package for conversion
- **webapp:** `fastapi`, `uvicorn`, `onnxruntime-genai`, `numpy`

## References

- [NVIDIA fine-tuning guide](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr)
- Base model: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- Stock genai checkpoint (speed baseline): [`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`](https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4)
