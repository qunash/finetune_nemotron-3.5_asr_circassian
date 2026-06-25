#!/usr/bin/env python3
"""Export a fine-tuned Nemotron-3.5-ASR `.nemo` to a portable INT8 ONNX streaming bundle.

Run once on the box you fine-tuned on (needs ``nemo_toolkit[asr]`` + torch). The result runs on
any CPU with ``nemotron_stream.py`` (ONNX Runtime + NumPy only — no torch/NeMo).

What it does, grounded in the cache-aware streaming + prompt internals of EncDecRNNTBPEModelWithPrompt:
  * exports the FastConformer encoder with ``prompt_index`` exposed as a real input (one graph per
    language), parity-checked against NeMo before writing weights;
  * verifies prompt_index actually conditions the output across multiple languages (catches the
    "prompt baked to one language" failure mode);
  * exports the RNN-T decoder+joint once (latency-independent), kept FP32 by default;
  * dynamic INT8-quantizes each encoder (MatMul + Conv, per-channel, external data) — quantizing the
    Conformer convolutions is what brings the encoder down to ~615 MB (matching the smallest public
    builds); use --quant-ops MatMul to keep convs FP32 if you prefer accuracy headroom;
  * dumps the mel filterbank + a config.json (prompt dictionary, cache shapes, per-latency chunk
    geometry) and verifies the NumPy front end matches NeMo's preprocessor.

Device: tracing/parity run on GPU when available (``--device``), which is far faster than CPU.
The INT8 step (``quantize_dynamic``) is a weight rewrite, not compute — it is CPU-only by nature
and GPU cannot accelerate it; fewer ``--latencies`` is the way to make it faster.

RAM-aware: encoders are exported one at a time into isolated temp dirs, the torch model is freed
before the (heavy) quantization pass, and weights use external-data files (no double-load).

    python export_quantize.py FT.nemo out_bundle --latencies low,high --device auto
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from nemotron_stream import log_mel  # NumPy front end shared with the runtime (verified below)

# Right-context -> chunk latency (NVIDIA Nemotron-3.5-ASR model card). Left context is fixed at 56.
RIGHT_CTX = {"ultra": 0, "low": 1, "balanced": 3, "medium": 6, "high": 13}
CHUNK_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}
LEFT_CTX = 56
LOG_GUARD = 2.0 ** -24

ENC_INPUTS = ["processed_signal", "processed_signal_length", "cache_last_channel",
              "cache_last_time", "cache_last_channel_len", "prompt_index"]
ENC_OUTPUTS = ["encoded", "encoded_len", "cache_last_channel_next",
               "cache_last_time_next", "cache_last_channel_len_next"]


def build_prompt_wrapper(torch, model, drop_extra):
    """Torch wrapper that inlines NeMo's _apply_prompt_to_encoded and exposes prompt_index.

    Mirrors NeMo's math exactly: it just replaces the frozen ``self._inference_prompt_index`` int
    with a tensor input, so one ONNX graph serves every language.
    """

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            self.prompt_kernel = model.prompt_kernel
            self.num_prompts = model.num_prompts
            self.drop_extra = drop_extra

        def forward(self, processed_signal, processed_signal_length,
                    cache_last_channel, cache_last_time, cache_last_channel_len, prompt_index):
            encoded, enc_len, ch_n, tm_n, ln_n = self.encoder.cache_aware_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra,
            )
            encoded = encoded.transpose(1, 2)              # (B, D, T) -> (B, T, D)
            B, T, _ = encoded.shape
            prompt = torch.zeros(B, T, self.num_prompts, dtype=encoded.dtype, device=encoded.device)
            prompt.scatter_(2, prompt_index.view(B, 1, 1).expand(-1, T, -1), 1.0)
            encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
            return encoded.transpose(1, 2), enc_len, ch_n, tm_n, ln_n

    return _Wrap().eval()


def _patch_torch(torch):
    """NeMo .nemo needs full unpickle; PyTorch >=2.9 needs the legacy ONNX exporter."""
    _orig = torch.load
    torch.load = functools.wraps(_orig)(lambda *a, **k: _orig(*a, **{**k, "weights_only": False}))
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 9):
        _oexp = torch.onnx.export
        torch.onnx.export = functools.wraps(_oexp)(
            lambda *a, **k: _oexp(*a, **(k if "dynamo" in k else {**k, "dynamo": False})))


def _consolidate(onnx, src: str, dst: Path, data_name: str) -> None:
    """Re-save scattered external weights into a single <name> + <name>.data pair."""
    model = onnx.load(src, load_external_data=True)
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    del model
    gc.collect()


def _resolve_device(torch, want: str):
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_prompt(ort, model, torch, enc_fp32: Path, langs, prompt_dict, ref_inputs, drop):
    """Confirm prompt_index conditions the encoder: ONNX matches NeMo per language AND distinct
    languages produce distinct outputs. Aborts if the prompt is effectively ignored — the failure
    mode that makes every language transcribe as English."""
    proc_sig, proc_len, cch, ctt, cln = ref_inputs
    feeds = {
        "processed_signal": proc_sig.cpu().numpy(),
        "processed_signal_length": proc_len.cpu().numpy(),
        "cache_last_channel": cch.cpu().numpy(),
        "cache_last_time": ctt.cpu().numpy(),
        "cache_last_channel_len": cln.cpu().numpy(),
    }
    sess = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    seen: dict[int, np.ndarray] = {}
    worst_ref = 0.0
    for lang in langs:
        if lang not in prompt_dict:
            print(f"  [verify] skip {lang!r} (not in prompt_dictionary)")
            continue
        idx = prompt_dict[lang]
        model.set_inference_prompt(lang)
        with torch.no_grad():
            er, _, _, _, _ = model.encoder.cache_aware_stream_step(
                processed_signal=proc_sig, processed_signal_length=proc_len,
                cache_last_channel=cch, cache_last_time=ctt, cache_last_channel_len=cln,
                keep_all_outputs=False, drop_extra_pre_encoded=drop)
            ref = model._apply_prompt_to_encoded(er).cpu().numpy()
        onx = sess.run(["encoded"], {**feeds, "prompt_index": np.array([idx], np.int64)})[0]
        d = float(np.abs(ref - onx).max())
        worst_ref = max(worst_ref, d)
        print(f"  [verify] {lang:6s} idx={idx:3d}  ONNX vs NeMo max|Δ|={d:.2e}")
        seen.setdefault(idx, onx)

    if worst_ref > 1e-3:
        raise SystemExit(f"ONNX encoder diverges from NeMo (max|Δ|={worst_ref:.2e}); aborting.")
    if len(seen) >= 2:
        a, b = list(seen.values())[:2]
        sens = float(np.abs(a - b).max())
        print(f"  [verify] prompt sensitivity across languages: max|Δ|={sens:.2e}")
        if sens < 1e-3:
            raise SystemExit("prompt_index is IGNORED — distinct languages give identical encoder "
                             "output. Transcription would default to one language. Aborting.")
    else:
        print("  [verify] WARNING: <2 known languages available; prompt sensitivity not checked")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a .nemo to an INT8 ONNX streaming bundle")
    ap.add_argument("nemo", help="path to the fine-tuned .nemo checkpoint")
    ap.add_argument("out", help="output bundle directory")
    ap.add_argument("--latencies", default="balanced",
                    help="comma list of ultra,low,balanced,medium,high (one encoder each)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="device for tracing/parity (quantization is always CPU)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify-langs", default="en-US,ady,kbd",
                    help="languages used to confirm prompt_index conditioning")
    ap.add_argument("--keep-fp32", action="store_true",
                    help="also keep the FP32 encoders (lets the runtime A/B test INT8 vs FP32)")
    ap.add_argument("--no-reduce-range", dest="reduce_range", action="store_false",
                    help="disable INT8 reduce_range (only safe on VNNI/AMX CPUs; default keeps it on)")
    ap.add_argument("--quant-ops", default="MatMul,Conv",
                    help="op types to INT8-quantize. Default 'MatMul,Conv' also quantizes the "
                         "Conformer convolutions (~615 MB encoder). Use 'MatMul' to keep convs in "
                         "FP32 (~885 MB, marginally safer accuracy on low-resource data).")
    ap.add_argument("--quantize-decoder", action="store_true",
                    help="also INT8-quantize the decoder_joint (~98 MB -> ~25 MB); slight accuracy "
                         "risk on the prediction network — off by default")
    ap.set_defaults(reduce_range=True)
    args = ap.parse_args()
    quant_ops = [s.strip() for s in args.quant_ops.split(",") if s.strip()]

    lats = [s.strip() for s in args.latencies.split(",") if s.strip()]
    for n in lats:
        if n not in RIGHT_CTX:
            raise SystemExit(f"unknown latency {n!r}; choose from {list(RIGHT_CTX)}")
    verify_langs = [s.strip() for s in args.verify_langs.split(",") if s.strip()]

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    logging.getLogger("nemo_logging").setLevel(logging.ERROR)

    import torch
    _patch_torch(torch)
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    from omegaconf import OmegaConf
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    dev = _resolve_device(torch, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.nemo} (device={dev})...")
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location=dev).eval().to(dev)
    if type(model).__name__ != "EncDecRNNTBPEModelWithPrompt":
        raise SystemExit(f"expected EncDecRNNTBPEModelWithPrompt, got {type(model).__name__}")

    subsampling = int(model.cfg.encoder.get("subsampling_factor", 8))
    vocab_size = int(model.tokenizer.vocab_size)
    num_prompts = int(model.num_prompts)
    prompt_dict = {k: int(v) for k, v in
                   OmegaConf.to_container(model.cfg.model_defaults.prompt_dictionary, resolve=True).items()}
    if "auto" not in prompt_dict:
        print("  WARNING: prompt key 'auto' absent; --language auto will fail at runtime")

    # --- tokenizer.model straight out of the .nemo tar ---
    with tarfile.open(args.nemo, "r:*") as tar:
        member = next(m for m in tar.getnames() if m.endswith("tokenizer.model"))
        (out / "tokenizer.model").write_bytes(tar.extractfile(member).read())

    # --- mel params + filterbank, then verify the NumPy front end matches NeMo ---
    pp = model.preprocessor._cfg
    n_fft = int(pp.n_fft)
    win = int(round(float(pp.window_size) * float(pp.sample_rate)))
    hop = int(round(float(pp.window_stride) * float(pp.sample_rate)))
    n_mels = int(pp.features)
    featurizer = model.preprocessor.featurizer
    preemph = float(getattr(featurizer, "preemph", 0.97) or 0.97)
    mag_power = float(getattr(featurizer, "mag_power", 2.0))
    log_guard = float(getattr(featurizer, "log_zero_guard_value", LOG_GUARD))
    normalize = str(getattr(featurizer, "normalize", "NA"))
    fb = featurizer.fb.detach().cpu().numpy().astype(np.float32).reshape(n_mels, -1)
    fb.tofile(out / "filterbank.bin")

    featurizer.dither = 0.0
    if hasattr(featurizer, "pad_to"):
        featurizer.pad_to = 0
    probe = (np.random.RandomState(0).randn(16000 * 2).astype(np.float32)) * 0.1
    with torch.no_grad():
        mel_nemo, mel_len = model.preprocessor(
            input_signal=torch.tensor(probe)[None].to(dev), length=torch.tensor([len(probe)]).to(dev))
    mel_nemo = mel_nemo[0].cpu().numpy()
    valid = int(mel_len[0])  # NeMo zero-fills frames >= this; comparing them inflates max|Δ|
    mel_np = log_mel(probe, fb, n_fft, hop, win, preemph, log_guard, mag_power)
    cols = min(valid, mel_nemo.shape[1], mel_np.shape[1])
    mel_diff = float(np.abs(mel_nemo[:, :cols] - mel_np[:, :cols]).max())
    print(f"mel front-end parity (NumPy vs NeMo, {cols} valid frames): max|Δ| = {mel_diff:.3e}  "
          f"(normalize={normalize})")
    if mel_diff > 1e-2:
        raise SystemExit("NumPy mel diverges from NeMo — front end would be wrong; aborting.")

    # --- export each latency's FP32 encoder (isolated temp dir -> no scattered weight files) ---
    probe_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(probe_wav.name, probe, 16000)
    probe_wav.close()

    cache_shapes = None
    lat_meta: dict[str, dict] = {}
    for name in lats:
        right = RIGHT_CTX[name]
        chunk = right + 1
        model.encoder.set_default_att_context_size([LEFT_CTX, right])
        model.encoder.setup_streaming_params(chunk_size=chunk, shift_size=chunk)
        scfg = model.encoder.streaming_cfg
        drop = getattr(scfg, "drop_extra_pre_encoded", 0)
        drop = int(drop[-1] if isinstance(drop, (list, tuple)) else drop)

        cch, ctt, cln = model.encoder.get_initial_cache_state(batch_size=1)
        cache_shapes = {"cache_last_channel": list(cch.shape),
                        "cache_last_time": list(ctt.shape),
                        "cache_last_channel_len": list(cln.shape)}

        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False,
                                             pad_and_drop_preencoded=True)
        buf.append_audio_file(probe_wav.name, stream_id=-1)
        proc_sig, proc_len = next(iter(buf))
        proc_sig, proc_len = proc_sig.to(dev), proc_len.to(dev)
        mel_chunk = int(proc_sig.shape[2])
        mel_shift = chunk * subsampling
        pre_encode = mel_chunk - mel_shift

        wrap = build_prompt_wrapper(torch, model, drop)
        model.set_inference_prompt("en-US" if "en-US" in prompt_dict else next(iter(prompt_dict)))
        prompt_idx = torch.tensor([model._inference_prompt_index], dtype=torch.long, device=dev)
        with torch.no_grad():
            enc_raw, _, _, _, _ = model.encoder.cache_aware_stream_step(
                processed_signal=proc_sig, processed_signal_length=proc_len,
                cache_last_channel=cch, cache_last_time=ctt, cache_last_channel_len=cln,
                keep_all_outputs=False, drop_extra_pre_encoded=drop)
            ref = model._apply_prompt_to_encoded(enc_raw)
            got, _, _, _, _ = wrap(proc_sig, proc_len, cch, ctt, cln, prompt_idx)
        parity = (got - ref).abs().max().item()
        if parity > 1e-4:
            raise SystemExit(f"[{name}] prompt wrapper diverges from NeMo (max|Δ|={parity:.2e}); aborting.")
        print(f"[{name}] chunk={CHUNK_MS[right]}ms  mel_chunk={mel_chunk}  pre_encode={pre_encode}  "
              f"drop={drop}  wrapper parity max|Δ|={parity:.2e}")

        enc_tmp = out / f"_enc_{name}"
        if enc_tmp.exists():
            shutil.rmtree(enc_tmp)
        enc_tmp.mkdir()
        with torch.no_grad():
            torch.onnx.export(
                wrap,
                (proc_sig, proc_len, cch, ctt, cln, prompt_idx),
                str(enc_tmp / "encoder.onnx"), input_names=ENC_INPUTS, output_names=ENC_OUTPUTS,
                opset_version=args.opset,
                dynamic_axes={"processed_signal": {0: "batch", 2: "time"},
                              "processed_signal_length": {0: "batch"},
                              "prompt_index": {0: "batch"},
                              "encoded": {0: "batch", 2: "time"},
                              "encoded_len": {0: "batch"}})
        fp32_enc = out / f"encoder.{name}.fp32.onnx"
        _consolidate(onnx, str(enc_tmp / "encoder.onnx"), fp32_enc, fp32_enc.name + ".data")
        shutil.rmtree(enc_tmp, ignore_errors=True)

        if name == lats[0]:  # confirm prompt conditioning once (same weights across latencies)
            print("Verifying prompt_index conditioning (FP32 ONNX vs NeMo)...")
            verify_prompt(ort, model, torch, fp32_enc, verify_langs, prompt_dict,
                          (proc_sig, proc_len, cch, ctt, cln), drop)

        lat_meta[name] = {
            "right_context": right, "chunk_ms": CHUNK_MS[right], "chunk_size_output": chunk,
            "mel_shift": mel_shift, "pre_encode_cache": pre_encode, "mel_chunk_frames": mel_chunk,
            "drop_extra": drop, "encoder_file": f"encoder.{name}.int8.onnx",
            "_fp32": fp32_enc.name,
        }

    os.unlink(probe_wav.name)

    # --- decoder+joint: latency-independent, kept FP32. Exported last because NeMo's
    # _prepare_for_export hooks can mutate modules used by the streaming parity checks above. ---
    print("Exporting decoder_joint (FP32)...")
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    with torch.no_grad():
        model.export(str(tmp / "m.onnx"), check_trace=False, onnx_opset_version=args.opset)
    dec_src = next(p for p in tmp.glob("*decoder_joint*.onnx"))
    _consolidate(onnx, str(dec_src), out / "decoder_joint.onnx", "decoder_joint.onnx.data")
    shutil.rmtree(tmp, ignore_errors=True)

    # Free torch weights (and GPU memory) before the memory-hungry quantization pass.
    del model, wrap
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # --- dynamic INT8 (per-channel, external data). CPU-only. ---
    # reduce_range keeps weights in [-64, 63] so U8S8 MatMul/Conv kernels don't saturate the int16
    # accumulator on non-VNNI CPUs (AVX2) — without it the encoder collapses to garbage there.
    # Quantizing Conv (the Conformer conv module is ~290 MB of FP32 weights) is what shrinks the
    # encoder from ~885 MB to ~615 MB; it is the default but can be disabled via --quant-ops MatMul.
    for name, meta in lat_meta.items():
        src = out / meta.pop("_fp32")
        dst = out / meta["encoder_file"]
        print(f"[{name}] quantizing {src.name} -> {dst.name} "
              f"(INT8 dynamic, ops={quant_ops}, per-channel, reduce_range={args.reduce_range})...")
        quantize_dynamic(model_input=str(src), model_output=str(dst),
                         weight_type=QuantType.QInt8, per_channel=True, reduce_range=args.reduce_range,
                         op_types_to_quantize=quant_ops, use_external_data_format=True)
        if args.keep_fp32:
            meta["encoder_fp32_file"] = src.name
        else:
            for f in src.parent.glob(src.name + "*"):
                f.unlink()

    decoder_file = "decoder_joint.onnx"
    if args.quantize_decoder:
        dec_int8 = out / "decoder_joint.int8.onnx"
        print(f"quantizing decoder_joint -> {dec_int8.name} (INT8 dynamic, ops={quant_ops})...")
        quantize_dynamic(model_input=str(out / "decoder_joint.onnx"), model_output=str(dec_int8),
                         weight_type=QuantType.QInt8, per_channel=True, reduce_range=args.reduce_range,
                         op_types_to_quantize=quant_ops, use_external_data_format=False)
        for f in out.glob("decoder_joint.onnx*"):
            f.unlink()
        decoder_file = dec_int8.name

    config = {
        "model_name": "nemotron-3.5-asr-streaming-0.6b (fine-tuned)",
        "sample_rate": 16000,
        "subsampling_factor": subsampling,
        "vocab_size": vocab_size,
        "blank_id": vocab_size,
        "num_prompts": num_prompts,
        "prompt_dictionary": prompt_dict,
        "mel": {"n_mels": n_mels, "n_fft": n_fft, "win_length": win, "hop_length": hop,
                "preemph": preemph, "mag_power": mag_power, "log_guard": log_guard,
                "normalize": normalize},
        "cache": cache_shapes,
        "encoder_inputs": ENC_INPUTS,
        "encoder_outputs": ENC_OUTPUTS,
        "decoder_file": decoder_file,
        "latencies": lat_meta,
        "default_latency": lats[0],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("\nBundle written to", out)
    total = 0
    for f in sorted(out.iterdir()):
        if f.is_file():
            total += f.stat().st_size
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
    print(f"  TOTAL {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
