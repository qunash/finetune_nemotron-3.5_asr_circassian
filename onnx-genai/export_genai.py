#!/usr/bin/env python3
"""Export a fine-tuned Nemotron-3.5-ASR `.nemo` to a Microsoft onnxruntime-genai bundle.

Produces the three-graph layout the onnxruntime-genai ``nemotron_speech`` runtime expects, so
``nemotron_genai.py`` can run the model on its fast native C++ engine:

    encoder.onnx (+ .data)   cache-aware FastConformer, INT4 MatMulNBits, prompt-conditioned
    decoder.onnx (+ .data)   RNN-T prediction net (2-layer LSTM, explicit h/c state I/O)
    joint.onnx   (+ .data)   RNN-T joiner (LogSoftmax over vocab+blank)
    genai_config.json        graph I/O names + mel/streaming geometry
    audio_processor_config.json   NeMo-compatible log-mel front-end params
    tokenizer.json / tokenizer_config.json / vocab.txt   SentencePiece -> Unigram (ORT-Extensions)

This is the genai-format sibling of ``../cpu_onnx/export_quantize.py``. The graph wrappers,
config generation, tokenizer conversion, and INT4 ``MatMulNBitsQuantizer`` call mirror
Microsoft's official ``tools/nemotron_export`` (microsoft/onnxruntime-genai, PR #1997). The one
piece that tool lacks is **prompt conditioning**: that tool targets the English model, whereas
this checkpoint is an ``EncDecRNNTBPEModelWithPrompt``. So the encoder here is exported with a
real ``lang_id`` input (one graph serves every language) and the export aborts if ``lang_id``
turns out to be ignored — the failure mode that makes every language transcribe as English.

Run on the box you fine-tuned on (needs ``nemo_toolkit[asr]`` + torch + onnx + onnxruntime):

    python export_genai.py FT.nemo out_genai --chunk-size 0.56

Then:  python nemotron_genai.py clip.wav --model out_genai --language ady
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import os
from pathlib import Path

import numpy as np

LEFT_CTX = 56  # this model's trained left context (matches the public int4 export)
SUBSAMPLING = 8

# Encoder I/O names — must match the onnxruntime-genai nemotron_speech runtime exactly.
ENC_IN = ["audio_signal", "length", "cache_last_channel", "cache_last_time",
          "cache_last_channel_len", "lang_id"]
ENC_OUT = ["outputs", "encoded_lengths", "cache_last_channel_next",
           "cache_last_time_next", "cache_last_channel_len_next"]


def _patch_torch(torch):
    """NeMo .nemo needs a full unpickle; torch>=2.9 routes ONNX export through dynamo by default."""
    _orig = torch.load
    torch.load = functools.wraps(_orig)(lambda *a, **k: _orig(*a, **{**k, "weights_only": False}))


def _consolidate(onnx, src: Path, dst: Path) -> None:
    """Re-save scattered/inline weights into a single <dst> + <dst>.data pair."""
    model = onnx.load(str(src), load_external_data=True)
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=dst.name + ".data", size_threshold=1024)
    del model
    gc.collect()


def build_encoder_wrapper(torch, model):
    """Streaming encoder wrapper with genai I/O. Inlines NeMo's _apply_prompt_to_encoded so
    ``lang_id`` selects the language slot — one graph serves every language."""

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = model.encoder
            self.prompt_kernel = model.prompt_kernel
            self.num_prompts = int(model.num_prompts)

        def forward(self, audio_signal, length, cache_last_channel, cache_last_time,
                    cache_last_channel_len, lang_id):
            audio_signal = audio_signal.transpose(1, 2)  # [B,T,M] -> [B,M,T] for NeMo
            encoded, enc_len, ch_n, tm_n, len_n = self.enc.forward_for_export(
                audio_signal=audio_signal, length=length,
                cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len)
            encoded = encoded.transpose(1, 2)  # [B,D,T] -> [B,T,D]
            B, T, _ = encoded.shape
            prompt = torch.zeros(B, T, self.num_prompts, dtype=encoded.dtype, device=encoded.device)
            prompt.scatter_(2, lang_id.view(B, 1, 1).expand(-1, T, -1), 1.0)
            encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
            return encoded, enc_len, ch_n, tm_n, len_n

    return Encoder().eval()


def build_decoder_wrapper(torch, decoder):
    """Prediction network with explicit LSTM state I/O; output [B, D, U] (genai layout)."""
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = decoder
            self.decoder._rnnt_export = True

        def forward(self, targets, h_in, c_in):
            g, (h_out, c_out) = self.decoder.predict(y=targets, state=(h_in, c_in), add_sos=False)
            return g.transpose(1, 2), h_out, c_out  # [B,1,D] -> [B,D,1]

    return Decoder().eval()


def build_joint_wrapper(torch, joint):
    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.joint = joint

        def forward(self, encoder_output, decoder_output):
            return self.joint.joint(encoder_output, decoder_output)

    return Joint().eval()


def verify_prompt(ort, enc_path: Path, slots: list[int], dummy: dict) -> None:
    """Abort if distinct lang_id values give identical encoder output (prompt ignored)."""
    sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    outs = []
    for sid in slots[:2]:
        feed = {**dummy, "lang_id": np.array([sid], np.int64)}
        outs.append(sess.run(["outputs"], feed)[0])
    if len(outs) == 2:
        delta = float(np.abs(outs[0] - outs[1]).max())
        print(f"  [verify] prompt sensitivity across slots {slots[:2]}: max|Δ|={delta:.2e}")
        if delta < 1e-4:
            raise SystemExit("lang_id is IGNORED — distinct slots give identical encoder output. "
                             "Every language would transcribe the same. Aborting.")


def write_configs(out: Path, n_mels, n_fft, hop, win, preemph, vocab_size, blank_id,
                  enc_hidden, enc_layers, dec_hidden, dec_layers, chunk_samples,
                  pre_encode, conv_context, max_symbols, is_prompt, dither) -> None:
    enc_inputs = {"audio_features": "audio_signal", "input_lengths": "length",
                  "cache_last_channel": "cache_last_channel", "cache_last_time": "cache_last_time",
                  "cache_last_channel_len": "cache_last_channel_len"}
    if is_prompt:
        enc_inputs["lang_id"] = "lang_id"
    genai = {"model": {
        "type": "nemotron_speech", "vocab_size": vocab_size, "num_mels": n_mels,
        "fft_size": n_fft, "hop_length": hop, "win_length": win, "preemph": preemph,
        "log_eps": 5.96046448e-08, "subsampling_factor": SUBSAMPLING, "left_context": LEFT_CTX,
        "conv_context": conv_context, "pre_encode_cache_size": pre_encode, "sample_rate": 16000,
        "chunk_samples": chunk_samples, "blank_id": blank_id, "max_symbols_per_step": max_symbols,
        "encoder": {"filename": "encoder.onnx", "hidden_size": enc_hidden,
                    "num_hidden_layers": enc_layers, "inputs": enc_inputs,
                    "outputs": {"encoder_outputs": "outputs", "output_lengths": "encoded_lengths",
                                "cache_last_channel_next": "cache_last_channel_next",
                                "cache_last_time_next": "cache_last_time_next",
                                "cache_last_channel_len_next": "cache_last_channel_len_next"}},
        "decoder": {"filename": "decoder.onnx", "hidden_size": dec_hidden,
                    "num_hidden_layers": dec_layers,
                    "inputs": {"targets": "targets", "lstm_hidden_state": "h_in",
                               "lstm_cell_state": "c_in"},
                    "outputs": {"outputs": "decoder_output", "lstm_hidden_state": "h_out",
                                "lstm_cell_state": "c_out"}},
        "joiner": {"filename": "joint.onnx",
                   "inputs": {"encoder_outputs": "encoder_output", "decoder_outputs": "decoder_output"},
                   "outputs": {"logits": "joint_output"}},
    }}
    (out / "genai_config.json").write_text(json.dumps(genai, indent=2))

    audio = {"model_type": "speech_features", "audio_params": {
        "sample_rate": 16000, "n_fft": n_fft, "hop_length": hop, "n_mels": n_mels,
        "window_length": win, "window_type": "hann", "fmin": 0, "fmax": 8000,
        "dither": dither, "preemphasis": preemph, "log_zero_guard_type": "add",
        "log_zero_guard_value": 1e-10, "normalize": "NA", "center": True, "mag_power": 2.0}}
    (out / "audio_processor_config.json").write_text(json.dumps(audio, indent=2))


def export_tokenizer(out: Path, model, vocab_size: int) -> None:
    """SentencePiece vocab -> ORT-Extensions Unigram tokenizer.json (T5Tokenizer path)."""
    tokens = []
    for i in range(vocab_size - 1):  # last id is blank
        try:
            d = model.tokenizer.ids_to_tokens([i])
            tokens.append(str(d[0] if isinstance(d, list) else d))
        except Exception:
            tokens.append(f"<unk_{i}>")
    tokens.append("<blank>")
    (out / "vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")

    vocab = [[t, 0.0 if t in ("<unk>", "<blank>") else -float(i)] for i, t in enumerate(tokens)]
    tok = {"version": "1.0", "truncation": None, "padding": None,
           "added_tokens": [{"id": 0, "content": "<unk>", "single_word": False, "lstrip": False,
                             "rstrip": False, "normalized": False, "special": True}],
           "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"},
           "pre_tokenizer": None, "post_processor": None, "decoder": None,
           "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab},
           "pretokenizer": {"pretokenizers": [{"type": "Metaspace", "add_prefix_space": False}]}}
    (out / "tokenizer.json").write_text(json.dumps(tok, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "T5Tokenizer", "unk_token": "<unk>", "model_max_length": 1024,
         "add_bos_token": False, "add_eos_token": False, "clean_up_tokenization_spaces": False},
        indent=2), encoding="utf-8")


def quantize_encoder(src: Path, dst: Path, method: str, block_size: int, accuracy_level: int) -> None:
    """FP32 MatMul weights -> INT4 MatMulNBits (weight-only, block-quant). Encoder only;
    decoder/joint stay FP32 (tiny, and the joiner runs every encoder step)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    if method == "k_quant":
        from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig
        algo = KQuantWeightOnlyQuantConfig()
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import RTNWeightOnlyQuantConfig
        algo = RTNWeightOnlyQuantConfig()
    q = MatMulNBitsQuantizer(model=onnx.load(str(src), load_external_data=True),
                             block_size=block_size, is_symmetric=True,
                             accuracy_level=accuracy_level, algo_config=algo)
    q.process()
    for f in (dst, Path(str(dst) + ".data")):
        if f.exists():
            f.unlink()
    q.model.save_model_to_file(str(dst), use_external_data_format=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a .nemo to an onnxruntime-genai streaming bundle")
    ap.add_argument("nemo", help="path to the fine-tuned .nemo checkpoint")
    ap.add_argument("out", help="output bundle directory")
    ap.add_argument("--chunk-size", type=float, default=0.56, choices=[0.08, 0.16, 0.32, 0.56, 1.12],
                    help="streaming chunk in seconds (0.56 = recommended 560 ms, matches public int4)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device for tracing")
    ap.add_argument("--no-quantize", action="store_true", help="keep the encoder FP32 (skip INT4)")
    ap.add_argument("--quant-method", default="rtn", choices=["rtn", "k_quant"],
                    help="INT4 algorithm (rtn = simplest/portable; k_quant needs neural-compressor)")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--accuracy-level", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    _patch_torch(torch)
    import onnx
    import onnxruntime as ort
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"Loading {args.nemo} (device={dev})...")
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location=dev).eval().to(dev)
    if not (hasattr(model, "prompt_kernel") and hasattr(model, "num_prompts")):
        raise SystemExit(f"{type(model).__name__} is not prompt-conditioned. This exporter targets "
                         "the multilingual EncDecRNNTBPEModelWithPrompt; for a plain English model "
                         "use Microsoft's tools/nemotron_export from onnxruntime-genai.")
    print(f"  model={type(model).__name__}  (prompt-conditioned, multilingual)")

    # --- streaming geometry (mirrors export_quantize.py; chunk geometry the runtime assumes) ---
    out_frames = int(round(args.chunk_size * 100)) // SUBSAMPLING  # encoder frames per chunk
    right = out_frames - 1
    model.encoder.set_default_att_context_size([LEFT_CTX, right])
    model.encoder.setup_streaming_params(chunk_size=out_frames, shift_size=out_frames)
    scfg = model.encoder.streaming_cfg
    pre_encode = getattr(scfg, "pre_encode_cache_size", 9)
    pre_encode = int(pre_encode[-1] if isinstance(pre_encode, (list, tuple)) else pre_encode)

    enc = model.encoder
    n_layers = int(getattr(enc, "num_layers", 24))
    d_model = int(getattr(enc, "d_model", 1024))
    conv = enc.layers[0].conv.conv
    conv_context = (conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size) - 1
    chunk_samples = int(args.chunk_size * 16000)
    mel_frames = pre_encode + out_frames * SUBSAMPLING

    pp = model.preprocessor._cfg
    n_fft, n_mels = int(pp.n_fft), int(pp.features)
    win = int(round(float(pp.window_size) * 16000))
    hop = int(round(float(pp.window_stride) * 16000))
    preemph = float(pp.get("preemph", 0.97) or 0.97)
    dither = float(pp.get("dither", 0.0) or 0.0)
    vocab_size = int(model.joint.num_classes_with_blank)
    blank_id = vocab_size - 1
    dec_hidden = int(getattr(model.decoder, "pred_hidden", 640))
    dec_layers = int(getattr(model.decoder, "pred_rnn_layers", 2))
    max_symbols = int(model.cfg.get("decoding", {}).get("greedy", {}).get("max_symbols", 10))
    print(f"  chunk={args.chunk_size}s right={right} mel_frames={mel_frames} pre_encode={pre_encode} "
          f"left={LEFT_CTX} conv_ctx={conv_context} vocab={vocab_size}")

    def t(*a, **k):
        return torch.zeros(*a, **k).to(dev)

    # --- encoder ---
    print("Exporting encoder...")
    enc_wrap = build_encoder_wrapper(torch, model)
    dummy = {"audio_signal": torch.randn(1, mel_frames, n_mels).to(dev),
             "length": torch.tensor([mel_frames], dtype=torch.int64).to(dev),
             "cache_last_channel": t(1, n_layers, LEFT_CTX, d_model),
             "cache_last_time": t(1, n_layers, d_model, conv_context),
             "cache_last_channel_len": t(1, dtype=torch.int64)}
    lang0 = torch.zeros(1, dtype=torch.int64).to(dev)
    enc_fp32 = out / "encoder.fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(enc_wrap, (*dummy.values(), lang0), str(enc_fp32),
                          input_names=ENC_IN, output_names=ENC_OUT, opset_version=args.opset,
                          do_constant_folding=True, dynamo=False,
                          dynamic_axes={"audio_signal": {0: "batch", 1: "time"}, "length": {0: "batch"},
                                        "cache_last_channel": {0: "batch"}, "cache_last_time": {0: "batch"},
                                        "cache_last_channel_len": {0: "batch"}, "lang_id": {0: "batch"},
                                        "outputs": {0: "batch", 1: "time_out"}, "encoded_lengths": {0: "batch"},
                                        "cache_last_channel_next": {0: "batch"},
                                        "cache_last_time_next": {0: "batch"},
                                        "cache_last_channel_len_next": {0: "batch"}})
    enc_fp32_c = out / "encoder.fp32c.onnx"
    _consolidate(onnx, enc_fp32, enc_fp32_c)
    for f in out.glob("encoder.fp32.onnx*"):
        f.unlink()

    pd = {k: int(v) for k, v in OmegaConf.to_container(
        model.cfg.model_defaults.prompt_dictionary, resolve=True).items()}
    (out / "prompt_dictionary.json").write_text(json.dumps(pd, indent=2, ensure_ascii=False))
    feed = {k: v.cpu().numpy() for k, v in dummy.items()}
    verify_prompt(ort, enc_fp32_c, sorted(set(pd.values())), feed)
    print(f"  prompt_dictionary.json written ({len(pd)} languages); ady/kbd ride uk-UA/bg-BG slots")

    # Free torch weights before the (memory-hungry) quantization pass.
    enc_path = out / "encoder.onnx"
    if args.no_quantize:
        _consolidate(onnx, enc_fp32_c, enc_path)
    else:
        print(f"Quantizing encoder -> INT4 MatMulNBits ({args.quant_method}, block={args.block_size})...")
        quantize_encoder(enc_fp32_c, enc_path, args.quant_method, args.block_size, args.accuracy_level)
    for f in out.glob("encoder.fp32c.onnx*"):
        f.unlink()

    # --- decoder ---
    print("Exporting decoder...")
    dec_wrap = build_decoder_wrapper(torch, model.decoder)
    dec_tmp = out / "decoder.tmp.onnx"
    with torch.no_grad():
        torch.onnx.export(dec_wrap, (t(1, 1, dtype=torch.int64), t(dec_layers, 1, dec_hidden),
                                     t(dec_layers, 1, dec_hidden)), str(dec_tmp),
                          input_names=["targets", "h_in", "c_in"],
                          output_names=["decoder_output", "h_out", "c_out"], opset_version=args.opset,
                          do_constant_folding=True, dynamo=False,
                          dynamic_axes={"targets": {0: "batch", 1: "target_len"}, "h_in": {1: "batch"},
                                        "c_in": {1: "batch"}, "decoder_output": {0: "batch", 2: "target_len"},
                                        "h_out": {1: "batch"}, "c_out": {1: "batch"}})
    _consolidate(onnx, dec_tmp, out / "decoder.onnx")
    for f in out.glob("decoder.tmp.onnx*"):
        f.unlink()

    # --- joint ---
    print("Exporting joint...")
    joint_wrap = build_joint_wrapper(torch, model.joint)
    joint_tmp = out / "joint.tmp.onnx"
    with torch.no_grad():
        torch.onnx.export(joint_wrap, (torch.randn(1, 1, d_model).to(dev),
                                       torch.randn(1, 1, dec_hidden).to(dev)), str(joint_tmp),
                          input_names=["encoder_output", "decoder_output"], output_names=["joint_output"],
                          opset_version=args.opset, do_constant_folding=True, dynamo=False,
                          dynamic_axes={"encoder_output": {0: "batch", 1: "time"},
                                        "decoder_output": {0: "batch", 1: "target_len"},
                                        "joint_output": {0: "batch", 1: "time", 2: "target_len"}})
    _consolidate(onnx, joint_tmp, out / "joint.onnx")
    for f in out.glob("joint.tmp.onnx*"):
        f.unlink()

    # --- configs + tokenizer ---
    print("Writing configs + tokenizer...")
    write_configs(out, n_mels, n_fft, hop, win, preemph, vocab_size, blank_id,
                  d_model, n_layers, dec_hidden, dec_layers, chunk_samples, pre_encode,
                  int(conv_context), max_symbols, True, dither)
    export_tokenizer(out, model, vocab_size)

    print("\nBundle written to", out)
    total = 0
    for f in sorted(out.iterdir()):
        if f.is_file():
            total += f.stat().st_size
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"  TOTAL {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
