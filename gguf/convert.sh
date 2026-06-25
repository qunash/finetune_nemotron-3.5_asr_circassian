#!/usr/bin/env bash
# Convert a fine-tuned Nemotron-3.5-ASR .nemo to a GGUF bundle for parakeet.cpp.
#
# Why: parakeet.cpp runs this exact model (FastConformer-CacheAware-RNNT + prompt)
# on CPU with a tight C++ transducer loop and ggml block-quantized weights — much
# faster than the Python/ONNX streaming path, as one portable binary. It validates
# WER 0 vs NeMo per language. (https://github.com/mudler/parakeet.cpp)
#
# Pipeline: grab parakeet-cli (prebuilt) + the official converter -> .nemo to f32
# GGUF -> re-quantize to q8_0 (near-lossless) and any extra types you pass.
#
# Usage:
#   gguf/convert.sh /path/to/finetuned.nemo [out_dir] [quant ...]
# Examples:
#   gguf/convert.sh model.nemo gguf_out                # q8_0 (default, recommended)
#   gguf/convert.sh model.nemo gguf_out q8_0 q4_k      # also build the smallest
#
# Env:
#   PK_VERSION  parakeet.cpp release tag (default below); CLI + converter are pinned
#               together so the GGUF schema matches.
#   PK_DIR      where to place the binary + converter (default: ./third_party/parakeet.cpp)
#   PK_BUILD=1  build natively from source instead of the portable prebuilt binary —
#               enables host AVX2/AVX-512/VNNI for the best CPU speed (needs cmake + g++).
set -euo pipefail

PK_VERSION="${PK_VERSION:-v0.3.2}"
NEMO="${1:?usage: gguf/convert.sh <finetuned.nemo> [out_dir] [quant ...]}"
OUT="${2:-gguf_out}"
if [ "$#" -gt 2 ]; then shift 2; QUANTS=("$@"); else QUANTS=(q8_0); fi

PK_DIR="${PK_DIR:-$(pwd)/third_party/parakeet.cpp}"
mkdir -p "$PK_DIR" "$OUT"

# --- 1. parakeet-cli: native build (fastest) or portable prebuilt (no compiler) ---
CLI=""
if [ "${PK_BUILD:-0}" = "1" ]; then
    [ -d "$PK_DIR/src/.git" ] || git clone --recursive --branch "$PK_VERSION" \
        https://github.com/mudler/parakeet.cpp "$PK_DIR/src"
    cmake -S "$PK_DIR/src" -B "$PK_DIR/src/build" -DPARAKEET_BUILD_CLI=ON -DGGML_NATIVE=ON
    cmake --build "$PK_DIR/src/build" -j
    CLI="$PK_DIR/src/build/examples/cli/parakeet-cli"
else
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64)        asset="bin-linux-cpu-x64" ;;
        Linux/aarch64|Linux/arm64) asset="bin-linux-cpu-arm64" ;;
        Darwin/arm64)        asset="bin-macos-metal-arm64" ;;
        Darwin/x86_64)       asset="bin-macos-cpu-x64" ;;
        *) echo "no prebuilt for $(uname -s)/$(uname -m); re-run with PK_BUILD=1" >&2; exit 1 ;;
    esac
    if [ ! -d "$PK_DIR/bin" ]; then
        url="https://github.com/mudler/parakeet.cpp/releases/download/$PK_VERSION/parakeet-$PK_VERSION-$asset.tar.gz"
        echo "Fetching $url"
        mkdir -p "$PK_DIR/bin"
        curl -fsSL "$url" | tar -xz -C "$PK_DIR/bin"
    fi
    shopt -s globstar nullglob
    for c in "$PK_DIR"/bin/**/parakeet-cli "$PK_DIR"/bin/parakeet-cli; do CLI="$c"; break; done
fi
[ -x "$CLI" ] || { echo "parakeet-cli not found/executable at '$CLI'" >&2; exit 1; }
echo "Using CLI: $CLI"

# --- 2. Official converter (pinned to the same tag) + its one extra dep ---
CONV="$PK_DIR/convert_parakeet_to_gguf.py"
curl -fsSL "https://raw.githubusercontent.com/mudler/parakeet.cpp/$PK_VERSION/scripts/convert_parakeet_to_gguf.py" -o "$CONV"
python -c "import gguf" 2>/dev/null || pip install -q gguf

# --- 3. .nemo -> f32 GGUF (lossless reference), then quantize ---
F32="$OUT/nemotron.f32.gguf"
echo "Converting $NEMO -> $F32 (this loads NeMo; run in your fine-tuning env)"
python "$CONV" --model "$NEMO" --dtype f32 --output "$F32"

for q in "${QUANTS[@]}"; do
    [ "$q" = "f32" ] && continue
    "$CLI" quantize "$F32" "$OUT/nemotron.$q.gguf" "$q"
done

# --- 4. Report sizes and the exact --lang keys baked into the model ---
echo; echo "== GGUF bundle ($OUT) =="
ls -la "$OUT"/*.gguf | awk '{printf "  %-28s %7.1f MB\n", $NF, $5/1e6}'
echo; echo "== Languages you can pass to --lang =="
python - "$F32" <<'PY'
import sys
try:
    from gguf import GGUFReader
    r = GGUFReader(sys.argv[1])
    f = r.fields.get("parakeet.prompt.dictionary.keys")
    if not f:
        print("  (no prompt dictionary — model is not multilingual)")
    else:
        keys = [bytes(f.parts[f.data[i]]).decode("utf-8") for i in range(len(f.data))]
        print("  " + ", ".join(keys))
except Exception as e:
    print(f"  (could not read keys: {e}); try: parakeet-cli info {sys.argv[1]}")
PY
echo; echo "Done. Transcribe with: gguf/transcribe.py --model $OUT/nemotron.${QUANTS[0]}.gguf --input clip.wav --lang ady"
