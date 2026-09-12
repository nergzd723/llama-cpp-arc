#!/usr/bin/env bash
# Download Qwen3.8-Flash-Next UD-IQ3_XXS (82 GB, 3 shards) and the shared MTP draft head.
#
# Usage: scripts/download-model.sh [models-dir]
# Env:   MTP_QUANT=Q4_K_M (1.9 GB, default) or Q8_0 (2.8 GB)
set -euo pipefail

MODELS=${1:-$(cd "$(dirname "$0")/.." && pwd)/models}
REPO=unsloth/Qwen3.8-Flash-Next-GGUF
MTP_QUANT=${MTP_QUANT:-Q4_K_M}
mkdir -p "$MODELS"

FILES=(
    "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
    "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf"
    "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf"
    "MTP/mtp-Qwen3.8-Flash-Next-shared-${MTP_QUANT}.gguf"
)

if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" "${FILES[@]}" --local-dir "$MODELS"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" "${FILES[@]}" --local-dir "$MODELS"
else
    for f in "${FILES[@]}"; do
        mkdir -p "$MODELS/$(dirname "$f")"
        curl -L -C - -o "$MODELS/$f" "https://huggingface.co/$REPO/resolve/main/$f"
    done
fi

echo "== done. Model entry point:"
echo "   $MODELS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
echo "   $MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-${MTP_QUANT}.gguf"
