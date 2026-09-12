#!/usr/bin/env bash
# Capture the routing profile the expert cache needs: which experts each layer
# picks for typical prompts. Two contrasting workloads are traced and merged;
# the fork's README says one merged CSV per model is enough. Takes a few
# minutes: each trace generates 512 tokens at your stock decode speed.
#
# Usage: scripts/profile.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

mkdir -p "$(dirname "$PROFILE")"
mapfile -t ARGS < <(common_args)

trace() {
    local out=$1 prompt=$2
    echo "== tracing -> $out"
    MOE_TRACE_OUT="$out" "$BIN/llama-moe-trace" -m "$MODEL" "${ARGS[@]}" \
        -c 4096 -n 512 -p "$prompt" >/dev/null
    echo "   rows: $(wc -l < "$out")"
}

trace "$(dirname "$PROFILE")/qwen38-chat.csv" \
"You are a helpful assistant. Explain to a friend, in plain language, how a mixture-of-experts language model routes tokens, why only a few experts run per token, and what that means for running large models on a home PC. Then answer three follow-up questions a curious beginner would ask."

trace "$(dirname "$PROFILE")/qwen38-code.csv" \
"Write a Python module implementing a token-bucket rate limiter with a small test suite using pytest. Include type hints, docstrings, and a short README section. Then refactor it to support multiple keys with an LRU-bounded dictionary and explain the trade-offs."

cat "$(dirname "$PROFILE")/qwen38-chat.csv" "$(dirname "$PROFILE")/qwen38-code.csv" > "$PROFILE"
echo "== merged profile: $PROFILE ($(wc -l < "$PROFILE") rows)"
