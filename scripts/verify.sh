#!/usr/bin/env bash
# Correctness and speed check: run the same deterministic prompt with the cache
# off and on, diff the outputs, and print decode/prefill rates from the server.
# The cache is supposed to be bit-identical to baseline on CUDA; this is the
# check that it stays so on Vulkan/SYCL, where the fork has never been tested.
#
# Usage: scripts/verify.sh            (uses SLOTS from env.sh for the "on" run)
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

PROMPT=${PROMPT:-"Write a bash script that renames every .jpeg file in a directory tree to .jpg, skipping files whose target already exists, and explain each line briefly."}
N_PREDICT=${N_PREDICT:-256}
OUT=${OUT:-$REPO_ROOT/verify-out}
mkdir -p "$OUT"

run_once() {
    local slots=$1 tag=$2
    SLOTS=$slots "$(dirname "$0")/serve.sh" --port "$PORT" > "$OUT/server-$tag.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 600); do
        curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
        sleep 2
    done
    curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0, "seed": 42, "cache_prompt": False}))' "$PROMPT" "$N_PREDICT")" \
        > "$OUT/resp-$tag.json"
    python3 - "$OUT/resp-$tag.json" "$tag" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); t = r.get("timings", {})
print(f"{sys.argv[2]:>8}: prompt {t.get('prompt_per_second', 0):7.2f} tok/s   decode {t.get('predicted_per_second', 0):7.2f} tok/s   tokens {t.get('predicted_n', 0)}")
open(sys.argv[1] + ".txt", "w").write(r.get("content", ""))
EOF
    kill "$pid"; wait "$pid" 2>/dev/null || true
}

run_once 0 "cache-off"
run_once "$SLOTS" "cache-on"

if cmp -s "$OUT/resp-cache-off.json.txt" "$OUT/resp-cache-on.json.txt"; then
    echo "== outputs identical: cache is exact on this backend"
else
    echo "== OUTPUTS DIFFER: see $OUT/resp-cache-*.json.txt"
    echo "   retry with GGML_VK_DISABLE_FUSION=1 (Vulkan) or GGML_SYCL_ENABLE_OPT=0 (SYCL); if still different, do not use the cache"
fi
grep -h 'init_moe_expert_cache' "$OUT"/server-cache-on.log || echo "== cache did not engage, check server-cache-on.log"
