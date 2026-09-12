#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next with the static expert cache on the Arc GPU.
#
# Usage: scripts/serve.sh                # cache on (SLOTS from env.sh, default 48)
#        SLOTS=0 scripts/serve.sh        # stock baseline, no cache
#        USE_MTP=1 scripts/serve.sh      # add the MTP draft head on the CPU
#        SLOTS=56 CTX=32768 scripts/serve.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

mapfile -t ARGS < <(common_args)

CACHE=()
if [ "$SLOTS" -gt 0 ]; then
    [ -f "$PROFILE" ] || { echo "profile $PROFILE missing, run scripts/profile.sh first" >&2; exit 1; }
    CACHE=(--moe-cache-profile "$PROFILE" --moe-cache-slots "$SLOTS")
fi

SPEC=()
if [ "$USE_MTP" = "1" ]; then
    SPEC=(-md "$MTP" -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1)
fi

echo "== threads=$THREADS slots=$SLOTS ctx=$CTX mtp=$USE_MTP async_cpu=$ASYNC_CPU"
echo "== look for 'init_moe_expert_cache: expert cache: 48 layers x $SLOTS slots, ... uploaded to Vulkan0' in the log"
exec "$BIN/llama-server" -m "$MODEL" "${ARGS[@]}" "${CACHE[@]}" "${SPEC[@]}" \
    -fit off -c "$CTX" -np 1 --cache-reuse 256 --jinja \
    --host "$HOST" --port "$PORT" "$@"
