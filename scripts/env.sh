#!/usr/bin/env bash
# Shared settings for profile.sh / serve.sh / verify.sh. Override any of these in the environment.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# BACKEND=sycl selects the SYCL build (llama.cpp/build-sycl/bin, built with
# BACKEND=sycl BUILD_DIR=build-sycl scripts/build.sh) and loads the oneAPI environment;
# the default is the Vulkan build in llama.cpp/build/bin. BIN= overrides either.
BACKEND=${BACKEND:-vulkan}
if [ "$BACKEND" = "sycl" ]; then
    BIN=${BIN:-$REPO_ROOT/llama.cpp/build-sycl/bin}
    if [ -z "${ONEAPI_ROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u
    fi
    export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
else
    BIN=${BIN:-$REPO_ROOT/llama.cpp/build/bin}
fi
MODELS=${MODELS:-$REPO_ROOT/models}

MODEL=${MODEL:-$MODELS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}
MTP=${MTP:-$MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf}
PROFILE=${PROFILE:-$REPO_ROOT/profiles/qwen38-merged.csv}

# One thread per physical core. The video's author lost half his speed to
# hyperthreads spin-waiting; 12 threads on a 6-core CPU dropped him to 6 tok/s.
PHYS_CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
THREADS=${THREADS:-${PHYS_CORES:-6}}
THREADS_BATCH=${THREADS_BATCH:-$(nproc)}   # prefill may use all hardware threads

# 12 GB card: 48 slots is the video's setting. Each slot costs ~89 MiB across the
# 48 layers at IQ3_XXS (1.86 MiB per expert). ~60 fits at 16k context, ~55 at 64k.
SLOTS=${SLOTS:-48}
CTX=${CTX:-16384}

# Number of MoE layers whose routed experts stay in system RAM (--n-cpu-moe).
# 99 = all 48 layers on the CPU (the video's setting). Lower it to put whole
# expert layers into spare VRAM: each layer is ~950 MiB at IQ3_XXS.
NCPUMOE=${NCPUMOE:-99}

# 0 = keep the CPU cold chain synchronous (the setting you were given).
# 1 = fork default, overlaps CPU and GPU work, worth a few percent if stable.
ASYNC_CPU=${ASYNC_CPU:-0}

# 1 = add the MTP draft head on the CPU. In the video it bought under 1 tok/s and
# costs ~2 GB of RAM; leave it off until the base run is tuned.
USE_MTP=${USE_MTP:-0}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}

common_args() {
    local a=(
        -ngl 99 --n-cpu-moe "$NCPUMOE"
        -t "$THREADS" -tb "$THREADS_BATCH"
        --load-mode mmap
        -fa on -ctk q8_0 -ctv q8_0
        -b 2048 -ub 512
    )
    if [ "$ASYNC_CPU" = "0" ]; then a+=(--no-sched-async-cpu); fi
    printf '%s\n' "${a[@]}"
}
