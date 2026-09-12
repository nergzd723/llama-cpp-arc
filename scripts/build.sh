#!/usr/bin/env bash
# Build thecodacus/llama.cpp (static MoE expert cache) for an Intel Arc GPU.
#
# The fork's `perf` branch carries a mis-merged TurboQuant change in the Vulkan
# shaders that breaks every Vulkan build. The expert cache itself never touches
# the Vulkan backend, so this script pins the fork at a known commit and then
# restores the untouched upstream ggml-vulkan directory from the exact upstream
# commit that the fork merged. Everything else in the fork stays as-is.
#
# Usage:
#   scripts/build.sh                 # Vulkan build (recommended for Arc B580)
#   BACKEND=sycl scripts/build.sh    # SYCL build, needs oneAPI, untested here
#
# Env overrides: SRC (source dir), BUILD_DIR (default build), JOBS (parallel jobs), LLAMA_CURL (ON/OFF)
set -euo pipefail

FORK_URL=https://github.com/thecodacus/llama.cpp
FORK_SHA=27c54b4bbcefadedcec6397477cc2e866c1db716      # branch perf, 2026-09-07
UPSTREAM_URL=https://github.com/ggml-org/llama.cpp
UPSTREAM_SHA=8c1a25166b6b1339edd635165c7d8fd65326ae82  # upstream merge point of that commit, 2026-09-03

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$REPO_ROOT/llama.cpp}
BACKEND=${BACKEND:-vulkan}
BUILD_DIR=${BUILD_DIR:-build}
JOBS=${JOBS:-$(nproc)}
LLAMA_CURL=${LLAMA_CURL:-OFF}

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 "$FORK_URL" "$SRC"
fi
cd "$SRC"

echo "== pinning fork at $FORK_SHA"
git fetch --depth 1 origin "$FORK_SHA"
git checkout -q --detach "$FORK_SHA"

echo "== restoring upstream ggml-vulkan from $UPSTREAM_SHA"
git fetch --depth 1 "$UPSTREAM_URL" "$UPSTREAM_SHA"
git checkout FETCH_HEAD -- ggml/src/ggml-vulkan
git status --short ggml/src/ggml-vulkan | head -5

# Backend-neutral fixes the fork lacks, kept as patches so the tree stays reproducible.
# Currently: scripts/patches/sycl-moe-cache-negative-ids.patch teaches the SYCL mul_mat_id
# about the -1 ids of the hot/cold expert split (otherwise the expert cache aborts on SYCL).
for p in "$REPO_ROOT"/scripts/patches/*.patch; do
    [ -e "$p" ] || continue
    if git apply --reverse --check "$p" >/dev/null 2>&1; then
        echo "== patch already applied: $(basename "$p")"
    else
        echo "== applying patch: $(basename "$p")"
        git apply "$p"
    fi
done

COMMON=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL="$LLAMA_CURL"
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON)

case "$BACKEND" in
    vulkan)
        # Ubuntu deps: cmake build-essential libvulkan-dev glslc spirv-headers glslang-tools mesa-vulkan-drivers vulkan-tools
        cmake -B "$BUILD_DIR" "${COMMON[@]}" -DGGML_VULKAN=ON
        ;;
    sycl)
        # Intel oneAPI Base Toolkit required. Not verified in this repo; see README.
        # setvars.sh reads unset variables (OCL_ICD_FILENAMES), so relax -u around it
        set +u
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh
        set -u
        cmake -B "$BUILD_DIR" "${COMMON[@]}" -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
              -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
        ;;
    *)
        echo "unknown BACKEND=$BACKEND (vulkan|sycl)" >&2; exit 1
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS" --target llama-server llama-cli llama-bench llama-moe-trace
echo "== built:"
ls -1 "$BUILD_DIR"/bin/llama-server "$BUILD_DIR"/bin/llama-cli "$BUILD_DIR"/bin/llama-bench "$BUILD_DIR"/bin/llama-moe-trace
