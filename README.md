# llama-cpp-arc

Run **Qwen3.8 Flash Next** (177B on disk, 6B active) on an **Intel Arc B580 12 GB** with **48 GB of system RAM**, reproducing the setup from the Codacus video *"Is Frontier Class Local AI Finally Practical?"*: his static MoE expert cache in `thecodacus/llama.cpp`, the IQ3_XXS quant, memory-mapped weights, and one thread per physical core. The video got 16.6 tok/s stock and 24.4 tok/s with the cache on an RTX 3060 12 GB. This repo does the same on Intel through the Vulkan backend.

## Reality check first

| Setup | Decode tok/s | Notes |
|---|---|---|
| Video: RTX 3060 12 GB, 6-core Ryzen, DDR4, CUDA | 16.6 stock, 24.4 cached | measured by the author |
| GenerelSchwerz fork, RTX 5070 Ti 16 GB, Core Ultra 9 285K | 47.0 | CUDA-only runtime, needs ~50 GB RAM pinned, 14.4 GB VRAM. Not portable |
| **This repo: Arc B580, 48 GB RAM, Vulkan** | expect 12 to 18 stock, 15 to 25 cached | untested on Intel by anyone; your CPU and RAM speed decide |

Decode with the experts on the CPU is bound by your CPU cores and RAM bandwidth, not the GPU. The cache moves the hottest experts onto the B580 so the CPU does less per token. Whatever you get, it will track the 3060 numbers only if your CPU is at least a modern 6-core with dual-channel memory.

## What is different from just cloning the fork

The fork's default `perf` branch does not compile with `-DGGML_VULKAN=ON`: a later TurboQuant change was mis-merged into the Vulkan shaders (duplicate helpers in `flash_attn_base.glsl`, macro redefinitions in `types.glsl`). The expert cache never touches the Vulkan backend, so `scripts/build.sh` pins the fork at commit `27c54b4` and restores the untouched upstream `ggml/src/ggml-vulkan` directory from upstream commit `8c1a251`, the exact commit the fork had merged. Nothing else changes. The only feature lost is TurboQuant KV types on Vulkan, which this setup does not use.

The cache code itself is backend-neutral: it picks the first GPU device, allocates the expert packs through the generic backend API, and the graph split is plain `get_rows`, `mul_mat_id` and `add`. The fork's log line says `uploaded to CUDA0` on NVIDIA; on this build it will say `Vulkan0`.

## Requirements

Hardware

- Arc B580 with **Resizable BAR enabled** in the BIOS. Without it, Arc loses 20 to 25 percent.
- 48 GB RAM. Do not run anything else heavy alongside; the model wants the page cache.
- NVMe SSD with about 100 GB free. The 27 GB n-gram table is read lazily from disk on every token, so a slow disk shows up as slow decode.

Software (Ubuntu 24.04 or newer)

- Kernel 6.12+ with the `xe` driver bound to the B580; 6.14+ preferred.
- Mesa 25.0+ with the ANV Vulkan driver; Mesa 26.1+ has the newer cooperative-matrix path for Battlemage. `vulkaninfo --summary` must list the B580.
- Packages: `cmake build-essential git libvulkan-dev glslc spirv-headers glslang-tools mesa-vulkan-drivers vulkan-tools python3`

## Steps

```bash
scripts/build.sh              # ~20 min; produces llama.cpp/build/bin/llama-server etc.
scripts/download-model.sh     # 82 GB IQ3_XXS + 1.9 GB MTP draft, into ./models
SLOTS=0 scripts/serve.sh      # 1. baseline, stock flags, no cache
scripts/profile.sh            # 2. record which experts fire (~5 min)
scripts/serve.sh              # 3. cache on, 48 slots
scripts/verify.sh             # 4. cache off vs on: outputs must be identical; prints tok/s
```

All knobs live in `scripts/env.sh` and can be overridden per run, for example `SLOTS=56 CTX=32768 USE_MTP=1 scripts/serve.sh`.

Confirm the cache engaged. The server log must contain:

```
init_moe_expert_cache: expert cache: 48 layers x 48 slots, ... MiB uploaded to Vulkan0
```

A warning instead means it fell back to baseline. `pack allocation failed` means too many slots for the VRAM left; the warning prints the maximum that fits.

## The 48 GB RAM plan

Sizes below come from the actual tensor headers of the unsloth UD-IQ3_XXS GGUF.

| Component | Size | Where it lives |
|---|---|---|
| Routed experts, 48 layers | 44.7 GiB | system RAM via mmap, page cache |
| One expert (gate+up+down) | 1.86 MiB | cached copies go to VRAM |
| Dense per-layer weights + output head | 3.7 GiB | VRAM |
| Token embeddings | 0.5 GiB | RAM |
| N-gram table | 26.8 GiB | SSD, rows read on demand |
| KV cache at 16k / 64k context, q8_0 | 0.2 / 0.8 GiB | VRAM |

The video's RAM sweep on this exact quant: decode holds full speed down to a 24 GB cap, because only the frequently routed experts need to stay resident. Prompt processing needs most of the experts and was near full speed at a 48 GB cap, over 100 tok/s at 40 GB, and 34 tok/s at 32 GB. Your 48 GB total leaves roughly 40 GB of page cache after the OS and the server process, so expect full-speed decode and slightly reduced long-prompt speed.

Rules that follow from this:

- Keep `--load-mode mmap`. Never use `--load-mode none` or `mlock` on 48 GB; the experts alone exceed usable RAM.
- Stay on **UD-IQ3_XXS**. UD-Q4_K_XL grows the experts to about 68 GiB, which still decodes but streams 25+ GB from SSD on every long prompt, the same cliff the video hit at a 32 GB cap.
- Leave MTP off at first. It costs ~2 GB of RAM and bought under 1 tok/s in the video because the CPU side is the bottleneck.
- Do not run a second model at the same time. The video's planner/worker swap needs both to share RAM; on 48 GB, use one.

## The 12 GB VRAM plan

Usable VRAM is about 11.5 GiB. Dense weights take 3.7 GiB, compute buffers about 0.8 GiB, and keep 0.9 GiB free for prompt workspace. At 16k context that leaves room for about 60 slots per layer; at 64k, about 55. Start at 48 as in the video and raise until `pack allocation failed`, then back off by 4. More slots means a higher fraction of routed experts served from the GPU.

## Threads

`scripts/env.sh` sets `-t` to the number of physical cores. This is the single biggest setting in the video: 12 threads on a 6-core CPU spent 65 percent of the time spin-waiting and ran at 6 tok/s; 6 threads ran at 24.4. Override with `THREADS=n` if the auto-detect is wrong.

## If something is off

- **Outputs differ between cache off and on** in `scripts/verify.sh`: retry with `GGML_VK_DISABLE_FUSION=1`, then `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`. If they still differ, do not use the cache and report it upstream with the two output files.
- **Garbage text even with the cache off**: compare against `-ngl 0` on the same prompt. Intel Vulkan had an unresolved garbage-output report for a Qwen3.5 MoE on Meteor Lake graphics. If CPU-only is fine and Vulkan is not, try a newer Mesa or the SYCL build.
- **A second GPU or the iGPU shows up first**: set `GGML_VK_VISIBLE_DEVICES=<index of the B580>` from `vulkaninfo --summary`.
- **Slow first prompt after start**: the page cache is cold; the second prompt is the real number.
- **Speed swings between runs**: check `THREADS`, then `ASYNC_CPU=1` versus `0`.

## SYCL instead of Vulkan

`BACKEND=sycl scripts/build.sh` configures the same tree against oneAPI. SYCL is Intel's official path and lists the B580, and one community comparison had it a few percent faster at MoE token generation, but B580 users report Vulkan as the more reliable and faster backend overall and the fork's SYCL tree carries unverified TurboQuant additions. Nothing in this repo was tested with SYCL. If you try it and outputs are wrong with the cache on, set `GGML_SYCL_ENABLE_OPT=0`.

## Fallbacks that work today without this fork

- Stock upstream `ggml-org/llama.cpp` master with `-DGGML_VULKAN=ON` runs the model with the baseline flags in `scripts/env.sh`; it has `qwen4exp`, lazy n-gram loading and `--spec-type draft-mtp`, just no expert cache.
- Upstream PR ggml-org/llama.cpp#27861 is a backend-neutral dynamic LRU expert cache that contributors ran on Vulkan and SYCL. It is decode-only, still a draft, and it does not stack with MTP.

## Sources

- Codacus, "Is Frontier Class Local AI Finally Practical?" (YouTube, 2026-09-06) and the fork https://github.com/thecodacus/llama.cpp
- GenerelSchwerz fork wiki, "Notable Runs" and "CUDA MoE Expert Cache" pages
- unsloth/Qwen3.8-Flash-Next-GGUF on Hugging Face
- ggml-org/llama.cpp discussions #12570 and #23313 on Arc B580 Vulkan and SYCL performance

See `docs/analysis.md` for the full analysis of the 47 tok/s claim and the three cache implementations.
