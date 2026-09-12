# Qwen3.8 Flash Next on an Arc B580 12 GB: can the "47.00 tok/s on public moe-cache" claim be matched with Vulkan or SYCL?

Date: 2026-09-12. Sources checked: GenerelSchwerz/llama.cpp wiki (Notable Runs, Flash Next 47 Run Evidence, CUDA MoE Expert Cache, Grouped MoE Decode, 12/16 GB setup guides), the `moe-cache` branch source, thecodacus/llama.cpp `perf` and `fable5/*` branches, upstream PR ggml-org/llama.cpp#27861, the unsloth GGUF tensor headers, Qwen3.8-Flash-Next config.json, and llama.cpp discussions #12570 / #23313 on Arc B580.

## Short answer

No. The 47 tok/s figure cannot be matched on a B580 with Vulkan or SYCL, and the branch that produced it cannot run on either backend at all.

* The GenerelSchwerz `moe-cache` implementation is a CUDA runtime, not a portable llama.cpp feature.
* The measurement used 16 GB of VRAM (peak 14.4 GB), a Core Ultra 9 285K and 64 GB DDR5. It does not fit in 12 GB even on CUDA.
* Two other implementations of the same idea are backend-neutral in code. One of them reports 16.6 to 24.4 tok/s on an RTX 3060 12 GB with this exact model and quant. That is the realistic ceiling class for a B580, and it is untested on Intel.

Expected on a B580 with a dual-channel DDR5 desktop CPU:

| Configuration | Decode tok/s |
|---|---|
| Stock llama.cpp, Vulkan or SYCL, all experts on CPU | 12 to 18 |
| Plus backend-neutral hot-expert cache, if it works on Intel | 15 to 24 |
| Plus MTP draft on CPU | times 1.1 to 1.3 |
| GenerelSchwerz 47.00 on RTX 5070 Ti 16 GB | not reachable |

## What the claim actually is

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB |
| CPU, RAM | Intel Core Ultra 9 285K, 64 GB |
| Backend | CUDA 13.3, `moe-cache` branch commit b46f7f7a4 |
| Model | unsloth Qwen3.8-Flash-Next UD-Q3_K_XL, 90 GB |
| Expert cache | 80 slots per layer |
| Loading | `--load-mode none --lazy-mode on` |
| Context, batch | 12288, 4096; KV q8_0 |
| Workload | 158-token prompt, 1024 generated tokens |
| Decode | 46.99 tok/s |
| Prefill | 95.31 tok/s |
| Peak VRAM | 14,384 MiB |
| Minimum free RAM during run | 2,290 MiB |
| Speculative decoding, prefetch | both off |

The run is documented with server logs and looks genuine. It is a single short-prompt, single-stream measurement, and the wiki itself says it is not a universal speedup. The fork's own 12 GB guide does not list Qwen3.8 at all.

## Why the fork's cache cannot run on Vulkan or SYCL

* The whole feature lives in `ggml/src/ggml-cuda/moe-cache.cu` (11,387 lines), its header (1,120 lines) and 78 hook sites in `ggml-cuda.cu`.
* It uses cudaMemcpyAsync and cudaMemcpyBatchAsync, CUDA streams and events, cudaMallocHost pinned memory, CUDA graph reuse, three custom kernels for grouped plan/gather decode, and mapped MMQ/MMVQ dispatch.
* llama.cpp reaches it only through registry lookups named `ggml_backend_moe_cache_*` that only the CUDA backend registers. On Vulkan or SYCL the flag silently does nothing.
* The wiki states: "Do not use the expert-cache flags on Vulkan, ROCm/HIP, Metal, SYCL, CPU-only, or another non-CUDA backend."
* Porting means rewriting the grouped-decode runtime as Vulkan compute shaders and SYCL kernels. That is months of work, not a build flag.

## The two backend-neutral alternatives

### thecodacus/llama.cpp, static profile cache

* Branch `perf` (default, commit 27c54b4, 2026-09-07). A routing profile picks the hottest experts per layer; they are packed into VRAM once at load; decode runs hot experts on GPU and cold ones on CPU; results are merged exactly.
* The code is backend-neutral: it takes the first GPU device, allocates through `ggml_backend_alloc_ctx_tensors_from_buft`, fills slabs with `ggml_backend_tensor_set`, and the graph is only `get_rows`, `mul_mat_id` and `add`. The README line "uploaded to CUDA0" is just the buffer name on the author's machine.
* It carries `--lazy-mode`, `--load-mode`, `--spec-type draft-mtp`, and `llama-moe-trace` (which records routing through the scheduler eval callback, so it is backend-neutral too).
* Reported for this model: RTX 3060 12 GB, UD-IQ3_XXS, 56 slots plus MTP: 16.6 to 24.4 tok/s.
* Problem found in this session: `perf` does not compile with `-DGGML_VULKAN=ON`. The Vulkan shader tree is mis-merged. `flash_attn_base.glsl` duplicates two helpers that now live in `fa_types.glsl`, and after removing those, `types.glsl` redefines the TurboQuant macros. The older `fable5/moe-expert-cache` branch (2026-07-24) has the cache but no `qwen4exp` architecture and no `--lazy-mode`, so it cannot load Qwen3.8 anyway. Upstream master has `qwen4exp`, `--lazy-mode` and `draft-mtp`, but no expert cache.
* The two env-var prefill optimizations in the README are CUDA-only.

### Upstream PR ggml-org/llama.cpp#27861, dynamic LRU cache

* 407 lines in `src/llama-moecache.cpp`, no backend-specific code. Tested by contributors on Vulkan (AMD RX 7600: +15 to 20%; dual RX 6950XT: +116%), HIP and SYCL.
* Reported for this model: 18.4 to 24.2 tok/s on two RTX 3090.
* Limits: decode-only, so it does not stack with MTP; Vulkan top-k fusion misfire (workaround `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`); SYCL in-place reorder conflict (workaround `GGML_SYCL_ENABLE_OPT=0`); slot exhaustion under slow uploads; still a draft. Its base predates upstream `--lazy-mode`, so the 26.8 GiB n-gram table must be mmapped.

## Does the model itself run on Vulkan and SYCL?

The `qwen4exp` graph needs GATED_DELTA_NET, SOLVE_TRI, TRI, CUMSUM, SSM_CONV, TOP_K, FILL, SGN, SOFTPLUS, multi-rope, flash attention with q8_0 KV, `mul_mat_id` over IQ2_S / IQ3_S / IQ4_NL, and I32 `get_rows`. Both backends implement all of them in current source. The SYCL docs list the Arc B580 as supported. Mesa ANV on Battlemage exposes cooperative matrix and integer dot product, and B580 users report Vulkan working out of the box.

One caution: a Vulkan garbage-output report for Qwen3.5-35B-A3B on an Intel Meteor Lake iGPU was closed unresolved. Compare a temperature-0 answer against CPU-only output before trusting any Intel run.

## Sizing on 12 GB, from the real UD-IQ3_XXS tensor headers

| Component | Size |
|---|---|
| Routed experts, 48 layers | 44.7 GiB (47 layers IQ2_S gate/up + IQ4_NL down, 1 layer IQ3_S) |
| One expert slab (gate+up+down) | 1.86 MiB mean, 2.22 MiB max |
| Dense per-layer weights on GPU | 3.2 GiB total |
| Output head | 0.5 GiB |
| Token embeddings (stay on CPU) | 0.5 GiB |
| N-gram PLE table | 26.8 GiB IQ4_NL, read lazily from disk |
| KV at 16k / 64k context, q8_0 | about 0.2 / 0.8 GiB |

Slot budget on the B580: about 11.5 GiB usable, minus 3.7 GiB dense, 0.8 GiB compute, 0.9 GiB headroom and KV, leaves 5.3 to 5.9 GiB. At 1.86 MiB per slab across 48 layers that is roughly 60 slots per layer at 16k context and 55 at 64k. The 48 slots in your codacus command fit.

RAM: experts plus token embeddings are about 46 GiB resident, plus the 2.8 GB MTP draft, KV and OS. 64 GB works. UD-Q4_K_XL scales the experts by about 1.5x to roughly 68 GiB, so it wants 96 GB or it will page through mmap.

## Vulkan or SYCL on the B580?

Start with Vulkan.

* Community results on the B580 consistently favour Vulkan; prompt processing is reported at roughly twice SYCL.
* Cooperative matrix works on Mesa ANV without tuning.
* Both backend-neutral caches were exercised on Vulkan by contributors; neither was on Intel SYCL without workarounds.
* SYCL is Intel's official path, lists the B580, and is sometimes a few percent faster at token generation on MoE models, but it has more garbage-output reports in 2026 and its reorder optimization does not cover IQ types.

For this workload the GPU backend barely matters during decode. Experts run on the CPU, so CPU cores and RAM bandwidth set the number. Bench both backends with `llama-bench` and keep whichever is faster and correct.

## Recipe

Build Vulkan:

```
cmake -B build -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench llama-moe-trace
```

Baseline, stock flags:

```
./build/bin/llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 --n-cpu-moe 99 --load-mode mmap --lazy-mode on -fa on \
  -ctk q8_0 -ctv q8_0 -c 16384 -b 2048 -ub 512 --jinja
```

Static cache, once a Vulkan-buildable thecodacus branch exists:

```
MOE_TRACE_OUT=qwen38-chat.csv ./build/bin/llama-moe-trace -m <gguf> -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 -p "<chat prompt>"
MOE_TRACE_OUT=qwen38-code.csv ./build/bin/llama-moe-trace -m <gguf> -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 -p "<code prompt>"
cat qwen38-chat.csv qwen38-code.csv > qwen38-merged.csv
./build/bin/llama-server -m <gguf> --moe-cache-profile qwen38-merged.csv --moe-cache-slots 48 \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 --n-cpu-moe 99 -t <physical cores> --load-mode mmap --lazy-mode on -fit off -fa on \
  -ctk q8_0 -ctv q8_0 -c 16384 -np 1 -b 2048 -ub 512 --jinja
```

Then verify: same prompt at temperature 0 with `--moe-cache-slots 0` and with 48; outputs must match. If they differ on Vulkan, retry with `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`.

## Not verified here

No Intel GPU was available, so nothing ran on a B580. The static cache has never been reported on Intel hardware. Build results above are compile-only.
