# Arc B580 run: <date>

## Machine

| Item | Value |
|---|---|
| CPU, physical cores / threads | |
| RAM, total / free before runs | |
| GPU, VRAM reported by `vulkaninfo` | |
| Resizable BAR | yes / no |
| Kernel, GPU driver module | |
| Mesa / ANV version (`vulkaninfo` driverInfo) | |
| Disk holding the model, type | |
| OS | |

## Build

| Item | Value |
|---|---|
| `scripts/build.sh` result | ok / failed at ... |
| Fork commit, upstream Vulkan commit | 27c54b4, 8c1a251 |
| `llama-cli --list-devices` | |

## Runs

Measure with the second of two identical requests, 256 tokens, temperature 0. One row per configuration.

| # | Config (SLOTS, THREADS, ASYNC_CPU, MTP, CTX) | Prompt tok/s | Decode tok/s | Cache engaged | Notes |
|---|---|---|---|---|---|
| 1 | baseline: SLOTS=0 | | | n/a | |
| 2 | SLOTS=48 | | | | |
| 3 | | | | | |

Long prompt (about 4000 tokens), best config: prompt ___ tok/s, decode ___ tok/s, SSD streaming observed: yes / no.

## Correctness

- `scripts/verify.sh`: identical / differ
- Env var needed, if any: none / `GGML_VK_DISABLE_FUSION=1` / `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`
- Baseline text quality on Vulkan vs `-ngl 0`: same / different

## Headline

Baseline ___ tok/s, cached ___ tok/s, best config ___. Blockers: none / <first error line and the step>.

## Logs

`results/logs/build.log`, `results/logs/baseline.log`, `results/logs/cache48.log`, `results/logs/verify.log`, others: ...
