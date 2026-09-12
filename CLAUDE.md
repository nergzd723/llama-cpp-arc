# Runbook for an agent on the real machine

You are running inside a clone of `llama-cpp-arc` on the target machine: an Intel Arc B580 with 12 GB VRAM, 48 GB of system RAM, Linux, an NVMe SSD. Your job is to reproduce, on Vulkan, the Codacus video result for Qwen3.8 Flash Next: a stock baseline first, then the static expert cache from `thecodacus/llama.cpp`, then a tuned number. Everything in `scripts/` was written and compile-tested in a container without a GPU. Nothing has run on a B580 before you. Treat every step as an experiment and record what actually happens.

Read `README.md` first for the reasoning behind the flags and the RAM/VRAM budgets. Do not re-derive them.

## Ground rules

- One llama-server at a time. Kill the previous one before starting the next; the model wants all of the page cache.
- Never use `--load-mode none`, `mlock`, or the Q4 quant on this machine. The experts alone are 44.7 GiB at IQ3_XXS.
- Do not edit anything under `llama.cpp/` except through `scripts/build.sh`. If a source change is needed, write it up in the results file instead.
- Do not change BIOS settings, kernel, or drivers yourself. Report what is missing and stop at that step.
- Keep every server log. Put them under `results/logs/` and reference them in the report.
- Time budget: the model download is 84 GB, the build about 20 minutes, each server start about 1 to 3 minutes while the page cache warms. Plan for several hours, and check in with the user after the baseline number.

## Step 0: preflight

Run and paste the output of each into `results/<date>-b580.md` (start from `results/TEMPLATE.md`):

```bash
uname -r                                        # want 6.12+ for the xe driver
lspci -nn | grep -i -E 'vga|display|3d'          # expect an Intel Battlemage device, id 8086:e20b for the B580
lsmod | grep -E '^xe |^i915 '                   # want xe bound to the B580
sudo lspci -vvv -s "$(lspci -nn | grep -i -E 'e20b|battlemage' | cut -d' ' -f1)" | grep -i -E 'Resizable BAR|BAR 2|Region 2' | head -3
vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion' | head -8
lscpu | grep -E 'Model name|^CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core|Socket'
free -g
df -h .                                         # need ~100 GB free for the model
```

Go/no-go:

- `vulkaninfo` must list the B580 with the Intel open-source driver (`driverName` contains `Intel open-source Mesa driver`, ANV). If it lists only `llvmpipe`, the Vulkan driver is missing: install `mesa-vulkan-drivers`, then stop and report if it still does not appear.
- Resizable BAR should show a BAR of 12 GB or 16 GB. If it is 256 MB, note it; runs will still work but 20 to 25 percent slower. Do not try to change it yourself.
- RAM must be 48 GB or more with under 4 GB used by other processes.
- Note the physical core count. `scripts/env.sh` auto-detects it; confirm the value it prints later matches.

## Step 1: build

```bash
sudo apt-get install -y cmake build-essential git libvulkan-dev glslc spirv-headers glslang-tools mesa-vulkan-drivers vulkan-tools python3 python3-pip curl
scripts/build.sh 2>&1 | tee results/logs/build.log
llama.cpp/build/bin/llama-cli --list-devices
```

Success: four binaries listed at the end of the script, and `--list-devices` shows `Vulkan0` with the B580's name and roughly 12 GB. If the build fails in the Vulkan shaders, the fix in `scripts/build.sh` did not apply; check that `git status` inside `llama.cpp/` shows modified files under `ggml/src/ggml-vulkan/` and that the two SHAs printed by the script match the ones in the file. Do not try to hand-patch shaders; report the first error line.

## Step 2: model

```bash
pip install -U huggingface_hub      # provides the `hf` CLI
scripts/download-model.sh 2>&1 | tail -5
ls -l models/UD-IQ3_XXS/ models/MTP/
```

Expected sizes: shard 1 about 11 MB (metadata only), shard 2 about 49.6 GB, shard 3 about 32.4 GB, MTP shared Q4_K_M about 1.9 GB. If a shard is short, rerun the script; downloads resume.

## Step 3: baseline, no cache

```bash
SLOTS=0 scripts/serve.sh > results/logs/baseline.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health; do sleep 5; done
```

Check the log for:

- the Vulkan device line and `Vulkan0 model buffer size` near 3.7 GiB, `CPU_Mapped model buffer size` near 45 GiB;
- a line about the per-layer embedding table being read lazily, or no error about 26 GiB allocations;
- no `unsupported op` fallbacks scrolling during generation.

Then run the same request twice (the first warms the page cache, only the second counts):

```bash
for i in 1 2; do curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain in detail how a token-bucket rate limiter works and write one in Python.","n_predict":256,"temperature":0,"seed":42,"cache_prompt":false}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print(f"prompt {t[\"prompt_per_second\"]:.1f} tok/s  decode {t[\"predicted_per_second\"]:.1f} tok/s")'; done
```

Also read the generated text once. It must be coherent English and Python. If it is repeating punctuation or garbage, stop the cache work: rerun the same prompt with `-ngl 0` appended (`SLOTS=0 scripts/serve.sh -ngl 0`) to confirm the CPU path is fine, record both outputs, and report a Vulkan correctness problem with the Mesa version. Nothing after this step is meaningful on a backend that produces wrong text.

Record the baseline decode number. This is the number the cache has to beat; on the video's 3060 machine it was 16.6.

## Step 4: profile and cache

```bash
pkill -f llama-server; sleep 2
scripts/profile.sh 2>&1 | tee results/logs/profile.log      # two traces of 512 tokens each
wc -l profiles/qwen38-merged.csv                              # expect a few tens of thousands of rows
scripts/serve.sh > results/logs/cache48.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health; do sleep 5; done
grep -E 'init_moe_expert_cache|expert cache' results/logs/cache48.log
```

The line must read `init_moe_expert_cache: expert cache: 48 layers x 48 slots, <N> MiB uploaded to Vulkan0`. Expected N is about 4300. Any warning instead names the reason:

- `no GPU device`: the Vulkan backend did not load; go back to step 1.
- `no CPU-resident MoE layers`: `--n-cpu-moe 99` did not take effect; check the serve command in the log.
- `pack allocation failed`: too many slots; the warning prints the maximum, set `SLOTS` to that minus 4.
- `cannot open profile` or `no decode rows`: step 4's profile is missing or empty.

Then rerun the two-request measurement from step 3 and record the decode number. Then:

```bash
pkill -f llama-server; sleep 2
scripts/verify.sh 2>&1 | tee results/logs/verify.log
```

`verify.sh` must print `outputs identical`. If it prints `OUTPUTS DIFFER`, rerun it with `GGML_VK_DISABLE_FUSION=1` exported; if still different, with `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`. Record which, if any, made them identical and the speed cost. If neither does, the cache is not usable on this driver: report it with both output files and stop tuning.

## Step 5: tune

Change one knob at a time, measure with the same two-request method, and log every run in the results table. Order of expected payoff:

1. `THREADS`: physical cores first, then minus one, then plus two. The video lost 75 percent of its speed to over-subscription.
2. `SLOTS`: 48, 56, 64, until `pack allocation failed`, then back off by 4. Watch `results/logs/` for allocation failures on the first long prompt; the README says to keep about 900 MB free.
3. `ASYNC_CPU=1`: the fork's default overlap; worth a few percent if the numbers are stable across three runs.
4. `USE_MTP=1`: costs 2 GB of RAM; in the video it bought under 1 tok/s. Try it last.
5. `CTX`: 16384 for measurements; only raise it for real use, and re-check slots.

Also measure one long prompt of about 4000 tokens once in the best configuration, to record prompt-processing speed and whether the SSD is streaming (watch `iostat -x 5` or `free -g` during the prompt).

## Step 6: report

Fill in `results/<date>-b580.md` from the template, commit it together with the logs and the profile CSV, and push to the same branch you found this file on:

```bash
git add results profiles/qwen38-merged.csv
git commit -m "results: Arc B580 run on <date>"
git push
```

Put the headline in the commit body: baseline decode, cached decode, best configuration, verify result, and any blocker. If you stopped early, say at which step and paste the exact error.

## Optional: SYCL

Only if Vulkan fails correctness or is clearly slow, and only if oneAPI is already installed on the machine. Build into a separate directory so the Vulkan build is kept:

```bash
BACKEND=sycl BUILD_DIR=build-sycl scripts/build.sh
BIN=$PWD/llama.cpp/build-sycl/bin SLOTS=0 scripts/serve.sh
```

The fork's SYCL tree carries unverified TurboQuant additions and may not compile; a failure there is a finding, not something to fix here. If it runs, repeat steps 3 to 5 with `BIN` pointing at it and add a SYCL column to the results table. If outputs differ with the cache on, try `GGML_SYCL_ENABLE_OPT=0`.
