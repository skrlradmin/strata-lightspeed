# LIGHTSPEED_PLAN — Strata decode speedups for "Harpy"

Instructions for OpenAI Codex. **Base commit: `mach10x/strata@a0116b2782a5b82622358c04a06496485d12b2d6` (engine 0.1.43.1)**, the build Harpy runs; this fork's `main` contains that tree plus this file. All file:line references below were checked against a0116b2 — line numbers drift, so re-grep the quoted symbol before editing.
Follow projects **in order**, **one branch + one PR per project** (branch names below). Do not add other `.md` files.

## Target machine (Harpy)
- Ubuntu; RTX 5090 32 GB (sm_120, 403 W power cap), Driver 595.99.02, CUDA 13.2, nvcc 13.1.115, `CMAKE_CUDA_ARCHITECTURES=120`; PCIe Gen5 x16 (engine probe 52.5 GB/s).
- Ryzen 7 9850X3D (Zen 5, single CCD, full AVX-512 incl. VBMI/VNNI, 96 MB L3); 2x32 GB DDR5-6000.
- Workload: Qwen3.8-Flash-Next 125B IQ3_S + MTP, experts partly in RAM (resident RAM mode) and partly in a VRAM cache.

## Global rules (every PR)

### 1. Benchmark hygiene
- Close ComfyUI and every other GPU user (`nvidia-smi` must show only strata). Keep desktop/gnome-shell load low (no browser video, no compositing-heavy apps); ideally run from a TTY/SSH.
- Check swap before each set (`free -h`, `swapon --show`). Baseline had 4.5 GB swap in use: `sudo swapoff -a && sudo swapon -a` when RAM allows, and record swap usage in results.
- The engine prints a low-VRAM warning on Harpy and suggests `--vram-reserve-mib 960`. Pick one setting (recommended: always pass `--vram-reserve-mib 960`, or always omit it) and keep it **identical across baseline and PR runs**; record it.
- The server **ignores `cache_prompt:false`**: for every cold-prefill measurement, **restart the server** before the run. Decode-only sets may reuse one server but must say so.
- Same binary flags, same env (no `STRATA_*` vars unless the PR's feature needs one — then list it), same prompt files, temp 0, 512 generated tokens, **median of 5 runs** (report all 5 + min/max).

### 2. Correctness: token/logit tolerance, not byte identity
Greedy temp-0 output is **not** byte-identical run-to-run on the baseline itself (sha256 differed every run: the multi-token CPU kernels round differently by group size, and group sizes depend on MTP draft acceptance — see #152 comment at `src/kernels/cpu/native_expert.cpp:116-119`; GPU reductions/async timing add more). So:
1. **First measure the baseline's own run-to-run variance** (P0, step 4): N ≥ 10 runs, same prompt, recording per step the generated token id, the top-1 logit, and the top-k (k=5) logits — add an env-gated dump (e.g. `STRATA_LS_LOGITS=path`, off by default) if no equivalent exists.
2. From those runs compute, pairwise vs a reference run: (a) **first-divergence index** (first token that differs), (b) **top-1 agreement rate** over teacher-forced steps (feed the reference tokens, compare argmax), (c) **max |Δlogit|** and p99 |Δlogit| over the top-5 entries on teacher-forced steps.
3. **Acceptance for a PR** (teacher-forced on the reference token sequence, so divergence doesn't cascade): top-1 agreement ≥ baseline's own minimum pairwise agreement (and ≥ 99.5%); max |Δlogit| ≤ 1.5x the baseline's max run-to-run |Δlogit|; median first-divergence index (free-running) not worse than baseline's by more than its own spread. Thresholds live in `bench/lightspeed/tolerance.json` (from P0) and the PR must print the comparison.
4. Kernel-level **parity tests** remain exact where they are today (integer sums vs ggml); run with `-DSTRATA_BUILD_TESTS=ON` (`CMakeLists.txt:99`), `STRATA_NATIVE_EXPERTS=ON` (default, `:104`):
   `ctest -R "native_expert_parity|expert_parity|iq_multi_parity|expert_multi_test|expert_cache_per_layer_test|expert_arena_load|pool_affinity"`. Add new parity cases for every new kernel.

### 3. Profiling & PR content
`perf stat`/`perf record` for CPU, `nsys`/`ncu` for GPU. PR description: what changed; benchmark table (baseline vs PR); tolerance check output; parity results; profile excerpt. Every new behaviour has an off switch; default-on only if it wins.

---

## P0 — Baseline (branch `lightspeed/p0-baseline`)
**Measured baseline (Harpy, a0116b2, already done — import it, don't re-invent it):**
- Command: `engine/strata --serve --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S gguf --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --resident-experts` — no `STRATA_*` env vars.
- Runtime state: `pcie_frac` auto-dropped 0.55 → 0.40 at the first request (the 0.1.42 `PcieAuto` refinement, `src/program/generate.cpp:1698-1705`); 12,160 cache slots (23.08 GiB VRAM); 27.17 GiB experts pinned in RAM; 7 CPU-pool workers.
- Decode: **median 198.9 tok/s** (set A: 180.5 / 198.2 / 202.0 / 205.6 / 198.9) and **207.3 tok/s** (set B). Note the ~8 tok/s gap between sets and the 180.5 outlier: run-to-run noise is ~±4%, so per-PR gains below ~3% need more runs.
- Cold short-prompt prefill: 293.7 tok/s (67-token prompt). Long prompt (6,241 tokens): prefill 4,088 tok/s, decode 179.3 tok/s.
- MTP acceptance 60–64%; expert-cache hit rate 97.8–99.6%.
- Raw files on Harpy: `~/ai/bench/strata_baseline/` (copy into the repo as below).

**Files:** `bench/lightspeed/run.sh`, `bench/lightspeed/prompts/{short.txt,long_6241.txt,decode.txt}`, `bench/lightspeed/results/baseline_a0116b2.json`, `bench/lightspeed/tolerance.json`, `bench/lightspeed/compare.py` (no new .md).
**Steps:**
1. Build Release, CUDA arch 120 (default at `CMakeLists.txt:158-159`; CUDA ≥ 13 enforced for ≥120 at `:186`), `STRATA_BUILD_TESTS=ON`; run the ctest set above.
2. `run.sh`: starts the server with the exact command above (+ the chosen `--vram-reserve-mib` setting), applies the hygiene checks (GPU users, swap, load average) and refuses/warns if they fail; restarts the server for cold-prefill runs; 5 runs → median; captures stderr lines (PCIe probe, pcie_frac, slots, pool workers, hit rate, MTP acceptance).
3. Import `~/ai/bench/strata_baseline/` numbers into `baseline_a0116b2.json`; re-run once to confirm within noise.
4. **Variance study** (Global rule 2): ≥10 runs, logit dumps, compute pairwise metrics, write `tolerance.json`. `compare.py` does the teacher-forced comparison for later PRs.
5. Instrument (env-gated, e.g. `STRATA_LS_STATS=1`): histogram of `nt` per expert group in `native_gu_rows` / `native_down_rows` — **verify the ~1.4 tokens/group** claim (comment at `native_expert.cpp:161`).
6. **Bound check for the CPU pool** (decides P2's direction): `perf stat` + memory-bandwidth counters (AMD uProf or `perf stat -e` DF/UMC events) on pool workers. The source disagrees with itself: `native_expert.cpp:137` says "bound by the codebook lookups, ~5 GB/s per core"; `iq_avx512.cpp:34-35` says "the pool is memory-bound" (~30 GB/s, 6 cores); `include/strata/core/expert_cache.hpp:3-5` measures 663.6 MB/token at ~40 GB/s = 16.2 ms of a ~53 ms token, 96% not overlapped (1.055 of 19.035 ms hidden). Record achieved GB/s vs DDR5-6000 dual-channel peak (~96 GB/s) and whether hot rows fit 96 MB L3.
**Acceptance:** baseline + tolerance files committed; nt histogram and bound analysis in `results/`.

---

## P1 — Best-core detection & main-thread pinning (est. a few %), branch `lightspeed/p1-core-pin`
**Context (a0116b2):** the CPU pool **already pins workers one per physical core and avoids SMT siblings** (topology in `src/kernels/cpu/pool.cpp:~60-335`, Linux helper `src/kernels/cpu/pool_affinity_linux.hpp`, tests `pool_affinity_linux_test.cpp`, `pool_affinity_test.cpp`). It does **not** read ACPI CPPC rankings and does not place the main/launch thread on the best core (no `acpi_cppc`/`highest_perf` anywhere). So P1 = add ranking + main-thread placement on top of the existing topology code, not a rewrite.
**Files:** `src/kernels/cpu/pool.cpp`, `src/kernels/cpu/pool_affinity_linux.hpp`, main-thread setup in `src/program/generate.cpp` (CLI parsing near `:2034`), tests next to `pool_affinity_linux_test.cpp` (fixture sysfs trees).
**Steps:**
1. Read `/sys/devices/system/cpu/cpu*/acpi_cppc/highest_perf`; rank physical cores; fall back silently if absent.
2. Optional short boost probe (≤50 ms per top-ranked candidate) to break ties.
3. Pin the main/launch (GPU-driving) thread to the best core; exclude that core **and its SMT sibling** from the worker list (with 8 cores: main on 1, 7 workers — matches today's 7 workers).
4. `--no-core-pin` / `STRATA_NO_CORE_PIN=1` restores current behaviour; log the layout.
**Acceptance:** tolerance check passes; parity + pool_affinity tests pass; decode median ≥ baseline (target +1–4%, needs ≥10 runs given ±4% noise).

---

## P2 — AVX-512 IQ3_S kernel (est. 3–15% decode), branch `lightspeed/p2-iq3s-avx512`
**Context (a0116b2):**
- `int native_gu_mt_min(int gu_type)` at `native_expert.cpp:115-131`: `STRATA_IQ_MT_MIN` default 2 (`:120-121`). New in 0.1.40+: `STRATA_IQ3S_MT1=1` makes IQ3_S use the kernel for 1 token, **but only on AVX2-without-AVX-512 CPUs** (`!cpu_avx512_ok()`, `:128-129`) — on Harpy 1-token IQ3_S groups still go to ggml `vec_dot`.
- `native_gu_rows` at `:134`; AVX-512 dispatch `iq512_gu_rows` at `:173` (guard `nt >= mt_min && …` at `:171`); ggml fallback `vec_dot` at `:185`. `native_down_rows` at `:199`, its `mt_min` at `:204`, `vec_dot` at `:223`.
- `src/kernels/cpu/iq_avx512.cpp`: IQ3_S = `Fmt<21>` (`:130`, 110-byte blocks). **Default decode is 16 scalar `iq3s_grid` lookups + `_mm512_set_epi32`** (`:141-147`); the `_mm512_i32gather_epi32` at `:139` is **opt-in** (`STRATA_IQ_GATHER=1`, `:31-33`; measured 3–5% slower on Zen 4).
- **NT=1 kernel already exists** (`gu_rows_nt` `case 1`, `:217`) but is unreachable by default because of `mt_min`; comment `native_expert.cpp:119` says MT_MIN=1 measured −1..−3% on IQ3_S AVX-512 — the new kernel must beat that.
- **Software prefetch already exists**: `STRATA_IQ_PREFETCH`, default 2048 B (`iq_avx512.cpp:34-40`), two `_mm_prefetch(..., _MM_HINT_T0)` per block (`:166-167`).
- Inner loop per token: masked sub (sign) → `maddubs` → `madd` → add. No `dpbusd` in the IQ kernels (`iq_avx2.cpp`, `iq_avx512.cpp`); VNNI `dpbusd` is already used for Q2_0 in `src/kernels/cpu/expert.cpp:133,183,212` — reuse that style.
**Steps:**
1. Use P0's bound check: memory-bound → prioritise 5; lookup-bound → prioritise 2–4.
2. Replace the **scalar lookups** (not the gather) for IQ3_S: the 512 x 4-byte grid via byte permutes (`_mm512_permutexvar_epi8` / `_mm512_permutex2var_epi8`, VBMI) over register-resident grid slices, or a 2-level decomposition; bit-identical to `iq3s_grid`.
3. Try `_mm512_dpbusd_epi32` to fuse `maddubs+madd`; must be exact (watch `maddubs` int16 saturation vs dpbusd int32).
4. 1-token path: tune NT=1 (more rows in flight for ILP), then enable it on AVX-512 for IQ3_S (extend `native_gu_mt_min`, e.g. a `STRATA_IQ3S_MT1` path that doesn't require `!cpu_avx512_ok()`) **only if** faster than ggml `vec_dot`. This changes lone-token rounding → must pass the tolerance check, not byte identity.
5. L3-aware row tiling (per-worker share of 96 MB), retune `STRATA_IQ_PREFETCH` for Zen 5, try `_MM_HINT_NTA`/non-temporal loads for rows used once.
6. Intrinsics first; `.S` asm only for the inner loop if measurably faster, intrinsics kept as fallback.
7. Switch `STRATA_IQ_LS=0` → current code. Runtime VBMI check.
**Tests:** `native_expert_parity` (CMake `:1491`, synthetic pairs + `tests/data/native_experts/iq3_s.txt`), `iq_multi_parity` (`:802`), `expert_parity` (`:1574`), `expert_multi_test` (`:1563`); add nt=1..8 cases; microbench via `native_expert_parity`.
**Acceptance:** exact parity; microbench faster at nt=1 and nt=2–3; tolerance check passes; decode median ≥ +3%.

---

## P3 — Async expert swaps (est. 3–10%), branch `lightspeed/p3-async-swaps`
**Context (a0116b2):**
- `adapt_every` default 4 (`generate.cpp:581`), `adapt_swaps` 96 (`:611`); trigger sites `:9517`, `:11531`, `:12082`, `:13265`.
- Residency table upload: `res_put` lambda at `generate.cpp:6163-6166` = **synchronous `cudaMemcpy` of the whole `host_res` table + `cudaStreamSynchronize`**; called via `res_upload` (`:8405`) from `apply_pending` (`:8413`, sets/clears `pending`, ~`:8426-8433`), and in a second loop's `apply_pending` at `:12990`.
- Swap copies already run on a non-blocking `adapt_stream` with `adapt_ev` (e.g. `:8340`, `:12979`). **0.1.43 added `STRATA_ADAPT_OVERLAP=1`** (`adapt_overlap()`, `:1368-1385`, opt-in; per-layer `AdaptGate` in `strata/core/verify.hpp`), which hides copies behind the next window — **but it is off in resident RAM mode unless `STRATA_ADAPT_OVERLAP_RESIDENT=1`** (`:8366-8370`), and Harpy runs `--resident-experts`. It measured −21% on an RTX 5090 Laptop and +4–8% on A4000/P100.
- Slot fills: `ExpertCache::fill_slot` → `cudaMemcpyAsync` at `src/core/expert_cache.cpp:682`; `fill_slot_queued` at `:731` (stream 0). The `cudaHostAlloc` at `:451` is **HIP-only staging** (`#if defined(STRATA_USE_HIP)` `:447`).
- `pcie_frac`: default 0.55 for native packs (`generate.cpp:575`, `:2775`), scaled by `pcie_frac_for_gbps` (`:1694-1696`), then refined by `PcieAuto` (`:1698-1705`, up to 3 updates; Harpy goes 0.55 → 0.40). `--pcie-frac` at `:2034`.
**Steps:**
1. First A/B the existing switches on Harpy: `STRATA_ADAPT_OVERLAP=1` with and without `STRATA_ADAPT_OVERLAP_RESIDENT=1`; if one wins and passes tolerance, consider just changing defaults for this config.
2. Double-buffered device residency table: two `d_res` buffers, `cudaMemcpyAsync` into the inactive one on the adapt stream, event-wait on the compute stream, swap pointer — remove the `cudaStreamSynchronize` from `res_put` on the decode path.
3. Upload only the changed entries (scatter of `pending`) instead of the whole table.
4. Per-layer `pcie_frac` (extend `PcieAuto` per layer from miss rates); global flag stays an override.
5. Later, behind a flag: router-ahead prefetch of layer L+1's likely experts.
**Tests:** `expert_cache_per_layer_test` (`CMakeLists.txt:556`), `expert_arena_load` (`:326`), new test for table consistency under concurrent swaps; nsys trace showing no host stall at swap time.
**Acceptance:** tolerance check passes; decode median ≥ +3%; nsys shows the synchronous table copy gone from the critical path.

---

## P4 — GPU IQ3_S verify kernel for sm_120 (est. 0–10%, hard; profile first), branch `lightspeed/p4-sm120-mma`
**Context (a0116b2):** `src/kernels/cuda/iq_kernels.cu` still uses dp4a only (`#include "strata/kernels/dp4a.hpp"` `:8`, `ggml_cuda_dp4a` → `STRATA_DP4A` `:71`), no mma, nothing sm_120-specific. Arch default 120 (`CMakeLists.txt:159`). Prefill uses ggml MMQ.
**Steps:**
1. `ncu`/`nsys` on the MTP verify window: if IQ3_S GPU expert time < ~10% of a token, stop and report.
2. `mma.sync` int8 (m16n8k32) kernel via inline PTX for multi-token verify (nt = drafts+1), IQ3_S decoded into registers/shared.
3. Optional `120a` build (`CMAKE_CUDA_ARCHITECTURES=120a`) behind an option; dp4a remains default.
4. Parity test vs the dp4a kernel (new `*_parity.cpp` registered like `iq_multi_parity`).
**Acceptance:** exact integer parity; ncu speedup; tolerance check passes; decode median not slower.

---

## Source-fact log
- @1678de3 corrections (kept): gather opt-in; NT=1 exists but gated by MT_MIN; prefetch exists; `expert_cache.cpp` HIP-only `cudaHostAlloc`; "~19" is ms not tok/s; swap copies async, only the table upload is synchronous.
- @a0116b2 updates: native_expert dispatch moved (`native_gu_mt_min` 115-131, `native_gu_rows` 134, iq512 call 173, vec_dot 185, down 199/204/223); new `STRATA_IQ3S_MT1` (AVX2-only); `iq_avx512.cpp` lines unchanged except prefetch 166-167, Fmt<21> 130; pool already does SMT-aware worker pinning; `expert_cache.cpp` :76→:451 (HIP), :278→:682, :327→:731; CMake arch :78→:159, BUILD_TESTS :99, test registrations moved; generate.cpp swap code now `res_put` 6163 / `res_upload` 8405 / `apply_pending` 8413 & 12990 (old ~6040-6056 gone); `PcieAuto` (0.1.42) and `STRATA_ADAPT_OVERLAP` (0.1.43) are new.
