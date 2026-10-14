# LIGHTSPEED_PLAN — Strata decode speedups for "Harpy"

Instructions for OpenAI Codex. Base commit: `mach10x/strata@1678de3`. Follow projects **in order**, **one branch + one PR per project**. Do not add other `.md` files.

## Target machine (Harpy)
- Ubuntu, RTX 5090 32 GB (sm_120), Ryzen 7 9850X3D (Zen 5, single CCD, full AVX-512 incl. VBMI/VNNI, 96 MB L3), 64 GB RAM.
- Workload: Qwen3.8-Flash-Next 125B IQ3_S + MTP, ~196 tok/s decode, experts partly in RAM (CPU pool + VRAM tier).

## Global rules (apply to every PR)
1. **Benchmark protocol:** fixed prompt file (`bench/lightspeed/prompt.txt`, committed in P0), temperature 0, 512 generated tokens, **median of 5 runs**, same flags every time. Report tok/s and the engine's own CPU-pool / per-token timing lines.
2. **Correctness:** greedy output must be **byte-identical** to the P0 baseline output (`bench/lightspeed/baseline_output.txt`). Note: with the default `STRATA_IQ_MT_MIN=2`, output can depend on how many drafts a verify window held (issue #152, comment in `native_expert.cpp:88-91`); keep every flag identical between runs. If a change alters rounding by design (e.g. P2 routing nt=1 differently), record a new baseline in that PR, explain why, and show parity tests still pass.
3. **Parity tests** must pass: build with `-DSTRATA_BUILD_TESTS=ON` (and `STRATA_NATIVE_EXPERTS=ON`, the default), run `ctest -R "native_expert_parity|expert_parity|iq_multi_parity|expert_multi_test|expert_cache_per_layer_test"`. Add new parity cases for every new kernel.
4. **Profiling:** `perf stat`/`perf record` for CPU work, `nsys` / `ncu` for GPU work. Attach summaries to the PR.
5. Every new behaviour gets an env/CLI switch to turn it off; default-on only if it wins the benchmark.
6. PR description: what changed, benchmark table (baseline vs PR, median + min/max), parity results, profile excerpt.

---

## P0 — Baseline (do first, branch `lightspeed/p0-baseline`)
**Scope:** reproducible benchmark harness, no engine changes.
**Files:** `bench/lightspeed/run.sh`, `bench/lightspeed/prompt.txt`, `bench/lightspeed/baseline_output.txt`, `bench/lightspeed/results/` (CSV/JSON only — no new .md).
**Steps:**
1. Build Release with CUDA (`CMAKE_CUDA_ARCHITECTURES` defaults to 120 at `CMakeLists.txt:77-78`) and `STRATA_BUILD_TESTS=ON`.
2. `run.sh`: runs `strata generate` 5x with the fixed prompt, temp 0, 512 tokens; captures tok/s, stderr timing lines (pool time, PCIe probe → `pcie_frac`), and output text; prints the median.
3. Instrument (env-gated, off by default, e.g. `STRATA_LS_STATS=1`): histogram of tokens per expert group `nt` passed to `native_gu_rows` / `native_down_rows`. **Verify the ~1.4 tokens/group figure** (comment at `native_expert.cpp:95-96`).
4. **Bound check for the CPU pool (critical for P2):** `perf stat -e cycles,instructions,` plus memory-bandwidth counters (or `perf mem`/AMD uProf) on the pool workers. Decide **memory-bound vs codebook-lookup-bound**: the source disagrees with itself — `native_expert.cpp:81-82` says "bound by the codebook lookups, ~5 GB/s per core", `iq_avx512.cpp:34-36` says "the pool is memory-bound" (~30 GB/s, 6 cores), and `expert_cache.hpp:3-5` measures 663.6 MB/token at ~40 GB/s. Record GB/s achieved vs Harpy's DDR5 peak, and whether the hot rows fit the 96 MB L3.
5. Record baseline numbers and the result of each check in `bench/lightspeed/results/p0.json`.
**Acceptance:** 5-run median is stable (spread < 2%); baseline output saved; nt histogram and bound analysis committed.

---

## P1 — Best-core detection & thread pinning (est. a few %), branch `lightspeed/p1-core-pin`
**Scope:** pin the main/launch thread to the highest-performance core, CPU pool workers to the other physical cores, avoid SMT siblings.
**Files:** new `src/platform/cpu_topology.{hpp,cpp}`; pool startup in `src/kernels/cpu/` (pool implementation, see `pool_test.cpp`); thread start in `src/program/generate.cpp`; CLI flag parsing (near `generate.cpp:1138`).
**Steps:**
1. Read `/sys/devices/system/cpu/cpu*/acpi_cppc/highest_perf` and `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`; fall back gracefully if absent.
2. Short boost probe (~50 ms per candidate core, only among the top-ranked CPPC cores) to break ties; cache the result per boot if cheap.
3. `pthread_setaffinity_np`: main thread → best core; workers → remaining physical cores, one per core, never the SMT sibling of the main core.
4. Flag `--no-core-pin` / `STRATA_NO_CORE_PIN=1` to disable; log the chosen layout at startup.
5. Unit test for the topology parser using fixture sysfs trees.
**Acceptance:** byte-identical output; parity tests pass; median tok/s ≥ baseline (target +1–4%); no regression with the flag off.

---

## P2 — AVX-512 IQ3_S kernel (est. 3–15% decode), branch `lightspeed/p2-iq3s-avx512`
**Context (verified @1678de3):**
- Dispatch: `native_gu_rows` in `src/kernels/cpu/native_expert.cpp:79-123` → `iq512_gu_rows` (line 103) when `nt >= mt_min` and AVX-512; else AVX2 `iq256_gu_rows`; else ggml `vec_dot` (line 111). `mt_min` = `STRATA_IQ_MT_MIN`, default 2 (line 92), so **1-token groups go to ggml vec_dot**. Down rows: `native_down_rows` lines 125+ (same rule, line 130).
- `src/kernels/cpu/iq_avx512.cpp`: IQ3_S = type 21, `Fmt<21>` (110-byte blocks). **Default path is 16 scalar `iq3s_grid` lookups + `_mm512_set_epi32`** (lines ~141-146). The `_mm512_i32gather_epi32` at line 139 is **opt-in** (`STRATA_IQ_GATHER=1`, line 33) and measured 3–5% slower on Zen 4.
- **An NT=1 path already exists** (`gu_rows_nt` case 1); it is just not reached by default because of `mt_min`. The comment at line 91 says MT_MIN=1 measured −1..−3% on IQ3_S — the new kernel must beat that.
- **Software prefetch already exists**: `STRATA_IQ_PREFETCH`, default 2048 bytes ahead, two `_mm_prefetch(..., _MM_HINT_T0)` per block (lines 34-39, 162-166).
- Inner loop per token: masked sub (sign) → `maddubs` → `madd` → add. `iq_avx2.cpp` uses `maddubs`/`madd`; no `dpbusd` anywhere in the CPU kernels.
**Steps:**
1. Depends on P0's bound check. If **memory-bound**, prioritise steps 4–5; if **lookup-bound**, prioritise steps 2–3.
2. Replace the **scalar lookups** (not the gather) for IQ3_S: the grid has 512 entries of 4 bytes. Implement the lookup with byte permutes (`_mm512_permutexvar_epi8` / `_mm512_permutex2var_epi8` = vpermb / vpermi2b) over the grid held in registers (split by byte lane / index bits as needed), or a 2-level decomposition. Keep results bit-identical to `iq3s_grid`.
3. Consider `_mm512_dpbusd_epi32` (VNNI) to fuse `maddubs+madd` where the scale application allows bit-exact int32 sums (check for int16 saturation in maddubs vs dpbusd — must match ggml's integer sums).
4. 1-token path: tune the NT=1 instantiation (decode cost is not amortised; fewer accumulators, more ILP across rows), then make it reachable for IQ3_S (e.g. per-type `mt_min` = 1 when the new kernel is active) **only if** it beats ggml vec_dot. Note this changes rounding vs ggml (#152) → new baseline justified in PR.
5. L3-aware tiling over rows (r0..r1 chunks sized for 96 MB L3 / per-worker share), tune `STRATA_IQ_PREFETCH` distance for Zen 5, try `_MM_HINT_NTA` / non-temporal streaming for rows used once.
6. Intrinsics first. Hand-written asm (`.S`) only for the inner loop, only if it beats the compiler measurably; keep intrinsics as fallback behind a switch.
7. Gate with `STRATA_IQ_LS=0` (fall back to current code). Compile-time check for AVX512VBMI; runtime check via `cpu_avx512_ok()`-style detection.
**Tests:** extend `native_expert_parity` (`--synthetic iq3_s/...`, `tests/data/native_experts/iq3_s.txt`), `iq_multi_parity`, `expert_parity` with nt=1..8 cases for the new path; microbench in `native_expert_parity`.
**Acceptance:** parity exact vs ggml integer sums; microbench faster at nt=1 and nt=2-3; end-to-end median ≥ +3%; output byte-identical (or new baseline justified per step 4).

---

## P3 — Async expert swaps (est. 3–10%), branch `lightspeed/p3-async-swaps`
**Context (verified @1678de3):**
- Adaptive swaps run every `adapt_every` rounds (default 4, `generate.cpp:359`, flag `--adapt-every`), up to `adapt_swaps` = 96 (line 381); triggered at `generate.cpp:5375` and `:6195`.
- Swap copies already run on a **non-blocking refill/adapt stream** (`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`, e.g. lines 2211, 4426, ~6029) with `adapt_ev`. The `apply_pending` lambda (`generate.cpp:~6037-6046`) waits/queries the event, commits, then does a **synchronous `cudaMemcpy` of the whole residency table** `d_res` (line ~6045). The same pattern exists in the single-session path near line 4430.
- Slot fills: `ExpertCache::fill_slot` uses `cudaMemcpyAsync` (`src/core/expert_cache.cpp:278`); `fill_slot_queued` at :327 on stream 0. `expert_cache.cpp:76` is a **HIP-only** blocking staging `cudaHostAlloc` (inside `#if defined(STRATA_USE_HIP)`), not the CUDA pinned arena.
- `pcie_frac`: default 0.55 for native packs (0.2 for the Q2_0 pack), scaled by a PCIe bandwidth probe (`generate.cpp:355, 1721-1734, 2236`).
**Steps:**
1. Double-buffer the device residency table: two `d_res` buffers, update the inactive one with `cudaMemcpyAsync` on the copy stream, flip the pointer with an event wait on the compute stream (no host sync).
2. Make sure the copy stream is a dedicated stream with low priority; host source buffers pinned.
3. Only copy the changed residency entries (scatter of the `pending` list) instead of the full table.
4. Per-layer `pcie_frac` tuning (measure per-layer miss rate vs PCIe time); keep the global flag as override.
5. Later (separate commit, behind a flag): router-ahead prefetch — start fetching layer L+1's likely experts once layer L's router is known.
**Tests:** `expert_cache_per_layer_test`, `expert_arena_load`, new test for the double-buffered table (consistency under concurrent swaps). nsys trace showing no host stall at the swap point.
**Acceptance:** byte-identical output; median ≥ +3%; nsys shows the sync `cudaMemcpy` gone from the decode critical path.

---

## P4 — GPU IQ3_S verify kernel for sm_120 (est. 0–10%, hard; profile first), branch `lightspeed/p4-sm120-mma`
**Context:** `src/kernels/cuda/iq_kernels.cu` is a dp4a transcription of llama.cpp (`ggml_cuda_dp4a` → `STRATA_DP4A`, lines 8, 54+), no mma, nothing sm_120-specific. Arch default 120 (`CMakeLists.txt:77-78`; CUDA ≥ 13 required for ≥120, line 105). Prefill uses ggml MMQ.
**Steps:**
1. Profile first with `ncu` on the verify window (MTP, multi-token): if IQ3_S GPU expert time < ~10% of a token, stop and report.
2. Write an `mma.sync` int8 (m16n8k32) kernel via inline PTX for the multi-token verify (nt = drafts+1), decoding IQ3_S into registers/shared memory.
3. Build with `120a` (`CMAKE_CUDA_ARCHITECTURES=120a`) behind an option; keep dp4a path as default fallback.
4. Parity test vs the dp4a kernel (new `*_parity.cpp` registered in CMake, like `iq_parity` / `mmvq_multi_parity`).
**Acceptance:** parity exact (integer sums); ncu shows kernel speedup; end-to-end median not slower; byte-identical output.

---

## Source facts checked @1678de3 (summary)
- Correct: dispatch in `native_expert.cpp` ~79-123; `iq512_gu_rows`; IQ3_S gather at `iq_avx512.cpp:139`; no dpbusd in `iq_avx2.cpp`; MT_MIN default 2; ~1.4 tokens/group (comment line 95-96, still verify); CPU pool largest single cost, 96% exposed (1.055 of 19.035 ms hidden); `cudaMemcpyAsync` at `expert_cache.cpp:278`; `pcie_frac` 0.55; `CMakeLists.txt:78`; tests `native_expert_parity`, `expert_parity`.
- Corrected: gather is opt-in (default is scalar lookups); NT=1 kernel exists but is gated by MT_MIN; prefetch exists (default 2048 B); `expert_cache.cpp:76` is HIP-only staging, not the pinned arena; adaptive code is at `generate.cpp:~6027-6062` and the sync `cudaMemcpy` at ~6045 (plus a twin path near 4424-4434); copies already use a non-blocking stream — only the residency table update is synchronous; "~19 tok/s" isn't in `expert_cache.hpp` — it says 19.035 **ms** pipeline time (16.2 ms pool on a ~53 ms token).
