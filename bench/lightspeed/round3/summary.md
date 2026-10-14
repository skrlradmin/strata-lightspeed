# Round 3, Harpy, 2026-10-14 13:46-14:03 PT
Setup: commit a0116b2, the baseline command plus --vram-reserve-mib 960 (strata-iq3_s-bench.json), STRATA_DECODE_TIMING=1 and STRATA_PREFILL_TIMING=1 in every run, temperature 0, 512 tokens. ComfyUI and llama-27b were stopped during the runs and restarted afterwards. The server was restarted before every session.

## 1. Profile (config A + STRATA_VERIFY_PROFILE=1; kept separate from the throughput numbers; the profiler itself slows decode)
- Decode tok/s under the profiler: short 167.9/167.8/167.8, long 159.5
- Short prompt, per window (2.25 tokens/window, 228 windows): 13.38 ms = verify 11.0 [GPU-reach wait 9.16 + per-layer host 1.22 (CPU pool 1.00, plan 0.07, actq 0.04) + stage 0.1] + draft 1.31 + commit 0.01
- GPU stages, total 11.15 ms/window, grouped:
  - attention/mixer 3.28 (GDN 2.02, QSA 1.26, of which QSA attention kernel 0.50)
  - hyper-connections (hc norm/down/up) 2.87
  - resident (VRAM) experts 1.72
  - shared experts + quant 0.80
  - waits A/B 0.70
  - router + plan 0.52
  - copy + combine 0.40
  - head 0.38
  - waitCPU 0.32
  - PCIe experts 0.15
  - gap 0.03
- Long prompt (6241 tokens): 13.48 ms/window, 2.15 tokens/window, GPU-reach wait 9.51, CPU 0.86; the GPU total is 11.06 with attention 0.56 and scores+topk 0.24
## 2. Balanced A vs C (6 pairs per length, order AC/CA alternating; restart, then 1 discarded warm-up, then the measured request)
Note: the warm-up fills the prompt cache, so the measured request reuses 60/67 (short) or 6234/6241 (long) tokens. Prefill below is therefore taken from the cold warm-up request.
- short A: decode median 201.2 [196.4-205.9], prefill 290.4
- short C: decode median 207.2 [197.4-209.0], prefill 296.3
  - per-pair C-A: +5.1, +6.3, +5.6, +0.9, -3.9, +0.6 % (sign test p=0.22, Wilcoxon exact p=0.16)
- long A: decode median 203.6 [189.1-203.9], prefill 4041
- long C: decode median 203.9 [188.8-204.5], prefill 4040
  - per-pair C-A: +0.2, +0.3, +3.3, +0.0, +0.2, -0.2 % (sign test p=0.22, Wilcoxon exact p=0.09)
- sha256: all cold warm-up outputs on the short prompt are identical (c3c4d308b4). The measured, cache-reusing outputs vary in both configs (short: A 2 variants, C 3; long: 2-3), and the long warm-ups have 2 variants. So the greedy output is not bit-stable across runs, in A as well as C.
- Per-run detail (drafts, tokens/window, CPU pool ms, cache hit, PCIe experts, pcie_frac start/final): results.csv
