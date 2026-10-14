# Lightspeed: paired A/B/C comparison, Harpy, 2026-10-14 13:00-13:10 PT
Commit a0116b2 (clean tree), engine 0.1.43.1, CUDA arch 120, driver 595.99.02 / CUDA 13.2 / nvcc 13.1.115. Hardware is in hardware.txt.
Command: serve/server.py --engine strata --config strata-iq3_s-bench.json --port 8080. The engine args are the baseline's plus --vram-reserve-mib 960, used in every run.
Stats env added in every run: STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1. STRATA_IQ_MT_MIN was unset.
A = baseline; B = +STRATA_ADAPT_OVERLAP=1; C = +STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_OVERLAP_RESIDENT=1
Order: interleaved ABC x 6 cycles. The server was restarted before every run (fresh process, cold prompt cache, all 67 prompt tokens read). Temperature 0, 512 tokens.
ComfyUI (comfyui.service) and llama-27b were stopped for the duration and both were restarted afterwards.
Overlap lines:
- A: "adaptive overlap: off: not in the resident RAM mode unless STRATA_ADAPT_OVERLAP_RESIDENT=1"
- B: "STRATA_ADAPT_OVERLAP=1 is off: not in the resident RAM mode unless STRATA_ADAPT_OVERLAP_RESIDENT=1" (so B behaves the same as A)
- C: "adaptive overlap (STRATA_ADAPT_OVERLAP=0 turns it off): the adaptive tier's copies land while the next window runs"
Decode tok/s, median [min-max]: A 179.8 [174.6-181.1], B 180.0 [176.0-180.5], C 182.7 [178.7-184.3]
Prefill tok/s, median: A 294.8, B 295.3, C 296.0
Per-cycle decode deltas vs A, cycles 1-6: B -0.7,-0.3,+3.4,-0.1,+0.7,-0.3 % / C +1.2,-0.6,+5.6,+3.4,-0.3,+1.7 %
All 18 outputs are identical (sha256 c3c4d308b4...). Drafts accepted 284-285 of 460-465, 2.24-2.25 tokens/window, cache hit 98.1%, pcie_frac 0.55 at start, then 0.45 or 0.40 after PcieAuto.
No bytes-moved figure exists for the CPU pool. Strata reports CPU-pool ms/window, CPU expert entries per layer-window and the count of experts read over PCIe (in results.csv and the decode timing lines).
Swap: 4422 MiB used before, 1932 MiB after.
Per-run details: results.csv. Raw logs: *_engine.log, *_server_stdout.log. Monitoring: dmon.log, vmstat.log, mpstat.log. Earlier baseline: part1_baseline/.

## Note on comparability

The 199-207 tok/s decode numbers from the first baseline (part1_baseline) and round 3 came from a warm server that reused cached prompt tokens. The ~180 tok/s numbers in round 2 came from a freshly restarted, fully cold server. The two methods aren't comparable, and the gap is not a regression. Byte-identical greedy output only holds on cold, freshly restarted runs.

Round 3 (round3/summary.md): the verify profile shows GPU compute dominates each window (GDN/mixer plus hyper-connections are ~55% of GPU time), while PCIe experts and CPU waits together are ~0.5 ms. Adaptive overlap (STRATA_ADAPT_OVERLAP=1 + STRATA_ADAPT_OVERLAP_RESIDENT=1) showed no statistically significant gain, so it stays experimental.
