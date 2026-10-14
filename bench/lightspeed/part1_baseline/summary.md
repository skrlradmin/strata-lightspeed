# Strata baseline on Harpy, 2026-10-14 (PT)
- Commit a0116b2782a5b82622358c04a06496485d12b2d6 (main, clean tree, no local diff). Engine 0.1.43.1. Release build, CMAKE_CUDA_ARCHITECTURES=120 (sm_120)
- HW: Ryzen 7 9850X3D (8C/16T), RTX 5090 32 GB (403 W cap left unchanged), 2x32 GB DDR5 G.Skill F5-6000J3036G32G running at 6000 MT/s; PCIe Gen5 x16 under load (pcie_underload.csv)
- SW: driver 595.99.02, CUDA 13.2 (driver), nvcc 13.1.115; uname/os-release/gcc are in software.txt
- Launch: ~/ai/Strata/run-iq3_s.sh -> serve/server.py --engine strata --config strata-iq3_s.json --port 8080
  engine: engine/strata --serve --pack Strata-data/packs/iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf ...-00002-of-00002.gguf --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp Strata-data/mtp/rt --max-context 32768 --kv int8 --resident-experts
- Env: no STRATA_* vars set (STRATA_IQ_MT_MIN unset = default). Engine env adds CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0.
- pcie_frac: 0.55 at startup (PCIe probe 52.5 GB/s), then auto-refined to 0.40 on the first request. Expert cache auto = 12160 slots (23.08 GiB). Experts in RAM: 27.17 GiB page-locked. CPU pool: 7 workers + host thread. Adaptive overlap off. Startup warning: only 282 MiB of VRAM free.
- Request: server /v1/chat/completions, temperature 0 (greedy), max_tokens 512, thinking off. All runs hit the same warm server process.
- Prompt: prompt.txt (sha256 a4f68532...), 67 tokens. Every run generated exactly 512 tokens (finish=length).
## Set A (bench.py, runs 1-5). Only run 1 had a cold prompt cache; runs 2-5 reused 60 tokens and read 7.
| run | prefill tok/s | decode tok/s | drafts acc/off |
|1|293.7 (67 read)|180.5|286/447|
|2|295.4*|198.2|280/472|
|3|316.7*|202.0|288/464|
|4|284.6*|205.6|289/471|
|5|267.2*|198.9|279/476|
Median: prefill 293.7, decode 198.9. * = 7-token suffix only, so not a real prefill measurement.
## Set B (bench2.py nocache, cache_prompt:false was ignored, so all 5 runs reused 60 tokens): decode 210.5/205.3/201.7/207.3/213.9, median 207.3
## Long prompt (prompt_long.txt, 6241 tokens, cold): prefill 4088 tok/s (1527 ms), decode 179.3 tok/s, drafts 281/494
- run1 output: run1_output.txt sha256 75dcfd9352879e47eb645fcf85ddc6d48c3f182d4141f56e42ddd805fcbf2334. NOTE: the 512-token greedy outputs differ in every run (sha256 per run is in the *_stdout.log files).
- Monitoring: dmon.log, mpstat.log, vmstat.log (set B + long run only). For set A: gpu_during.csv, top_during.log. Under load the GPU averaged ~366 W, ~45 C, SM clock ~2860 MHz.
- Other load: gnome-shell ~99% of one core, ComfyUI (916 MiB VRAM, ~20% CPU), grok-bot ~25%. Swap: 4.5 GB already used before the runs.
