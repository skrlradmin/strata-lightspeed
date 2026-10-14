import json,re,glob,statistics,csv
rows=[]
for cyc in range(1,7):
  for cfg in "ABC":
    t=f"{cfg}_c{cyc}"
    r=json.loads(open(f"{t}_result.jsonl").read().strip().splitlines()[-1])
    log=open(f"{t}_engine.log").read()
    g=lambda p:(re.search(p,log) or [None,None])[1]
    tm=r["timings"]
    ov=[l for l in log.splitlines() if "adaptive overlap" in l or "STRATA_ADAPT_OVERLAP=1 is off" in l]
    pf=re.findall(r"pcie_frac ([0-9.]+)",log)
    dt=g(r"(strata decode timing:.*)")
    rows.append(dict(tag=t,cfg=cfg,cycle=cyc,prompt_tokens=r["usage"]["prompt_tokens"],cached=tm["cache_n"],gen_tokens=tm["predicted_n"],finish=r["finish"],
      prefill_ms=tm["prompt_ms"],prefill_tps=tm["prompt_per_second"],decode_ms=tm["predicted_ms"],decode_tps=tm["predicted_per_second"],
      drafts_acc=tm["draft_n_accepted"],drafts_off=tm["draft_n"],accept_rate=round(tm["draft_n_accepted"]/tm["draft_n"],3),
      tokens_per_window=g(r"([0-9.]+) tokens/window"),ms_per_window=g(r"([0-9.]+) ms/window"),gpu_wait_ms=g(r"GPU-reach wait ([0-9.]+)"),cpu_pool_ms=g(r"CPU ([0-9.]+)\]"),
      cache_hit=g(r"decode expert cache hit rate: ([0-9.]+)%"),pcie_experts=g(r"; ([0-9]+) more read by the GPU over PCIe"),
      pcie_frac_start=pf[0] if pf else None,pcie_frac_final=pf[-1] if pf else None,overlap_line=ov[-1] if ov else "",sha256=r["sha256"]))
with open("results.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for cfg in "ABC":
  d=[x["decode_tps"] for x in rows if x["cfg"]==cfg]; p=[x["prefill_tps"] for x in rows if x["cfg"]==cfg]
  print(cfg,"decode",d,"med",statistics.median(d),"min",min(d),"max",max(d),"| prefill med",statistics.median(p),min(p),max(p))
for cyc in range(1,7):
  a=[x for x in rows if x["cycle"]==cyc]
  print(cyc,"B-A %+.1f%%"%(100*(a[1]["decode_tps"]/a[0]["decode_tps"]-1)),"C-A %+.1f%%"%(100*(a[2]["decode_tps"]/a[0]["decode_tps"]-1)))
for x in rows: print(x["tag"],x["decode_tps"],x["prefill_tps"],x["gen_tokens"],x["cached"],x["drafts_acc"],x["drafts_off"],x["tokens_per_window"],x["cpu_pool_ms"],x["cache_hit"],x["pcie_experts"],x["pcie_frac_start"],x["pcie_frac_final"],x["sha256"][:10])
for c in "ABC": print(c,[x["overlap_line"] for x in rows if x["cfg"]==c][0])
