import json,re,glob,statistics as st,csv,itertools,math
rows=[]
for f in sorted(glob.glob("*_result.jsonl")):
    tag=f[:-13]; L=[json.loads(x) for x in open(f).read().strip().splitlines() if x.startswith("{")]
    log=open(tag+"_engine.log").read()
    dt=re.findall(r"tokens/window, ([0-9.]+) ms/window.*?GPU-reach wait ([0-9.]+).*?CPU ([0-9.]+)\]",log)
    tpw=re.findall(r"([0-9.]+) tokens/window",log); pf=re.findall(r"pcie_frac ([0-9.]+)",log)
    hit=re.findall(r"cache hit rate: ([0-9.]+)%",log); pe=re.findall(r"; ([0-9]+) more read by the GPU over PCIe",log)
    w,m=L[0],L[-1]
    parts=tag.split("_")
    rows.append(dict(tag=tag,phase="profile" if parts[0]=="prof" else "throughput",length=parts[1] if parts[0]=="prof" else parts[0],
      pair=parts[1] if parts[0]!="prof" else "",cfg="A" if parts[0]=="prof" else parts[2],
      prompt_tokens=m["usage"]["prompt_tokens"],cold_prefill_tps=w["timings"]["prompt_per_second"],cold_prefill_ms=w["timings"]["prompt_ms"],
      measured_cache_n=m["timings"]["cache_n"],gen_tokens=m["timings"]["predicted_n"],decode_tps=m["timings"]["predicted_per_second"],decode_ms=m["timings"]["predicted_ms"],
      drafts_acc=m["timings"]["draft_n_accepted"],drafts_off=m["timings"]["draft_n"],tokens_per_window=tpw[-1] if tpw else "",ms_per_window=dt[-1][0] if dt else "",
      gpu_wait_ms=dt[-1][1] if dt else "",cpu_pool_ms=dt[-1][2] if dt else "",cache_hit=hit[-1] if hit else "",pcie_experts=pe[-1] if pe else "",
      pcie_frac_start=pf[0] if pf else "",pcie_frac_final=pf[-1] if pf else "",sha256_warmup=w["sha256"],sha256=m["sha256"]))
with open("results.csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def wil(d):
    d=[x for x in d if x!=0];n=len(d);r=sorted(range(n),key=lambda i:abs(d[i]));rk=[0]*n
    for j,i in enumerate(r): rk[i]=j+1
    W=sum(rk[i] for i in range(n) if d[i]>0);mu=n*(n+1)/4
    tot=0;ext=0
    for s in itertools.product([0,1],repeat=n):
        w=sum(rk[i] for i in range(n) if s[i]);tot+=1;ext+= abs(w-mu)>=abs(W-mu)-1e-9
    return W,ext/tot
def sign(d):
    d=[x for x in d if x!=0];n=len(d);k=sum(x>0 for x in d);k=min(k,n-k)
    return min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
for L in ("short","long"):
    t=[r for r in rows if r["phase"]=="throughput" and r["length"]==L]
    for c in "AC":
        d=[r["decode_tps"] for r in t if r["cfg"]==c];p=[r["cold_prefill_tps"] for r in t if r["cfg"]==c]
        print(L,c,"decode med %.1f [%.1f-%.1f]"%(st.median(d),min(d),max(d)),"prefill(cold warmup) med %.1f [%.1f-%.1f]"%(st.median(p),min(p),max(p)),"tpw",sorted(set(r["tokens_per_window"] for r in t if r["cfg"]==c)),"sha",sorted(set(r["sha256"][:10] for r in t if r["cfg"]==c)),"warm sha",sorted(set(r["sha256_warmup"][:10] for r in t if r["cfg"]==c)),"cache_n",sorted(set(r["measured_cache_n"] for r in t if r["cfg"]==c)))
    dl=[]
    for p in range(1,7):
        a=[r for r in t if r["pair"]==f"p{p}" and r["cfg"]=="A"][0];c=[r for r in t if r["pair"]==f"p{p}" and r["cfg"]=="C"][0]
        dl.append(c["decode_tps"]-a["decode_tps"]);print(" pair",p,"AC" if p%2 else "CA","A %.1f C %.1f delta %+.1f (%+.1f%%)"%(a["decode_tps"],c["decode_tps"],dl[-1],100*dl[-1]/a["decode_tps"]))
    print(" sign p=%.3f  wilcoxon W=%s p=%.3f"%(sign(dl),*wil(dl)))
for r in rows:
    if r["phase"]=="profile": print(r["tag"],r["decode_tps"],r["cold_prefill_tps"],r["prompt_tokens"],r["sha256"][:10])
