import json,time,urllib.request,hashlib,statistics,sys
p=open("prompt.txt").read()
res=[]
for i in range(1,6):
    body={"model":"strata","stream":False,"max_tokens":512,"temperature":0,"messages":[{"role":"user","content":p}],"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False}}
    t0=time.perf_counter()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=3600)
    raw=r.read().decode(); wall=time.perf_counter()-t0
    open(f"run{i}_response.json","w").write(raw)
    j=json.loads(raw); t=j.get("timings",{}); u=j.get("usage",{})
    txt=j["choices"][0]["message"].get("content") or ""
    if i==1:
        open("run1_output.txt","w").write(txt)
    d=dict(run=i,wall_s=round(wall,3),usage=u,timings=t,finish=j["choices"][0].get("finish_reason"),sha256=hashlib.sha256(txt.encode()).hexdigest())
    print(json.dumps(d),flush=True); res.append(d)
json.dump(res,open("results.json","w"),indent=1)
for k in ("prompt_per_second","predicted_per_second"):
    v=[x["timings"].get(k) for x in res]
    print(k,v,"median",statistics.median([x for x in v if x is not None]) if any(v) else None)
