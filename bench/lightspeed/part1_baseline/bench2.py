import json,time,urllib.request,hashlib,sys,subprocess
tag,pf,n,extra=sys.argv[1],sys.argv[2],int(sys.argv[3]),json.loads(sys.argv[4])
p=open(pf).read()
for i in range(1,n+1):
    body={"model":"strata","stream":False,"max_tokens":512,"temperature":0,"messages":[{"role":"user","content":p}],"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False},**extra}
    t0=time.perf_counter()
    if i==1: subprocess.Popen("sleep 1.5; nvidia-smi -q -d PCIE > %s_pcie_underload.txt"%tag,shell=True)
    raw=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=3600).read().decode()
    wall=time.perf_counter()-t0
    open(f"{tag}_run{i}_response.json","w").write(raw)
    j=json.loads(raw);txt=j["choices"][0]["message"].get("content") or ""
    open(f"{tag}_run{i}_output.txt","w").write(txt)
    print(json.dumps(dict(run=i,wall_s=round(wall,3),usage=j.get("usage"),timings=j.get("timings"),finish=j["choices"][0].get("finish_reason"),sha256=hashlib.sha256(txt.encode()).hexdigest())),flush=True)
