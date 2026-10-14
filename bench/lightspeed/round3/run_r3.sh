#!/bin/bash
D=~/ai/bench/lightspeed/round3; cd $D; S=~/ai/Strata
cp ../bench2.py ../strata-iq3_s-bench.json ../prompt.txt ../part1_baseline/prompt_long.txt .
sed -i 's#/home/dutch/ai/bench/lightspeed/engine_current.log#/home/dutch/ai/bench/lightspeed/round3/engine_current.log#' strata-iq3_s-bench.json
sha256sum prompt.txt prompt_long.txt > prompts.sha256
echo "start $(date)"; { free -m | grep Swap; grep pswp /proc/vmstat; } > swap_before.txt
systemctl is-active comfyui llama-27b > services_before.txt
sudo -n systemctl stop comfyui && echo comfy stopped
~/ai/scripts/h3-mode.sh on
nvidia-smi dmon -s pucvmt -d 1 > dmon.log 2>&1 & DM=$!
vmstat -t 1 > vmstat.log 2>&1 & VM=$!
mpstat -P ALL 1 > mpstat.log 2>&1 & MP=$!
session() { # tag cfg promptfile nreq extraenv
  tag=$1; cfg=$2; pf=$3; n=$4; E="STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1 $5"
  [ $cfg = C ] && E="$E STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_OVERLAP_RESIDENT=1"
  echo "$tag cfg=$cfg prompt=$pf env: $E" | tee ${tag}_env.txt
  rm -f engine_current.log
  (cd $S; env -u STRATA_IQ_MT_MIN $E .venv/bin/python serve/server.py --engine strata --config $D/strata-iq3_s-bench.json --port 8080 > $D/${tag}_server_stdout.log 2>&1 & echo $! > $D/srv.pid)
  for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/v1/models | grep -q loaded && break; sleep 5; done
  sleep 3
  python3 bench2.py $tag $pf $n "{}" > ${tag}_result.jsonl 2>&1
  sleep 2
  kill $(cat srv.pid); for i in $(seq 1 30); do pgrep -f "Strata/engine/strata" >/dev/null || break; sleep 2; done
  pkill -f "Strata/engine/strata"; sleep 3
  cp engine_current.log ${tag}_engine.log; rm -f ${tag}_pcie_underload.txt
  free -m | grep Swap | sed "s/^/$tag /" >> swap_per_session.txt
}
for i in 1 2 3; do session prof_short_$i A prompt.txt 1 "STRATA_VERIFY_PROFILE=1"; done
session prof_long_1 A prompt_long.txt 1 "STRATA_VERIFY_PROFILE=1"
for len in short long; do pf=prompt.txt; [ $len = long ] && pf=prompt_long.txt
  for p in 1 2 3 4 5 6; do
    if [ $((p%2)) = 1 ]; then o="A C"; else o="C A"; fi
    for c in $o; do session ${len}_p${p}_$c $c $pf 2 ""; done
  done
done
kill $DM $VM $MP
{ free -m | grep Swap; grep pswp /proc/vmstat; } > swap_after.txt
rm -f srv.pid engine_current.log
~/ai/scripts/h3-mode.sh off
sudo -n systemctl start comfyui && echo comfy started
systemctl is-active comfyui llama-27b > services_after.txt
echo "ALLDONE $(date)"
