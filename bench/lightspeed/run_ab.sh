#!/bin/bash
D=~/ai/bench/lightspeed; cd $D
S=~/ai/Strata
echo "start $(date)"; free -m | grep Swap > swap_before.txt; grep pswp /proc/vmstat >> swap_before.txt
sudo -n systemctl stop comfyui && echo comfy stopped
~/ai/scripts/h3-mode.sh on
nvidia-smi dmon -s pucvmt -d 1 > dmon.log 2>&1 & DM=$!
vmstat -t 1 > vmstat.log 2>&1 & VM=$!
mpstat -P ALL 1 > mpstat.log 2>&1 & MP=$!
for cyc in 1 2 3 4 5 6; do for cfg in A B C; do
  tag=${cfg}_c$cyc
  E="STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1"
  [ $cfg = B ] && E="$E STRATA_ADAPT_OVERLAP=1"
  [ $cfg = C ] && E="$E STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_OVERLAP_RESIDENT=1"
  echo "$tag env: $E" | tee ${tag}_env.txt
  rm -f engine_current.log
  (cd $S; env -u STRATA_IQ_MT_MIN $E .venv/bin/python serve/server.py --engine strata --config $D/strata-iq3_s-bench.json --port 8080 > $D/${tag}_server_stdout.log 2>&1 & echo $! > $D/srv.pid)
  for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/v1/models | grep -q loaded && break; sleep 5; done
  sleep 3
  python3 bench2.py $tag prompt.txt 1 "{}" > ${tag}_result.jsonl 2>&1
  sleep 2
  kill $(cat srv.pid); for i in $(seq 1 30); do pgrep -f "Strata/engine/strata" >/dev/null || break; sleep 2; done
  pkill -f "Strata/engine/strata"; sleep 3
  cp engine_current.log ${tag}_engine.log
  free -m | grep Swap >> swap_per_session.txt
done; done
kill $DM $VM $MP
free -m | grep Swap > swap_after.txt; grep pswp /proc/vmstat >> swap_after.txt
~/ai/scripts/h3-mode.sh off
sudo -n systemctl start comfyui && echo comfy started
echo "ALLDONE $(date)"
