#!/bin/bash
# chain_sweep.sh QPID (node-138, evq-v45; run detached): once the v45 queue QPID has exited (its EXIT trap has released the
# servers, judge and tunnels), restart them as on 10-31 (same cards, ports, adapter, pid files; release-lab.txt is
# release.txt without the judge, which sweep_queue.sh releases only when the node-136 queue q136.pid is gone too) and
# start sweep_queue.sh detached: default bcp + val again (resume: only missing / infra rows run), then val of the arms
# w20, d1, d2, d3, w5, w2, w1, linear (the last four resume the H100's rows, h100_copy.sh), then bcp of the arms linear,
# w1, w2, w5, w20 (bcp d3, d2, d1: the node-136 queue). A card someone else holds is left alone (serve_f2.sh skips a card with more than
# 4000 MiB used; start_judge.sh needs BUDGET + 6 GiB free on node-67 G0/G1): the sweep runs on the servers that answer,
# and without the judge or any server it releases what started and stops. Log: chain.log here.
X=/mnt/ssd4/user/search-v9/logs/evq-v45; D=/mnt/ssd4/user/search-v9; cd $X
QPID=${1:?QPID}; A=/home/user/octo-env/models/adapter-v45-search-v9-rest
G68=user@10.0.1.68; G134=user@198.51.100.134; G67=user@198.51.100.67
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
log() { echo "$(date '+%F %T %Z') $*" >> chain.log; }
log "waiting for queue $QPID"
while kill -0 $QPID 2>/dev/null; do sleep 60; done
log "queue $QPID exited; queue.log: $(tail -n 2 $D/eval/queue.log | tr '\n' ' ' | cut -c1-300)"
sleep 30
log "cards after release: 138 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
log "68 $($SSH -n $G68 'nvidia-smi -i 4,5 --query-gpu=index,memory.used --format=csv,noheader' 2>&1 | tr '\n' ' ')"
log "134 $($SSH -n $G134 'nvidia-smi -i 6,7 --query-gpu=index,memory.used --format=csv,noheader' 2>&1 | tr '\n' ' ')"
log "67 $($SSH -n $G67 'nvidia-smi -i 0,1 --query-gpu=index,memory.used,memory.free --format=csv,noheader' 2>&1 | tr '\n' ' ')"
JL="$G67 /home/user/octo-env/search-v9-rest/judge-1030/judge.pid sglang.launch_server.*--port 32500"
export JUDGE_RELEASE=$JL JUDGE_PEERS=$X/q136.pid
j=$($SSH -n $G67 'curl -sf -m 5 http://10.0.2.167:32500/v1/models > /dev/null && echo "judge pid (running)"' 2>&1)
for i in $(seq 24); do  # the judge first (unless one runs): wait up to 2 h for room on node-67 G0/G1
  [[ $j == "judge pid"* ]] && break
  j=$($SSH -n $G67 'cd /home/user/octo-env/search-v9-rest/judge-1030 && JUDGE_GPUS=0,1 bash start_judge.sh' 2>&1); log "judge: $j"
  [[ $j == "judge pid"* ]] || sleep 300
done
if [[ $j == "judge pid"* ]]; then
  log "138: $(CU=/usr/local/cuda-12.9 SGLANG_PY=$D/sglang-venv/bin/python MODEL=$D/models/Qwen3.5-4B \
    SGL_PATH=$D/cursor-ba8072c/sglang:$D/runtime/code-local LORAS=rq1-v45=$A \
    bash serve_f2.sh $X 0:32420 1:32421 2:32422 3:32423 4:32424 2>&1 | tr '\n' ' ')"
  log "68: $($SSH $G68 "cd /home/user/octo-env/search-v9-rest && CU=/usr/local/cuda-12.4 LORAS=rq1-v45=$A bash -s -- \$PWD 4:32410 5:32411" \
    < serve_f2.sh 2>&1 | tr '\n' ' ')"
  log "134: $($SSH -n $G134 "R=/home/user/search-v9-134; cd \$R/eval-v45 && TMPDIR=/mnt/disk4/user/tmp CU=/usr/local/cuda-12.9 \
    SGLANG_PY=/home/user/octo-env/sglang-venv/bin/python MODEL=/home/user/octo-env/models/Qwen3.5-4B \
    SGL_PATH=\$R/cursor/sglang:\$R/code-9d3e21c4 LORAS=rq1-v45=$A \
    bash \$R/code-9d3e21c4/results/search-v9-rest-1028/scripts/serve_f2.sh \$PWD 6:32416 7:32417" 2>&1 | tr '\n' ' ')"
  T="ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  setsid $T -L 127.0.0.1:32410:127.0.0.1:32410 -L 127.0.0.1:32411:127.0.0.1:32411 $G68 > tunnel68.log 2>&1 < /dev/null & echo $! > tunnel68.pid
  setsid $T -L 127.0.0.1:32416:127.0.0.1:32416 -L 127.0.0.1:32417:127.0.0.1:32417 $G134 > tunnel134.log 2>&1 < /dev/null & echo $! > tunnel134.pid
  setsid $T -L 127.0.0.1:32500:10.0.2.167:32500 $G67 > tunnel67.log 2>&1 < /dev/null & echo $! > tunnel67.pid
  for i in $(seq 60); do  # up to 15 min for the servers and the judge
    up=""; for p in 32420 32421 32422 32423 32424 32410 32411 32416 32417; do
      curl -sf -m 5 http://127.0.0.1:$p/health > /dev/null && up="$up,127.0.0.1:$p"; done
    ju=$(curl -sf -m 5 http://127.0.0.1:32500/v1/models > /dev/null && echo up)
    [ "$(tr ',' '\n' <<< "${up#,}" | grep -c .)" -eq 9 ] && [ -n "$ju" ] && break; sleep 15
  done
fi
. ./env.sh; export SERVERS=${up#,} JUDGE_URL
if [ -z "${ju:-}" ] || [ -z "$SERVERS" ]; then
  log "judge ${ju:-down}, servers '${SERVERS}': releasing, no sweep"; SERVERS=${SERVERS:-none} RELEASE_ONLY=1 bash sweep_queue.sh release-lab.txt; exit 1
fi
log "servers up: $SERVERS; judge up"
E=()
for s in bcp val; do E+=(v45 $A $s -); done
for a in w20 d1 d2 d3 w5 w2 w1 linear; do E+=(v45 $A val $a); done
for a in linear w1 w2 w5 w20; do E+=(v45 $A bcp $a); done  # bcp d3 d2 d1: the node-136 queue
SELF_PIDFILE=$X/sweep.pid setsid nohup bash sweep_queue.sh release-lab.txt "${E[@]}" > sweep.out 2>&1 < /dev/null & echo $! > sweep.pid
log "sweep queue pid $(cat sweep.pid): ${#E[@]} args"
