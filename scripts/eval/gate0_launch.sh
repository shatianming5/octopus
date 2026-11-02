#!/bin/bash
# Gate 0 in one command (docs/cross-branch-read.md "Gate 0"; scripts/eval/gate0_shard.sh runs each shard). From a
# controller with ssh to node-94 (H94), node-136 (H136), node-68 (H68) and node-134 (H134); it only sends commands, data
# moves between the hosts on the LAN. SHARD_HOSTS ("136 68 68"; node-94's CPU is IRAOD's): one shard per entry, in order
# (a host listed twice runs two); each shard runs 2 x CONC (6) episodes (gate0_shard.sh), 36 in all at ~1.2 cores each.
# GPUs are shared with this user's IRAOD jobs (user, 10-22): pick-gpus takes a GPU that holds nothing (<= 1000 MiB) or
# only this user's iraod processes, samples it PICK_S (60) s, keeps RESERVE_MIB (10240) above its peak plus ACT_MIB (4096)
# for SGLang's activations, and sets the host's MEM_FRAC so that every picked GPU keeps that room; a GPU whose SGLang
# share would fall under MIN_MIB (16384) is left out. Other users' GPUs are never picked.
#   gate0_launch.sh start SHA     1. checks that every shard host reaches node-136 through node-134 (the done markers go
#                                 there); 2. clears node-136's old markers; 3. starts each serving host's release watch
#                                 (serve-watch, MAX_H); 4. starts SGLang (base model) on the picked GPUs of node-136 and
#                                 node-94, each start bounded by START_S (1800) s and failing on a dead server, and aborts
#                                 when a server's KV pool cannot hold its busiest load (MIN_KV, KV_PER_EP); 5. starts the shards,
#                                 each host with its shard-wait. Any exit before step 5 is done (a failed start, Ctrl-C, a
#                                 killed controller, a dropped ssh) marks abort, through node-183 and through the shard hosts
#                                 -> node-134, and the watches stop both hosts' servers
#   gate0_launch.sh status SHA    shard rows, markers, GPUs
#   gate0_launch.sh stop SHA      stops the dev_evals, their episodes, the tunnels and the servers (abort, or past MAX_H)
#   gate0_launch.sh collect SHA   every shard's rows and trial dirs to node-134 under the same paths; fails listing the
#                                 missing episodes unless every bundle has 4 scored episodes in both arms; then gate0.py
# On the hosts (start and stop run these): shard-wait OUT SHARD... marks each shard done on node-136 once its dev_evals
# exited (at once when the shard never started one), then stops this host's tunnel; serve-watch OUT NSHARDS stops this
# host's servers when all shards are done, on abort, or after MAX_H hours, and writes nvidia-smi to OUT/released.txt;
# stop-host OUT; check OUT (on node-134); pick-gpus (on a serving host). MAX_H (24): track A smoke episodes took 289-335 s
# on average (718 s at most), ~7.5 min in the dc6e4d6 run; 2312 episodes at 36 at once are ~64 rounds, ~8 h at 7.5 min,
# ~11 h if shared GPUs make them 10 min, plus a last episode's agent timeout (up to 1 h).
# Every host has CODE=/home/user/harbor-udocker/code-SHA and OUT=/home/user/octo-env/gate0-SHA. node-94 and node-68
# reach node-136 only through node-134 (198.51.100.134); node-94's servers listen on 10.0.1.94. DRY=1: print the
# remote commands.
set -euo pipefail
H94=${H94:-"-J node-138-intern user@10.0.1.94"} H136=${H136:-"-J node-183-intern user@10.0.2.36"}
H68=${H68:-node-68-intern} H134=${H134:-node-134-intern} MAX_H=${MAX_H:-24} SHARD_HOSTS=(${SHARD_HOSTS:-136 68 68})
host() { case $1 in 94) echo "$H94" ;; 136) echo "$H136" ;; 68) echo "$H68" ;; esac; }
HOSTS=$(printf '%s\n' "${SHARD_HOSTS[@]}" | awk '!seen[$0]++')  # each shard host once
TO136="ssh -o BatchMode=yes user@198.51.100.134 ssh -o BatchMode=yes user@10.0.2.36"
to136() { if hostname -I | grep -qw 10.0.2.36; then bash -c "$1"; else
  ssh -o BatchMode=yes user@198.51.100.134 "ssh -o BatchMode=yes user@10.0.2.36 $(printf %q "$1")"; fi; }
on() { local h=$1; shift; if [ -n "${DRY:-}" ]; then echo "+ ssh $h: $*"; else ssh $h "$*"; fi; }
code() { echo /home/user/harbor-udocker/code-$1; }
out() { echo /home/user/octo-env/gate0-$1; }
stop_servers() { on "$H136" "RL_ENV_FILE=$1/scripts/rl_env/rl_run.coupled.env $1/scripts/rl_env/eval_serve.sh stop" || true
  on "$H94" "RL_ENV_FILE=$1/scripts/rl_env/rl_run.coupled.env $1/scripts/rl_env/eval_serve.sh stop" || true; }

case $1 in
start)
  C=$(code $2) O=$(out $2); ENV="RL_ENV_FILE=$C/scripts/rl_env/rl_run.coupled.env"; W="setsid nohup bash $C/scripts/eval/gate0_launch.sh"
  NS=${#SHARD_HOSTS[@]}
  for h in $HOSTS; do on "$(host $h)" "$TO136 true" || { echo "gpu-$h cannot reach node-136 through node-134"; exit 1; }; done
  on "$H136" "rm -rf $O/done && mkdir -p $O/done && { MAX_H=$MAX_H $W serve-watch $O $NS > $O/serve-watch.log 2>&1 < /dev/null & }"
  abort() { echo "abort: the watches stop the servers"; on "$H136" "touch $O/done/abort" || true
    for h in $HOSTS; do [ $h = 136 ] || on "$(host $h)" "$TO136 touch $O/done/abort" || true; done; }
  trap abort EXIT; trap 'exit 130' INT TERM HUP  # until the shards run (SIGKILL of the controller: stop SHA)
  on "$H94" "mkdir -p $O && { MAX_H=$MAX_H $W serve-watch $O $NS > $O/serve-watch.log 2>&1 < /dev/null & }"
  PICK="PICK_S=${PICK_S:-60} RESERVE_MIB=${RESERVE_MIB:-10240} ACT_MIB=${ACT_MIB:-4096} MIN_MIB=${MIN_MIB:-16384} bash $C/scripts/eval/gate0_launch.sh pick-gpus"
  P136=$(ssh $H136 "$PICK") || { echo "node-136: GPU query failed"; exit 1; }  # read-only, also under DRY
  P94=$(ssh $H94 "$PICK") || { echo "node-94: GPU query failed"; exit 1; }
  read -r G136 F136 B136 <<< "$P136"; read -r G94 F94 B94 <<< "$P94"; [ "$G136" != none ] || G136=; [ "$G94" != none ] || G94=
  N136=$(echo "$G136" | awk -F, '{print NF}'); N94=$(echo "$G94" | awk -F, '{print NF}')
  echo "picked: node-136 ${G136:-none} (MEM_FRAC ${F136:--}), node-94 ${G94:-none} (MEM_FRAC ${F94:--})"
  [ $((N136 + N94)) -gt 0 ] || { echo "no usable GPU"; exit 1; }
  [ $((N136 + N94)) -ge 4 ] || echo "only $((N136 + N94)) servers: MAX_H ($MAX_H h) may be too short"
  T="timeout ${START_S:-1800}"
  p136=; [ -z "$G136" ] || { on "$H136" "cd $C && SERVE_HOST=10.0.2.36 MEM_FRAC=$F136 BUSY_MIB=$B136 $ENV $T scripts/rl_env/eval_serve.sh start $G136" & p136=$!; }
  p94=; [ -z "$G94" ] || { on "$H94" "cd $C && SERVE_HOST=10.0.1.94 PORT_BASE=30310 MEM_FRAC=$F94 BUSY_MIB=$B94 $ENV $T scripts/rl_env/eval_serve.sh start $G94" & p94=$!; }
  ok=1; [ -z "$p136" ] || wait $p136 || ok=0; [ -z "$p94" ] || wait $p94 || ok=0
  [ $ok = 1 ] || { echo "a server start failed"; exit 1; }
  # the busiest server holds 2 x ceil(shards x c / servers) episodes at once (c: CONC capped at the servers, as the
  # shards' rotation does), each up to KV_PER_EP (32768) tokens of context
  N=$((N136 + N94)); c=$(( ${CONC:-6} < N ? ${CONC:-6} : N ))
  MIN_KV=${MIN_KV:-$(( 2 * ( (NS * c + N - 1) / N ) * ${KV_PER_EP:-32768} ))}
  kv() { on "$1" "for p in \$(seq $2 $(($2 + $3 - 1))); do echo \$p \$(curl -sf -m 10 http://$4:\$p/get_server_info | python3 -c 'import json, sys; print(json.load(sys.stdin)[\"max_total_num_tokens\"])' || echo failed); done"; }
  [ -n "${DRY:-}" ] || { [ -z "$G136" ] || kv "$H136" 30300 $N136 10.0.2.36; [ -z "$G94" ] || kv "$H94" 30310 $N94 10.0.1.94; } |
    while read -r port t; do echo "KV $port: $t tokens"; [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -ge $MIN_KV ] || { echo "server $port: KV $t, need $MIN_KV"; exit 1; }; done

  for h in $HOSTS; do
    mine=$(for i in "${!SHARD_HOSTS[@]}"; do if [ "${SHARD_HOSTS[$i]}" = $h ]; then echo $i; fi; done)
    for s in $mine; do on "$(host $h)" "N136=$N136 N94=$N94 CONC=${CONC:-6} bash $C/scripts/eval/gate0_shard.sh $s $NS $O" || echo "shard $s did not start"; done
    on "$(host $h)" "{ $W shard-wait $O $(echo $mine) > $O/shard-wait.log 2>&1 < /dev/null & }" || echo "gpu-$h's shard-wait did not start"
  done
  trap - EXIT INT TERM HUP
  echo "started; progress: gate0_launch.sh status $2" ;;
status)
  O=$(out $2)
  for h in $HOSTS; do on "$(host $h)" "hostname; wc -l $O/off-*.jsonl $O/event-*.jsonl 2>/dev/null | grep -v total"; done
  on "$H136" "ls $O/done; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '"
  on "$H94" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '" ;;
stop)
  C=$(code $2) O=$(out $2)
  for h in $HOSTS; do on "$(host $h)" "bash $C/scripts/eval/gate0_launch.sh stop-host $O" || true; done
  on "$H136" "touch $O/done/abort" || true; stop_servers $C ;;
collect)
  C=$(code $2) O=$(out $2); R=user@198.51.100.134; NS=${#SHARD_HOSTS[@]}; first=1
  for h in $HOSTS; do
    f=""; for i in "${!SHARD_HOSTS[@]}"; do if [ "${SHARD_HOSTS[$i]}" = $h ]; then f="$f off-$i.jsonl event-$i.jsonl jobs-off-$i jobs-event-$i"; fi; done
    [ $first = 0 ] || f=" tasks.txt$f"
    on "$(host $h)" "$([ $first = 0 ] || echo "rsync -a $C/ $R:$C/ && ")cd $O && $([ $first = 0 ] || echo "ls tasks > tasks.txt && ")rsync -a$f $R:$O/"; first=0
  done
  all() { for i in $(seq 0 $((NS - 1))); do printf ' %s-%s.jsonl' $1 $i; done; }
  on $H134 "set -o pipefail; cd $O && cat$(all off) > off.jsonl && cat$(all event) > event.jsonl && bash $C/scripts/eval/gate0_launch.sh check $O && PYTHONPATH=$C ~/octo-env/areal/.venv/bin/python $C/scripts/eval/gate0.py off=off.jsonl event=event.jsonl | tee gate0.txt" ;;
pick-gpus)  # "GPUS MEM_FRAC BUSY_MIB" of this host, or "none"; a failed query exits non-zero
  python3 - <<'PY'
import os, subprocess, time
q = lambda f: [l.split(", ") for l in subprocess.check_output(["nvidia-smi", f"--query-{f}", "--format=csv,noheader,nounits"], text=True).splitlines() if l.strip()]
env = lambda k: int(os.environ[k])
total, idx = {}, {}
for i, uuid, t in q("gpu=index,uuid,memory.total"):
    total[int(i)], idx[uuid] = int(t), int(i)
peak, now = {}, {}
for n in range(env("PICK_S") // 10 + 1):
    for i, used in q("gpu=index,memory.used"):
        peak[int(i)] = max(peak.get(int(i), 0), int(used)); now[int(i)] = int(used)
    if n < env("PICK_S") // 10:
        time.sleep(10)
other, iraod = set(), set()  # GPUs with a process that is not this user's iraod job; GPUs with one that is
for uuid, pid in q("compute-apps=gpu_uuid,pid"):
    try:
        ok = os.stat(f"/proc/{pid}").st_uid == os.getuid() and b"iraod" in open(f"/proc/{pid}/cmdline", "rb").read().lower()
    except OSError:
        ok = False
    (iraod if ok else other).add(idx[uuid])
other |= {g for g in total if now[g] > 1000 and g not in iraod}  # memory held by a process we cannot see: not ours
# sglang 0.5.19: static memory (weights + KV) = MEM_FRAC x the GPU's free memory at its start (torch.cuda.mem_get_info),
# so with MEM_FRAC <= keep(g) / total[g] the static part stays <= keep(g); SGLang's activations and cuda graphs come on
# top (ACT_MIB), and IRAOD keeps its sampled peak + RESERVE_MIB. If IRAOD drops while SGLang measures free memory, the
# static part grows by that drop and IRAOD's room shrinks below RESERVE_MIB by the same amount.
keep = lambda g: total[g] - peak[g] - env("RESERVE_MIB") - env("ACT_MIB")
cand = sorted((g for g in total if g not in other), key=keep, reverse=True)
for k in range(len(cand), 0, -1):  # the most GPUs (largest keep first) on which SGLang still gets MIN_MIB of static
    pick = cand[:k]
    frac = min(0.88, min(keep(g) / total[g] for g in pick))
    if all(frac * (total[g] - now[g]) >= env("MIN_MIB") for g in pick):
        print(",".join(map(str, sorted(pick))), f"{int(frac * 1000) / 1000:.3f}", max(peak[g] for g in pick) + 1024)
        break
else:
    print("none")
PY
  ;;
check)  # every bundle of OUT/tasks.txt has samples 0-3 scored (last row per episode has y) in both arms
  python3 - "$2" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1]); want = {(b, s) for b in o.joinpath("tasks.txt").read_text().split() for s in range(4)}
bad = 0
for arm in ("off", "event"):
    last = {}
    for line in o.joinpath(f"{arm}.jsonl").read_text().splitlines():
        r = json.loads(line); last[(Path(r["task_dir"]).name, r["sample"])] = r
    missing = sorted(k for k in want if "y" not in last.get(k, {}))
    print(f"{arm}: {len(want) - len(missing)}/{len(want)} scored")
    for b, s in missing:
        print(f"  missing {arm} {b} sample {s}: {last.get((b, s), {}).get('error', 'no row')}")
    bad += len(missing)
sys.exit(1 if bad else 0)
PY
  ;;
shard-wait)
  O=$2; shift 2
  for s in "$@"; do
    for f in off event; do p=$(cat $O/$f-$s.pid 2>/dev/null || echo 0); while [ "$p" != 0 ] && kill -0 $p 2>/dev/null; do sleep 60; done; done
    until to136 "touch $O/done/shard-$s"; do sleep 60; done; echo "$(date) shard $s done"
  done
  p=$(cat $O/tunnel.pid 2>/dev/null || echo 0); ps -o args= -p $p 2>/dev/null | grep -q '^ssh -N' && kill $p && echo "tunnel $p stopped" || true ;;
stop-host)  # this host's dev_evals (each its own session), their episodes' harbor runs (each its own session: SIGINT
  O=$2      # first, so Harbor removes its container), the tunnel
  for f in $O/off-*.pid $O/event-*.pid; do p=$(cat $f 2>/dev/null) || continue  # a stale pid file may name another group
    ps -o args= -p "$p" 2>/dev/null | grep -q -- "-m octopus.eval.dev_eval .*--jobs-dir $O/jobs-" && kill -- -"$p" && echo "stopped $(basename $f .pid)" || true; done
  pg=$(pgrep -u $USER -f "harbor run -p $O/jobs-" | xargs -r -I{} ps -o pgid= -p {} | sort -u) || true
  for g in $pg; do kill -INT -- -$g 2>/dev/null || true; done; sleep 20
  for g in $pg; do kill -9 -- -$g 2>/dev/null || true; done
  p=$(cat $O/tunnel.pid 2>/dev/null || echo 0); ps -o args= -p $p 2>/dev/null | grep -q '^ssh -N' && kill $p || true
  echo "$(hostname): stopped; left: $(pgrep -u $USER -fc "$O/jobs-" || true)" ;;
serve-watch)
  O=$2 NS=$3 C=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); end=$(( $(date +%s) + MAX_H * 3600 ))
  until m=$(to136 "ls $O/done" 2>/dev/null || true); [ "$(echo "$m" | grep -c '^shard-')" -ge $NS ] || echo "$m" | grep -qx abort ||
    [ "$(date +%s)" -ge $end ]; do sleep 60; done
  RL_ENV_FILE=$C/scripts/rl_env/rl_run.coupled.env $C/scripts/rl_env/eval_serve.sh stop; sleep 10
  # abort may come before a start had written its pid files: stop once more after they exist
  ! echo "$m" | grep -qx abort || { sleep 60; RL_ENV_FILE=$C/scripts/rl_env/rl_run.coupled.env $C/scripts/rl_env/eval_serve.sh stop; }
  { date; echo "markers: $(echo $m)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; } | tee $O/released.txt ;;
*) sed -n 2,30p "$0"; exit 1 ;;
esac
