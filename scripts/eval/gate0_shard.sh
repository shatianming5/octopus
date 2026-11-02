#!/bin/bash
# Gate 0 (docs/cross-branch-read.md "Gate 0"): one container host's shard of both arms, arm 1 (read tool off) and arm 3
# (OCTOPUS_READ_TOOL=1 OCTOPUS_READ_EVENT=1), k=4, the base model, the settings of scripts/rl_env/rl_run.coupled.env.
#   gate0_shard.sh SHARD NSHARDS OUT_DIR      (DRY=1: print the commands, start nothing)
# Bundles (user, 10-21): all of DATA (default $OCTOPUS_HARBOR/datasets)/coupled-a-val plus 244 of DATA/coupled-a-train drawn with random.Random(0) from its
# sorted bundle ids (289: Val 45 + Train 244); coupled-a-test never enters Gate 0. OUT_DIR/tasks holds them as symlinks,
# made once (the same set on every host from the same data). Shard SHARD (0-based) of NSHARDS takes a contiguous slice
# of their sorted names (dev_eval --offset/--limit).
# Run from the code dir's copy of this file on each container host (gate0_launch.sh: node-94, node-136, node-68). SGLang, the
# base model, RL_ENV_FILE=<code dir>/scripts/rl_env/rl_run.coupled.env:
#   node-136: SERVE_HOST=10.0.2.36 eval_serve.sh start <its free GPUs>    (ports 30300-...)
#   node-94: SERVE_HOST=10.0.1.94 PORT_BASE=30310 eval_serve.sh start <its free GPUs>    (ports 30310-...)
# node-136 is reached on 127.0.0.1:30320-... through node-134's sshd (LAN; node-68/94 cannot reach 10.0.2.36; 30300-30304
# stay free for other runs' tunnels), node-94 directly (172.18.160.x). N136 (5), N94 (8): each host's servers. Each
# dev_eval sends each new episode to its least loaded server (octopus/serving/route.py), so the servers share the load.
# Settings: OUT_DIR/run, made here from the env file when missing. Trials clone goldens as fuse overlays (UDOCKER_OVERLAY, default
# $OCTO_ENV/bin/fuse-overlayfs): a tar copy of a 2-3 GB golden on a busy disk outlasts the 900 s clone timeout. CONC (all servers): each
# arm's concurrent episodes, one per server on CONC of them, rotated by shard so the shards spread over all servers;
# both arms of a shard use the same servers; a shard runs 2 x CONC episodes at once. Both arms add --env OCTOPUS_AUTO_FORK=1 (Gate
# 0's environment fork per bundle task, #222). A rerun skips finished episodes and reruns infra errors. Each shard's rows
# (OUT_DIR/{off,event}-SHARD.jsonl) point at trial dirs on its own host: `gate0_launch.sh collect SHA` gathers every
# shard's rows and trial dirs on node-134 under the same paths and runs scripts/eval/gate0.py there.
set -euo pipefail
SHARD=$1 NSHARDS=$2 OUT=$3 TASKS=$3/tasks
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); E=$C/scripts/rl_env/rl_run.coupled.env
source "$E"
DATA=${DATA:-$OCTOPUS_HARBOR/datasets}
for v in $(compgen -e | grep '^GROK_' | grep -vx GROK_MODELS_BASE_URL); do unset "$v"; done  # pre-rename names: dev_eval refuses them
export UDOCKER_OVERLAY=${UDOCKER_OVERLAY:-$OCTO_ENV/bin/fuse-overlayfs}
if [ ! -d "$TASKS" ]; then mkdir -p "$TASKS.tmp"; python3 - "$DATA" "$TASKS.tmp" <<'PY'
import os, random, sys
data, out, n = sys.argv[1], sys.argv[2], 244
ids = lambda s: sorted(d for d in os.listdir(f"{data}/coupled-a-{s}") if os.path.isdir(f"{data}/coupled-a-{s}/{d}"))
for split, picked in (("val", ids("val")), ("train", random.Random(0).sample(ids("train"), n))):
    for b in picked:
        os.symlink(f"{data}/coupled-a-{split}/{b}", f"{out}/{b}")
PY
  mv "$TASKS.tmp" "$TASKS"; fi
N=$(find "$TASKS" -mindepth 1 -maxdepth 1 | wc -l); PER=$(( (N + NSHARDS - 1) / NSHARDS ))
SERVERS=""; TUNNEL=(ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15)
for i in $(seq 0 $((${N136:-5} - 1))); do SERVERS+=127.0.0.1:$((30320 + i)),; TUNNEL+=(-L "$((30320 + i)):10.0.2.36:$((30300 + i))"); done
for i in $(seq 0 $((${N94:-8} - 1))); do SERVERS+=10.0.1.94:$((30310 + i)),; done
SERVERS=${SERVERS%,}
TUNNEL+=(user@198.51.100.134)
IFS=, read -ra ALL <<< "$SERVERS"; CONC=${CONC:-${#ALL[@]}}; [ "$CONC" -le ${#ALL[@]} ] || CONC=${#ALL[@]}
mine() { local k out=""; for k in $(seq 0 $((CONC - 1))); do out+=${ALL[$(( ($1 * CONC + k) % ${#ALL[@]} ))]},; done; echo "${out%,}"; }
D=(nice -n 10 "$AREAL_PY" -m octopus.eval.dev_eval --run "$OUT/run" --tasks-dir "$TASKS" --k 4 --parallel 1
   --offset $((SHARD * PER)) --limit "$PER")
run() { if [ -n "${DRY:-}" ]; then echo "+ $*"; else "$@"; fi; }
bg() { local name=$1; shift  # detached, output to OUT/NAME.log, pid to OUT/NAME.pid
  if [ -n "${DRY:-}" ]; then echo "+ $* > $OUT/$name.log &"; return; fi
  setsid "$@" > "$OUT/$name.log" 2>&1 < /dev/null & echo $! > "$OUT/$name.pid"; }
echo "shard $SHARD/$NSHARDS: task dirs $((SHARD * PER))..$((SHARD * PER + PER - 1)) of $N; code $C; overlay $UDOCKER_OVERLAY"
[ -x "$UDOCKER_OVERLAY" ] || { echo "no fuse-overlayfs at $UDOCKER_OVERLAY"; exit 1; }
cd "$C"; run mkdir -p "$OUT"
[ -f "$OUT/run/episode_settings.json" ] || run env RL_ENV_FILE="$E" scripts/rl_env/rl_run.sh settings "$OUT/run" $EXTRA rollout.use_lora=false
[ "${N136:-5}" = 0 ] || curl -sf -m 5 http://127.0.0.1:$((30319 + ${N136:-5}))/v1/models > /dev/null || { bg tunnel "${TUNNEL[@]}"; [ -n "${DRY:-}" ] || sleep 5; }
[ -n "${DRY:-}" ] || for s in ${SERVERS//,/ }; do curl -sf -m 5 "http://$s/v1/models" > /dev/null || { echo "$s down"; exit 1; }; done
export PYTHONPATH=$CURSOR/areal:$C LITELLM_LOCAL_MODEL_COST_MAP=True
for arm in off event; do
  X=(--env OCTOPUS_AUTO_FORK=1); [ $arm = off ] || X+=(--env OCTOPUS_READ_TOOL=1 --env OCTOPUS_READ_EVENT=1)
  # both arms of a shard on the same servers
  bg $arm-$SHARD "${D[@]}" --servers "$(mine $SHARD)" "${X[@]}" --out "$OUT/$arm-$SHARD.jsonl" --jobs-dir "$OUT/jobs-$arm-$SHARD"
done
[ -n "${DRY:-}" ] || echo "started: pids in $OUT/{off,event}-$SHARD.pid, logs beside them"
