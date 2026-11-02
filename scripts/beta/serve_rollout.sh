#!/bin/bash
# Rollout SGLang servers on Cluster Beta for a trainer elsewhere (an Alpha trainer's EXTRA_SERVERS: search_launch.sh
# EXTRA_ROLLOUT): one server per GPU g at SERVE_HOST:PORT0+g, with this code dir's scripts/beta/search.env (SGLANG_PY
# MODEL_VL ADAPTER0 CURSOR LORA_RANK LORA_KEEP MEM_FRAC CUDA_HOME PORT0) and the flags of scripts/rl_env/rl_run.sh serve().
#   SERVE_HOST=IP [JUDGE=G] serve_rollout.sh start DIR [GPUS=0,1,2,3]   returns 0 once every /v1/models answers
#   serve_rollout.sh stop DIR                                             in the same job (srun --jobid=J --overlap)
# JUDGE=G (a GPU not in GPUS): also the judge (JUDGE_MODEL, served as "judge", lib.sh judge_start's flags) at
# SERVE_HOST:JUDGE_PORT, for search_launch.sh JUDGE_HOST=SERVE_HOST; its log DIR/sglang-judge.log is the guards' JUDGE_LOG.
# As a queue task of a held job (scripts/beta/hold.sbatch): scripts/beta/tasks/serve-rollout.sh. The servers outlive the
# task (setsid) and keep their GPUs after it ends, while the hold runs the next queued task at once: queue nothing else
# on that job until they are stopped (serve_rollout.sh stop).
# Start them fresh for every trainer start, a resume too: requests a stopped trainer left in flight leak the servers' LoRA
# count, and the next start's unload of the old adapters (rl_run.sh start) then hangs holding the LoRA lock (v9, 3x).
# stop walks the ppid tree from each DIR/sglang-*.pid and signals exact pids: no pgrep -f, no process-group kill.
set -uo pipefail
source "$(dirname "$0")/search.env"
case "${1:-}" in
start)
  D=${2:?DIR}; GPUS=${3:-0,1,2,3}; J=${JUDGE:-}; : "${SERVE_HOST:?SERVE_HOST}"
  [[ ,$GPUS, != *,$J,* ]] || { echo "JUDGE GPU $J is in GPUS $GPUS"; exit 1; }
  mkdir -p "$D"; rm -f "$D"/READY "$D"/FAILED "$D"/sglang-*.pid
  export CUDA_HOME PATH=$CUDA_HOME/bin:$(dirname "$SGLANG_PY"):$PATH SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
  export TZ=America/New_York  # log times as Alpha's (Beta nodes: UTC): search_step3_check.sh's step windows read them
  for g in ${GPUS//,/ } $J; do  # never start on a GPU someone else is using
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)  # no number: not known free
    [[ $used =~ ^[0-9]+$ ]] && [ "$used" -le "${BUSY_MIB:-1000}" ] || { echo "GPU $g busy or unreadable (${used:-no answer} MiB)"; exit 1; }
  done
  W=""  # PORT:NAME of every server started
  launch() {  # launch NAME GPU PORT ARGS...: one SGLang on GPU at SERVE_HOST:PORT, its exact pid in D/sglang-NAME.pid
    local n=$1 g=$2 p=$3; shift 3; W="$W $p:$n"
    CUDA_VISIBLE_DEVICES=$g setsid bash -c 'echo $$ > "$0"; exec "$@"' "$D/sglang-$n.pid" "$SGLANG_PY" -m sglang.launch_server \
      --host "$SERVE_HOST" --port "$p" --tp-size 1 --attention-backend triton --sampling-backend pytorch \
      --disable-prefill-cuda-graph "$@" > "$D/sglang-$n.log" 2>&1 < /dev/null &
    for _ in $(seq 100); do [ -s "$D/sglang-$n.pid" ] && break; sleep 0.1; done
  }
  for g in ${GPUS//,/ }; do  # the cursor-patched SGLang ahead of the venv's; its shims import CODE's octopus/
    PYTHONPATH=$CURSOR/sglang:$CODE SGLANG_CACHE_DIR=$D/sglang-cache launch "g$g" "$g" $((PORT0 + g)) --model-path "$MODEL_VL" \
      --mem-fraction-static "$MEM_FRAC" --context-length 262144 --enable-cache-report --enable-lora --lora-paths rq1-v0="$ADAPTER0" \
      --max-lora-rank "$LORA_RANK" --max-loaded-loras $((LORA_KEEP + 4)) --max-loras-per-batch 4 --lora-backend triton \
      --lora-strict-loading --mm-attention-backend triton_attn
  done
  [ -z "$J" ] || SGLANG_CACHE_DIR=$D/sglang-cache-judge launch judge "$J" "$JUDGE_PORT" --model-path "$JUDGE_MODEL" \
    --served-model-name judge --mem-fraction-static 0.85 --context-length 32768
  for _ in $(seq 240); do  # at most ~20 min; a server that exits fails the start
    ok=1
    for w in $W; do curl -sf -m 3 "http://$SERVE_HOST:${w%%:*}/v1/models" > /dev/null || ok=0; done
    [ $ok = 1 ] && break
    for w in $W; do n=${w#*:}; p=$(cat "$D/sglang-$n.pid" 2>/dev/null); [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null && { echo "server $n (pid $p) exited"; tail -n 5 "$D/sglang-$n.log"; ok=-1; }; done
    [ $ok = -1 ] && break
    sleep 5
  done
  if [ "$ok" != 1 ]; then echo "servers failed to come up (stop them: serve_rollout.sh stop $D)"; touch "$D/FAILED"; exit 1; fi
  S=""; for g in ${GPUS//,/ }; do S="$S${S:+,}$SERVE_HOST:$((PORT0 + g))"; done
  echo "servers $S${J:+; judge up at http://$SERVE_HOST:$JUDGE_PORT/v1}"; touch "$D/READY"
  ;;
stop)
  D=${2:?DIR}
  desc() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
  PIDS=""; for f in "$D"/sglang-*.pid; do p=$(cat "$f" 2>/dev/null)  # still an SGLang server (not a reused pid)
    grep -qF sglang.launch_server "/proc/$p/cmdline" 2>/dev/null && PIDS="$PIDS $p $(desc "$p")"; done
  PIDS=$(echo $PIDS | tr ' ' '\n' | sort -un | tr '\n' ' ')
  for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done
  alive=""; for _ in $(seq 40); do alive=""; for p in $PIDS; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done; [ -z "$alive" ] && break; sleep 2; done
  for p in $alive; do kill -KILL "$p" 2>/dev/null; done
  rm -f "$D"/READY "$D"/sglang-*.pid
  echo "stopped: $(echo $PIDS | wc -w) pids; survivors: ${alive:-none}"
  ;;
*) echo "usage: SERVE_HOST=IP [JUDGE=G] serve_rollout.sh start DIR [GPUS] | serve_rollout.sh stop DIR"; exit 2 ;;
esac
