#!/bin/bash
# SGLang servers for dev evaluation: one per GPU on ports 30300+i, each with the given LoRA adapters loaded, named as
# training names them (NAME = rq1-v<step>: dev_eval --lora). As rl_run.sh's servers, they only answer /generate: dev_eval
# runs training's AReaL proxy, which renders and parses (octopus/rl/proxy.py).
# --mm-attention-backend triton_attn: SGLang's sm90 default (fa3) needs a newer sgl_kernel than the pinned 0.3.5 and crashed
# the vision warmup on H200 (NVwulf); sm89 (node-94) already defaults to triton_attn.
#   eval_serve.sh start GPU[,GPU...] [NAME=ADAPTER_DIR ...]   (prints the --servers list for dev_eval)
#   eval_serve.sh stop
# Machine settings (SGLANG_PY MODEL_VL STATE, optional CUDA_HOME and CURSOR, rl_run.sh's cursor overlay dir, which needs CODE
# too, the code dir its shims import: the episodes then need OCTOPUS_CURSOR=1; LORA_RANK, default 32, SERVE_HOST, the address to listen on, default 127.0.0.1 (a LAN address: dev_eval on another host reaches it; SGLang has no auth, so LAN only), PORT_BASE, default 30300, and BUSY_MIB, the MiB a GPU may already use, default 4000, MEM_FRAC, default 0.85, and SGLANG_ARGS, more SGLang flags, e.g.
# the read tool's mount flags, from the environment) from $RL_ENV_FILE (default ~/octo-env/rl_run.env).
set -uo pipefail
source "${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}"
D=$STATE/eval-serve-$(hostname -s); mkdir -p "$D"  # per host: STATE may be on a filesystem other hosts share
stop() {  # each server is its own session (setsid): signal the group, so scheduler children go with a dead launcher
  local p pids; pids=$(cat "$D"/*.pid 2>/dev/null); for p in $pids; do kill -- "-$p" 2>/dev/null; done
  alive() { for p in $pids; do kill -0 -- "-$p" 2>/dev/null && return 0; done; return 1; }
  for _ in $(seq 60); do alive || break; sleep 2; done
  for p in $pids; do kill -9 -- "-$p" 2>/dev/null; done; rm -f "$D"/*.pid; }
case "${1:-}" in
start)
  stop; GPUS=$2; shift 2; LORAS=("$@")
  LORA=(); [ ${#LORAS[@]} = 0 ] || LORA=(--enable-lora --lora-paths "${LORAS[@]}" --max-lora-rank ${LORA_RANK:-32}
    --max-loaded-loras $(( ${#LORAS[@]} + 1 )) --max-loras-per-batch $(( ${#LORAS[@]} + 1 )) --lora-backend triton
    --lora-strict-loading)  # no NAME=DIR: the base model alone (dev_eval with rollout.use_lora=false, no --lora)
  for g in ${GPUS//,/ }; do used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -gt "${BUSY_MIB:-4000}" ] && { echo "GPU $g busy (${used} MiB)"; exit 1; }; done  # SGLang takes 85%: room for a small tenant
  [ -n "${CUDA_HOME:-}" ] && export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
  export PATH="$(dirname "$SGLANG_PY"):$PATH" SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_CACHE_DIR="$STATE/sglang-cache"
  if [ -n "${CURSOR:-}" ]; then  # the cursor-patched SGLang, its shims import CODE (docs/cursor-protocol.md "Deployment")
    old=$(grep -L 'sys.modules\[__name__\]' "$CURSOR"/sglang/sglang/srt/octopus_*.py 2>&1)
    [ -z "$old" ] || { echo "$CURSOR/sglang has copies, not shims ($old): re-run patches/apply_sglang_cursor.py and apply_sglang_mount.py on it"; exit 1; }
    export PYTHONPATH=$CURSOR/sglang:$CODE
  fi
  i=0; SERVERS=""
  for g in ${GPUS//,/ }; do
    PORT=$((${PORT_BASE:-30300} + i)); SERVERS="$SERVERS${SERVERS:+,}${SERVE_HOST:-127.0.0.1}:$PORT"
    CUDA_VISIBLE_DEVICES=$g setsid bash -c 'echo $$ > "$0"; exec "$@"' "$D/sglang-g$g.pid" "$SGLANG_PY" -m sglang.launch_server --model-path "$MODEL_VL" --host "${SERVE_HOST:-127.0.0.1}" --port $PORT \
      --tp-size 1 --mem-fraction-static ${MEM_FRAC:-0.85} --attention-backend triton --sampling-backend pytorch --context-length 262144 \
      --disable-prefill-cuda-graph --enable-cache-report "${LORA[@]}" --mm-attention-backend triton_attn ${SGLANG_ARGS:-} \
      > "$D/sglang-g$g.log" 2>&1 < /dev/null &
    for _ in $(seq 100); do [ -s "$D/sglang-g$g.pid" ] && break; sleep 0.1; done  # written by the session leader itself (setsid forks under job control, so $! can name an exited pid)
    i=$((i + 1))
  done
  i=0; for g in ${GPUS//,/ }; do s=${SERVE_HOST:-127.0.0.1}:$((${PORT_BASE:-30300} + i)); i=$((i + 1))
    until curl -sf -m 3 "http://$s/v1/models" > /dev/null; do  # a server that died while loading fails the start
      kill -0 -- "-$(cat "$D/sglang-g$g.pid")" 2>/dev/null || { echo "server on GPU $g exited: $D/sglang-g$g.log"; exit 1; }
      sleep 10; done; done
  echo "$SERVERS"
  ;;
stop) stop; echo stopped ;;
*) sed -n 2,5p "$0"; exit 1 ;;
esac
