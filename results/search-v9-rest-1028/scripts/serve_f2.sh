#!/bin/bash
# serve_f2.sh STATE PAIRS...: (re)start our 4B rollout servers (trial-136: mem 0.88 + mamba ratio 0.5; no HiCache, which
# the read tool's mount patch refuses). STATE holds sglang-cache and the pid/log files, OV (default STATE) the
# cursor-ba8072c/sglang overlay and code-local, CU the CUDA toolkit (the JIT kernels need its nvcc); PAIRS are gpu:port.
# LORAS: the startup adapters (default rq1-v41). BUDGET (MiB, for a GPU shared with another user): launch only with
# BUDGET + 6 GiB free, at --mem-fraction-static BUDGET / (free memory now). An estimate: SGLang 0.5.19 sizes the server
# to the fraction of the free memory at its distributed init (bootstrap pre_model_load_memory; the log's "Load weight
# begin. avail mem"), so it takes ~BUDGET (+ prefill activations) only if the other user's memory is the same then; memory
# freed in between makes it larger by the same factor. KV_TOKENS and MAMBA_SLOTS (both, recommended on a shared GPU) fix
# the pools instead: --max-total-tokens (the KV pool is min(profiled, given)) and --max-mamba-cache-size (used as given,
# taken from the remaining memory first; SGLang fails at start if that is short, it does not shrink it). Scale the pool
# sizes, not the footprint: README. share_watch.sh stops a server when the other user needs the room.
# Without BUDGET: 0.88 on a GPU with <= 4000 MiB used. Another host's paths (newhost.sh): SGLANG_PY (the SGLang venv's
# python), MODEL (Qwen3.5-4B), SGL_PATH (PYTHONPATH: the cursor+mount SGLang overlay, then the code dir).
S=$1; shift; OV=${OV:-$S}; CU=${CU:-/usr/local/cuda-12.9}
PY=${SGLANG_PY:-/home/user/octo-env/sglang-venv/bin/python}
L=/home/user/search-v9-run/areal-exp/checkpoints/user/rq1-rl/search-v9-rest/default/weight_update_v41
LORAS=${LORAS:-rq1-v41=$L}
export PATH=$(dirname "$PY"):$CU/bin:$PATH CUDA_HOME=$CU
export PYTHONPATH=${SGL_PATH:-$OV/cursor-ba8072c/sglang:$OV/code-local} SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_CACHE_DIR=$S/sglang-cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=4
mkdir -p $S/sglang-cache
for pg in "$@"; do
  g=${pg%:*}; port=${pg#*:}; f=$S/rollout-g$g.pid
  if [ -f $f ] && tr '\0' ' ' < /proc/$(cat $f)/cmdline 2>/dev/null | grep -q "sglang.launch_server.*--port $port"; then
    p=$(cat $f); kill -TERM -- -$p 2>/dev/null || kill $p; for _ in $(seq 30); do kill -0 $p 2>/dev/null || break; sleep 2; done; kill -KILL -- -$p 2>/dev/null
  fi
done
sleep 5
for pg in "$@"; do
  g=${pg%:*}; port=${pg#*:}
  read -r used free < <(nvidia-smi -i $g --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ,)
  if [ -n "${BUDGET:-}" ]; then
    [ -n "$free" ] && [ "$free" -ge $((BUDGET + 6144)) ] || { echo "GPU $g: ${free:-unreadable} MiB free, skip"; continue; }
    mem=$(awk -v b=$BUDGET -v f=$free 'BEGIN {printf "%.3f", b / f}')
  else
    [ -n "$used" ] && [ "$used" -le 4000 ] || { echo "GPU $g busy (${used:-unreadable} MiB), skip"; continue; }
    mem=0.88
  fi
  CUDA_VISIBLE_DEVICES=$g setsid "$PY" -m sglang.launch_server --model-path ${MODEL:-/home/user/octo-env/models/Qwen3.5-4B} --host 127.0.0.1 --port $port \
    --tp-size 1 --mem-fraction-static $mem --mamba-full-memory-ratio 0.5 \
    ${KV_TOKENS:+--max-total-tokens $KV_TOKENS} ${MAMBA_SLOTS:+--max-mamba-cache-size $MAMBA_SLOTS} \
    --attention-backend triton --sampling-backend pytorch --context-length 262144 --disable-prefill-cuda-graph --enable-cache-report \
    --enable-lora --lora-paths $LORAS --max-lora-rank 64 --max-loaded-loras 44 --max-loras-per-batch 4 --lora-backend triton \
    --lora-strict-loading --mm-attention-backend triton_attn > $S/rollout-g$g.log 2>&1 < /dev/null &
  echo $! > $S/rollout-g$g.pid; echo "gpu $g port $port pid $!"
done
