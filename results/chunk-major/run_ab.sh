#!/bin/bash
# actor_ab.py on a host (results/chunk-major/README.md "Use in training"): run_ab.sh GPUS LOG GROUPS [actor_ab options]
# GPUS: comma-separated card ids (one rank each); GROUPS: comma-separated file names under $D/spool. The caller's env sets
# the configuration (OCTOPUS_CHUNK_MAJOR, OCTOPUS_CHUNK_TOKENS, OCTOPUS_BLOCK_CHECKPOINT) and where things are: D (scratch:
# src/ this code at the pushed commit (or SRC), spool/ copied groups, lora/ the adapter every run starts from: r64 on the
# resolved targets, node-134's adapter-v0-4b-text-r64 with B drawn from N(0, 1e-3), seed 0, as bench.py), PY (the AReaL
# venv's python), AREAL (the patched AReaL overlay), MODEL (Qwen3.5-4B-text).
set -e
GPUS=$1 LOG=$2 FILES=$3  # (not GROUPS: a bash builtin array)
shift 3
export TMPDIR=$D/tmp TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$D/tmp/triton} TORCHINDUCTOR_CACHE_DIR=$D/tmp/inductor
export PYTHONPATH=$AREAL:${SRC:=$D/src} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8
export OCTOPUS_INIT_LORA=$D/lora OCTOPUS_ACT_GPU_GB=0
mkdir -p "$TMPDIR" "$D/logs"
cd "$SRC"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" >> "$D/logs/$LOG"
exec "$PY" -m torch.distributed.run --nproc-per-node $(( $(tr -cd , <<< "$GPUS" | wc -c) + 1 )) \
    --master-port $(( 29700 + ${GPUS%%,*} )) results/chunk-major/actor_ab.py "$MODEL" "$D/spool/${FILES//,/,$D/spool/}" \
    actor.lora_rank=64 actor.lora_alpha=64 "$@" >> "$D/logs/$LOG" 2>&1
