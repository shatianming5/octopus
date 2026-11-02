#!/bin/bash
# bench.py on node-134 (results/chunk-major/README.md): run134.sh GPUS LOG MB [bench args...]
# GPUS: comma-separated card ids (one rank each). torchrun through the venv python: its script shebang names 138's path. Everything under /mnt/disk4/user/chunk-major-p2 (the 138 venv and
# AReaL overlay copied there, this code at the pushed commit, the micro-batches of select_mbs.py).
set -e
GPUS=$1 LOG=$2 MB=$3
shift 3
D=/mnt/disk4/user/chunk-major-p2
export TMPDIR=/mnt/disk4/user/tmp/chunk-major-p2 TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/disk4/user/tmp/chunk-major-p2/triton}
export TORCHINDUCTOR_CACHE_DIR=/mnt/disk4/user/tmp/chunk-major-p2/inductor
export PYTHONPATH=$D/areal-overlay:$D/code OCTOPUS_SGLANG_PY=/home/user/octo-env-data/sglang-venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$D/logs"
cd $D/code
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" >> "$D/logs/$LOG"
exec $D/.venv/bin/python -m torch.distributed.run --nproc-per-node $(( $(tr -cd , <<< "$GPUS" | wc -c) + 1 )) --master-port $(( 29600 + ${GPUS%%,*} )) \
    results/chunk-major/bench.py "$D/mbs/$MB" /mnt/disk4/user/octo-env-data/models/Qwen3.5-4B-text "$@" \
    >> "$D/logs/$LOG" 2>&1
