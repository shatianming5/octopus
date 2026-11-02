#!/usr/bin/env bash
set -euo pipefail
root=/mnt/disk2/user/octopus-contention/r2-history-serving-integration-20260912-bb698a
campaign=/mnt/disk2/user/octopus-contention/r2-current-admission-gpu-20260912
blanket=/mnt/disk2/user/octopus-contention/r2-history-matched-20260912-d3c687/evidence/blanket-1601.json
cd "$root/native-source"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1601
export PYTHONPATH=src:.:tests
/mnt/disk2/user/octopus-contention/.venv/bin/python -m benchmarks.r2_resolved_followup \
  --freeze "$campaign/input/measurement-freeze-s1601.json" \
  --trace "$campaign/export/s1601/e2-normal-b6_semantic_value-1601.json.gz" \
  --fixed "$campaign/export/s1601/e2-normal-b6_semantic_fixed-1601.json.gz" \
  --queue "$campaign/export/s1601/e2-normal-b6_queue_aware-1601.json.gz" \
  --apc "$campaign/export/s1601/e2-normal-projected_apc-1601.json.gz" \
  --blanket "$blanket" --mode next_reuse \
  --source 807d46de552db10e274d2910e3676b1e62bd9934 \
  --policy-source 95481254b87a8bf9b0ffe5fc84f511bc97ebbde0 \
  --seed 1601 --blocks 9216 --bytes 8455716864 \
  --output "$root/evidence/lower1601-next_reuse.json"
