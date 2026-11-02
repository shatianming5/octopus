#!/usr/bin/env bash
set -euo pipefail
root=/mnt/disk2/user/octopus-contention/r2-keep-continuity-20260912-1e12c6
campaign=/mnt/disk2/user/octopus-contention/r2-physical-pressure-discovery-20260912
prior=/mnt/disk2/user/octopus-contention/r2-resolved-history-followup-20260912-d3c687
cd "$root/native-source"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1601
export PYTHONPATH=src:.:tests
/mnt/disk2/user/octopus-contention/.venv/bin/python -m benchmarks.r2_resolved_followup \
  --freeze "$campaign/input/measurement-freeze-p10752.json" \
  --trace "$campaign/export/p10752-s1702/e2-normal-b6_semantic_value-1702.json.gz" \
  --fixed "$campaign/export/p10752-s1702/e2-normal-b6_semantic_fixed-1702.json.gz" \
  --queue "$campaign/export/p10752-s1702/e2-normal-b6_queue_aware-1702.json.gz" \
  --apc "$campaign/export/p10752-s1702/e2-normal-projected_apc-1702.json.gz" \
  --blanket "$prior/evidence/upper1702-blanket.json" \
  --mode next_reuse --source 3e7cf232f8aa254fd63712d9d75fa6fb486cf1be \
  --policy-source e869df19b30c36f07864be65baad7f789ff73325 \
  --seed 1702 --blocks 10752 --bytes 9865003008 \
  --output "$root/evidence/upper1702-next_reuse.json"
