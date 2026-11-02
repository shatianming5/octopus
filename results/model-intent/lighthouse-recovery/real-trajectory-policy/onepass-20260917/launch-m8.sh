#!/usr/bin/env bash
set -uo pipefail

TASK=/home/user/octopus-m8-pilot-20260917-c84g1
RUN=$TASK/run-m8-onepass-c84g1
PYTHON=$TASK/venv/bin/python
LOG=$TASK/logs/m8-onepass.log
TERMINAL=$TASK/artifacts/m8-terminal.json

start_epoch=$(date +%s.%N)
start_iso=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
set +e
timeout --signal=TERM --kill-after=30 900 \
  env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$TASK/source:$TASK/source/src" \
  "$PYTHON" "$TASK/source/benchmarks/real_trajectory_policy.py" train \
  --data "$TASK/data/encoded.json" \
  --recipe "$TASK/config/frozen-recipe.json" \
  --output "$RUN" >"$LOG" 2>&1
returncode=$?
set -e
end_epoch=$(date +%s.%N)
end_iso=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
elapsed=$(awk -v s="$start_epoch" -v e="$end_epoch" 'BEGIN {printf "%.3f", e-s}')
printf '{\n  "returncode": %s,\n  "started_at": "%s",\n  "ended_at": "%s",\n  "elapsed_seconds": %s,\n  "gpu_index": 0,\n  "gpu_uuid": "GPU-a4a7a4a7-1cff-3992-8333-1f4784e967e8",\n  "run_path": "%s",\n  "log_path": "%s"\n}\n' \
  "$returncode" "$start_iso" "$end_iso" "$elapsed" "$RUN" "$LOG" >"$TERMINAL"
exit "$returncode"
