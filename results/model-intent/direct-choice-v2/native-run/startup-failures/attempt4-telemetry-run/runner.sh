#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/user/octopus-direct-choice-1b-20260914
PYTHON=/home/user/octopus-r2-20260913/venv/bin/python
MODEL=/home/user/octopus-r2-20260913/model/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
GPU=9

cd "$ROOT" || exit 90
mkdir -p run
start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
printf '%s\n' "$$" > run/runner.pid
printf '%s\n' "$start_iso" > run/start.utc
nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > run/gpu.before.csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > run/compute.before.csv 2>&1 || true

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/home/user/octopus-r2-20260913/venv/bin:$PATH
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED=1601
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_NO_USAGE_STATS=1

set +e
/usr/bin/timeout --signal=TERM --kill-after=10s 600s \
  "$PYTHON" -m benchmarks.semantic_choice infer --execute-model \
  --inputs "$ROOT/public/inputs.json" \
  --model-path "$MODEL" \
  --output "$ROOT/responses"
rc=$?
set -e

end_epoch=$(date +%s)
end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > run/gpu.after.csv 2>&1 || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > run/compute.after.csv 2>&1 || true
printf '%s\n' "$end_iso" > run/end.utc
printf '{"exit_code":%d,"start_utc":"%s","end_utc":"%s","elapsed_seconds":%d,"gpu":%d}\n' \
  "$rc" "$start_iso" "$end_iso" "$((end_epoch-start_epoch))" "$GPU" > run/terminal.json
exit "$rc"
