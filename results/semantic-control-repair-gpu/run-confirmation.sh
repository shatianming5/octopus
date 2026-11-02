#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RESULT_DIR="$ROOT/results/semantic-control-repair-gpu"
PY="/mnt/disk2/user/octopus-contention/.venv/bin/python"
MODEL="/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
COMMIT="${OCTOPUS_COMMIT:-fd582d0126958aa1a056e65684d0e31d7f7f1276}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"

mkdir -p "$RESULT_DIR/traces" "$RESULT_DIR/logs"
STATUS="$RESULT_DIR/run-status.txt"
trap 'echo "exit_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$?" | tee -a "$STATUS"' EXIT
{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "pwd=$ROOT"
  echo "commit=$COMMIT"
  echo "cuda_visible_devices=$GPU"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,driver_version --format=csv
} | tee "$STATUS"

export PATH="/mnt/disk2/user/octopus-contention/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OCTOPUS_COMMIT="$COMMIT"
export OCTOPUS_MODEL_PATH="$MODEL"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

if [[ ! -x "$PY" ]]; then
  echo "missing python: $PY" | tee -a "$STATUS"
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "missing model snapshot: $MODEL" | tee -a "$STATUS"
  exit 2
fi

visible_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d ' ')"
echo "visible_uuid=$visible_uuid" | tee -a "$STATUS"
echo "$visible_uuid" > "$RESULT_DIR/gpu-uuid.txt"

echo "STAGE lifecycle $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
"$PY" -m pytest -q \
  tests/test_gpu_admission.py::test_semantic_soft_gpu_preemption_consumer_binding_and_no_lease \
  --junitxml="$RESULT_DIR/lifecycle-junit.xml" \
  2>&1 | tee "$RESULT_DIR/logs/lifecycle.log"
echo "STAGE lifecycle_done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${PIPESTATUS[0]}" | tee -a "$STATUS"

CONFIG="$RESULT_DIR/real-tool-gpu6-protocol.json"
for method in apc queue_aware semantic_soft; do
  echo "STAGE trace_$method $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
  "$PY" -m benchmarks.admission \
    --config "$CONFIG" \
    --condition tight_burst \
    --method "$method" \
    --seed 101 \
    --mode real_tool \
    --phase main \
    --output "$RESULT_DIR/traces/real_tool-tight_burst-101-${method}.json" \
    2>&1 | tee "$RESULT_DIR/logs/real_tool-tight_burst-101-${method}.log"
  echo "STAGE trace_${method}_done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${PIPESTATUS[0]}" | tee -a "$STATUS"
done

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
echo COMPLETE | tee "$RESULT_DIR/COMPLETE"
