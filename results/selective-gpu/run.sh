#!/usr/bin/env bash
# Isolated Selective B5 GPU acceptance orchestrator. Not a source mutation.
set -euo pipefail

ROOT="/mnt/disk2/user/octopus-selective-gpu-a9638a5-4007b0b5"
RESULT_DIR="$ROOT/results/selective-gpu"
PY="/mnt/disk2/user/octopus-contention/.venv/bin/python"
MODEL="/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
COMMIT="a9638a585a6de38900945d3d88ac03d10b3af513"
GPU="6"
PROTOCOL="$RESULT_DIR/protocol.json"
RECORDS="$RESULT_DIR/records.jsonl"
SUMMARY="$RESULT_DIR/summary.json"
JUNIT="$RESULT_DIR/logs/junit.xml"
STATUS="$RESULT_DIR/execution/status.txt"
WALL_START_FILE="$RESULT_DIR/execution/wall-start-utc.txt"
WALL_END_FILE="$RESULT_DIR/execution/wall-end-utc.txt"
RESOURCE_WALL_FILE="$RESULT_DIR/execution/resource-wall-seconds.txt"
KERNEL_FILE="$RESULT_DIR/execution/kernel-time.txt"

cd "$ROOT"
mkdir -p "$RESULT_DIR"/{logs,freeze,execution,cleanup}

export PATH="/mnt/disk2/user/octopus-contention/.venv/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/tests"
export OCTOPUS_COMMIT="$COMMIT"
export OCTOPUS_MODEL_PATH="$MODEL"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

wall_start_ns="$(date +%s%N)"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$WALL_START_FILE"
printf 'unmeasured; no tokens-to-kernel conversion\n' > "$KERNEL_FILE"
echo "$$" > "$RESULT_DIR/execution/orchestrator-pid.txt"

{
  echo "phase=freeze"
  echo "start_utc=$(cat "$WALL_START_FILE")"
  echo "host=$(hostname)"
  echo "commit=$COMMIT"
  echo "cuda_visible_devices=$GPU"
  echo "deploy=$ROOT"
  echo "model=$MODEL"
  echo "python=$PY"
  echo "protocol=$PROTOCOL"
} | tee "$STATUS" "$RESULT_DIR/freeze/identity.txt"

nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv \
  | tee "$RESULT_DIR/freeze/nvidia-smi-before.csv"
nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ' | tee "$RESULT_DIR/freeze/gpu-uuid.txt"
hostname | tee "$RESULT_DIR/freeze/host.txt"
printf '%s\n' "$COMMIT" | tee "$RESULT_DIR/freeze/commit.txt"
printf '%s\n' "a09a35458c702b33eeacc393d103063234e8bc28" | tee "$RESULT_DIR/freeze/model_revision.txt"
printf '%s\n' "$MODEL" | tee "$RESULT_DIR/freeze/model_path.txt"
printf '%s\n' "$PY" | tee "$RESULT_DIR/freeze/python.txt"
cp -a "$PROTOCOL" "$RESULT_DIR/freeze/protocol.json"

if [[ ! -x "$PY" ]]; then
  echo "missing python: $PY" | tee -a "$STATUS"
  echo "missing python: $PY" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "missing model snapshot: $MODEL" | tee -a "$STATUS"
  echo "missing model snapshot: $MODEL" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "missing protocol: $PROTOCOL" | tee -a "$STATUS"
  echo "missing protocol: $PROTOCOL" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ -e "$RECORDS" ]]; then
  echo "records already exist; refusing to mutate: $RECORDS" | tee -a "$STATUS"
  echo "records already exist" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ -e "$SUMMARY" ]]; then
  echo "summary already exists; refusing to mutate: $SUMMARY" | tee -a "$STATUS"
  echo "summary already exists" > "$RESULT_DIR/FAILED"
  exit 2
fi

"$PY" - <<'PY' | tee "$RESULT_DIR/freeze/environment.txt"
import os, sys
print("python", sys.version.replace("\n", " "))
print("executable", sys.executable)
print("OCTOPUS_COMMIT", os.environ.get("OCTOPUS_COMMIT"))
print("OCTOPUS_MODEL_PATH", os.environ.get("OCTOPUS_MODEL_PATH"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE"))
print("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE"))
print("HF_DATASETS_OFFLINE", os.environ.get("HF_DATASETS_OFFLINE"))
print("VLLM_ENABLE_V1_MULTIPROCESSING", os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))
import torch, vllm, pytest
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("pytest", pytest.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device_count", torch.cuda.device_count())
    print("torch.cuda.get_device_name", torch.cuda.get_device_name(0))
    print("torch.cuda.total_memory", torch.cuda.get_device_properties(0).total_memory)
PY

record_cleanup() {
  local rc="$1"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$WALL_END_FILE"
  "$PY" - <<PY | tee "$RESOURCE_WALL_FILE"
from datetime import datetime, timezone
from pathlib import Path
start = Path("$WALL_START_FILE").read_text().strip()
end = Path("$WALL_END_FILE").read_text().strip()
fmt = "%Y-%m-%dT%H:%M:%SZ"
s = datetime.strptime(start, fmt).replace(tzinfo=timezone.utc)
e = datetime.strptime(end, fmt).replace(tzinfo=timezone.utc)
print(int((e - s).total_seconds()))
PY
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv \
    | tee "$RESULT_DIR/cleanup/nvidia-smi-after.csv" || true
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv \
    | tee "$RESULT_DIR/cleanup/gpu6-compute-apps.csv" || true
  ps -o pid,ppid,etime,cmd -p "$$" --no-headers | tee "$RESULT_DIR/cleanup/orchestrator-ps.txt" || true
  pgrep -af "test_gpu_selective|benchmarks.selective|pytest" | tee "$RESULT_DIR/cleanup/matching-processes.txt" || true
  echo "orchestrator_exit=$rc" | tee "$RESULT_DIR/cleanup/orchestrator-exit.txt"
}

trap 'rc=$?; record_cleanup "$rc"' EXIT

echo "phase=gpu_correctness" | tee -a "$STATUS"
set +e
"$PY" -m pytest -q tests/test_gpu_selective.py \
  --junitxml="$JUNIT" \
  > "$RESULT_DIR/logs/pytest.log" 2>&1
pytest_rc=$?
set -e
echo "pytest_exit=$pytest_rc" | tee "$RESULT_DIR/execution/pytest-exit.txt"
echo "pytest_exit=$pytest_rc" | tee -a "$STATUS"
if [[ "$pytest_rc" -ne 0 ]]; then
  echo "correctness selectors failed; stopping before 48-record matrix" | tee -a "$STATUS"
  echo "correctness_failed" > "$RESULT_DIR/FAILED"
  exit "$pytest_rc"
fi

echo "phase=execute_gpu" | tee -a "$STATUS"
set +e
"$PY" -m benchmarks.selective \
  --execute-gpu \
  --protocol "$PROTOCOL" \
  --output "$RECORDS" \
  > "$RESULT_DIR/logs/execute-gpu.log" 2>&1
gpu_rc=$?
set -e
echo "execute_gpu_exit=$gpu_rc" | tee "$RESULT_DIR/execution/execute-gpu-exit.txt"
echo "execute_gpu_exit=$gpu_rc" | tee -a "$STATUS"
if [[ -f "$RECORDS" ]]; then
  wc -l < "$RECORDS" | tee "$RESULT_DIR/execution/record-count.txt"
fi
if [[ "$gpu_rc" -ne 0 ]]; then
  echo "execute-gpu failed; preserving records if any" | tee -a "$STATUS"
  echo "execute_gpu_failed" > "$RESULT_DIR/FAILED"
fi

if [[ -f "$RECORDS" ]]; then
  echo "phase=cpu_report" | tee -a "$STATUS"
  set +e
  CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.selective \
    --report "$RECORDS" \
    --output "$SUMMARY" \
    > "$RESULT_DIR/logs/report.log" 2>&1
  report_rc=$?
  set -e
  echo "report_exit=$report_rc" | tee "$RESULT_DIR/execution/report-exit.txt"
  echo "report_exit=$report_rc" | tee -a "$STATUS"
  if [[ "$report_rc" -ne 0 ]]; then
    echo "cpu report failed" | tee -a "$STATUS"
    echo "report_failed" >> "$RESULT_DIR/FAILED"
  fi
else
  echo "no records.jsonl; skipping cpu report" | tee -a "$STATUS"
  report_rc=1
fi

if [[ "$gpu_rc" -eq 0 && "${report_rc:-1}" -eq 0 ]]; then
  echo "complete" | tee "$RESULT_DIR/DONE"
  echo "complete" | tee -a "$STATUS"
  exit 0
fi
exit 1
