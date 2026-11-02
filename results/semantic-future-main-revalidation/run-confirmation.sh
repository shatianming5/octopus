#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/user/octopus-main-revalidation-4007b0b5"
cd "$ROOT"
RESULT_DIR="$ROOT/results/semantic-future-main-revalidation"
PY="/mnt/disk2/user/octopus-contention/.venv/bin/python"
MODEL="/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
COMMIT="ccc073837c845c9a195eff2e66b06906e14f2057"
GPU="${CUDA_VISIBLE_DEVICES:-6}"

mkdir -p "$RESULT_DIR/traces" "$RESULT_DIR/logs" "$RESULT_DIR/identity"
STATUS="$RESULT_DIR/run-status.txt"
CONFIG="$RESULT_DIR/real-tool-gpu6-protocol.json"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "commit=$COMMIT"
  echo "cuda_visible_devices=$GPU"
  echo "deploy=$ROOT"
  echo "model=$MODEL"
  echo "python=$PY"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv
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
  echo "missing python: $PY" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "missing model snapshot: $MODEL" | tee -a "$STATUS"
  echo "missing model snapshot: $MODEL" > "$RESULT_DIR/FAILED"
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" | tee -a "$STATUS"
  echo "missing config: $CONFIG" > "$RESULT_DIR/FAILED"
  exit 2
fi

visible_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d ' ')"
echo "visible_uuid=$visible_uuid" | tee -a "$STATUS"
echo "$visible_uuid" > "$RESULT_DIR/gpu-uuid.txt"
printf '%s\n' "$visible_uuid" > "$RESULT_DIR/identity/device_uuid.txt"
hostname > "$RESULT_DIR/identity/host.txt"
printf '%s\n' "$COMMIT" > "$RESULT_DIR/identity/commit.txt"
printf '%s\n' "a09a35458c702b33eeacc393d103063234e8bc28" > "$RESULT_DIR/identity/model_revision.txt"
printf '%s\n' "$MODEL" > "$RESULT_DIR/identity/model_path.txt"

set +e
"$PY" - <<'PY' > "$RESULT_DIR/identity/environment.txt" 2>&1
import os, sys
print("python", sys.version.replace("\n", " "))
print("executable", sys.executable)
print("OCTOPUS_COMMIT", os.environ.get("OCTOPUS_COMMIT"))
print("OCTOPUS_MODEL_PATH", os.environ.get("OCTOPUS_MODEL_PATH"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE"))
print("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE"))
print("VLLM_ENABLE_V1_MULTIPROCESSING", os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))
import torch
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device_count", torch.cuda.device_count())
    print("torch.cuda.get_device_name", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("torch.cuda.total_memory", props.total_memory)
try:
    import vllm
    print("vllm", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    print("vllm_import_error", repr(exc))
PY
env_ec=$?
set -e
echo "STAGE environment_dump exit=$env_ec $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"

echo "STAGE lifecycle $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
set +e
"$PY" -m pytest -q \
  tests/test_gpu_admission.py::test_semantic_soft_gpu_preemption_consumer_binding_and_no_lease[semantic_future_only] \
  --junitxml="$RESULT_DIR/lifecycle-junit.xml" \
  > "$RESULT_DIR/logs/lifecycle.log" 2>&1
lifecycle_ec=$?
set -e
echo "STAGE lifecycle_done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$lifecycle_ec" | tee -a "$STATUS"
if [[ "$lifecycle_ec" -ne 0 ]]; then
  echo "lifecycle_failed exit=$lifecycle_ec" | tee -a "$STATUS"
  echo "lifecycle_failed exit=$lifecycle_ec" > "$RESULT_DIR/FAILED"
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
  exit "$lifecycle_ec"
fi

trace_fail=0
missing_trace=0
run_trace() {
  local method="$1"
  local seed="$2"
  local out="$RESULT_DIR/traces/real_tool-tight_burst-${seed}-${method}.json"
  local log="$RESULT_DIR/logs/real_tool-tight_burst-${seed}-${method}.log"
  echo "STAGE trace_${seed}_${method} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
  set +e
  "$PY" -m benchmarks.semantic_future run \
    --config "$CONFIG" \
    --method "$method" \
    --seed "$seed" \
    --phase main \
    --output "$out" \
    > "$log" 2>&1
  local ec=$?
  set -e
  echo "STAGE trace_${seed}_${method}_done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$ec" | tee -a "$STATUS"
  if [[ "$ec" -ne 0 ]]; then
    trace_fail=1
  fi
  if [[ ! -s "$out" ]]; then
    missing_trace=1
    echo "missing_or_empty_trace $out" | tee -a "$STATUS"
  fi
}

run_trace apc 1101
run_trace queue_aware 1101
run_trace semantic_naive_pin 1101
run_trace semantic_future_only 1101
run_trace queue_aware 1202
run_trace semantic_future_only 1202
run_trace apc 1202
run_trace semantic_naive_pin 1202
run_trace semantic_future_only 1303
run_trace semantic_naive_pin 1303
run_trace queue_aware 1303
run_trace apc 1303
run_trace semantic_naive_pin 1404
run_trace apc 1404
run_trace semantic_future_only 1404
run_trace queue_aware 1404

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
echo "trace_fail=$trace_fail missing_trace=$missing_trace" | tee -a "$STATUS"

if [[ "$trace_fail" -ne 0 || "$missing_trace" -ne 0 ]]; then
  echo "trace_chain_failed trace_fail=$trace_fail missing_trace=$missing_trace" | tee -a "$STATUS"
  echo "trace_chain_failed trace_fail=$trace_fail missing_trace=$missing_trace" > "$RESULT_DIR/FAILED"
  exit 1
fi

echo COMPLETE | tee "$RESULT_DIR/COMPLETE"
echo "CHAIN_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$STATUS"
