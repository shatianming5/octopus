#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/user/octopus-tiny-lora-step2-20260915
SOURCE="$ROOT/source"
DATA="$ROOT/data"
MODEL=/home/user/octopus-r2-20260913/model/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PYTHON=/home/user/octopus-r2-20260913/venv/bin/python
RUN="$ROOT/main-run"
DEV="$ROOT/dev-responses"
TEST="$ROOT/test-responses"
STATE="$ROOT/run"

cd "$SOURCE" || exit 90
test ! -e "$RUN" || exit 91
test ! -e "$DEV" || exit 92
test ! -e "$TEST" || exit 93
mkdir -p "$STATE"

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/home/user/octopus-r2-20260913/venv/bin:$PATH
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=9
export PYTHONHASHSEED=1601
export WORLD_SIZE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

overall_start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
overall_start_epoch=$(date +%s)
printf '%s\n' "$$" > "$STATE/runner.pid"
env | LC_ALL=C sort > "$STATE/environment.txt"
nvidia-smi -i 9 --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$STATE/gpu.before.csv"
nvidia-smi dmon -i 9 -s pucvmet -d 5 > "$STATE/gpu.dmon" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$STATE/gpu-monitor.pid"

finish() {
    rc=$1
    phase=$2
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    end_epoch=$(date +%s)
    nvidia-smi -i 9 --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$STATE/gpu.after.csv" 2>&1 || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$STATE/compute.after.csv" 2>&1 || true
    printf '{"exit_code":%d,"terminal_phase":"%s","start_utc":"%s","end_utc":"%s","elapsed_seconds":%d,"gpu":9}\n' \
        "$rc" "$phase" "$overall_start_iso" "$end_iso" "$((end_epoch-overall_start_epoch))" > "$STATE/terminal.json"
    exit "$rc"
}

run_phase() {
    name=$1
    limit=$2
    shift 2
    start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_epoch=$(date +%s)
    set +e
    /usr/bin/timeout --signal=TERM --kill-after=15s "$limit" \
        "$@" > "$STATE/$name.stdout" 2> "$STATE/$name.stderr"
    rc=$?
    set -e
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    end_epoch=$(date +%s)
    printf '{"phase":"%s","exit_code":%d,"start_utc":"%s","end_utc":"%s","elapsed_seconds":%d}\n' \
        "$name" "$rc" "$start_iso" "$end_iso" "$((end_epoch-start_epoch))" > "$STATE/$name.terminal.json"
    return "$rc"
}

run_phase train 3600s \
    "$PYTHON" -m benchmarks.tiny_lora train --execute-training \
    --model-path "$MODEL" \
    --training "$DATA/public/train-supervised.json" \
    --freeze "$DATA/freeze.json" \
    --output "$RUN" || { rc=$?; finish "$rc" train; }

run_phase dev 600s \
    "$PYTHON" -m benchmarks.tiny_lora infer --execute-model \
    --model-path "$MODEL" \
    --run "$RUN" \
    --split dev \
    --inputs "$DATA/public/dev-inputs.json" \
    --output "$DEV" || { rc=$?; finish "$rc" dev; }

run_phase test 600s \
    "$PYTHON" -m benchmarks.tiny_lora infer --execute-model \
    --model-path "$MODEL" \
    --run "$RUN" \
    --split test \
    --inputs "$DATA/public/test-inputs.json" \
    --output "$TEST" || { rc=$?; finish "$rc" test; }

finish 0 complete
