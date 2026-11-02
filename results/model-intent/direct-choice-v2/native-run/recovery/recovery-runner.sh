#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/user/octopus-direct-choice-1b-20260914
RUN="$ROOT/recovery-run"
PYTHON=/home/user/octopus-r2-20260913/venv/bin/python
CALL_STARTS="$RUN/call-starts.log"

cd "$ROOT" || exit 90
test ! -e "$ROOT/responses-recovery" || exit 91
mkdir -p "$RUN"
: > "$CALL_STARTS"

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/home/user/octopus-r2-20260913/venv/bin:$PATH
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=9
export PYTHONHASHSEED=1601
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CALL_STARTS

start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
printf '%s\n' "$$" > "$RUN/runner.pid"
nvidia-smi -i 9 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$RUN/gpu.before.csv"

set +e
/usr/bin/timeout --signal=KILL 600s \
  "$PYTHON" "$ROOT/recovery.py" > "$RUN/inference.stdout" 2> "$RUN/inference.stderr" &
timeout_pid=$!
printf '%s\n' "$timeout_pid" > "$RUN/timeout.pid"

python_pid=
for _ in $(seq 1 50); do
    python_pid=$(pgrep -P "$timeout_pid" | head -n1)
    test -n "$python_pid" && break
    sleep 0.1
done
printf '%s\n' "$python_pid" > "$RUN/python.pid"

(
    sleep 120
    calls=$(wc -l < "$CALL_STARTS")
    printf '%s\n' "$calls" > "$RUN/calls-at-120s.txt"
    if test "$calls" -eq 0 && kill -0 "$python_pid" 2>/dev/null; then
        printf '%s\n' "$python_pid" > "$RUN/killed-zero-call-python.pid"
        kill -KILL "$python_pid"
    fi
) &
monitor_pid=$!

wait "$timeout_pid"
rc=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
set -e

end_epoch=$(date +%s)
end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nvidia-smi -i 9 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$RUN/gpu.after.csv" 2>&1 || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$RUN/compute.after.csv" 2>&1 || true
printf '{"exit_code":%d,"start_utc":"%s","end_utc":"%s","elapsed_seconds":%d,"gpu":9,"calls_started":%d}\n' \
  "$rc" "$start_iso" "$end_iso" "$((end_epoch-start_epoch))" "$(wc -l < "$CALL_STARTS")" > "$RUN/terminal.json"
exit "$rc"
