#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?reference directory required}
SIGNAL=${2:-octopus-json-repeat-stock-reference}
PYTHON=/home/ubuntu/repos/octopus/.venv/bin/python
REMOTE=user@198.51.100.10
PORT=20134
RDIR=/home/user/octopus-lighthouse-agent7-json-repeat-stock-v1-20260916
RUN="$DATA/execution"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)
STATUS=failed
DECISION=EXECUTION_FAILURE
DETAIL="stock reference exited before a terminal output"

mkdir -p "$RUN"
exec > >(tee "$RUN/job.log") 2>&1

finish() {
  local rc=$?
  local finished elapsed
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  elapsed=$(($(date +%s) - START_SECONDS))
  if [[ ! -e "$RUN/terminal.json" ]]; then
    "$PYTHON" - "$RUN/terminal.json" "$STATUS" "$DECISION" "$DETAIL" \
      "$STARTED_AT" "$finished" "$elapsed" "$rc" <<'PY'
import json
import sys
from pathlib import Path

path, status, decision, detail, started, finished, elapsed, rc = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "octopus-json-repeat-stock-terminal-v1",
    "status": status,
    "decision": decision,
    "detail": detail,
    "started_at": started,
    "finished_at": finished,
    "elapsed_seconds": int(elapsed),
    "exit_code": int(rc),
}, sort_keys=True, indent=2) + "\n")
PY
  fi
  tmux wait-for -S "$SIGNAL-done" || true
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT

"$PYTHON" - "$RUN/started.json" "$STARTED_AT" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "octopus-json-repeat-stock-start-v1",
    "started_at": sys.argv[2],
    "pid": os.getpid(),
    "gpu_budget_seconds": 300,
    "wall_budget_seconds": 450,
    "planned_model_requests": 1,
}, sort_keys=True, indent=2) + "\n")
PY
tmux wait-for -S "$SIGNAL-started" || true

timeout --signal=KILL 450 ssh -T -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
  "cd '$RDIR' && env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout --signal=KILL 300 ./runtime/python -m benchmarks.lighthouse_fresh_reference run --freeze reference-freeze.json --output execution/reference-output.json"

scp -P "$PORT" -o BatchMode=yes \
  "$REMOTE:$RDIR/execution/reference-output.json" \
  "$RUN/reference-output.json"

STATUS=complete
DECISION=REFERENCE_COMPLETE
DETAIL="one exact-input schema-constrained stock fresh-vLLM request completed"
