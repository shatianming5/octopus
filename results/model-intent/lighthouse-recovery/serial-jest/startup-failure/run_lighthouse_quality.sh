#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?repository root required}
PYTHON=/home/ubuntu/repos/octopus/.venv/bin/python
DATA=${2:-"$ROOT/results/model-intent/lighthouse-quality"}
if [[ "$DATA" != /* ]]; then
  DATA="$ROOT/$DATA"
fi
SIGNAL=${3:-octopus-lighthouse-quality}
TASK_REPO="$DATA/assets/task-repo"
RUN="$DATA/execution"
FREEZE="$DATA/freeze.json"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)
STATUS=failed
DECISION=EXECUTION_FAILURE
DETAIL="quality preflight exited before a terminal official result"

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
    "schema": "octopus-lighthouse-quality-terminal-v1",
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
    "schema": "octopus-lighthouse-quality-start-v1",
    "started_at": sys.argv[2],
    "pid": os.getpid(),
    "gpu_budget_seconds": 1170,
    "cpu_evaluator_budget_seconds": 1800,
    "planned_model_episodes": 1,
}, sort_keys=True, indent=2) + "\n")
PY
tmux wait-for -S "$SIGNAL-started"

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m benchmarks.lighthouse_quality solve \
  --freeze "$FREEZE" \
  --output-dir "$RUN/solver"

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m benchmarks.lighthouse_quality evaluate \
  --freeze "$FREEZE" \
  --task-repo "$TASK_REPO" \
  --output-dir "$RUN/solver"

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m benchmarks.lighthouse_quality report \
  --freeze "$FREEZE" \
  --output-dir "$RUN/solver"

if [[ -f "$RUN/solver/official.log" ]]; then
  gzip -n9 "$RUN/solver/official.log"
fi

DECISION=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "$RUN/solver/result.json")
STATUS=complete
DETAIL="single-method official Lighthouse quality preflight completed"
