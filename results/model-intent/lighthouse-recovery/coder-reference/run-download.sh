#!/usr/bin/env bash
set -euo pipefail
cd /home/user/octopus-qwen-coder-reference-v1-20260915
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/home/user/octopus-r2-20260913/venv/bin/python
finish() {
  local rc=$?
  "$python" - "$rc" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
Path('download-terminal.json').write_text(json.dumps({
    'exit_code': int(sys.argv[1]), 'finished_at': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
  tmux wait-for -S qwen-coder-reference-download-v1-done
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT
"$python" - "$$" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
pid = int(sys.argv[1])
Path('download-job.json').write_text(json.dumps({
    'pid': pid, 'start_ticks': int((Path('/proc')/str(pid)/'stat').read_text().split()[21]),
    'started_at': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
tmux wait-for -S qwen-coder-reference-download-v1-started
timeout --signal=TERM --kill-after=5s 1770s "$python" download-verified.py > download.log 2>&1
