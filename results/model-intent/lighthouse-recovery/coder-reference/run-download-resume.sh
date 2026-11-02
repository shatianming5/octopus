#!/usr/bin/env bash
set -euo pipefail
cd /home/user/octopus-qwen-coder-reference-v1-20260915
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/home/user/octopus-r2-20260913/venv/bin/python
test ! -e download-result.json
test -f download-terminal.json
finish() {
  local rc=$?
  "$python" - "$rc" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
Path('download-resume-terminal.json').write_text(json.dumps({
    'exit_code': int(sys.argv[1]), 'finished_at': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
  tmux wait-for -S qwen-coder-reference-resume-v1-done
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT
remaining=$("$python" - "$$" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
root = Path('.')
spent = 0.0
for directory in (root/'attempt-1', root):
    start = json.loads((directory/'download-job.json').read_text())['started_at']
    end = json.loads((directory/'download-terminal.json').read_text())['finished_at']
    spent += (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
limit = json.loads((root/'transport-amendment.json').read_text())['aggregate_active_download_job_wall_seconds_max']
remaining = int(limit - spent - 5)
if remaining <= 0:
    raise RuntimeError('aggregate active download budget exhausted')
pid = int(sys.argv[1])
with Path('download-resume-job.json').open('x') as stream:
    json.dump({
        'pid': pid, 'start_ticks': int((Path('/proc')/str(pid)/'stat').read_text().split()[21]),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'prior_active_job_wall_seconds': spent, 'timeout_seconds': remaining,
    }, stream, indent=2)
    stream.write('\n')
print(remaining)
PY
)
tmux wait-for -S qwen-coder-reference-resume-v1-started
timeout --signal=TERM --kill-after=5s "${remaining}s" "$python" download-resume.py > download-resume.log 2>&1
