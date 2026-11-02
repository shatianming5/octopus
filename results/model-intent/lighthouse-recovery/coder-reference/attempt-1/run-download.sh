#!/usr/bin/env bash
set -u
cd /home/user/octopus-qwen-coder-reference-v1-20260915 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
python=/home/user/octopus-r2-20260913/venv/bin/python
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
timeout --signal=TERM --kill-after=5s 1800s "$python" download.py > download.log 2>&1
rc=$?
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
