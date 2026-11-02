#!/usr/bin/env bash
set -u
cd /home/user/octopus-lighthouse-raw-evidence-v1-20260915 || exit 1
export CUDA_VISIBLE_DEVICES=9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:/home/user/octopus-r2-20260913/venv/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=/home/user/octopus-lighthouse-fresh-reference-v1-20260915
python - "$out" "$$" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
p = Path('/proc') / sys.argv[2] / 'stat'
row = {'pid': int(sys.argv[2]), 'start_ticks': int(p.read_text().split()[21]),
       'started_at': datetime.now(timezone.utc).isoformat(), 'gpu': 9}
(Path(sys.argv[1]) / 'job-identity.json').write_text(json.dumps(row, indent=2) + '\n')
PY
timeout --signal=TERM --kill-after=5s 300s python -m benchmarks.lighthouse_fresh_reference run \
  --freeze "$out/freeze.json" --output "$out/result.json" > "$out/engine.log" 2>&1
rc=$?
python - "$out" "$rc" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
(Path(sys.argv[1]) / 'terminal.json').write_text(json.dumps({
    'exit_code': int(sys.argv[2]), 'finished_at': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
tmux wait-for -S lighthouse-fresh-reference-v1-terminal
exit "$rc"
