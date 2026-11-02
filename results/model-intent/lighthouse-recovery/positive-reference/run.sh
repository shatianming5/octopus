#!/usr/bin/env bash
set -euo pipefail
cd /home/user/octopus-lighthouse-repository-search-v1-20260915
export CUDA_VISIBLE_DEVICES=9 CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:/home/user/octopus-r2-20260913/venv/bin:$PATH
export PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=/home/user/octopus-lighthouse-positive-reference-v1-20260915
finish() {
  local rc=$?
  python - "$out" "$rc" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
(Path(sys.argv[1])/'terminal.json').write_text(json.dumps({'exit_code':int(sys.argv[2]),'finished_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
PY
  tmux wait-for -S lighthouse-positive-reference-v1-done
  exit "$rc"
}
trap finish EXIT
python - "$out" "$$" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
pid=int(sys.argv[2]); row={'pid':pid,'start_ticks':int((Path('/proc')/str(pid)/'stat').read_text().split()[21]),'started_at':datetime.now(timezone.utc).isoformat()}
(Path(sys.argv[1])/'job-identity.json').write_text(json.dumps(row,indent=2)+'\n')
PY
timeout --signal=TERM --kill-after=5s 300s python -m benchmarks.lighthouse_fresh_reference run \
  --freeze "$out/freeze.json" --output "$out/result.json" > "$out/engine.log" 2>&1
