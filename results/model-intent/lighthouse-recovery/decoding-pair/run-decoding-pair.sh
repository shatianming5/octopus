#!/usr/bin/env bash
set -u
ROOT=/home/user/octopus-lighthouse-decoding-pair-v1-20260916
PY=/home/user/octopus-r2-20260913/venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export CUDA_VISIBLE_DEVICES=9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="/home/user/octopus-r2-20260913/venv/bin:/usr/local/cuda-12.9/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
job_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
job_start_epoch=$(date +%s)
printf "%s\n" "$job_start" > run/job-start.txt
run_arm() {
  arm=$1
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_epoch=$(date +%s)
  printf "%s\n" "$start" > "run/${arm}-start.txt"
  timeout --kill-after=5s 290s "$PY" -m benchmarks.lighthouse_fresh_reference run --freeze "run/${arm}-freeze.json" --output "run/${arm}-result.json" > "run/${arm}.log" 2>&1 &
  pid=$!
  printf "%s\n" "$pid" > "run/${arm}.pid"
  nvidia-smi -i 9 --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "run/${arm}-active-gpu.csv" 2>&1 || true
  wait "$pid"
  rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  end_epoch=$(date +%s)
  printf "%s\n" "$end" > "run/${arm}-end.txt"
  printf "%s\n" "$rc" > "run/${arm}.rc"
  printf "%s\n" "$((end_epoch-start_epoch))" > "run/${arm}-wall-seconds.txt"
}
run_arm constrained
constrained_rc=$(cat run/constrained.rc)
run_arm unconstrained
unconstrained_rc=$(cat run/unconstrained.rc)
job_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
job_end_epoch=$(date +%s)
export job_start job_end job_start_epoch job_end_epoch constrained_rc unconstrained_rc
nvidia-smi -i 9 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits > run/final-compute-processes.csv 2>&1 || true
"$PY" - <<'PY'
import json, os
from pathlib import Path
r=Path("run")
def text(name, default=None):
    p=r/name
    return p.read_text().strip() if p.exists() else default
payload={
  "status":"completed" if os.environ["constrained_rc"]=="0" and os.environ["unconstrained_rc"]=="0" else "failed",
  "job_start":os.environ["job_start"], "job_end":os.environ["job_end"],
  "wall_seconds":int(os.environ["job_end_epoch"])-int(os.environ["job_start_epoch"]),
  "arms":{
    a:{"returncode":int(text(f"{a}.rc","-999")),"start":text(f"{a}-start.txt"),"end":text(f"{a}-end.txt"),"wall_seconds":int(text(f"{a}-wall-seconds.txt","0")),"result_exists":(r/f"{a}-result.json").exists(),"log":f"run/{a}.log","active_gpu":text(f"{a}-active-gpu.csv")}
    for a in ("constrained","unconstrained")
  },
  "total_model_requests_max":2,
  "final_compute_processes":text("final-compute-processes.csv","")
}
(r/"terminal-status.json").write_text(json.dumps(payload,indent=2)+"\n")
PY
printf "%s\n" "$job_end" > run/job-end.txt
