#!/bin/bash
#SBATCH -J octopus-validate
#SBATCH -p debug-h200x4
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=180G
#SBATCH -t 2:00:00
# One-GPU check of the NVwulf deployment, no training: SGLang (context 262144, the zero rank-64 adapter as rq1-v0), then
# dev_eval on the first bundle of the first 4 bundles-v2-valid eval repos x k=2 through the Singularity backend
# (fork depth 2: this file does not set OCTOPUS_FORK_DEPTH; 2 is the runtime default, verified in octopus/runtime/tree.py).
# bundles-v2-valid is what this launcher was written for; track A is not on this path. CURSOR is unset: this check
# does not start the cursor/mount overlay. Every 15 s the job's CPU time, memory and /dev/shm use go to usage.tsv.
#   sbatch -o ROOT/logs/validate-%j.log validate.sbatch ROOT
set -uo pipefail
ROOT=$1
export RL_ENV_FILE=$ROOT/code/scripts/nvwulf/rl_run.env LORA_RANK=64
source "$RL_ENV_FILE"
D=$ROOT/logs/validate-$SLURM_JOB_ID; mkdir -p "$D/tasks"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
while sleep 15; do
  echo "$(date +%s) $(awk '/usage_usec/ {print $2}' "$CG/cpu.stat") $(cat "$CG/memory.current") $(du -sb /dev/shm 2>/dev/null | cut -f1)"
done > "$D/usage.tsv" &
python3 - "$ROOT/datasets/bundles-v2-valid" "$D/tasks" <<'EOF'
import json, os, sys
src, dst = sys.argv[1:]
evals = json.load(open(f"{src}.splits.json"))["eval"]
for repo in sorted(evals)[:4]:
    os.symlink(f"{src}/{evals[repo][0]}", f"{dst}/{evals[repo][0]}")
EOF
echo "tasks: $(ls "$D/tasks" | tr '\n' ' ')"
SERVERS=$(timeout 1800 bash "$CODE/scripts/rl_env/eval_serve.sh" start 0 rq1-v0="$ADAPTER0" | tail -1)
echo "servers: $SERVERS"
if [[ $SERVERS == 127.0.0.1:* ]]; then
  cd "$CODE" && "$AREAL_PY" -m octopus.eval.dev_eval --servers "$SERVERS" --lora rq1-v0 --tokenizer "$MODEL_TEXT" \
    --out "$D/dev_eval.jsonl" --k 2 --parallel 8 --jobs-dir "$D/jobs" --tasks-dir "$D/tasks"
fi
bash "$CODE/scripts/rl_env/eval_serve.sh" stop
cp "$STATE/eval-serve-$(hostname -s)/sglang-g0.log" "$D/"
kill %1
