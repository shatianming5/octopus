#!/bin/bash
#SBATCH -J octopus-rl
#SBATCH -p h200x8-long
#SBATCH --gres=gpu:h200:8
#SBATCH -c 62
#SBATCH --mem=1300G
#SBATCH -t 2-00:00:00
# Bundle RL (Qwen3.5-4B, rank-64 all-linear LoRA) on one h200x8 node with scripts/nvwulf/rl_run.env: 5 SGLang rollout +
# 3 FSDP training GPUs, the bundles-v2-valid train split (what this launcher was written for; track A is not on this path). AReaL saves a recover checkpoint after every step, so a job that
# hits the time limit is continued by resubmitting with its trial name (ROOT/state/trial).
#   sbatch -o ROOT/logs/train-%j.log train.sbatch ROOT [TRIAL]
set -uo pipefail
ROOT=$1
export RL_ENV_FILE=$ROOT/code/scripts/nvwulf/rl_run.env
source "$RL_ENV_FILE"
python3 - "$ROOT/datasets/bundles-v2-valid" <<'EOF'
import json, sys
src = sys.argv[1]
train = json.load(open(f"{src}.splits.json"))["train"]
open(f"{src}.train.txt", "w").write("".join(f"{src}/{b}\n" for repo in sorted(train) for b in train[repo]))
EOF
bash "$CODE/scripts/rl_env/rl_run.sh" start ${2:-} || exit 1
tail --pid="$(cat "$STATE/trainer.pid")" -f /dev/null
bash "$CODE/scripts/rl_env/rl_run.sh" stop
