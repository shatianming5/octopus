#!/bin/bash
#SBATCH -J octopus-swe-pilot
#SBATCH -p alpha
#SBATCH -A cluster_project
#SBATCH --qos=priority
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH -c 48
#SBATCH --mem=700G
#SBATCH -t 1-00:00:00
#SBATCH --signal=B:SIGTERM@900
# The SWE RL pilot on Alpha (scripts/alpha/rl_run.env) as one 24 h job that keeps its allocation to the end (lib.sh hold):
# an oracle trial on this node (the Singularity backend must score 1.0 here), Val of the base (the zero adapter rq1-v0)
# on coupled-a-val (45 bundles, k=4, all 4 GPUs), training (rl_run.sh: 2 rollout + 2 training GPUs) to 15 steps, Val of
# the step-15 adapter with the same settings (P/run, made once from the env file), servers, k and tasks; then queued
# tasks (P/queue/*.sh, e.g. `train_to 40 && val v40 "$(adapter 40)"`) until the time limit. Each part is skipped when done
# (P/val-NAME.done; dev_eval resumes a partial file; rl_run.sh start TRIAL resumes training), so a new job with the same
# TRIAL continues. A failed stage stops its processes and the job holds. TERM 15 min before the limit: the exit trap
# stops the trainer (AReaL keeps a recover checkpoint per step) and every SGLang server, removes this user's /dev/shm trials.
#   sbatch -o ROOT/logs/pilot-%j.log ROOT/code-SHA/scripts/alpha/pilot.sbatch ROOT/code-SHA [TRIAL]
# (Slurm runs a copy of this file, so the code dir is an argument.) Results: ROOT/swe/pilot/TRIAL/val-v*.jsonl.
set -uo pipefail
C=$1; TRIAL=${2:-swe-read-alpha-$(date +%m%d%H%M)}
export RL_ENV_FILE=$C/scripts/alpha/rl_run.env CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"; [ ${#G[@]} = 4 ] || { echo "want 4 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }
NR=${SPLIT_ROLLOUT:-2}  # rollout GPUs, the rest train (a continuation may change it: SPLIT_ROLLOUT=3)
export INFER_GPUS=$(IFS=,; echo "${G[*]:0:NR}") TRAIN_GPUS=$(IFS=,; echo "${G[*]:NR}")
source "$RL_ENV_FILE"; source "$C/scripts/alpha/lib.sh"
P=$PILOT/$TRIAL; mkdir -p "$P"; VAL=$OCTOPUS_HARBOR/datasets/coupled-a-val
say "job $SLURM_JOB_ID on $(hostname), GPUs $CUDA_VISIBLE_DEVICES (rollout $INFER_GPUS, train $TRAIN_GPUS), trial $TRIAL, code $C"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
cleanup() {
  stop_stage; singularity instance stop --all > /dev/null 2>&1  # trials a killed episode left (overlay mode)
  find /dev/shm -maxdepth 1 -user "$USER" -name 'octopus-*' -exec rm -rf {} + 2>/dev/null
  say "exit; GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM

oracle() {  # one oracle trial on this node: the reference fixes in a Singularity container score 1.0
  local O=$P/oracle-$SLURM_JOB_ID T r; T=$(ls -d "$VAL"/*/ | head -1); T=${T%/}
  bgrun "$OCTOPUS_HARBOR" "$O.log" env PYTHONPATH="$C" "$OCTOPUS_HARBOR/bin/harbor" run -p "$T" -a oracle -e "$OCTOPUS_HARBOR_ENV" \
    -n 1 -o "$O" --job-name oracle --environment-build-timeout-multiplier 6 -y
  r=$("$AREAL_PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["verifier_result"]["rewards"]["reward"])' "$O"/oracle/*/result.json 2>/dev/null)
  [ "$r" = 1.0 ] || { say "oracle trial reward '$r' ($O.log)"; return 1; }; say "oracle 1.0 ($T)"
}
val() {  # val NAME ADAPTER_DIR: the adapter served as rq1-NAME on all 4 GPUs, coupled-a-val x 4 into P/val-NAME.jsonl
  [ -f "$P/val-$1.done" ] && return 0
  [ -f "$P/run/episode_settings.json" ] || bash "$C/scripts/rl_env/rl_run.sh" settings "$P/run" $EXTRA > "$P/settings.log" 2>&1 \
    || { say "settings failed: $P/settings.log"; return 1; }
  local S rc; S=$(timeout 1800 bash "$C/scripts/rl_env/eval_serve.sh" start "$CUDA_VISIBLE_DEVICES" "rq1-$1=$2" | tail -1)
  [[ $S == 127.0.0.1:* ]] || { say "eval_serve: $S"; return 1; }
  say "val $1 ($2) on $S"
  bgrun "$C" "$P/val-$1.log" env PYTHONPATH="$CURSOR/areal:$C" "$AREAL_PY" -m octopus.eval.dev_eval --run "$P/run" \
    --tasks-dir "$VAL" --k 4 --parallel 12 --servers "$S" --lora "rq1-$1" --tokenizer "$MODEL_TEXT" --out "$P/val-$1.jsonl" \
    --jobs-dir "$P/jobs-val-$1"; rc=$?
  bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null
  [ $rc = 0 ] && touch "$P/val-$1.done"; say "val $1 exit $rc: $(tail -n 3 "$P/val-$1.log" | tr '\n' ' ')"; return $rc
}

# setup.sbatch's SIFs and oracle sweep may still run when this job starts (user: both pilots queue at once): wait for
# ROOT/swe/READY, written once that sweep passed; it names the code dir to run (a fix made while this job queued): when
# that is another one, its copy of this script takes over in this allocation
until [ -f "$OCTO_ENV/swe/READY" ]; do sleep 60 & BG=$!; wait $BG; BG=""; done; say "READY: $(head -1 "$OCTO_ENV/swe/READY")"
takeover "$OCTO_ENV/swe/READY" scripts/alpha/pilot.sbatch
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 >> "$P/gpu-$SLURM_JOB_ID.csv" 2>&1 &  # per-GPU use
oracle && val v0 "$ADAPTER0" && train_to "$STEPS" && val "v$STEPS" "$(adapter "$STEPS")" && say "gate runs done: $P/val-v0.jsonl $P/val-v$STEPS.jsonl"
stop_stage; hold
