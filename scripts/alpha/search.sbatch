#!/bin/bash
#SBATCH -J octopus-search-pilot
#SBATCH -p alpha
#SBATCH -A cluster_project
#SBATCH --qos=priority
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH -t 1-00:00:00
#SBATCH --signal=B:SIGTERM@900
# The Search RL pilot on Alpha (scripts/alpha/search.env), one job: the BM25 retrieval service of the OpenResearcher
# corpus (its 57 GB docs.arrow + index copied to /dev/shm first: random reads off NFS are slow) and the judge (JUDGE_MODEL)
# (SGLang, served as "judge") for the whole job (with JUDGE_HOST none here: a rollout job's lib.sh serve_rollout owns it); then Val of the base (the zero adapter rq1-v0) on the 500 OpenResearcher
# validation questions (k=1, the judge at evaluation's temperature, 3 policy servers on the non-judge GPUs); then
# training (rl_run.sh: 2 rollout + 1 training GPU) to STEPS; then Val of the step-STEPS adapter, same settings (P/run, made
# once from the env file), servers, k and questions; then queued tasks (P/queue/*.sh, lib.sh hold) until the time limit:
# the job keeps its allocation to the end (user, 10-22). Each part is skipped when done (P/val-NAME.done; the eval resumes
# a partial file; rl_run.sh start TRIAL resumes training), so a new job with the same TRIAL continues. A failed stage
# stops its processes and the job holds. TERM 15 min before the limit: the exit trap stops the trainer, every SGLang
# server (policy and judge) and the service, and removes the /dev/shm copy.
#   sbatch -o ROOT/logs/search-%j.log ROOT/code-SHA/scripts/alpha/search.sbatch ROOT/code-SHA [TRIAL]
# Results: ROOT/search/pilot/TRIAL/{val-v0,val-vSTEPS}.jsonl.
set -uo pipefail
C=$1; TRIAL=${2:-search-read-alpha-$(date +%m%d%H%M)}
export RL_ENV_FILE=$C/scripts/alpha/search.env CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"; [ ${#G[@]} = 4 ] || { echo "want 4 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }
NR=${SPLIT_ROLLOUT:-2}  # rollout GPUs after the judge's, the rest train (SPLIT_ROLLOUT=0 with EXTRA_SERVERS: all rollout remote)
NJ=1; [ -n "${JUDGE_HOST:-}" ] && NJ=0  # JUDGE_HOST: the judge runs there (lib.sh serve_rollout in a rollout job), none here
export JUDGE_GPU=${G[0]} INFER_GPUS=$(IFS=,; echo "${G[*]:NJ:NR}") TRAIN_GPUS=$(IFS=,; echo "${G[*]:NR+NJ}")
source "$RL_ENV_FILE"
source "$C/scripts/alpha/lib.sh"
P=$PILOT/$TRIAL; mkdir -p "$P" "$STATE"; SHM=/dev/shm/$USER-search-$SLURM_JOB_ID
say "job $SLURM_JOB_ID on $(hostname), GPUs $CUDA_VISIBLE_DEVICES (judge ${JUDGE_HOST:-$JUDGE_GPU}, rollout $INFER_GPUS, train $TRAIN_GPUS), trial $TRIAL, code $C"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
SP=""; JP=""
cleanup() {
  stop_stage
  [ -n "$JP" ] && kill -- "-$JP" 2>/dev/null; [ -n "$SP" ] && kill -- "-$SP" 2>/dev/null; sleep 5
  [ -n "$JP" ] && kill -9 -- "-$JP" 2>/dev/null; [ -n "$SP" ] && kill -9 -- "-$SP" 2>/dev/null
  rm -rf "$SHM"
  say "exit; GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM

takeover "$OCTO_ENV/search/CODE" scripts/alpha/search.sbatch  # ROOT/search/CODE: a code dir with a fix made while queued
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 >> "$P/gpu-$SLURM_JOB_ID.csv" 2>&1 &  # per-GPU use
if grep -qx "$SLURM_JOB_ID" "$OCTO_ENV/search/HOLD" 2>/dev/null; then  # a job id in ROOT/search/HOLD: no stages (the trial
  # runs elsewhere), the allocation waits for tasks in its own queue
  P=$PILOT/hold-$SLURM_JOB_ID; say "on hold (ROOT/search/HOLD), tasks: $P/queue/*.sh"; hold
fi

# 1. retrieval service (bm25, 8 processes on one port) and judge, each its own process group
services() {
local root=$SHM  # a failed copy (job 144687: the target vanished mid-copy, cause unknown) serves from /projects instead
mkdir -p "$SHM/search-scratch" && cp -r "$SEARCH_DATA/search-scratch/openresearcher" "$SHM/search-scratch/" \
  || { say "copy to $SHM failed: serving from $SEARCH_DATA"; rm -rf "$SHM"; root=$SEARCH_DATA; }
(cd "$C" && exec setsid "$AREAL_PY" -m octopus.envs.search.service serve --corpus openresearcher --root "$root" --backend bm25 \
  --tokenizer "$OCTO_ENV/models/Qwen3-0.6B-tokenizer" --port "$SEARCH_PORT" --workers 8) > "$P/search-service-$SLURM_JOB_ID.log" 2>&1 < /dev/null & SP=$!
[ -n "${JUDGE_HOST:-}" ] || judge_start 127.0.0.1 "$P/judge-$SLURM_JOB_ID.log"
up "$OCTOPUS_SEARCH_URL/health" $SP 120 || { say "search service down: $P/search-service-$SLURM_JOB_ID.log"; return 1; }
up "$OCTOPUS_JUDGE_URL/models" "${JP:-$$}" 180 || { say "judge down: $OCTOPUS_JUDGE_URL"; return 1; }  # remote: waits 30 min
say "search service $(curl -sf -m 5 "$OCTOPUS_SEARCH_URL/health"); judge up at $OCTOPUS_JUDGE_URL"
}

# 2. Val: val NAME ADAPTER_DIR, the adapter served as rq1-NAME on the 3 non-judge GPUs, the 500 validation questions x 1
val() {
  [ -f "$P/val-$1.done" ] && return 0
  [ -f "$P/run/episode_settings.json" ] || bash "$C/scripts/rl_env/rl_run.sh" settings "$P/run" $EXTRA > "$P/settings.log" 2>&1 \
    || { say "settings failed: $P/settings.log"; return 1; }
  local S rc; S=$(timeout 1800 bash "$C/scripts/rl_env/eval_serve.sh" start "${G[1]},${G[2]},${G[3]}" "rq1-$1=$2" | tail -1)
  [[ $S == 127.0.0.1:* ]] || { say "eval_serve: $S"; return 1; }
  say "val $1 ($2) on $S"
  bgrun "$C" "$P/val-$1.log" env PYTHONPATH="$CURSOR/areal:$C" "$AREAL_PY" -m octopus.eval.search_dev_eval --set validation \
    --dataset-dir "$SEARCH_DATA/datasets/OpenResearcher__OpenResearcher-Corpus-Gold-Doc" --run "$P/run" --servers "$S" \
    --lora "rq1-$1" --tokenizer "$MODEL_TEXT" --judge-url "$OCTOPUS_JUDGE_URL" --judge-model judge --k 1 --parallel 16 \
    --out "$P/val-$1.jsonl" --jobs-dir "$P/jobs-val-$1"; rc=$?
  bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null
  [ $rc = 0 ] && touch "$P/val-$1.done"; say "val $1 exit $rc: $(tail -n 3 "$P/val-$1.log" | tr '\n' ' ')"; return $rc
}
services && val v0 "$ADAPTER0" && train_to "$STEPS" && val "v$STEPS" "$(adapter "$STEPS")" && say "gate runs done: $P/val-v0.jsonl $P/val-v$STEPS.jsonl"
stop_stage; hold
