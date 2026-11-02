#!/bin/bash
# Search v5 on Cluster Beta (user, 10-23): from step 0, one data pass over the 3040 Search train questions with
# rl.branch_credit=gate (search's rule, a trainer setting: octopus/rl/group.py search_gate; v5 ran it under the old name search_gate; EXTRA_ADD below), then Val (the 500 OpenResearcher validation questions, k=1) of
# that step's adapter and of the base rq1-v0 on this hardware (Alpha's v0 0.360 ran on H200), paired. A queue task of a
# held Beta job (ROOT/hold.sbatch runs ROOT/queue/*.sh) on its 4 GB200s in the Alpha baseline's single-node topology
# (search-read-alpha-full-1022 steps 1-5, tasks/search-full.sh): judge GPU0 (Qwen3-32B: this run's record; now JUDGE_MODEL), rollout GPU1-2 (16 episodes
# each), training GPU3. Services and Val as scripts/alpha/search.sbatch; one pass as alpha/search_epoch_stop.sh had it: stop at
# the step boundary after every train qid has a finished episode (or 5 steps without a new one once 3000 have; the
# zero-variance filter draws ~60 questions a step), that step's adapter from the saver or the recover checkpoint.
# Resumable: a later job running it with the same TRIAL resumes training from its recover checkpoint and skips finished
# parts (P/train.done, P/val-*.done). Smoke test: STOP_STEP=N stops after step N, VAL_LIMIT=N runs Val on N questions,
# EXTRA_ADD (more overrides, e.g. rl.filter_zero_variance=false). No time limit of its own: scripts/beta/search_v5_deadline.sh.
#   bash search_v5.sh CODE TRIAL      results: ROOT/search/pilot/TRIAL/{val-vK,val-v0}.jsonl, paired.txt
[ -n "${SEARCH_V5_CLEAN:-}" ] || exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  SLURM_JOB_ID="$SLURM_JOB_ID" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" STOP_STEP="${STOP_STEP:-}" VAL_LIMIT="${VAL_LIMIT:-}" \
  EXTRA_ADD="${EXTRA_ADD:-}" SEARCH_V5_CLEAN=1 bash "$0" "$@"  # not the shared home's profile (Alpha's PYTHONPATH, TRITON_CACHE_DIR, SLURM_*)
set -uo pipefail
C=$1; TRIAL=$2
export RL_ENV_FILE=$C/scripts/beta/search.env CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"; [ ${#G[@]} = 4 ] || { echo "want 4 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }
export JUDGE_GPU=${G[0]} INFER_GPUS=${G[1]},${G[2]} TRAIN_GPUS=${G[3]}
export EXTRA_ADD="rl.branch_credit=gate${EXTRA_ADD:+ $EXTRA_ADD}"; source "$RL_ENV_FILE"
source "$C/scripts/alpha/lib.sh"
P=$PILOT/$TRIAL; mkdir -p "$P" "$STATE"; SHM=/dev/shm/$USER-search-$SLURM_JOB_ID; Q=$SEARCH_DATA/search/train-qids.txt
M=$FILEROOT/logs/$USER/rq1-rl/$TRIAL/main.log
say "job $SLURM_JOB_ID on $(hostname), GPUs $CUDA_VISIBLE_DEVICES (judge $JUDGE_GPU, rollout $INFER_GPUS, train $TRAIN_GPUS), trial $TRIAL, code $C"
SP=""; JP=""
cleanup() {
  stop_stage
  [ -n "$JP" ] && kill -- "-$JP" 2>/dev/null; [ -n "$SP" ] && kill -- "-$SP" 2>/dev/null; sleep 5
  [ -n "$JP" ] && kill -9 -- "-$JP" 2>/dev/null; [ -n "$SP" ] && kill -9 -- "-$SP" 2>/dev/null
  rm -rf "$SHM"
  say "exit; GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM
up() { local i; for i in $(seq "$3"); do curl -sf -m 5 "$1" > /dev/null && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 10; done; return 1; }

# 1. retrieval service (bm25, 8 processes on one port; its index copied to /dev/shm) and judge, each its own process group
services() {
mkdir -p "$SHM/search-scratch" && cp -r "$SEARCH_DATA/search-scratch/openresearcher" "$SHM/search-scratch/" \
  || { say "copy to $SHM failed"; return 1; }  # never served from Alpha's copy: the index there is not ours to write
(cd "$C" && exec setsid "$AREAL_PY" -m octopus.envs.search.service serve --corpus openresearcher --root "$SHM" --backend bm25 \
  --tokenizer "$OCTO_ENV/models/Qwen3-0.6B-tokenizer" --port "$SEARCH_PORT" --workers 8) > "$P/search-service-$SLURM_JOB_ID.log" 2>&1 < /dev/null & SP=$!
CUDA_VISIBLE_DEVICES=$JUDGE_GPU PATH="$(dirname "$SGLANG_PY"):$CUDA_HOME/bin:$PATH" CUDA_HOME=$CUDA_HOME SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  SGLANG_CACHE_DIR="$STATE/sglang-cache-judge" setsid "$SGLANG_PY" -m sglang.launch_server --model-path "$JUDGE_MODEL" --served-model-name judge \
  --host 127.0.0.1 --port "$JUDGE_PORT" --tp-size 1 --mem-fraction-static 0.85 --context-length 32768 --attention-backend triton \
  --sampling-backend pytorch --disable-prefill-cuda-graph > "$P/judge-$SLURM_JOB_ID.log" 2>&1 < /dev/null & JP=$!
up "$OCTOPUS_SEARCH_URL/health" $SP 120 || { say "search service down: $P/search-service-$SLURM_JOB_ID.log"; return 1; }
up "http://127.0.0.1:$JUDGE_PORT/v1/models" $JP 180 || { say "judge down: $P/judge-$SLURM_JOB_ID.log"; return 1; }
say "search service $(curl -sf -m 5 "$OCTOPUS_SEARCH_URL/health"); judge up"
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
    ${VAL_LIMIT:+--limit $VAL_LIMIT} --out "$P/val-$1.jsonl" --jobs-dir "$P/jobs-val-$1"; rc=$?
  bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null
  [ $rc = 0 ] && touch "$P/val-$1.done"; say "val $1 exit $rc: $(tail -n 3 "$P/val-$1.log" | tr '\n' ' ')"; return $rc
}

# 3. one pass (or STOP_STEP steps), then that step's adapter (P/train.done: "K ADAPTER_DIR")
step() { grep -oE 'Train step [0-9]+' "$M" 2>/dev/null | tail -1 | awk '{print $3 + 0}'; }
alive() { kill -0 "$(cat "$STATE/trainer.pid")" 2>/dev/null || { say "trainer exited: $STATE/train-$TRIAL.log"; return 1; }; }
train_pass() {
  [ -f "$P/train.done" ] && return 0
  STEPS=190 bash "$C/scripts/rl_env/rl_run.sh" start "$TRIAL" || return 1
  local want n s last best=0 flat=0 K A; want=$(wc -l < "$Q"); last=$(step)
  mkdir -p "$JOBS_ROOT/jobs-$TRIAL"  # search_finished_qids.py lists it before the first episode starts
  while alive; do
    n=$("$AREAL_PY" "$C/scripts/alpha/search_finished_qids.py" "$JOBS_ROOT/jobs-$TRIAL" "$P/covered.txt" "$Q"); s=$(step)
    if [ "${s:-0}" -gt "${last:-0}" ]; then
      last=$s; if [ "$n" -gt "$best" ]; then best=$n; flat=0; else flat=$((flat + 1)); fi
      say "step $s: $n of $want train qids finished"
      [ -n "${STOP_STEP:-}" ] && [ "$s" -ge "$STOP_STEP" ] && break
    fi
    if [ -z "${STOP_STEP:-}" ] && { [ "$n" -ge "$want" ] || { [ "$n" -ge 3000 ] && [ "$flat" -ge 5 ]; }; }; then
      say "covered ($n of $want): stop at the next step boundary"
      while alive && [ "$(step)" -le "${s:-0}" ]; do sleep 10; done; break
    fi
    sleep 120 & BG=$!; wait $BG; BG=""
  done
  alive || return 1
  sleep 60  # that step's recover checkpoint
  bash "$C/scripts/rl_env/rl_run.sh" stop > /dev/null; K=$(step); A=$(final_adapter "$K") || return 1
  echo "$K $A" > "$P/train.done"; say "step-$K adapter $A"
}

services && train_pass && read -r K A < "$P/train.done" && val "v$K" "$A" && val v0 "$ADAPTER0" \
  && "$AREAL_PY" "$C/scripts/alpha/paired.py" "$P/val-v0.jsonl" "$P/val-v$K.jsonl" | tee "$P/paired.txt" && say "SEARCH V5 DONE (step $K)"
