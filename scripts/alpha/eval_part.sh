#!/bin/bash
# eval_part.sh K TASKS_DIR OUT NAME=ADAPTER_DIR ...: bundles 15K..15K+14 of TASKS_DIR, k=4, on this job's 4 GPUs with the
# settings of Val v0 (P/run), every adapter served together and evaluated at once (dev_eval --parallel 6 each), rows into
# P/OUT-NAME-partK.jsonl; RUN (env, default P/run): another settings dir, e.g. P/run-plan-control. Like val_part.sh, for the SWE full trial (Test: v0 and v36 on the same node and load)
R=/projects/cluster/octopus/octopus; N=$R/code-d57c294; K=$1 D=$2 O=$3; shift 3
export RL_ENV_FILE=$N/scripts/alpha/rl_run.env; source $RL_ENV_FILE; source $N/scripts/alpha/lib.sh
P=$PILOT/swe-read-alpha-full-1022
bash $CODE/scripts/rl_env/rl_run.sh unserve  # this host's rollout servers, if any are left
S=$(timeout 1800 bash $CODE/scripts/rl_env/eval_serve.sh start "$CUDA_VISIBLE_DEVICES" "$@" | tail -1)
[[ $S == 127.0.0.1:* ]] || { echo "eval_serve: $S"; exit 1; }
echo "$(date +%T) part $K of $D on $(hostname) $S: $*"
pids=""; cd $CODE
for a in "$@"; do n=${a%%=*}
  PYTHONPATH=$CURSOR/areal:$CODE $AREAL_PY -m octopus.eval.dev_eval --run ${RUN:-$P/run} --tasks-dir $D --k 4 --parallel 6 --servers $S \
    --lora $n --tokenizer $MODEL_TEXT --out $P/$O-$n-part$K.jsonl --jobs-dir $P/jobs-$O-$n-part$K --offset $((15 * K)) --limit 15 \
    > $P/$O-$n-part$K.log 2>&1 & pids="$pids $!"
done
rc=0; for p in $pids; do wait $p || rc=1; done
bash $CODE/scripts/rl_env/eval_serve.sh stop; echo "$(date +%T) part $K exit $rc"; exit $rc
