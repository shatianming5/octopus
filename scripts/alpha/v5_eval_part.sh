#!/bin/bash
# v5_eval_part.sh K SET OFFSET LIMIT: bundles OFFSET..OFFSET+LIMIT-1 of SWE SET (val = coupled-a-val, test = coupled-a-test),
# k=4, for SWE v5's step-36 adapter served as rq1-v5 on this job's 4 GPUs (dev_eval --parallel 12: the per-node load of the
# baseline's Val and Test), the settings of Val v0 (P/run), the code of the baseline's evals (eval_part.sh); rows into P/SET-v5-partK.jsonl
R=/projects/cluster/octopus/octopus; N=$R/code-d57c294; K=$1 SET=$2
export RL_ENV_FILE=$N/scripts/alpha/rl_run.env; source $RL_ENV_FILE; source $N/scripts/alpha/lib.sh
P=$PILOT/swe-read-alpha-v5-1023; A=$(ls -d $FILEROOT/checkpoints/$USER/rq1-rl/swe-read-alpha-v5-1023/default/*globalstep35 | head -1)
[ -f "$A/adapter_model.safetensors" ] || { echo "no step-36 adapter"; exit 1; }
bash $CODE/scripts/rl_env/rl_run.sh unserve  # this host's rollout servers, if any are left
S=$(timeout 1800 bash $CODE/scripts/rl_env/eval_serve.sh start "$CUDA_VISIBLE_DEVICES" rq1-v5=$A | tail -1)
[[ $S == 127.0.0.1:* ]] || { echo "eval_serve: $S"; exit 1; }
echo "$(date +%T) part $K ($SET $3+$4) on $(hostname) $S, adapter $A"
cd $CODE && PYTHONPATH=$CURSOR/areal:$CODE $AREAL_PY -m octopus.eval.dev_eval --run $P/run --tasks-dir $OCTOPUS_HARBOR/datasets/coupled-a-$SET \
  --k 4 --parallel 12 --servers $S --lora rq1-v5 --tokenizer $MODEL_TEXT --out $P/$SET-v5-part$K.jsonl --jobs-dir $P/jobs-$SET-v5-part$K \
  --offset $3 --limit $4 > $P/$SET-v5-part$K.log 2>&1; rc=$?
bash $CODE/scripts/rl_env/eval_serve.sh stop; echo "$(date +%T) part $K exit $rc"; exit $rc
