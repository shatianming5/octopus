#!/bin/bash
# one third of SWE Val v36 (coupled-a-val bundles 15K..15K+14, k=4) on this job's 4 GPUs, the settings of Val v0 (P/run)
R=/projects/cluster/octopus/octopus; N=$R/code-d57c294; K=$1
export RL_ENV_FILE=$N/scripts/alpha/rl_run.env; source $RL_ENV_FILE; source $N/scripts/alpha/lib.sh
P=$PILOT/swe-read-alpha-full-1022; A=$(ls -d $FILEROOT/checkpoints/$USER/rq1-rl/swe-read-alpha-full-1022/default/*globalstep35 | head -1)
[ -f "$A/adapter_model.safetensors" ] || { echo "no step-36 adapter"; exit 1; }
bash $CODE/scripts/rl_env/rl_run.sh unserve  # this host's rollout servers
S=$(timeout 1800 bash $CODE/scripts/rl_env/eval_serve.sh start "$CUDA_VISIBLE_DEVICES" rq1-v36=$A | tail -1)
[[ $S == 127.0.0.1:* ]] || { echo "eval_serve: $S"; exit 1; }
echo "$(date +%T) part $K on $(hostname) $S, adapter $A"
cd $CODE && PYTHONPATH=$CURSOR/areal:$CODE $AREAL_PY -m octopus.eval.dev_eval --run $P/run --tasks-dir $OCTOPUS_HARBOR/datasets/coupled-a-val \
  --k 4 --parallel 12 --servers $S --lora rq1-v36 --tokenizer $MODEL_TEXT --out $P/val-v36-part$K.jsonl --jobs-dir $P/jobs-val-v36-part$K \
  --offset $((15 * K)) --limit 15 > $P/val-v36-part$K.log 2>&1; rc=$?
bash $CODE/scripts/rl_env/eval_serve.sh stop; echo "$(date +%T) part $K exit $rc: $(tail -n 2 $P/val-v36-part$K.log | tr '\n' ' ')"; exit $rc
