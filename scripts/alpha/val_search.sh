#!/bin/bash
# val_search.sh NAME ADAPTER_DIR: Search Val (the 500 OpenResearcher validation questions, k=1, the judge already serving
# in this job at 127.0.0.1:32500, eval at its temperature 0) of one adapter on GPUs 1-3 of this job, the settings of
# Val v0 (P/run), into P/val-NAME.jsonl; run in the Search trainer's job with srun --overlap (not a queue task: two jobs
# hold on the Search trial's queue)
[ -n "${VAL_SEARCH_CLEAN:-}" ] || exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin LANG=${LANG:-C} \
  $(env | grep -E '^SLURM_[A-Z0-9_]*=' | tr '\n' ' ') CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  VAL_SEARCH_CLEAN=1 bash "$0" "$@"  # the environment of a controller's val (run_clean), not the srun caller's
R=/projects/cluster/octopus/octopus; N=$R/code-d57c294
export RL_ENV_FILE=$N/scripts/alpha/search.env; source $RL_ENV_FILE; source $N/scripts/alpha/lib.sh
P=$PILOT/search-read-alpha-full-1022; IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
curl -sf -m 10 "$OCTOPUS_JUDGE_URL/models" > /dev/null || { echo "judge down"; exit 1; }
S=$(timeout 1800 bash $CODE/scripts/rl_env/eval_serve.sh start "${G[1]},${G[2]},${G[3]}" "rq1-$1=$2" | tail -1)
[[ $S == 127.0.0.1:* ]] || { echo "eval_serve: $S"; exit 1; }
cd $CODE && PYTHONPATH=$CURSOR/areal:$CODE $AREAL_PY -m octopus.eval.search_dev_eval --set validation \
  --dataset-dir $SEARCH_DATA/datasets/OpenResearcher__OpenResearcher-Corpus-Gold-Doc --run $P/run --servers $S \
  --lora rq1-$1 --tokenizer $MODEL_TEXT --judge-url $OCTOPUS_JUDGE_URL --judge-model judge --k 1 --parallel 16 \
  --out $P/val-$1.jsonl --jobs-dir $P/jobs-val-$1 > $P/val-$1.log 2>&1; rc=$?
bash $CODE/scripts/rl_env/eval_serve.sh stop; echo "$(date +%T) val $1 exit $rc: $(tail -n 2 $P/val-$1.log | tr '\n' ' ')"; exit $rc
