#!/bin/bash
# without waiting for the SWE Test (that runs in 144123 alone): the Search full trial resumes from its recover in 144129 (judge GPU0, training GPU1-3)
# with 4 rollout servers in 144687 (10.0.4.25:32300-32303, 16 episodes each), user 10-23; 144123 is left out (07:19 TERM)
R=/projects/cluster/octopus/octopus; N=$(head -1 $R/search/CODE)
echo "$(date +%T) start (no Test wait: the Test runs in 144123 alone); code $N"
E='exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash -c'
rm -f $R/search/pilot/hold-144687/queue/07-search-servers.sh.log  # before the task is queued: a rerun waits for this start's line only
echo "$E 'export RL_ENV_FILE=$N/scripts/alpha/search.env INFER_GPUS=0,1,2,3 TRAIN_GPUS= SERVE_HOST=10.0.4.25; source \$RL_ENV_FILE; bash \$CODE/scripts/rl_env/rl_run.sh serve && exec sleep infinity'" \
  > $R/search/pilot/hold-144687/queue/07-search-servers.sh
until grep -q '^servers 10.0.4.25' $R/search/pilot/hold-144687/queue/07-search-servers.sh.log 2>/dev/null; do
  [ -e $R/search/pilot/hold-144687/queue/07-search-servers.sh.failed ] && { echo "$(date +%T) the Search servers failed"; exit 1; }; sleep 20; done
# fresh servers hold rq1-v0 only; printed, not checked here: rl_run.sh start unloads whatever else they hold
for p in 32300 32301 32302 32303; do echo "$p: $(srun --jobid=144687 --overlap -n1 -c1 curl -s -m 10 http://10.0.4.25:$p/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"] if m.get("parent")])')"; done
echo 144129 > $R/search/rl-state-search/owner  # 144123 trains no Search any more
echo "source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/search.sbatch $N search-read-alpha-full-1022 BATCH=16 STEPS=190 SAVE_EVERY=10 SPLIT_ROLLOUT=0 EPISODES_PER_SERVER=16 EXTRA_SERVERS=10.0.4.25:32300,10.0.4.25:32301,10.0.4.25:32302,10.0.4.25:32303" \
  > $R/swe/pilot/swe-read-alpha-full-1022/queue/13-search-resume.sh
echo "$(date +%T) queued the Search resume in 144129"
