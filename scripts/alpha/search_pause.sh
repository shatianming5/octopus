#!/bin/bash
# pause the Search full trial (user 10-23): after the next "Train step N done" and its recover, stop the trainer in 144129,
# then the 4 Search servers in 144687 (rl_run.sh unserve there: the Search ones only)
N=/projects/cluster/octopus/octopus/code-d57c294
M=/projects/cluster/octopus/octopus/search/areal-exp/logs/user/rq1-rl/search-read-alpha-full-1022/main.log
[ -f $M ] || { echo "no $M"; exit 1; }; n=$(grep -c 'Train step' $M); until [ $(grep -c 'Train step' $M) -gt $n ]; do sleep 10; done
echo "$(date +%T) $(grep -oE 'Train step [0-9]+/' $M | tail -1) done"; sleep 60
srun --jobid=144129 --overlap -n1 -c1 bash -c 'cd /tmp; RL_ENV_FILE=/projects/cluster/octopus/octopus/code-d57c294/scripts/alpha/search.env bash /projects/cluster/octopus/octopus/code-d57c294/scripts/rl_env/rl_run.sh stop'
srun --jobid=144687 --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=$N/scripts/alpha/search.env bash $N/scripts/rl_env/rl_run.sh unserve" \
  || { echo "$(date +%T) unserve of the Search servers failed"; exit 1; }  # the Search servers only
sleep 60; srun --jobid=144687 --overlap -n1 -c1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo "$(date +%T) paused; covered: $(python3 /projects/cluster/octopus/octopus/logs/search_finished_qids.py /projects/cluster/octopus/octopus/search/rl/jobs-search-read-alpha-full-1022 /projects/cluster/octopus/octopus/search/analysis-covered.txt)"
