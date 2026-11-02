#!/bin/bash
# v5 off 144123 before its 07:19 TERM: at the first step boundary after 06:15 stop the trainer, restart 144687's four
# SWE servers clean (a killed trainer can leave an adapter that will not unload), then resume on them alone, 16 each
M=/projects/cluster/octopus/octopus/swe/areal-exp/logs/user/rq1-rl/swe-read-alpha-v5-1023/main.log; VQ=/projects/cluster/octopus/octopus/swe/pilot/swe-read-alpha-v5-1023/queue; H=/projects/cluster/octopus/octopus/search/pilot/hold-144687/queue
[ -f $M ] || { echo "no $M"; exit 1; }
until [ $(date +%s) -ge $(date -d '2026-10-23 06:15' +%s) ]; do sleep 60; done
# no boundary by 07:10: stop anyway (resume from the last recover; the in-flight rollout is dropped and redone)
n=$(grep -c 'Train step' $M); until [ $(grep -c 'Train step' $M) -gt $n ] || [ $(date +%s) -ge $(date -d '2026-10-23 07:10' +%s) ]; do sleep 10; done
echo "$(date +%T) $(grep -oE 'Train step [0-9]+/' $M | tail -1) done (boundary: $([ $(grep -c 'Train step' $M) -gt $n ] && echo yes || echo 'no, 07:10 deadline'))"; sleep 60
srun --jobid=144129 --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=/projects/cluster/octopus/octopus/code-39c1180/scripts/alpha/rl_run.env bash /projects/cluster/octopus/octopus/code-39c1180/scripts/rl_env/rl_run.sh stop" || exit 1
srun --jobid=144687 --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=/projects/cluster/octopus/octopus/code-d57c294/scripts/alpha/rl_run.env bash /projects/cluster/octopus/octopus/code-d57c294/scripts/rl_env/rl_run.sh unserve" || exit 1
srun --jobid=144687 --overlap -n1 -c1 bash -c 'for p in $(ps -u user -o pid=,args= | awk "\$2==\"sleep\" && \$3==\"infinity\" {print \$1}"); do kill $p; done'  # its serve task ends
cp /projects/cluster/octopus/octopus/logs/swe-serve-10.0.4.25.sh $H/09-swe-servers.sh
until grep -q '^servers 10.0.4.25' $H/09-swe-servers.sh.log 2>/dev/null; do [ -e $H/09-swe-servers.sh.failed ] && exit 1; sleep 10; done
echo "source /projects/cluster/octopus/octopus/code-39c1180/scripts/alpha/lib.sh; run_clean scripts/alpha/pilot.sbatch /projects/cluster/octopus/octopus/code-39c1180 swe-read-alpha-v5-1023 BATCH=8 STEPS=36 SAVE_EVERY=6 SPLIT_ROLLOUT=0 EPISODES_PER_SERVER=16 BRANCH_CREDIT=gate EXTRA_SERVERS=10.0.4.25:31300,10.0.4.25:31301,10.0.4.25:31302,10.0.4.25:31303" > $VQ/02-swe-v5-4srv.sh
echo "$(date +%T) queued the 4-server resume"
