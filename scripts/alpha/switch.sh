#!/bin/bash
# switch.sh MAIN_LOG JOBID CODE ENV_FILE TASK_SRC TASK_DST: after MAIN_LOG's next "Train step N done" (+60 s for the
# recover checkpoint), stop that trial's trainer inside JOBID with CODE's rl_run.sh and ENV_FILE, then queue TASK_SRC as TASK_DST
M=$1 J=$2 C=$3 E=$4
[ -f "$M" ] || { echo "no $M"; exit 1; }  # a wrong path would make the [ -gt ] below true and stop mid-step
n=$(grep -c 'Train step' $M); echo "$(date +%T) at $n"
until [ $(grep -c 'Train step' $M) -gt $n ]; do sleep 10; done
echo "$(date +%T) $(grep -oE 'Train step [0-9]+/[0-9]+' $M | tail -1) done"; sleep 60
srun --jobid=$J --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=$E bash $C/scripts/rl_env/rl_run.sh stop"
cp "$5" "$6"; echo "$(date +%T) stopped, queued $6"
