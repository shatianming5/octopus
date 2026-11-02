#!/bin/bash
#SBATCH --partition=beta
#SBATCH --account=cluster_project
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --job-name=octo-hold
#SBATCH --output=/projects/cluster/octopus/beta/octopus/logs/hold-%j.out
# Resident controller on Cluster Beta: holds the allocation for its full wall time (user rule) and runs $OCTO_QUEUE/*.sh in
# name order, one at a time (task -> .running-JOB -> .done/.failed, output in <task>.log). No requeue.
# OCTO_QUEUE (sbatch --export=ALL,OCTO_QUEUE=<dir>, no commas): a job's own queue dir, so a task runs on the allocation
# it was queued for (jobs sharing one dir each take whichever task they rename first); default = ROOT/queue (216338's).
Q=${OCTO_QUEUE:-/projects/cluster/octopus/beta/octopus/queue}
mkdir -p "$Q"
echo "$(date) job $SLURM_JOB_ID on $(hostname) GPUs ${CUDA_VISIBLE_DEVICES:-?}; holding, watching $Q"
while :; do
  for t in "$Q"/*.sh; do
    [ -e "$t" ] || continue
    r="$t.running-$SLURM_JOB_ID"; mv "$t" "$r" || continue
    echo "$(date) run $t"; if bash "$r" > "$t.log" 2>&1; then mv "$r" "$t.done"; else mv "$r" "$t.failed"; fi
    echo "$(date) end $t"
  done
  sleep 60
done
