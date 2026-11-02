#!/bin/bash
# serve-rollout.sh QUEUE DIR SERVE_HOST [GPUS=0,1,2,3]: writes QUEUE/01-serve-rollout-$(basename DIR).sh, a queue task of a
# held Beta job (scripts/beta/hold.sbatch, OCTO_QUEUE=QUEUE) that starts, in a clean environment, rollout servers of this
# code dir (scripts/beta/serve_rollout.sh: SERVE_HOST:32300+g on each GPU g, state in DIR; JUDGE=G: the judge on GPU G too).
# Run it from the deployed code dir ROOT/code-SHA (its cursor overlay ROOT/cursor-SHA beside it). SERVE_HOST: the job's node,
# `srun --jobid=J --overlap -n1 -c1 hostname -I`. The task ends once they answer; they serve until
#   srun --jobid=J --overlap -n1 -c1 bash ROOT/code-SHA/scripts/beta/serve_rollout.sh stop DIR
set -euo pipefail
Q=${1:?QUEUE}; D=${2:?DIR}; H=${3:?SERVE_HOST}; G=${4:-0,1,2,3}
C=$(cd "$(dirname "$0")/../../.." && pwd); F=$Q/01-serve-rollout-$(basename "$D").sh
[ ! -e "$F" ] || { echo "$F exists"; exit 1; }
echo "exec env -i HOME=\"\$HOME\" USER=\"\$USER\" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
CUDA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\" CUDA_DEVICE_ORDER=PCI_BUS_ID SERVE_HOST=$H${JUDGE:+ JUDGE=$JUDGE} bash $C/scripts/beta/serve_rollout.sh start $D $G" > "$F.tmp"
mv "$F.tmp" "$F"; echo "queued $F"
