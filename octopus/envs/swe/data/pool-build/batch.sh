#!/bin/bash
# One screening batch, the 2026-10-06 pipeline (build-n560.sh / screen-mbs.sh / revalidate-n560.sh):
#   select -> regen -> baseline -> oracle+nop -> select_valid -> oracle+nop again -> finalize (drop rules) -> audit
# usage: batch.sh NAME KIND N          datasets/tasks-NAME (source), swesmith-NAME, swesmith-NAME-valid, swesmith-NAME-final
#        batch.sh NAME mb LISTFILE     multi-bug: copy the "dataset/task" dirs listed in LISTFILE (already regenerated)
#        batch.sh NAME src -           datasets/tasks-NAME already selected (tasks-no183)
#        batch.sh NAME ds -            datasets/swesmith-NAME already regenerated on node-94 (relay183.sh: node-183's regen_tasks.py is older)
# CONC (default 24) is the Harbor -n of the oracle and nop runs, which run in parallel; END=revalidate stops after the
# second pass (finalize + audit then run on node-94, which has the parquet).
set -u
NAME=$1 KIND=$2 ARG=$3
source /home/user/harbor-udocker/scripts/env.sh; export PYTHONPATH=$N/code
export UDOCKER_CREATE_SLOTS=${SLOTS:-$UDOCKER_CREATE_SLOTS}  # env.sh pins 4; relay183.sh passes SLOTS=8 on node-183
PY=$N/uv-tools/harbor/bin/python; AP=/mnt/disk1/user/areal/.venv/bin/python
B=/home/user/pool-build; D=$N/datasets; SRC=tasks-$NAME; DS=swesmith-$NAME
cd /home/user
log() { echo "$(date +%F_%T) $*"; }
run() {  # dataset agent job
  harbor run -p $D/$1 -a $2 -e octopus.envs.swe.udocker:UdockerEnvironment -n ${CONC:-24} -o $N/jobs \
    --job-name $3 --environment-build-timeout-multiplier 6 -y > $N/jobs/$3.console.log 2>&1
}
if [ "$KIND" = mb ]; then
  log select-mb; mkdir $D/$DS || exit 1; while read -r p; do cp -r $D/$p $D/$DS/; done < $ARG
elif [ "$KIND" != ds ]; then
  [ "$KIND" = src ] || { log select; $AP -m octopus.envs.swe.data.select_swesmith /home/user/swesmith-hf $ARG $SRC $KIND || exit 1; }
  log regen; $PY -m octopus.envs.swe.data.regen_tasks $SRC $DS || exit 1
fi
log baseline; $PY -m octopus.envs.swe.data.baseline_tasks $DS ${BASE:-16} > $N/jobs/baseline-$DS.jsonl 2> $N/jobs/baseline-$DS.err
log validate; run $DS oracle v1-$NAME-oracle & run $DS nop v1-$NAME-nop & wait
python3 -m octopus.envs.swe.data.select_valid $DS v1-$NAME-oracle v1-$NAME-nop $DS-valid || exit 1
log revalidate; run $DS-valid oracle v2-$NAME-oracle & run $DS-valid nop v2-$NAME-nop & wait
[ "${END:-}" = revalidate ] && { log BATCH_DONE; exit 0; }
log finalize; $AP $B/finalize_batch.py $NAME || exit 1
log audit; $AP -m octopus.envs.swe.data.audit_tasks /home/user/swesmith-hf $DS-final v2-$NAME-oracle v2-$NAME-nop 2>&1 | tee $B/audit-$NAME.txt
if [ -s "$B/unsound-$NAME.txt" ]; then
  mkdir -p "$D/$DS-unsound"
  while read -r t; do
    [ -d "$D/$DS-final/$t" ] && mv "$D/$DS-final/$t" "$D/$DS-unsound/"
  done < "$B/unsound-$NAME.txt"
  log "audit-drop $(grep -c . "$B/unsound-$NAME.txt") -> $D/$DS-unsound"
fi
[ "$KIND" = mb ] && { log mb_cover; $AP /home/user/octopus-sft/mb_cover.py $DS-final; }
log BATCH_DONE
