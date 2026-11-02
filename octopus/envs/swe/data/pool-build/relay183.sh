#!/bin/bash
# Runs on node-134 (the only host that reaches both): one batch whose Harbor runs happen on node-183.
# node-94: select, partition (tasks without a golden on 183 -> datasets/tasks-no183), regen (183's regen_tasks.py is older)
# -> dataset to 183 -> baseline .. revalidate there (batch.sh ds, END=revalidate) -> valid set + jobs back to node-94
# -> finalize + audit on node-94 (which has the parquet).      usage: relay183.sh NAME KIND N   (KIND=src: already selected)
set -u
NAME=$1 KIND=$2 ARG=$3; B=/home/user/pool-build; N=/home/user/harbor-udocker
H94="ssh -o BatchMode=yes user@10.0.1.94"; H183="ssh -o BatchMode=yes user@10.0.2.83"
ENV94="cd /home/user; source $N/scripts/env.sh; export PYTHONPATH=$N/code; AP=/mnt/disk1/user/areal/.venv/bin/python; PY=$N/uv-tools/harbor/bin/python"
log() { echo "$(date +%F_%T) $*"; }
# 183 sshd drops rapid successive connections (MaxStartups). pipefail so a dead writer is not a silent success.
copy() {
  local i
  for i in 1 2 3 4 5; do
    set -o pipefail
    $1 "$2" | $3 "$4"
    local rc=$?
    set +o pipefail
    [ $rc -eq 0 ] && return 0
    log "copy attempt $i failed rc=$rc"
    sleep 15
  done
  return 1
}
SELECT="\$AP -m octopus.envs.swe.data.select_swesmith /home/user/swesmith-hf $ARG tasks-$NAME $KIND &&"; [ "$KIND" = src ] && SELECT=""
if [ "$KIND" != ds ]; then  # ds: datasets/swesmith-NAME already regenerated on node-94
  log select; $H94 "$ENV94; $SELECT \$AP $B/partition.py tasks-$NAME $B/goldens-183.txt tasks-no183" || exit 1
  log regen; $H94 "$ENV94; \$PY -m octopus.envs.swe.data.regen_tasks tasks-$NAME swesmith-$NAME" || exit 1
fi
log copy-to-183; copy "$H94" "cd /home/user && tar c pool-build/batch.sh harbor-udocker/datasets/swesmith-$NAME" "$H183" "cd /home/user && tar x && mkdir -p $B/logs $N/jobs" || exit 1
# 183's env.sh has no fuse-overlayfs (full 3.3 GB container copies: ~1 task/min); my runs use a copy of node-94's binary.
RUN="UDOCKER_OVERLAY=$B/bin/fuse-overlayfs SLOTS=8 CONC=${CONC183:-48} BASE=${BASE183:-32} END=revalidate bash batch.sh $NAME ds -"
log run-183; for i in 1 2 3 4 5; do  # sshd on 183 drops rapid successive connections (MaxStartups); retry the launch
  $H183 "cd $B; setsid nohup bash -c '$RUN; echo EXIT \$?' > logs/$NAME.log 2>&1 < /dev/null & sleep 2; grep -q baseline logs/$NAME.log" && break
  log "launch attempt $i failed"; sleep 15; [ $i = 5 ] && exit 1
done
until $H183 "grep -q '^EXIT' $B/logs/$NAME.log" 2>/dev/null; do sleep 120; done
$H183 "cat $B/logs/$NAME.log"; $H183 "grep -q '^EXIT 0' $B/logs/$NAME.log" || exit 1
log copy-back; copy "$H183" "cd $N && tar c datasets/swesmith-$NAME-valid jobs/v1-$NAME-oracle jobs/v1-$NAME-nop jobs/v2-$NAME-oracle jobs/v2-$NAME-nop jobs/baseline-swesmith-$NAME.jsonl jobs/baseline-swesmith-$NAME.err jobs/swesmith-$NAME-valid.selection.json" "$H94" "cd $N && tar x" || exit 1
log copy-known-failing; copy "$H183" "cd $N/datasets/swesmith-$NAME && tar c */tests/known_failing.txt" "$H94" "cd $N/datasets/swesmith-$NAME && tar x" || exit 1
log finalize
$H94 "$ENV94; \$AP $B/finalize_batch.py $NAME" || exit 1
$H94 "$ENV94; \$AP -m octopus.envs.swe.data.audit_tasks /home/user/swesmith-hf swesmith-$NAME-final v2-$NAME-oracle v2-$NAME-nop 2>&1 | tee $B/audit-$NAME.txt" || exit 1
$H94 "$ENV94; if [ -s $B/unsound-$NAME.txt ]; then mkdir -p $N/datasets/swesmith-$NAME-unsound; while read -r t; do [ -d $N/datasets/swesmith-$NAME-final/\$t ] && mv $N/datasets/swesmith-$NAME-final/\$t $N/datasets/swesmith-$NAME-unsound/; done < $B/unsound-$NAME.txt; echo audit-drop \$(grep -c . $B/unsound-$NAME.txt); fi" || exit 1
log BATCH_DONE
