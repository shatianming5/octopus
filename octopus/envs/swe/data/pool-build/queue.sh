#!/bin/bash
# Runs batch.sh for each line "NAME KIND ARG" of QUEUEFILE, taking the first line and deleting it before each batch,
# so lines can be removed from (or appended to) the file while it runs.   usage: [RUNNER=relay183.sh] queue.sh QUEUEFILE
B=/home/user/pool-build; Q=$1; mkdir -p $B/logs; RUNNER=${RUNNER:-batch.sh}
while [ -s "$Q" ]; do
  line=$(head -1 "$Q"); sed -i 1d "$Q"
  set -- $line; [ -z "${1:-}" ] && continue
  echo "$(date +%F_%T) START $line"
  bash $B/$RUNNER "$@" > $B/logs/$1.log 2>&1; rc=$?
  echo "$(date +%F_%T) END $1 rc=$rc $(grep -E "^FINAL" $B/logs/$1.log | tail -1)"
done
echo "$(date +%F_%T) QUEUE_DONE"
