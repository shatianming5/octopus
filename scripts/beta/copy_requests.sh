#!/bin/bash
# copy_requests.sh DEST: every 20 s, rsync /dev/shm/bcp-trials-*/<job>/<trial>/agent/requests.jsonl (only that file of a trial,
# its relative path kept, only changed files) to DEST; runs until the job ends (srun --overlap step of the held job)
DEST=$1; mkdir -p "$DEST"; cd /dev/shm || exit 1
while :; do
  ls -d bcp-trials-* > /dev/null 2>&1 && rsync -a --prune-empty-dirs --include='bcp-trials-*/' --include='bcp-trials-*/*/' \
    --include='bcp-trials-*/*/*/' --include='bcp-trials-*/*/*/agent/' --include='bcp-trials-*/*/*/agent/requests.jsonl' --exclude='*' \
    bcp-trials-* "$DEST/" 2>> "$DEST.rsync.err"
  sleep 20
done
