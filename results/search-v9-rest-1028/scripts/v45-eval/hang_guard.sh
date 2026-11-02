# hang_guard.sh (node-138, 10-31): the 10-29 script's BCP eval can sit idle after its last row (README "BCP v0 result
# (10-29)"). Once test-v45-fork-k1.jsonl has 830 rows, unchanged for 10 min, while that eval (EVPID) still runs, TERM the
# queue QPID: its trap releases everything and the chain goes on. Exits when the eval or the queue ends by itself.
QPID=4003430; EVPID=4003718; R=/mnt/ssd4/user/search-v9/eval/test-v45-fork-k1.jsonl; L=/mnt/ssd4/user/search-v9/logs/evq-v45/hang_guard.log
log() { echo "$(date '+%F %T %Z') $*" >> $L; }
log "guard on queue $QPID, eval $EVPID"
while kill -0 $EVPID 2>/dev/null && kill -0 $QPID 2>/dev/null; do
  n=$(wc -l < $R); age=$(( $(date +%s) - $(stat -c %Y $R) ))
  if [ "$n" -ge 830 ] && [ $age -ge 600 ]; then log "$n rows, last write ${age}s ago, eval still running: TERM queue $QPID"; kill -TERM $QPID; exit 0; fi
  sleep 60
done
log "eval or queue ended by itself: nothing to do"
