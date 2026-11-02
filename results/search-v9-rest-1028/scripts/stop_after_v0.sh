#!/bin/bash
# User 10-29: run only BCP v0. When the queue logs "v0 bcp exit 0", TERM the queue: its TERM/EXIT traps stop the next
# eval (v41) it may just have started and release the servers, judge and tunnels in release.txt. Exits if the queue ends.
D=/mnt/ssd4/user/search-v9; Q=$(cat $D/logs/evq/queue.pid); L=$D/eval/queue.log
until grep -q "v0 bcp exit 0" $L; do
  tr "\0" " " < /proc/$Q/cmdline 2>/dev/null | grep -q search_eval_queue.sh || { echo "$(date) queue gone; exit"; exit; }
  sleep 2
done
tr "\0" " " < /proc/$Q/cmdline | grep -q search_eval_queue.sh && kill -TERM $Q && echo "$(date) v0 done; TERM queue $Q"
