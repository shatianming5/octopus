#!/bin/bash
# share_watch.sh PIDFILE PATTERN GPUS: back off on GPUs shared with another user. At start it records, per GPU in GPUS
# (comma-separated indices), the memory other users' processes hold; every 30 s, if on one GPU that has grown past the
# start and the GPU has under 6 GiB free (our own peaks alone never trigger it), it stops our server: the pid in PIDFILE,
# only while its command line matches PATTERN (e.g. 'sglang.launch_server.*--port 32412'; checked again right before the
# kill), TERM to its process group, KILL after 60 s. Exits once that pid is gone. The eval's episodes on that server then
# become infra-failure rows (search_eval_queue.sh fails the queue past MAX_INFRA); rerun the same eval to fill them once a
# server is back.
#   setsid nohup bash share_watch.sh rollout-g7.pid 'sglang.launch_server.*--port 32412' 7 >> share_watch.log 2>&1 < /dev/null &
P=$(cat "$1"); PAT=$2; ME=$(id -un)
say() { echo "$(date '+%F %T %Z') $*"; }
ours() { cat /proc/$P/cmdline 2>/dev/null | tr '\0' ' ' | grep -qE -- "$PAT"; }
others() {  # MiB held on GPU $1 by other users' processes
  u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u="$u" '$1 == u {print $2, $3}' \
    | while read -r p m; do [ "$(ps -o user= -p $p)" != "$ME" ] && echo $m; done | awk '{s += $1} END {print s + 0}'
}
ours || { say "pid ${P:-none} does not match '$PAT'"; exit 1; }
declare -A BASE; for g in ${3//,/ }; do BASE[$g]=$(others $g); done
say "watching $P on GPUs $3; other users' MiB at start: $(for g in ${3//,/ }; do printf '%s:%s ' $g ${BASE[$g]}; done)"
while ours; do
  for g in ${3//,/ }; do
    o=$(others $g); free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$o" -gt "${BASE[$g]}" ] && [ "$free" -lt 6144 ]; then
      ours || break 2
      say "GPU $g: other users $o MiB (start ${BASE[$g]}), $free MiB free: TERM $P"
      kill -TERM -- -$P 2>/dev/null || kill $P; for _ in $(seq 30); do kill -0 $P 2>/dev/null || exit 0; sleep 2; done
      kill -KILL -- -$P 2>/dev/null; say "KILL $P"; exit 0
    fi
  done
  sleep 30
done
say "pid $P gone"
