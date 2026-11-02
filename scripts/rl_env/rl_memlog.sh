#!/bin/bash
# One line per minute while the rl_run.sh trainer runs (re-reads its pid, so restarts keep logging): free memory, the
# trainer cgroup (anon, shmem, file), OOM kills, GPU memory, the three largest processes.
# usage: RL_ENV_FILE=... setsid nohup rl_memlog.sh >> $STATE/memlog.txt 2>&1 &
source "${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}"; P=$STATE/trainer.pid
while true; do
  if [ -f $P ] && kill -0 $(cat $P) 2>/dev/null; then
    f=/sys/fs/cgroup$(cut -d: -f3 /proc/$(cat $P)/cgroup 2>/dev/null)
    echo "$(date +%T) avail $(awk '/MemAvailable/ {printf "%d", $2/2^20}' /proc/meminfo)G cgroup $(( $(cat $f/memory.current 2>/dev/null || echo 0) / 2**30 ))G $(awk '/^(anon|shmem|file) / {printf "%s %dG ", $1, $2/2^30}' $f/memory.stat 2>/dev/null)oom_kill $(awk '/oom_kill / {print $2}' $f/memory.events 2>/dev/null) gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',') top $(ps -u $USER -o rss=,comm= --sort=-rss | head -3 | awk '{printf "%s:%dG ", $2, $1/2^20}')"
  fi
  sleep 60
done
