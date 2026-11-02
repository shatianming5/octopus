#!/bin/bash
# Supervision wait for an rl_run.sh run under rl_watchdog.sh: returns when "Train step TARGET/N done" appears, or
# $STATE/incidents.log gains a line other than "watchdog v2 up", or needs-main / run.done appears, or the train log shows
# "Training failed" after this wait began; prints why, the last step and the new incidents.
# usage: RL_ENV_FILE=... rl_wait.sh TARGET
source "${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}"; T=$1; ST=$STATE; TR=$(cat $ST/trial)
M=$FILEROOT/logs/$USER/rq1-rl/$TR/main.log; TL=$ST/train-$TR.log; I=$ST/incidents.log
n0=$(wc -l < $I 2>/dev/null || echo 0); f0=$(grep -c "Training failed" $TL 2>/dev/null); f0=${f0:-0}
step() { sed "s/\x1b\[[0-9;]*m//g" $M 2>/dev/null | grep -oE "Train step [0-9]+/[0-9]+ done" | tail -1; }
while true; do
  s=$(step); k=$(echo "$s" | grep -oE "[0-9]+" | head -1)
  [ -n "$k" ] && [ "$k" -ge "$T" ] && { why="reached $s"; break; }
  new=$(tail -n +$((n0 + 1)) $I 2>/dev/null | grep -v "watchdog v2 up")
  [ -n "$new" ] && { why="incident"; break; }
  [ -f $ST/needs-main ] && { why="needs-main"; break; }
  [ -f $ST/run.done ] && { why="run.done"; break; }
  f=$(grep -c "Training failed" $TL 2>/dev/null); [ "${f:-0}" -gt "$f0" ] && { why="Training failed in train log"; break; }
  sleep 60
done
echo "$(date "+%F %T") WAKE: $why | last: $(step)"
tail -n +$((n0 + 1)) $I 2>/dev/null | cut -c1-300 | tail -8
[ -f $ST/needs-main ] && cat $ST/needs-main | cut -c1-300
