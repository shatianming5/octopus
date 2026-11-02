#!/bin/bash
# search_v5_deadline.sh CODE TRIAL DEADLINE: the time limit search_v5.sh has not got (it trains until the pass is
# covered), run beside its running queue task on a Beta login node (reads the shared filesystem; acts on the held job's
# node with srun --overlap). Every minute, while training goes on: an ALERT line per hour without a new "Train step".
# At DEADLINE (date -d; the held job's end less the tail: Val vK + Val v0 after a stop) with training still on: stops the
# trainer at the next step boundary (or at DEADLINE + 30 min regardless: the last finished step's recover checkpoint).
# That ends the queue task (search_v5.sh's train_pass fails on a dead trainer, so it exports nothing and runs no Val, and
# its cleanup stops the judge and the search service). After that stop, or when the task ended before the deadline
# without P/train.done (a trainer crash), this writes the last finished step's adapter as P/train.done (lib.sh
# final_adapter) and queues 11-search-v5-val.sh, search_v5.sh again on TRIAL: with train.done it serves, runs Val vK and
# Val v0, pairs. Exits doing nothing when the pass ends on its own (train.done).
#   nohup bash -lc 'bash CODE/scripts/beta/search_v5_deadline.sh CODE TRIAL "2026-10-24 04:30 UTC"' > ROOT/logs/deadline-TRIAL.log 2>&1 &
# (bash -lc: Slurm's client is on PATH, with SLURM_CONF, only in a login shell)
[ -n "${DL_CLEAN:-}" ] || exec env -i HOME="$HOME" USER="$USER" LANG=C.UTF-8 SLURM_CONF="${SLURM_CONF:-}" \
  PATH="$(dirname "$(command -v srun)"):/usr/local/bin:/usr/bin:/bin" DL_CLEAN=1 bash "$0" "$@"  # not the profile's
set -uo pipefail
C=$1; TRIAL=$2; END=$(date -d "$3" +%s) || exit 1
source "$C/scripts/beta/search.env"; source "$C/scripts/alpha/lib.sh"
P=$PILOT/$TRIAL; M=$FILEROOT/logs/$USER/rq1-rl/$TRIAL/main.log; Q=$OCTO_ENV/queue
step() { grep -oE 'Train step [0-9]+' "$M" 2>/dev/null | tail -1 | awk '{print $3 + 0}'; }
f=$(ls "$Q"/*.sh.running-* 2>/dev/null | head -1); [ -n "$f" ] || { say "no running queue task in $Q"; exit 1; }
JOB=${f##*.running-}; squeue -h -j "$JOB" > /dev/null || { say "no Slurm client (start it with bash -lc)"; exit 1; }
job() {  # the held job still runs (its end leaves the task .running); gone only after 3 checks 20 s apart
  local i; for i in 1 2 3; do squeue -h -j "$JOB" 2>/dev/null | grep -q . && return 0; [ $i = 3 ] || sleep 20; done; return 1; }
training() { [ ! -f "$P/train.done" ] && [ -e "$f" ]; }
say "watching ${f##*/} (job $JOB), trial $TRIAL, code $C; deadline $(date -d @"$END")"
last=$(step); seen=$(date +%s); alerted=$seen
while training && [ "$(date +%s)" -lt "$END" ]; do
  job || { say "job $JOB is gone (3 squeue checks)"; exit 1; }
  s=$(step); now=$(date +%s)
  [ "${s:-0}" -gt "${last:-0}" ] && { last=$s; seen=$now; alerted=$now; }
  [ $((now - alerted)) -ge 3600 ] && { say "ALERT: no new Train step for $(((now - seen) / 60)) min (last: step ${last:-0}), $M"; alerted=$now; }
  sleep 60
done
if training; then
  s0=$(step); say "deadline: stopping the trainer at the step boundary after step $s0 (at $(date -d @$((END + 1800)) +%T) regardless)"
  while training && [ "$(step)" -le "$s0" ] && [ "$(date +%s)" -lt $((END + 1800)) ]; do sleep 10; done
fi
if training; then
  [ "$(step)" -gt "$s0" ] && sleep 60  # that step's recover checkpoint
  [ "$(cat "$STATE/trial")" = "$TRIAL" ] || { say "$STATE runs $(cat "$STATE/trial"), not $TRIAL: not stopped"; exit 1; }
  srun --jobid="$JOB" --overlap -n1 -c1 --mem=4G --gpus=4 env RL_ENV_FILE="$C/scripts/beta/search.env" \
    bash "$C/scripts/rl_env/rl_run.sh" stop || { say "stop failed"; exit 1; }
  while [ -e "$f" ] && job; do sleep 10; done  # the queue task ends (its train_pass sees the trainer gone)
  [ -e "$f" ] && { say "job $JOB is gone: no Val"; exit 1; }
  why="stopped at the deadline"
else
  [ -f "$P/train.done" ] && { say "the pass ended on its own ($P/train.done): nothing to do"; exit 0; }
  why="the queue task ended without train.done ($(ls "${f%.running-*}".* 2>/dev/null | tr '\n' ' '); a trainer crash?)"
  say "$why: Val of the last finished step"
fi
K=$(step); [ "${K:-0}" -ge 1 ] || { say "no finished step: nothing to evaluate"; exit 1; }
A=$(final_adapter "$K") || exit 1
echo "$K $A" > "$P/train.done"; say "step-$K adapter $A"
echo "exec bash $C/scripts/beta/search_v5.sh $C $TRIAL  # Val after search_v5_deadline.sh ($why) at step $K" \
  > "$Q/11-search-v5-val.sh"; say "queued $Q/11-search-v5-val.sh"
