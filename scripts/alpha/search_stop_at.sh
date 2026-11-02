#!/bin/bash
# search_stop_at.sh TRIAL K: on Alpha, after "Train step K done" (+60 s for the recover checkpoint; K=0: now) touch P/stopped
# (every guard of TRIAL exits on it), stop the trainer in TJ, then in each rollout job J of P/jobs ("TJ RJ [J ...]", "TJ TJ": none) its
# servers, its judge (lib.sh serve_rollout: the process group in P/judge-J.pid) and its task (the exact pid in
# P/rollout-J.pid), then END the trial's controller (search.sbatch, pid in P/controller.pid, written by search_launch.sh's task):
# its TERM trap runs the EXIT trap (stop_stage, kill judge + BM25 groups, rm /dev/shm copy). Then waits (10 min) for both hold
# loops' .failed markers and verifies every job's GPUs are free and ports 32300-32699 are closed. Resumes nothing.
# Servers outside these jobs (search_launch.sh EXTRA_ROLLOUT, JUDGE_HOST) are their owner's: scripts/beta/serve_rollout.sh stop.
# Exits 1 without stopping when P/stopped exists, TJ has ended (sacct) or no longer runs 20-TRIAL.sh (a stopped or
# ended run); 2 if NOT CLEAN after the stop.
T=${1:?}; K=${2:?}; N=$(cd "$(dirname "$0")/../.." && pwd); R=$(dirname "$N"); P=$R/search/pilot/$T; read -r TJ RJS < "$P/jobs" || exit 1
RJS=$(for j in $RJS; do [ "$j" = "$TJ" ] || echo "$j"; done)  # "TJ TJ": no rollout job (search_launch.sh RJ = TJ)
M=$R/search/areal-exp/logs/user/rq1-rl/$T/main.log; QT=$R/search/pilot/hold-$TJ/queue
source "$N/scripts/alpha/lib.sh"  # owner_running: sacct; an error or no answer counts as running, only a final state does not
live() { [ ! -e "$P/stopped" ] && { [ -e "$QT/20-$T.sh" ] || [ -e "$QT/20-$T.sh.running-$TJ" ]; } && owner_running "$TJ"; }  # a TERM'd TJ leaves .running-TJ
until c=$(grep -c 'Train step' "$M" 2>/dev/null); [ "${c:-0}" -ge "$K" ]; do live || { echo "$(date +%T) $T stopped or not running: not stopping"; exit 1; }; sleep 20; done
live || { echo "$(date +%T) $T stopped or not running: not stopping"; exit 1; }
touch "$P/stopped"; sleep 60
echo "$(date +%T) ${c:-0} steps done (stop at $K); stopping"
srun --jobid="$TJ" --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=$N/scripts/alpha/search.env bash $N/scripts/rl_env/rl_run.sh stop"
for RJ in $RJS; do
srun --jobid="$RJ" --overlap -n1 -c1 bash -c "cd /tmp; RL_ENV_FILE=$N/scripts/alpha/search.env bash $N/scripts/rl_env/rl_run.sh unserve"
# exact pids, cmdline checked (never pgrep/pkill -f)
pid=$(cat "$P/judge-$RJ.pid" 2>/dev/null)
if [ -n "$pid" ] && srun --jobid="$RJ" --overlap -n1 -c1 bash -c "tr '\0' ' ' < /proc/$pid/cmdline | grep -q 'sglang.launch_server.*--served-model-name judge '"; then
  srun --jobid="$RJ" --overlap -n1 -c1 bash -c "kill -- -$pid; for _ in \$(seq 60); do kill -0 -- -$pid 2> /dev/null || exit 0; sleep 2; done; kill -9 -- -$pid"
elif [ -n "$pid" ]; then echo "judge pid '$pid' of $RJ not a live judge: nothing to end"; fi
pid=$(cat "$P/rollout-$RJ.pid" 2>/dev/null || { [ "$RJS" = "$RJ" ] && cat "$P/rollout.pid" 2>/dev/null; })  # P/rollout.pid: a trial launched before serve_rollout (one rollout job)
if [ -n "$pid" ] && srun --jobid="$RJ" --overlap -n1 -c1 bash -c "tr '\0' ' ' < /proc/$pid/cmdline | grep -qx 'sleep infinity '"; then
  srun --jobid="$RJ" --overlap -n1 -c1 kill "$pid"
else echo "rollout pid '$pid' of $RJ not a live sleep infinity: nothing to end"; fi
done
pid=$(cat "$P/controller.pid" 2>/dev/null)
if [ -n "$pid" ] && srun --jobid="$TJ" --overlap -n1 -c1 bash -c "tr '\0' ' ' < /proc/$pid/cmdline| grep -q 'search.sbatch.*$T'"; then
  srun --jobid="$TJ" --overlap -n1 -c1 kill -TERM "$pid"
  for _ in $(seq 60); do srun --jobid="$TJ" --overlap -n1 -c1 kill -0 "$pid" 2>/dev/null|| break; sleep 5; done
else echo "controller pid '$pid' not a live search.sbatch of $T: nothing to end"; fi
for _ in $(seq 120); do [ -e "$QT/20-$T.sh.failed" ] && break; sleep 5; done
for RJ in $RJS; do QR=$R/search/pilot/hold-$RJ/queue
  for _ in $(seq 120); do [ -e "$QR/01-rollout-$T.sh.failed" ] && break; sleep 5; done; done
echo "queue state: $(ls "$QT" | grep "20-$T" | tr '\n' ' ')$(for RJ in $RJS; do echo -n "| $(ls "$R/search/pilot/hold-$RJ/queue" | grep "01-rollout-$T" | tr '\n' ' ')"; done)"
bad=0
for j in $TJ $RJS; do
  g=$(srun --jobid="$j" --overlap -n1 -c1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader| tr '\n' ' ')
  p=$(srun --jobid="$j" --overlap -n1 -c1 ss -ltnH| awk '{n=split($4,a,":"); if (a[n]>=32300 && a[n]<=32699) printf a[n] " "}')
  [ -n "$g" ] || { echo "job $j: no GPU answer"; bad=1; }
  echo "job $j: GPUs $g| open ports in 32300-32699: ${p:-none}"; [ -z "$p" ] || bad=1
  srun --jobid="$j" --overlap -n1 -c1 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits| awk '$1>2000{exit 1}' || bad=1
done
for j in $TJ $RJS; do Q=$R/search/pilot/hold-$j/queue
  compgen -G "$Q/*.running-*" > /dev/null && { echo "a task is still running in $Q"; bad=1; }; done
[ $bad = 0 ] && echo "CLEAN: ready for search_launch.sh" || { echo "NOT CLEAN: see above"; exit 2; }
