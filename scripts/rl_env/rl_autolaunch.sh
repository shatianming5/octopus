#!/bin/bash
# Deferred launch of trial TRIAL on the trainer host (node-94). Every 120 s: waits for MARKER (e.g. written over ssh by
# node-67 eval_gate.sh once the previous run's evals are done) AND every local GPU + the first $REMOTE host's REMOTE_GPUS
# (default 0-4) all < 1000 MiB. Then starts rl_launch.sh detached and guards the first step: "Training failed" in the
# train log or main.log before "Train step 1/N done" -> kill launcher + watchdog, rl_run.sh stop, log the traceback tail.
# No retry.
# usage: RL_ENV_FILE=... setsid nohup rl_autolaunch.sh TRIAL MARKER >> autolaunch.out 2>&1 < /dev/null &
E=${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}; source "$E"; D=$(dirname "$(readlink -f "$0")"); T=$1; MARK=$2; ST=$STATE
M=$FILEROOT/logs/$USER/rq1-rl/$T/main.log; TL=$ST/train-$T.log; RH=${REMOTE%%:*}; RG=${REMOTE_GPUS:-0,1,2,3,4}
log() { echo "$(date '+%F %T') $(hostname -s) autolaunch: $*"; }
busy() {  # GPUs at >= 1000 MiB (an unreachable remote host counts as busy)
  { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/^/local:/'
    [ -n "$RH" ] && { ssh -o BatchMode=yes -o ConnectTimeout=20 $RH \
      "nvidia-smi -i $RG --query-gpu=index,memory.used --format=csv,noheader,nounits" | sed 's/^/remote:/' \
      || echo "remote:unreachable, 99999"; }; } | awk -F', ' '$2 >= 1000 {printf "%s=%s ", $1, $2}'
}
log "armed: waiting for $MARK and free GPUs (local all, ${RH:-no remote} $RG < 1000 MiB), poll 120 s"
until [ -f $MARK ]; do sleep 120; done
log "marker present ($(cat $MARK)); waiting for GPUs"
while b=$(busy); [ -n "$b" ]; do sleep 120; done
mkdir -p $ST; RL_ENV_FILE=$E setsid nohup bash $D/rl_launch.sh $T > $ST/launch.out 2>&1 < /dev/null & LP=$!
log "LAUNCHED $T: rl_launch.sh pid $LP"
until [ -f $TL ]; do
  kill -0 $LP 2>/dev/null || [ -f $TL ] || { log "launcher exited before the train log: $(tail -n 5 $ST/start.log | tr '\n' '|')"; exit 1; }
  sleep 10
done
while ! sed 's/\x1b\[[0-9;]*m//g' $M 2>/dev/null | grep -qE "Train step 1/[0-9]+ done"; do
  if grep -qs "Training failed" $TL $M; then
    kill $LP 2>/dev/null; sleep 2; kill $(cat $ST/watchdog.pid 2>/dev/null) 2>/dev/null
    RL_ENV_FILE=$E bash $D/rl_run.sh stop >> $ST/stop.log 2>&1
    echo "$(date '+%F %T') first-step guard: 'Training failed' before step 1; launcher + watchdog killed, run stopped, no retry" >> $ST/incidents.log
    log "GUARD STOP: 'Training failed' before step 1; launcher + watchdog killed, rl_run.sh stop rc=$?. No retry. Train log tail:"
    sed 's/\x1b\[[0-9;]*m//g' $TL | tail -n 60
    exit 1
  fi
  sleep 10
done
log "first step OK: $(sed 's/\x1b\[[0-9;]*m//g' $M | grep -oE 'Train step 1/[0-9]+ done' | tail -1); guard exits, watchdog owns the run"
