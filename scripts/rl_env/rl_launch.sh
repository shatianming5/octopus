#!/bin/bash
# Start (or resume) trial TRIAL with rl_run.sh, then start rl_watchdog.sh once main.log and the train log exist.
# usage: RL_ENV_FILE=... setsid nohup rl_launch.sh TRIAL > $STATE/launch.out 2>&1 < /dev/null &
E=${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}; source "$E"; D=$(dirname "$(readlink -f "$0")"); T=$1; mkdir -p $STATE
RL_ENV_FILE=$E bash $D/rl_run.sh start $T > $STATE/start.log 2>&1 || exit 1
M=$FILEROOT/logs/$USER/rq1-rl/$T/main.log
until [ -f $M ] && [ -f $STATE/train-$T.log ]; do  # a trainer that exits before AReaL writes main.log: stop, release
  kill -0 "$(cat $STATE/trainer.pid)" 2>/dev/null || [ -f $M ] || {
    echo "trainer exited before $M"; RL_ENV_FILE=$E bash $D/rl_run.sh stop >> $STATE/stop.log 2>&1; exit 1; }
  sleep 30
done
RL_ENV_FILE=$E setsid nohup bash $D/rl_watchdog.sh > $STATE/watchdog.out 2>&1 < /dev/null & echo $! > $STATE/watchdog.pid
