#!/bin/bash
# Restart the running trial with changed env settings: after "Train step AFTER/N done" (+60 s for its recover checkpoint)
# stops the watchdog and the run, sets each KEY=VALUE in the env file (backup $RL_ENV_FILE.bak-<time>) and relaunches the
# same trial with rl_launch.sh (resumes after step AFTER). Example (bundle-e9): rl_restart.sh 10 EPISODES_PER_SERVER=10
# usage: RL_ENV_FILE=... setsid nohup rl_restart.sh AFTER KEY=VALUE... > $STATE/restart.out 2>&1 < /dev/null &
E=${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}; source "$E"; D=$(dirname "$(readlink -f "$0")"); N=$1; shift
T=$(cat $STATE/trial); M=$FILEROOT/logs/$USER/rq1-rl/$T/main.log
log() { echo "$(date "+%F %T") $*" | tee -a $STATE/incidents.log; }
until sed "s/\x1b\[[0-9;]*m//g" $M | grep -qE "Train step $N/[0-9]+ done"; do sleep 20; done
sleep 60  # the step's recover checkpoint and saver write right after the step line
log "restart: stop after step $N to set $*; resume $T"
kill $(cat $STATE/watchdog.pid) 2>/dev/null; sleep 2
RL_ENV_FILE=$E bash $D/rl_run.sh stop >> $STATE/stop.log 2>&1
cp $E $E.bak-$(date +%m%d%H%M)
for kv in "$@"; do
  k=${kv%%=*}; sed -i "s|^$k=.*|$kv|" $E
  grep -qx "$kv" $E || { log "restart: env edit $kv failed, NOT resuming"; exit 1; }
done
sleep 20
RL_ENV_FILE=$E setsid nohup bash $D/rl_launch.sh $T > $STATE/launch.out 2>&1 < /dev/null &
log "restart: relaunched (rl_launch.sh pid $!)"
