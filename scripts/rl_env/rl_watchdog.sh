#!/bin/bash
# Watchdog v2 for an rl_run.sh run (machine settings from $RL_ENV_FILE, like rl_run.sh), polls every 60 s; it resumes the
# same trial (AReaL recover checkpoint) after every stop:
#  - "Train step STEPS/N done" in main.log: DONE, writes $STATE/run.done, stops the run (releases every GPU), exit 0
#  - CUDA OOM in the new trainer log: stop, MB_TOKENS -10%, resume (max 3 OOM resumes, then NEEDS-MAIN)
#  - NaN/inf in a new stats table: resume once; a repeat -> NEEDS-MAIN, stop
#  - host memory (trainer cgroup shmem + RssAnon of rpc/trainer processes) > LIMIT_G: stop, MB_TOKENS -10%, resume (max 5 cuts)
#  - "Training failed with exception" in new main.log or train log: stop, resume (max 5 crashes). A second crash with an
#    update_weights_disk callback timeout -> NEEDS-MAIN with push timings, no resume
#  - trainer gone, SGLang server dead, or no new "Train step" for STALL_S: stop, resume (max 5 such resumes, then NEEDS-MAIN)
#  - Traceback with frames in $CODE/octopus: stop, NEEDS-MAIN (code bug: patch, then start the watchdog again)
# MB_TOKENS cuts are appended to the env file (the last assignment wins). Every action is logged to $STATE/incidents.log
# with time, step and evidence. NEEDS-MAIN writes $STATE/needs-main and exits 3.
# usage: RL_ENV_FILE=... [LIMIT_G=385] [STALL_S=7200] setsid nohup rl_watchdog.sh > $STATE/watchdog.out 2>&1 &
#        (rl_launch.sh starts it after the run is up)
E=${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}; source "$E"; D=$(dirname "$(readlink -f "$0")")
ST=$STATE; LIMIT_G=${LIMIT_G:-385}; STALL_S=${STALL_S:-7200}
T=$(cat $ST/trial); RD=$FILEROOT/logs/$USER/rq1-rl/$T; M=$RD/main.log; A=$RD/actor.log; R=$RD/rollout.log; TL=$ST/train-$T.log
log() { echo "$(date '+%F %T') $*" >> $ST/incidents.log; }
step_now() { grep -oE "Train step [0-9]+/[0-9]+ done" $M 2>/dev/null | tail -1; }
stop_run() { log "STOP ($1) last: $(step_now)"; RL_ENV_FILE=$E bash $D/rl_run.sh stop >> $ST/stop.log 2>&1; }
start_run() { RL_ENV_FILE=$E bash $D/rl_run.sh start $T >> $ST/resume.log 2>&1; log "RESUME rc=$? ($1), trial $T"; LASTN=$(grep -c "Train step" $M 2>/dev/null); LASTT=$(date +%s); OFF=$(wc -l < $TL); MOFF=$(wc -l < $M); }
cut_mb() { cur=$(source "$E"; echo $MB_TOKENS); cur=${cur:-262144}; new=$(( cur * 9 / 10 )); echo "MB_TOKENS=$new  # watchdog cut $(date '+%F %T'): $1" >> $E; log "MB_TOKENS $cur -> $new ($1)"; }
needs_main() { log "NEEDS-MAIN: $1"; echo "$(date '+%F %T') $1" > $ST/needs-main; exit 3; }
held_g() {
  f=/sys/fs/cgroup$(cut -d: -f3 /proc/$(cat $ST/trainer.pid)/cgroup 2>/dev/null)
  shm=$(awk '/^shmem / {printf "%d", $2 / 2^30}' $f/memory.stat 2>/dev/null)
  anon=$(for q in $(pgrep -u $USER -f "areal.infra.rpc.rpc_server|octopus.rl.train"); do awk '/^RssAnon/ {print $2}' /proc/$q/status 2>/dev/null; done | awk '{s += $1} END {printf "%d", s / 2^20}')
  echo $(( ${shm:-0} + ${anon:-0} ))
}
push_timings() {
  { echo "actor:"; tail -n 3000 $A 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "update_weights|adapter|Callback|timed out|push" | tail -n 8
    echo "rollout:"; tail -n 3000 $R 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Loading weights|update_weights|lora" | tail -n 6; } | cut -c1-200 | tr '\n' '|'
}
oom=0; nan=0; other=0; memcut=0; crashes=0; ucrash=0
LASTN=$(grep -c "Train step" $M 2>/dev/null); LASTN=${LASTN:-0}; LASTT=$(date +%s); OFF=$(wc -l < $TL); MOFF=$(wc -l < $M)
log "watchdog v2 up: trial $T, steps $STEPS, memory limit ${LIMIT_G}G, stall ${STALL_S}s, crash trigger on 'Training failed'"
while true; do
  sleep 60
  if grep -qE "Train step $STEPS/[0-9]+ done" $M 2>/dev/null; then log "DONE: step $STEPS of $STEPS"; echo done > $ST/run.done; sleep 20; stop_run "done, release GPUs"; log "released: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " ")"; exit 0; fi
  new=$(tail -n +$((OFF + 1)) $TL 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'); OFF=$(wc -l < $TL)
  newm=$(tail -n +$((MOFF + 1)) $M 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'); MOFF=$(wc -l < $M)
  both="$new
$newm"
  tp=$(cat $ST/trainer.pid 2>/dev/null)
  alive=0; kill -0 "$tp" 2>/dev/null && alive=1
  if echo "$new" | grep -q "OutOfMemoryError"; then
    if [ $oom -ge 3 ]; then stop_run "CUDA OOM"; needs_main "CUDA OOM 4th time at $(step_now)"; fi
    oom=$((oom + 1)); stop_run "CUDA OOM #$oom"; cut_mb "CUDA OOM #$oom"; start_run "after OOM #$oom"; continue
  fi
  if echo "$new" | grep -qE "(^|[ │])(-?nan|-?inf)([ │]|$)"; then
    if [ $nan -ge 1 ]; then stop_run "NaN/inf repeat"; needs_main "NaN/inf again after resume at $(step_now)"; fi
    nan=$((nan + 1)); stop_run "NaN/inf"; start_run "after NaN/inf, resume once"; continue
  fi
  if echo "$both" | grep -q "Training failed with exception"; then
    upd=0; tail -n 400 $A 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -qE "update_weights_disk failed|read timeout=600" && upd=1
    ev=$(echo "$both" | grep -m1 "Training failed with exception" | cut -c1-160)
    crashes=$((crashes + 1)); [ $upd -eq 1 ] && ucrash=$((ucrash + 1))
    log "CRASH #$crashes (update_weights timeout=$upd, repeats=$ucrash): $ev"
    if [ $ucrash -ge 2 ]; then stop_run "training failed, update_weights timeout repeated"; needs_main "update_weights_disk timeout repeated twice at $(step_now); push timings: $(push_timings)"; fi
    if [ $crashes -gt 5 ]; then stop_run "training failed"; needs_main "crash limit (5) reached: $ev"; fi
    stop_run "training failed"; start_run "after training failed #$crashes"; continue
  fi
  if [ $alive -eq 1 ]; then
    h=$(held_g); if [ "${h:-0}" -gt $LIMIT_G ]; then
      memcut=$((memcut + 1)); [ $memcut -gt 5 ] && { stop_run "host memory"; needs_main "host memory ${h}G above ${LIMIT_G}G after 5 cuts"; }
      log "HOST MEMORY ${h}G > ${LIMIT_G}G"; stop_run "host memory"; cut_mb "host memory ${h}G"; start_run "after host memory cut"; continue
    fi
  fi
  if [ $(grep -c "Train step" $M 2>/dev/null) -gt $LASTN ]; then LASTN=$(grep -c "Train step" $M); LASTT=$(date +%s); fi
  sgl_dead=0; for pf in $ST/serve/sglang-$(hostname -s)-g*.pid; do [ -f "$pf" ] && { kill -0 "$(cat $pf)" 2>/dev/null || sgl_dead=1; }; done
  stall=0; [ $(( $(date +%s) - LASTT )) -gt $STALL_S ] && stall=1
  if [ $alive -eq 0 ] || [ $sgl_dead -eq 1 ] || [ $stall -eq 1 ]; then
    why="trainer gone"; [ $sgl_dead -eq 1 ] && why="SGLang server dead"; [ $stall -eq 1 ] && why="stall > ${STALL_S}s"
    if tail -n 400 $TL | sed 's/\x1b\[[0-9;]*m//g' | grep -q "Traceback" && tail -n 400 $TL | grep -q "$CODE/octopus/"; then
      stop_run "$why"; needs_main "code traceback in octopus at $(step_now): see $TL"; fi
    other=$((other + 1)); [ $other -gt 5 ] && { stop_run "$why"; needs_main "resume limit (5) reached: $why"; }
    stop_run "$why"; start_run "after $why #$other"; continue
  fi
done
