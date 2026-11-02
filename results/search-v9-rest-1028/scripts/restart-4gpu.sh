#!/bin/bash
# search-v9-rest 10-29: relaunch from step 41 on 4 GPUs after the 2-GPU trainer was stopped (step 42's ppo_update hit
# AReaL's 2 h RPC timeout and was re-sent). Applies patches/areal-train-rpc-timeout.patch (PR #329) to the overlay,
# deploys stage-327 (#327 8deeecb) into code-local, relaunches with train4.sh (GPUs 1-4, fsdp d4) on the same 7
# rollout servers and starts watch-j (waits for a shared-server eval before releasing the services) and
# one_pass_stop5.py (falls back to total_train_steps 61). Any failed step before the launch aborts it.
set -u
B=/mnt/ssd4/user/search-v9; L=$B/logs/rest2; C=$B/runtime/code-local; O=$B/areal-overlay
SV=127.0.0.1:32302,127.0.0.1:32310,127.0.0.1:32311,127.0.0.1:32312,127.0.0.1:32313,127.0.0.1:32314,127.0.0.1:32315
say() { echo "$(date '+%F %T %Z') $*"; }
P=$(cat $B/logs/train-rest.pid)
tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q octopus.rl.train && { say "trainer $P still running; abort"; exit 1; }
pgrep -u user -f '^bash watch-j.sh|one_pass_stop|^bash switch-4gpu.sh' > /dev/null && { say "a watcher runs; abort"; exit 1; }
F=areal/infra/controller/train_controller.py
if ! grep -q 'http_timeout=86400.0' $O/$F; then
  cp -p $O/$F $B/runtime/train_controller.py.bak-1029 || { say "backup failed; abort"; exit 1; }
  (cd $O && patch -p1 < $L/areal-train-rpc-timeout.patch) || { say "patch failed; abort"; exit 1; }
  rm -f $O/areal/infra/controller/__pycache__/train_controller.*.pyc
fi
say "overlay: $(grep -c 'http_timeout=86400.0' $O/$F) patched call site"
mkdir -p $B/runtime/code-local-bak-1029/octopus/rl
[ -e $B/runtime/code-local-bak-1029/octopus/rl/actor.py ] || cp -p $C/octopus/rl/actor.py $C/octopus/rl/config.py $B/runtime/code-local-bak-1029/octopus/rl/ || { say "backup failed; abort"; exit 1; }
cp $B/runtime/stage-327/octopus/rl/{actor,config,block_checkpoint}.py $C/octopus/rl/ || { say "deploy failed; abort"; exit 1; }
say "deployed: $(cd $C/octopus/rl && md5sum actor.py config.py block_checkpoint.py | cut -c1-8 | tr '\n' ' ')"
[ -e $B/logs/train-rest-2gpu-1029.log ] || mv $B/logs/train-rest.log $B/logs/train-rest-2gpu-1029.log
bash $L/train4.sh "$SV" > $L/train4-launch.out 2>&1; cat $L/train4-launch.out
NP=$(cat $B/logs/train-rest.pid)
[ "$NP" != "$P" ] && kill -0 $NP 2>/dev/null || { say "relaunch failed"; exit 1; }
cd $L
setsid nohup bash watch-j.sh >> watch-j.log 2>&1 < /dev/null &
echo $! > watch-j.pid
setsid nohup python3 one_pass_stop5.py >> one_pass_stop.log 2>&1 < /dev/null & echo $! > one_pass_stop.pid
say "relaunched trainer $NP on GPUs 1-4; watch-j $(cat watch-j.pid), one_pass_stop5 $(cat one_pass_stop.pid)"
