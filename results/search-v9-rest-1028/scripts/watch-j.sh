#!/bin/bash
# search-v9-rest auto-release (node-138, layout of 10-28): when the trainer exits, stop its AReaL workers, 138's search
# service + tunnels, node-134's judge (port 32500) + rollout g2 (32302) and node-68's rollouts g0-g5 (32310-32315). The
# AReaL workers are found by command line (search-v9-rest); every other pid comes from our pid files and is checked
# against its expected command line (an SGLang server: its port) first. The trainer pid is re-read at each 60 s poll,
# but a poll that finds no live pid (old trainer gone, new pid not yet written, file missing) releases at once: stop
# this watcher before restarting the trainer and start it after the new pid is written.
B=/mnt/ssd4/user/search-v9; D=$B/logs/rest2
say() { echo "$(date '+%F %T %Z') $*"; }
pgkill() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q -- "$2" || { say "skip $1"; return; }
  kill -TERM -- -$1 2>/dev/null || kill $1; say "TERM $1 ($2)"; for _ in $(seq 30); do kill -0 $1 2>/dev/null || return; sleep 2; done; kill -KILL -- -$1 2>/dev/null; }
while kill -0 "$(cat $B/logs/train-rest.pid)" 2>/dev/null; do sleep 60; done
say "trainer $(cat $B/logs/train-rest.pid) exited; releasing"
w=$(ps -u user -o pid=,args= | awk '/areal\.infra\.rpc\.rpc_server|proxy\.proxy_rollout_server/ && /search-v9-rest/ {print $1}')
[ -n "$w" ] && { kill $w; sleep 10; kill -9 $w 2>/dev/null; }
# a shared-server eval (search_eval_shared.sh) still uses the rollout servers, the judge and the tunnels: release
# them once it ends (the trainer's GPUs above are already free)
while pgrep -u "$(id -un)" -f '^[^ ]*python[^ ]* -m octopus\.eval\.search_dev_eval' > /dev/null; do say "eval running; release waits"; sleep 300; done
pgkill $(cat $D/search.pid) octopus.envs.search.service
ssh -o BatchMode=yes -o ConnectTimeout=20 user@198.51.100.134 "$(declare -f say pgkill); L=/mnt/disk4/user/search-v9-state/rest2; pgkill \$(cat \$L/judge.pid) 'sglang.launch_server.*--port 32500 '; pgkill \$(cat \$L/rollout-g2.pid) 'sglang.launch_server.*--port 32302 '"
ssh -o BatchMode=yes -o ConnectTimeout=20 user@10.0.1.68 "$(declare -f say pgkill); L=/home/user/octo-env/search-v9-rest; for g in 0 1 2 3 4 5; do pgkill \$(cat \$L/rollout-g\$g.pid) \"sglang.launch_server.*--port 3231\$g \"; done"
for f in tunnel tunnel68; do pgkill $(cat $D/$f.pid) "ssh -N"; done
say released; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
