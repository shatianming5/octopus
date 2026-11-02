#!/bin/bash
# release_shared.sh LAB_PIDS JUDGE_PIDS (node-138, evq-v45; run detached): the lab servers, their tunnels and the node-67
# judge are shared by queues that end at different times (lab BCP queue, lab Val queue; the judge also by the node-136
# queue). Those queues release none of them: release-lab.txt is emptied (the BCP queue reads it at its release) and the
# judge's judge.pid holds "held" (their judge line then skips it). This script releases release-shared-lab.txt once every
# pid in LAB_PIDS (comma-separated) is gone, and the judge (judge.pid.shared) once every pid in JUDGE_PIDS is gone too.
# Log: release_shared.log.
X=/mnt/ssd4/user/search-v9/logs/evq-v45; cd $X; L=$X/release_shared.log
say() { echo "$(date '+%F %T %Z') $*" >> $L; }
pgkill() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -qE -- "$2" || { echo "skip ${1:-none} ($2)"; return; }
  kill -TERM -- -$1 2>/dev/null || kill $1; echo "TERM $1 ($2)"
  for _ in $(seq 30); do kill -0 $1 2>/dev/null || return; sleep 2; done; kill -KILL -- -$1 2>/dev/null; echo "KILL $1"; }
kill_file() { pgkill "$(cat "$1" 2>/dev/null)" "$2"; }
each() {  # each FILE: kill_file PIDFILE PATTERN on each line's host
  while read -r h f p; do
    [ -z "$h" ] || [[ $h == \#* ]] && continue
    if [ "$h" = local ]; then say "$(kill_file "$f" "$p")"
    else say "$h: $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" "$(declare -f kill_file pgkill); kill_file $f '$p'" 2>&1)"; fi
  done < "$1"
}
gone() { for p in ${1//,/ }; do kill -0 $p 2>/dev/null && return 1; done; return 0; }
LAB=$1; JUD=$2; lab=0
say "waiting: lab servers and tunnels after pids $LAB, judge after pids $LAB,$JUD"
while :; do
  if [ $lab = 0 ] && gone $LAB; then say "lab queues gone: releasing release-shared-lab.txt"; each release-shared-lab.txt; lab=1; fi
  if [ $lab = 1 ] && gone $JUD; then
    say "all queues gone: releasing the judge"
    echo "user@198.51.100.67 /home/user/octo-env/search-v9-rest/judge-1030/judge.pid.shared sglang.launch_server.*--port 32500" > judge-shared.txt
    each judge-shared.txt; say "released"; exit 0
  fi
  sleep 60
done
