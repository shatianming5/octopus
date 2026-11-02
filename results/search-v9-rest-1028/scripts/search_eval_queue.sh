#!/bin/bash
# search_eval_queue.sh RELEASE NAME ADAPTER SET [NAME ADAPTER SET ...]: run search_eval_shared.sh (same dir, same env:
# SERVERS, JUDGE_URL, EVAL_*) on each NAME ADAPTER SET in turn, one at a time. An eval fails when it exits non-zero or
# its summary has more than MAX_INFRA (default 0.02) of its rows as infra failures (search_dev_eval itself exits 0 on
# those); the queue stops at the first failure. Then, whether the queue finished, failed or got TERM, it releases what
# RELEASE lists (a TERM during the release is ignored). Each RELEASE line is "HOST PIDFILE PATTERN": HOST "local" or an
# ssh target, PIDFILE on that host, PATTERN an extended regex the pid's command line must match (e.g. an SGLang server's
# port); the pid's process group gets TERM, then KILL after 60 s. A pid whose command line does not match is skipped and
# logged. At start the log shows each line's pid and command line. Log: OUT/queue.log (OUT as in search_eval_shared.sh).
#   SERVERS=... JUDGE_URL=... setsid nohup bash search_eval_queue.sh release.txt v0 ADAPTER0 bcp v41 ADAPTER41 val \
#     > /dev/null 2>&1 < /dev/null &
set -uo pipefail
[ $# -ge 4 ] && [ $((($# - 1) % 3)) -eq 0 ] || { echo "usage: $0 RELEASE NAME ADAPTER SET ..."; exit 1; }
: "${SERVERS:?}" "${JUDGE_URL:?}"
RELEASE=$(realpath "$1"); shift
cd "$(dirname "$0")"; OUT=${EVAL_OUT:-/mnt/ssd4/user/search-v9/eval}; mkdir -p "$OUT"
say() { echo "$(date '+%F %T %Z') $*" >> "$OUT/queue.log"; }
pgkill() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -qE -- "$2" || { echo "skip ${1:-none} ($2)"; return; }
  kill -TERM -- -$1 2>/dev/null || kill $1; echo "TERM $1 ($2)"
  for _ in $(seq 30); do kill -0 $1 2>/dev/null || return; sleep 2; done; kill -KILL -- -$1 2>/dev/null; echo "KILL $1"; }
show() { p=$(cat "$1" 2>/dev/null); echo "${p:-none}: $([ -n "$p" ] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160)"; }
each() {  # each FUNC: FUNC PIDFILE PATTERN on each RELEASE line's host
  while read -r h f p; do
    [ -z "$h" ] || [[ $h == \#* ]] && continue
    if [ "$h" = local ]; then say "$($1 "$f" "$p")"
    else say "$h: $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" "$(declare -f $1 pgkill); $1 $f '$p'" 2>&1)"; fi
  done < "$RELEASE"
}
kill_file() { pgkill "$(cat "$1" 2>/dev/null)" "$2"; }
EV=""
release() {
  trap '' TERM
  [ -n "$EV" ] && { kill -TERM "$EV" 2>/dev/null; wait "$EV"; }  # the eval's own cleanup: trials, search service
  each kill_file; say "released"
}
say "queue $* on $SERVERS, judge $JUDGE_URL; release:"; each show
trap release EXIT; trap 'say "TERM"; exit 143' TERM
while [ $# -gt 0 ]; do
  bash search_eval_shared.sh "$1" "$2" "$3" & EV=$!; wait $EV; rc=$?; EV=""
  case $3 in bcp) S=$OUT/test-$1-fork-k1.log ;; val) S=$OUT/val-$1-fork-k1.log ;; esac
  n=$(grep -a '^n=' "$S" 2>/dev/null | tail -n 1 | sed 's/^n=\([0-9]*\).*/\1/'); i=$(grep -a '^infra failures' "$S" 2>/dev/null | tail -n 1 | grep -o '[0-9]*$')
  say "$1 $3 exit $rc: scored ${n:-?}, infra ${i:-?}"
  [ $rc -eq 0 ] || exit $rc
  awk -v n="${n:-0}" -v i="${i:-0}" -v m="${MAX_INFRA:-0.02}" 'BEGIN {exit !(i > m * (n + i))}' && { say "$1 $3: infra over ${MAX_INFRA:-0.02}"; exit 1; }
  shift 3
done
