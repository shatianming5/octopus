#!/bin/bash
# sweep_queue.sh RELEASE NAME ADAPTER SET ARM [NAME ADAPTER SET ARM ...] (10-31 v45 width/depth sweep, node-138): run
# search_eval_arm.sh (same dir, same env: SERVERS, JUDGE_URL, EVAL_*) on each entry in turn, one at a time. ARM "-" is the
# run's own settings (outputs as search_eval_shared.sh's); another ARM uses the settings dir dirname(RUN)/run-sw-ARM and
# outputs OUT/{test,val}-sw-ARM-NAME-fork-k1.*. Unlike search_eval_queue.sh, a failed eval (non-zero exit, or more than
# MAX_INFRA, default 0.02, of its rows infra failures) is logged and the queue goes on; the failures are listed at the end.
# Then, whether it finished or got TERM, it releases RELEASE as search_eval_queue.sh does (same line format, a TERM during
# the release is ignored). RELEASE_ONLY=1: release at once, no eval (chain_sweep.sh when a server or the judge is missing).
# The judge serves two such queues at once (lab and node-136), so it is not in RELEASE: JUDGE_RELEASE is its line, and the
# queue releases it only if none of the pid files in JUDGE_PEERS names a live process (checked under a lock, after
# removing SELF_PIDFILE, its own pid file: of two queues ending together the second releases it). Log: OUT/QUEUE_LOG
# (default queue.log).
set -uo pipefail
[ -n "${RELEASE_ONLY:-}" ] && [ $# -eq 1 ] || { [ $# -ge 5 ] && [ $((($# - 1) % 4)) -eq 0 ]; } \
  || { echo "usage: $0 RELEASE NAME ADAPTER SET ARM ..."; exit 1; }
: "${SERVERS:?}" "${JUDGE_URL:?}"
RELEASE=$(realpath "$1"); shift
cd "$(dirname "$0")"; OUT=${EVAL_OUT:-/mnt/ssd4/user/search-v9/eval}; mkdir -p "$OUT"
R0=/home/user/search-v9-run/areal-exp/logs/user/rq1-rl/search-v9-rest
say() { echo "$(date '+%F %T %Z') $*" >> "$OUT/${QUEUE_LOG:-queue.log}"; }
pgkill() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -qE -- "$2" || { echo "skip ${1:-none} ($2)"; return; }
  kill -TERM -- -$1 2>/dev/null || kill $1; echo "TERM $1 ($2)"
  for _ in $(seq 30); do kill -0 $1 2>/dev/null || return; sleep 2; done; kill -KILL -- -$1 2>/dev/null; echo "KILL $1"; }
show() { p=$(cat "$1" 2>/dev/null); echo "${p:-none}: $([ -n "$p" ] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160)"; }
each() {  # each FUNC [FILE]: FUNC PIDFILE PATTERN on each line's host of FILE (default RELEASE)
  while read -r h f p; do
    [ -z "$h" ] || [[ $h == \#* ]] && continue
    if [ "$h" = local ]; then say "$($1 "$f" "$p")"
    else say "$h: $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" "$(declare -f $1 pgkill); $1 $f '$p'" 2>&1)"; fi
  done < "${2:-$RELEASE}"
}
kill_file() { pgkill "$(cat "$1" 2>/dev/null)" "$2"; }
EV=""; FAILED=""
release() {
  trap '' TERM
  [ -n "$EV" ] && { kill -TERM "$EV" 2>/dev/null; wait "$EV"; }  # the eval's own cleanup: trials, search service
  say "sweep failures:${FAILED:- none}"; each kill_file
  if [ -n "${JUDGE_RELEASE:-}" ]; then
    exec 8> "$OUT/.judge-release.lock"; flock 8; [ -n "${SELF_PIDFILE:-}" ] && rm -f "$SELF_PIDFILE"
    live=""; for f in ${JUDGE_PEERS:-}; do p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && live="$live $f"; done
    if [ -n "$live" ]; then say "judge kept: live$live"; else each kill_file <(echo "$JUDGE_RELEASE"); fi
    flock -u 8
  fi
  say "released"
}
say "sweep $* on $SERVERS, judge $JUDGE_URL; release:"; each show; [ -n "${JUDGE_RELEASE:-}" ] && each show <(echo "$JUDGE_RELEASE")
trap release EXIT; trap 'say "TERM"; exit 143' TERM
[ -n "${RELEASE_ONLY:-}" ] && { say "RELEASE_ONLY"; exit 1; }
while [ $# -gt 0 ]; do
  N=$1; A=$2; S=$3; ARM=$4; shift 4
  if [ "$ARM" = - ]; then L=""; RUN=$R0; else L=sw-$ARM; RUN=$(dirname "$R0")/run-$L; fi
  T=${L:+$L-}$N-fork-k1
  EVAL_RUN=$RUN EVAL_ARM=$L bash search_eval_arm.sh "$N" "$A" "$S" & EV=$!; wait $EV; rc=$?; EV=""
  case $S in bcp) F=$OUT/test-$T.log ;; val) F=$OUT/val-$T.log ;; esac
  n=$(grep -a '^n=' "$F" 2>/dev/null | tail -n 1 | sed 's/^n=\([0-9]*\).*/\1/'); i=$(grep -a '^infra failures' "$F" 2>/dev/null | tail -n 1 | grep -o '[0-9]*$')
  say "$S $T exit $rc: scored ${n:-?}, infra ${i:-?}"
  if [ $rc -ne 0 ] || awk -v n="${n:-0}" -v i="${i:-0}" -v m="${MAX_INFRA:-0.02}" 'BEGIN {exit !(i > m * (n + i))}'; then
    FAILED="$FAILED $S:$T"; say "$S $T failed (exit $rc or infra over ${MAX_INFRA:-0.02}): next"
  fi
done
