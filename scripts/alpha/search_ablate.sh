#!/bin/bash
# search_ablate.sh [--rerun] QUEUE NAME ADAPTER RUN SETS K LABEL ARM...: queue an eval-only ablation of one adapter on a
# held job's queue (docs/ablations.md). Run on Alpha from the deployed code (ROOT/code-SHA/scripts/alpha/search_ablate.sh:
# the tasks run that code's search_eval.sh).
#   QUEUE: a hold queue dir (ROOT/search/pilot/hold-JOB/queue; lib.sh hold runs its *.sh in name order, one at a time).
#   NAME ADAPTER: as search_eval.sh (v<N>, the adapter dir). RUN: the run's settings dir (config.yaml, episode_settings.json).
#   SETS: val | bcp | both. K: samples per question.
#   ARM: A:KEY=VALUE[,KEY=VALUE...] or A: (the baseline, RUN's own settings); KEY one of KNOBS below (KEY= drops it).
#     Each arm A gets RUN's copy with its changes in episode_settings.json, dirname(RUN)/run-LABEL-A (as P/run-newcode-d0
#     with "OCTOPUS_FORK_DEPTH": "0"), and one task per set, NN-val-LABEL-A-NAME.sh or NN+1-bcp-LABEL-A-NAME.sh (NN: env,
#     default 60; every Val first), its output ROOT/search/pilot/eval/{val,test}-LABEL-A-NAME-fork-kK.jsonl.
#     EVAL_PARALLEL (env) is passed through.
# Idempotent: an arm's existing settings dir must hold the same settings (else refused); a task queued, running, done,
# .skip-* or .deferred-* is skipped, a failed one queued again (search_dev_eval resumes from the rows it has). --rerun also
# queues the done ones again: a run that ended with infra failures ends done, and its rerun runs only those episodes.
set -euo pipefail
RERUN=0; [ "${1:-}" = --rerun ] && { RERUN=1; shift; }
Q=${1:?QUEUE}; NAME=${2:?NAME}; A=${3:?ADAPTER}; RUN=${4:?RUN}; SETS=${5:?SETS}; K=${6:?K}; L=${7:?LABEL}; shift 7
N=$(cd "$(dirname "$0")/../.." && pwd); NN=${NN:-60}
# the episode knobs an arm may change (docs/search-env.md "Limits"): a typo, a host setting or a credential fails here
KNOBS="FORK_DEPTH|MAX_BRANCHES|MAX_THREADS|BRANCH_MAX_TURNS|MAIN_MAX_CALLS|SEARCH_BUDGET|ANSWER_RESERVE|THREAD_BUDGET|CONTEXT_WINDOW|READ_TOOL|REWIND_TOOL|REWIND_MAX|LINEAR"
Q=$(realpath -e "$Q"); RUN=$(realpath -e "$RUN")
[ -d "$Q" ] && [ -f "$RUN/episode_settings.json" ] && [ -f "$RUN/config.yaml" ] || { echo "no queue $Q or no run settings in $RUN"; exit 1; }
[[ $NAME =~ ^v[0-9]+$ && $K =~ ^[1-9][0-9]*$ && $L =~ ^[A-Za-z0-9._-]+$ && $NN =~ ^[0-9]+$ ]] || { echo "want NAME v<N>, K > 0, LABEL [A-Za-z0-9._-]+"; exit 1; }
[ -d "$A" ] || { echo "no adapter dir $A"; exit 1; }
case $SETS in val|bcp) ;; both) SETS="val bcp" ;; *) echo "SETS must be val, bcp or both"; exit 1 ;; esac
[ $# -gt 0 ] || { echo "no ARM"; exit 1; }
declare -A S
for arm in "$@"; do  # every arm checked before anything is made
  a=${arm%%:*}; c=${arm#*:}
  [[ $arm == *:* && $a =~ ^[A-Za-z0-9._-]+$ && -z ${S[$a]+x} ]] || { echo "ARM '$arm': want A:KEY=VALUE[,...], A unique"; exit 1; }
  [[ -z $c || $c =~ ^OCTOPUS_($KNOBS)=[A-Za-z0-9_.:/+-]*(,OCTOPUS_($KNOBS)=[A-Za-z0-9_.:/+-]*)*$ ]] \
    || { echo "ARM '$arm': each change OCTOPUS_KNOB=VALUE, KNOB one of $KNOBS"; exit 1; }
  S[$a]=$(python3 - "$RUN/episode_settings.json" "$c" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
for k, v in (kv.split("=", 1) for kv in filter(None, sys.argv[2].split(","))):
    s.pop(k, None) if v == "" else s.update({k: v})
print(json.dumps(s, indent=1, sort_keys=True))  # as config.save_settings writes it
EOF
)
  D=$(dirname "$RUN")/run-$L-$a
  [ ! -d "$D" ] || [ "$(cat "$D/episode_settings.json")" = "${S[$a]}" ] || { echo "$D exists with other settings: refused"; exit 1; }
done
for arm in "$@"; do
  a=${arm%%:*}; D=$(dirname "$RUN")/run-$L-$a
  if [ ! -d "$D" ]; then
    rm -rf "$D.tmp"; mkdir "$D.tmp"; cp "$RUN/config.yaml" "$D.tmp/"; echo "${S[$a]}" > "$D.tmp/episode_settings.json"; mv "$D.tmp" "$D"
  fi
  for set in $SETS; do
    n=$NN; [ "$set" = bcp ] && n=$((NN + 1)); t=$Q/$n-$set-$L-$a-$NAME.sh
    if [ -e "$t" ] || { [ -e "$t.done" ] && [ $RERUN = 0 ]; } || compgen -G "$t.running-*" > /dev/null \
      || compgen -G "$t.skip-*" > /dev/null || compgen -G "$t.deferred-*" > /dev/null; then echo "skip $t"; continue; fi
    echo "exec env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=\$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID ${EVAL_PARALLEL:+EVAL_PARALLEL=$EVAL_PARALLEL} bash $N/scripts/alpha/search_eval.sh $N $NAME $A $set $D $K 0 $L-$a" > "$t.tmp"
    mv "$t.tmp" "$t"; echo "queued $t ($D)"
  done
done
