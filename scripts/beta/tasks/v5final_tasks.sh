#!/bin/bash
# v5final_tasks.sh K ADAPTER: writes the v5-final eval tasks (coordinator 10-24): ROOT/queue/13-v5final-bcp-fork-k2.sh (216338)
# and ROOT/queue-2/{21..24}-*.sh (216715) (Q1/Q2: other dirs to write them to, e.g. to diff them). Fork tasks run search_eval.sh
# once: infra failures are not rerun, as in every other arm (not scored), except when 21 resumes 13's --out after 13 was cut
# by 216338's end or ended .failed: search_dev_eval then reruns 13's rows with an error too; a linear task needs its fork output complete (as many qid x sample rows, scored or infra failure, as the
# set has) and the fork task's .done, and takes CAP = round(mean calls of the scored rows), THREAD_BUDGET default (32K).
# Historic: C = code-d3ca0c2, whose search_eval.sh still took RUN - (its ROOT/search/pilot/eval/run); current search_eval.sh requires RUN.
set -euo pipefail
K=$1; A=$2; BR=/projects/cluster/octopus/beta/octopus; C=$BR/code-d3ca0c2; O=$BR/search/pilot/eval
Q1=${Q1:-$BR/queue} Q2=${Q2:-$BR/queue-2}
TAG=v5-final-v$K
ENV='env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 SLURM_JOB_ID="$SLURM_JOB_ID" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CUDA_DEVICE_ORDER=PCI_BUS_ID'
LEAK='BUSY_MIB=6000'  # 216715's node only (GPUs 2-3: 5.2 / 4.7 GB no process owns)
complete() {  # complete FILE N: N qid x sample rows in FILE (scored or infra failure)
  echo "$BR/areal/.venv/bin/python -c 'import json, sys; last = {}
for l in open(sys.argv[1]):
    r = json.loads(l); last[(r[\"qid\"], r[\"sample\"])] = r
sys.exit(0 if len(last) == int(sys.argv[2]) else 1)' $1 $2"; }
cap() {  # cap FILE: round(mean calls) over the scored last rows
  echo "\$($BR/areal/.venv/bin/python -c 'import json, sys; last = {}
for l in open(sys.argv[1]):
    r = json.loads(l); last[(r[\"qid\"], r[\"sample\"])] = r
ok = [r for r in last.values() if \"error\" not in r]; print(round(sum(r[\"calls\"] for r in ok) / len(ok)))' $1)"; }
fork() {  # fork SET N EXTRA_ENV: a fork k=2 task body
  local f=$O/$([ $1 = val ] && echo val || echo test)-$TAG-fork-k2.jsonl
  cat <<EOF
exec $ENV $3 SEARCH_EVAL_CLEAN=1 bash $C/scripts/beta/search_eval.sh $C v$K $A $1 - 2 0 v5-final
EOF
}
lin() {  # lin SET N DONE_TEST EXTRA_ENV: a linear-32K k=2 task body, CAP from the fork output (DONE_TEST: the fork task's .done)
  local f=$O/$([ $1 = val ] && echo val || echo test)-$TAG-fork-k2.jsonl
  cat <<EOF
F=$f
{ $3; } && $(complete $f $2) || { echo "fork output not complete or not done: \$F" >&2; exit 1; }
CAP=$(cap $f); echo "\$(date) CAP \$CAP (round mean calls of \$F's scored rows), THREAD_BUDGET default"
exec $ENV $4 OCTOPUS_MAIN_MAX_CALLS=\$CAP SEARCH_EVAL_CLEAN=1 bash $C/scripts/beta/search_eval.sh $C v$K $A $1 - 2 1 v5-final
EOF
}
H="# v5-final (search-v5-beta-1023b step $K, $A) eval, coordinator 10-24; scripts/beta/search_eval.sh, main d3ca0c2"
{ echo "$H: BCP fork k=2 on 216338 (216715's 21 completes it if cut)"; fork bcp 1660 ""; } > $Q1/13-v5final-bcp-fork-k2.sh
{ echo "$H: BCP fork k=2; done if 13 finished on 216338, else resumes the same --out after 13 stopped"
  echo "T=$BR/queue/13-v5final-bcp-fork-k2.sh"
  echo "[ -e \$T.done ] && $(complete $O/test-$TAG-fork-k2.jsonl 1660) && { echo '13 finished on 216338'; exit 0; }"
  echo 'alive() { local i; for i in 1 2 3; do squeue -h -j 216338 2>/dev/null | grep -q . && return 0; sleep 20; done; return 1; }  # 216338 still runs'
  echo 'while { [ -e $T ] || ls $T.running-* > /dev/null 2>&1; } && alive; do sleep 300; done  # 13 queued or running: never two writers'
  echo "[ -e \$T.done ] && $(complete $O/test-$TAG-fork-k2.jsonl 1660) && { echo '13 finished on 216338'; exit 0; }"
  echo 'if [ -e $T.failed ]; then how="13 failed"; elif ls $T.running-* > /dev/null 2>&1; then how="13 cut by 216338'"'"'s end"; elif [ -e $T ]; then how="13 not run"; elif [ -e $T.done ]; then how="13 done, rows short"; else how="no 13 state"; fi'
  echo 'echo "$(date) running 13'"'"'s --out ($how): rows with an error in it are rerun too"'
  fork bcp 1660 "$LEAK"; } > $Q2/21-v5final-bcp-fork-k2.sh
{ echo "$H: Val fork k=2"; fork val 1000 "$LEAK"; } > $Q2/22-v5final-val-fork-k2.sh
{ echo "$H: Val linear-32K k=2, CAP from 22"; lin val 1000 "[ -e $BR/queue-2/22-v5final-val-fork-k2.sh.done ]" "$LEAK"; } > $Q2/23-v5final-val-lin32.sh
{ echo "$H: BCP linear-32K k=2, CAP from 21 (or 13)"; lin bcp 1660 "[ -e $BR/queue-2/21-v5final-bcp-fork-k2.sh.done ] || [ -e $BR/queue/13-v5final-bcp-fork-k2.sh.done ]" "$LEAK"; } > $Q2/24-v5final-bcp-lin32.sh
for f in $Q1/13-v5final-bcp-fork-k2.sh $Q2/2[1-4]-v5final-*.sh; do bash -n $f && echo "ok $f"; done
