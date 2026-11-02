#!/bin/bash
# after SWE v5's step 36: Val and Test of its adapter (v5_eval_part.sh), the whole Test on 145532 (48 cores, alphagpu21: where
# the baseline's Test ran) and the Val halves on 144129 and 144687 when they can start by 12:40 and end by 14:00 (144129 is
# TERMed ~14:02, a TERMed part stays .running), else the whole Val on 145532 after the Test (it runs until 10-24 07:19); each
# set merged and paired with the baseline's (swe-read-alpha-full-1022) v0 and v36 when it is done, plus fork / branch rates
R=/projects/cluster/octopus/octopus; P=$R/swe/pilot/swe-read-alpha-v5-1023; B=$R/swe/pilot/swe-read-alpha-full-1022
CK=$R/swe/areal-exp/checkpoints/user/rq1-rl/swe-read-alpha-v5-1023/default; H=$R/search/pilot/hold-144687/queue; H2=$R/search/pilot/hold-145532/queue
T='exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash '$R'/logs/v5_eval_part.sh'
before() { [ $(date +%s) -lt $(date -d "$1" +%s) ]; }
killsleep() { srun --jobid=$1 --overlap -n1 -c1 bash -c 'for p in $(ps -u user -o pid=,args= | awk "\$2==\"sleep\" && \$3==\"infinity\" {print \$1}"); do kill $p; done'; }  # its serve task ends
part() { task[$1]=$2; echo "$T $1 $3 $4 $5" > $2; }  # part K TASK SET OFFSET LIMIT
done_part() {  # done_part K SET DEADLINE: 0 once part K's log has its last line, 1 when its task failed or DEADLINE passed
  until [ -f $P/$2-v5-part$1.log ] && grep -q 'infra failures' $P/$2-v5-part$1.log; do
    [ -e ${task[$1]}.failed ] && { echo "$(date +%T) part $1 failed: ${task[$1]}.log"; return 1; }
    before "$3" || { echo "$(date +%T) $3: part $1 $2 not done"; return 1; }; sleep 60; done
  echo "$(date +%T) part $1 $2: $(tail -n 1 $P/$2-v5-part$1.log | cut -c1-200)"; }
report() {  # report SET BASELINE_V0 BASELINE_V36: P/SET-v5.jsonl paired with both, then each one's fork and branch rates
  python3 $R/logs/paired.py $2 $P/$1-v5.jsonl; python3 $R/logs/paired.py $3 $P/$1-v5.jsonl
  python3 - $2 $3 $P/$1-v5.jsonl <<'PY'
import json, sys
for f in sys.argv[1:]:  # a rerun's last row counts, error rows not
    rows = [r for r in {(r["task_dir"], r["sample"]): r for r in map(json.loads, open(f))}.values() if "error" not in r]
    bc = [b for r in rows for b in r.get("branch_credit") or []]
    print(f.rsplit("/", 1)[-1], "episodes", len(rows), "forked %.3f" % (sum(bool(r.get("forks")) for r in rows) / len(rows)),
          "branches/episode %.2f" % (sum(r.get("branches") or 0 for r in rows) / len(rows)), "branches scored", len(bc),
          *(("self-solved %.3f landed %.3f" % (sum(b["solved"] for b in bc) / len(bc), sum(b["landed"] for b in bc) / len(bc)),) if bc else ()))
PY
  echo "$(date +%T) $1 DONE"; }
touch $P/val-v36.done  # the controller's own one-node Val v36 is skipped: the parts below replace it
until ls $CK/*globalstep35/adapter_model.safetensors > /dev/null 2>&1 && grep -qs 'gate runs done' $P/queue/*.log; do
  before '2026-10-23 14:17' || { echo "$(date +%T) 144129 ended, no gate line: training not finished (continue elsewhere): $P/queue"; exit 1; }; sleep 60; done
echo "$(date +%T) step-36 adapter, trainer done"
part 2 $H2/02-eval-v5-part2.sh test 0 44
halves=0; if before '2026-10-23 12:40'; then part 0 $P/queue/03-eval-v5-part0.sh val 0 23; part 1 $H/10-eval-v5-part1.sh val 23 22; halves=1; fi
killsleep 145532; killsleep 144687; echo "$(date +%T) queued: Test on 145532$([ $halves = 1 ] && echo ', Val halves on 144129 / 144687' || echo ' (after 12:40: no Val halves)')"
if [ $halves = 1 ] && done_part 0 val '2026-10-23 14:00' && done_part 1 val '2026-10-23 14:00'; then
  cat $P/val-v5-part0.jsonl $P/val-v5-part1.jsonl > $P/val-v5.jsonl; report val $B/val-v0.jsonl $B/val-v36.jsonl
else
  part 3 $H2/03-eval-v5-part3.sh val 0 45; echo "$(date +%T) the whole Val queued on 145532 after the Test"
fi
rc=0  # the sets are independent: a failed Test still leaves the whole Val's merge and pairing
done_part 2 test '2026-10-24 07:00' && { cp $P/test-v5-part2.jsonl $P/test-v5.jsonl; report test $B/test-rq1-v0.jsonl $B/test-rq1-v36.jsonl; } || rc=1
if [ -n "${task[3]}" ]; then done_part 3 val '2026-10-24 07:00' && { cp $P/val-v5-part3.jsonl $P/val-v5.jsonl; report val $B/val-v0.jsonl $B/val-v36.jsonl; } || rc=1; fi
echo "$(date +%T) V5 EVAL $([ $rc = 0 ] && echo DONE || echo 'ended with a failed part')"; exit $rc
