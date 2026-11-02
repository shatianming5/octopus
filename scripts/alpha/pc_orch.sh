#!/bin/bash
# 144123 after the SWE Test (three task files 12-14 in its queue): the Test's merge and paired CI, then the plan-control
# evaluation (eval only, user 10-23): coupled-a-val, rq1-v0 and rq1-v36 at once on 144123 with P/run-plan-control
# (P/run with rl.plan_control: true), parts of 15 bundles in order while each still ends before 07:00 (a part's time:
# the last part's, the first one's 1.5 x a Test part's); then pc_stats.py on the parts that ran
R=/projects/cluster/octopus/octopus; P=$R/swe/pilot/swe-read-alpha-full-1022
Q=$R/search/pilot/search-read-alpha-full-1022/queue
A=$(ls -d $R/swe/areal-exp/checkpoints/user/rq1-rl/swe-read-alpha-full-1022/default/*globalstep35 | head -1)
[ -n "$A" ] && [ -f $P/run-plan-control/config.yaml ] || { echo "no adapter or no P/run-plan-control"; exit 1; }
for k in 0 1 2; do t=$Q/1$((2 + k))-test-part$k.sh
  until [ -e $t.done ]; do [ -e $t.failed ] && { echo "$(date +%T) Test part $k failed: $t.log"; exit 1; }
    [ $(date +%s) -lt $(date -d '2026-10-23 07:05' +%s) ] || { echo "$(date +%T) Test part $k not done by 07:05"; exit 1; }; sleep 60; done; done
for n in rq1-v0 rq1-v36; do cat $P/test-$n-part0.jsonl $P/test-$n-part1.jsonl $P/test-$n-part2.jsonl > $P/test-$n.jsonl; done
echo "$(date +%T) TEST DONE"; python3 $R/logs/paired.py $P/test-rq1-v0.jsonl $P/test-rq1-v36.jsonl
last=$(( ( $(stat -c %Z $Q/14-test-part2.sh.done) - $(stat -c %Z $Q/13-test-part1.sh.done) ) * 3 / 2 ))  # ctime: the rename
end=$(date -d '2026-10-23 07:00' +%s); parts=""
E="exec env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=\$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES RUN=$P/run-plan-control bash $R/logs/eval_part.sh"
for k in 0 1 2; do
  [ $(( $(date +%s) + last )) -lt $end ] || { echo "$(date +%T) part $k would end after 07:00 (~${last}s): stop here"; break; }
  t=$Q/1$((5 + k))-pc-part$k.sh; s=$(date +%s)
  echo "$E $k $R/harbor/datasets/coupled-a-val pc rq1-v0=$R/models/adapter-v0-4b-text-r64 rq1-v36=$A" > $t
  until [ -e $t.done ]; do [ -e $t.failed ] && { echo "$(date +%T) pc part $k failed: $t.log"; exit 1; }
    [ $(date +%s) -lt $(date -d '2026-10-23 07:05' +%s) ] || { echo "$(date +%T) pc part $k not done by 07:05: left out"; break 2; }; sleep 60; done
  last=$(( $(date +%s) - s )); parts="$parts $k"; echo "$(date +%T) pc part $k done in ${last}s"
done
[ -n "$parts" ] || exit 1
for n in rq1-v0 rq1-v36; do : > $P/pc-$n.jsonl; for k in $parts; do cat $P/pc-$n-part$k.jsonl >> $P/pc-$n.jsonl; done; done
python3 $R/logs/pc_stats.py $P/pc-rq1-v0.jsonl $P/pc-rq1-v36.jsonl
echo "$(date +%T) PC DONE (parts$parts)"
