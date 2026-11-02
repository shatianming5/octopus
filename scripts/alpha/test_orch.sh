#!/bin/bash
# after Val v36 (val36_orch.sh): Test on coupled-a-test (44 bundles) for rq1-v0 and rq1-v36 in three jobs at once
# (144129 part 0, 144687 part 1, 144123 part 2 if before 06:00 on 10-23, else after part 0 in 144129), then the merge
# into P/test-rq1-v0.jsonl, P/test-rq1-v36.jsonl and the paired bootstrap
R=/projects/cluster/octopus/octopus; P=$R/swe/pilot/swe-read-alpha-full-1022
D=$R/harbor/datasets/coupled-a-test
until grep -q 'VAL36 DONE' $R/logs/val36_orch.log; do sleep 60; done; echo "$(date +%T) Val v36 done"
A=$(ls -d $R/swe/areal-exp/checkpoints/user/rq1-rl/swe-read-alpha-full-1022/default/*globalstep35 | head -1)
[ -n "$A" ] || { echo "$(date +%T) no step-36 adapter"; exit 1; }
T="exec env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=\$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES bash $R/logs/eval_part.sh"
M="test rq1-v0=$R/models/adapter-v0-4b-text-r64 rq1-v36=$A"
task[0]=$P/queue/11-test-part0.sh; echo "$T 0 $D $M" > ${task[0]}
task[1]=$R/search/pilot/hold-144687/queue/06-test-part1.sh; echo "$T 1 $D $M" > ${task[1]}
if [ $(date +%s) -lt $(date -d '2026-10-23 06:00' +%s) ]; then task[2]=$R/search/pilot/search-read-alpha-full-1022/queue/11-test-part2.sh
else task[2]=$P/queue/12-test-part2.sh; fi; echo "$T 2 $D $M" > ${task[2]}
echo "$(date +%T) queued"
for k in 0 1 2; do for n in rq1-v0 rq1-v36; do
  until [ -f $P/test-$n-part$k.log ] && grep -q 'infra failures' $P/test-$n-part$k.log; do
    [ -e ${task[$k]}.failed ] && { echo "$(date +%T) part $k failed: ${task[$k]}.log"; exit 1; }; sleep 60; done
  echo "$(date +%T) part $k $n: $(tail -n 1 $P/test-$n-part$k.log | cut -c1-200)"; done; done
for n in rq1-v0 rq1-v36; do cat $P/test-$n-part0.jsonl $P/test-$n-part1.jsonl $P/test-$n-part2.jsonl > $P/test-$n.jsonl; done
python3 $R/logs/paired.py $P/test-rq1-v0.jsonl $P/test-rq1-v36.jsonl
echo "$(date +%T) TEST DONE"
