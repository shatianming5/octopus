#!/bin/bash
# the SWE Test's merge and paired CI once its last part is done in 144123, then 4 SWE rollout servers there for v5
P=/projects/cluster/octopus/octopus/swe/pilot/swe-read-alpha-full-1022; Q=/projects/cluster/octopus/octopus/search/pilot/search-read-alpha-full-1022/queue
until [ -e $Q/14-test-part2.sh.done ]; do [ -e $Q/14-test-part2.sh.failed ] && { echo "part 2 failed"; exit 1; }; sleep 60; done
for n in rq1-v0 rq1-v36; do cat $P/test-$n-part0.jsonl $P/test-$n-part1.jsonl $P/test-$n-part2.jsonl > $P/test-$n.jsonl; done
echo "$(date +%T) TEST DONE"; python3 /projects/cluster/octopus/octopus/logs/paired.py $P/test-rq1-v0.jsonl $P/test-rq1-v36.jsonl
