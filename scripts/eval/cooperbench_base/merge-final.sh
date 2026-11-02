#!/bin/bash
# merge the three shard outputs into the single base train6 output on node-134 (all rows kept; the analysis takes each episode's last row)
B=/home/user/harbor-udocker
OUT=$B/evals/s1b-base-train6-122f874.jsonl
cp $OUT $B/s1b/merged-before-136-68.bak.jsonl
scp -q -o BatchMode=yes user@10.0.2.36:/home/user/harbor-udocker/s1b/evals-136.jsonl $B/s1b/shard-136-main.jsonl || echo FAIL-136main
scp -q -o BatchMode=yes user@10.0.2.36:/home/user/harbor-udocker/s1b/evals-136-pal.jsonl $B/s1b/shard-136-pal.jsonl || echo FAIL-136pal
cat $B/s1b/shard-136-main.jsonl $B/s1b/shard-136-pal.jsonl $B/s1b/evals-134-pal.jsonl >> $OUT
echo rows=$(wc -l < $OUT)
python3 $B/s1b/analyze-final.py $OUT
