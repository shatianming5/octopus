#!/bin/bash
# judge_loop.sh NAME BUILDLOG WORKERS: judge datasets/NAME batch by batch while its build runs (the manifest grows per
# repo), until the build has ended and every task is labelled
W=/mnt/disk6/user/coupled-s1a
name=$1; build=$2; workers=$3
until [ -f /home/user/harbor-udocker/datasets/$name.manifest.jsonl ]; do sleep 60; done
while true; do
  ended=$(grep -c "^exit" $build)
  $W/scripts/judge.sh $name --workers $workers >> $W/logs/judge-$name.log 2>&1
  [ "$ended" -ge 1 ] && python3 $W/scripts/unlabelled.py $name && break
  sleep 120
done
echo "exit judge $name"
