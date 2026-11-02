#!/bin/bash
# stage2.sh NAME: after its judge loop, reduce -> refresh -> reduce (NAME-r2) -> accept -> timing of the accepted bundles;
# each container stage sharded by repo over node-134 (8 slots) and node-138 (16) (shard.py, shard_agent.sh),
# then the stage once more on node-134 over the whole dataset (it skips every bundle with a result, so only lost ones run)
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
DS=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
PY=/home/user/octo-env/areal/.venv/bin/python
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
BB="-m octopus.envs.swe.data.build_bundles"
name=$1
cd $W
wait_remote() {  # $1 shard dir: until every remote host with a job there has its .done
  for j in $1/*.job.json; do [ -e "$j" ] || continue; h=$(basename $j .job.json); until [ -f $1/$h.done ]; do sleep 60; done; done
}
until grep -q "exit judge $name" logs/judge-loop-$name.log 2>/dev/null; do sleep 60; done
[ -f $DS/$name-r1.reduce.json ] || nice -n 19 $PY $BB reduce $W/swesmith-hf $name $name-r1 $W/labels/$name.jsonl - $name >> logs/reduce1-$name.log 2>&1 || exit 1
# refresh: own share through a dataset name that links to the cut dataset
d=$W/shards/refresh-$name; $PY scripts/shard.py plan refresh $name $name-r1
ln -sfn $DS/$name-r1 $DS/$name-r1-g134; cp $d/g134.manifest.jsonl $DS/$name-r1-g134.manifest.jsonl
nice -n 19 $HP $BB refresh $name $name-r1-g134 8 >> logs/refresh-$name.log 2>&1; cp $DS/$name-r1-g134.refresh.jsonl $d/g134.refresh.jsonl
wait_remote $d; $PY scripts/shard.py merge refresh $name $name-r1
nice -n 19 $HP $BB refresh $name $name-r1 8 >> logs/refresh-$name.log 2>&1 || exit 1
[ -f $DS/$name-r2.reduce.json ] || nice -n 19 $PY $BB reduce $W/swesmith-hf $name $name-r2 $W/labels/$name.jsonl $DS/$name-r1.refresh.jsonl $name >> logs/reduce2-$name.log 2>&1 || exit 1
mkdir -p $DS/$name-r2.baseline && cp -n $DS/$name.baseline/*.json $DS/$name-r2.baseline/
scripts/accept_timing.sh $name-r2 || exit 1
echo "exit stage2 $name"
