#!/bin/bash
# accept_timing.sh NAME: accept of datasets/NAME, then timing of its accepted bundles; each sharded by repo over node-134
# (8 slots) and node-138 (16) (shard.py, shard_agent.sh), then once more on node-134 over the whole dataset
# (it skips every bundle with a result, so only lost ones run)
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
DS=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
PY=/home/user/octo-env/areal/.venv/bin/python
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
BB="-m octopus.envs.swe.data.build_bundles"
n=$1
cd $W
wait_remote() {  # $1 shard dir: until every remote host with a job there has its .done
  for j in $1/*.job.json; do [ -e "$j" ] || continue; h=$(basename $j .job.json); until [ -f $1/$h.done ]; do sleep 60; done; done
}
# accept
if [ ! -f $DS/$n.acceptance-1c.json ]; then  # an accept already finished (e.g. its shards merged by hand) is kept
d=$W/shards/accept-$n; $PY scripts/shard.py plan accept $n
ln -sfn $DS/$n $DS/$n-g134; ln -sfn $DS/$n.baseline $DS/$n-g134.baseline
cp $d/g134.manifest.jsonl $DS/$n-g134.manifest.jsonl
nice -n 19 $HP $BB accept $n-g134 8 >> logs/accept-$n.log 2>&1; cp $DS/$n-g134.acceptance-1c.jsonl $d/g134.accept.jsonl
wait_remote $d; $PY scripts/shard.py merge accept $n
nice -n 19 $HP $BB accept $n 8 >> logs/accept-$n.log 2>&1 || exit 1
fi
$HP -c "
import json
rep = json.load(open('$DS/$n.acceptance-1c.json'))
with open('$DS/$n.accepted.jsonl', 'w') as f:
    f.writelines(l for l in open('$DS/$n.manifest.jsonl') if rep.get(json.loads(l)['bundle'], {}).get('accepted'))"
# timing
d=$W/shards/timing-$n; $PY scripts/shard.py plan timing $n
nice -n 19 $HP $BB timing $n $d/g134.manifest.jsonl 8 >> logs/timing-$n.log 2>&1; cp $d/g134.manifest.jsonl.timing.jsonl $d/g134.timing.jsonl
wait_remote $d; $PY scripts/shard.py merge timing $n
nice -n 19 $HP $BB timing $n $DS/$n.accepted.jsonl 8 >> logs/timing-$n.log 2>&1
echo "exit accept_timing $n"
