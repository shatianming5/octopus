#!/bin/bash
# reduce_subset.sh NAME TAG BUNDLE...: reduce -> refresh -> reduce -> accept -> timing of only these bundles of datasets/NAME
# (on node-134, 8 at once) as NAME-TAG-r2, e.g. bundles an older reduce dropped; NAME-TAG links to NAME with their manifest
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
D=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
PY=/home/user/octo-env/areal/.venv/bin/python
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
BB="-m octopus.envs.swe.data.build_bundles"
name=$1; sub=$1-$2; shift 2
cd $W
python3 -c "
import json, sys
want = set(sys.argv[1:])
open('$D/$sub.manifest.jsonl', 'w').writelines(l for l in open('$D/$name.manifest.jsonl') if json.loads(l)['bundle'] in want)" "$@"
ln -sfn $D/$name $D/$sub; ln -sfn $D/$name.baseline $D/$sub.baseline
nice -n 19 $PY $BB reduce $W/swesmith-hf $sub $sub-r1 $W/labels/$name.jsonl - $name > logs/reduce1-$sub.log 2>&1 || exit 1
nice -n 19 $HP $BB refresh $sub $sub-r1 8 > logs/refresh-$sub.log 2>&1 || exit 1
nice -n 19 $PY $BB reduce $W/swesmith-hf $sub $sub-r2 $W/labels/$name.jsonl $D/$sub-r1.refresh.jsonl $name > logs/reduce2-$sub.log 2>&1 || exit 1
mkdir -p $D/$sub-r2.baseline && cp $D/$name.baseline/*.json $D/$sub-r2.baseline/
nice -n 19 $HP $BB accept $sub-r2 8 > logs/accept-$sub-r2.log 2>&1 || exit 1
python3 -c "
import json
rep = json.load(open('$D/$sub-r2.acceptance-1c.json'))
open('$D/$sub-r2.accepted.jsonl', 'w').writelines(l for l in open('$D/$sub-r2.manifest.jsonl') if rep[json.loads(l)['bundle']]['accepted'])"
nice -n 19 $HP $BB timing $sub-r2 $D/$sub-r2.accepted.jsonl 8 > logs/timing-$sub-r2.log 2>&1
echo "exit reduce_subset $sub"
