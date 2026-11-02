#!/bin/bash
# recheck_fix.sh NAME: after clean_globs grew, the bundles of datasets/NAME whose cleanup now deletes other paths
# (affected.py) are accepted and timed again (node-134, 8 at once) under NAME-fix (a link to NAME with their manifest);
# every bundle of NAME and NAME-valid first gets the current clean step (build_bundles clean); the new results are
# appended to NAME's logs (the last result of a bundle wins) and NAME's acceptance json rewritten from its log
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
D=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
PY=/home/user/octo-env/areal/.venv/bin/python
BB="-m octopus.envs.swe.data.build_bundles"
n=$1; f=$1-fix
cd $W
$PY scripts/affected.py $n > $W/affected-$n.json || exit 1
$HP $BB clean $D/$n/* $D/$n-valid/* || exit 1
python3 -c "
import json
aff = set(json.load(open('$W/affected-$n.json')))
open('$D/$f.manifest.jsonl', 'w').writelines(l for l in open('$D/$n.accepted.jsonl') if json.loads(l)['bundle'] in aff)"
[ -s $D/$f.manifest.jsonl ] || { echo "exit recheck_fix $n (none affected)"; exit 0; }
ln -sfn $D/$n $D/$f; ln -sfn $D/$n.baseline $D/$f.baseline
nice -n 19 $HP $BB accept $f 8 >> logs/accept-$f.log 2>&1 || exit 1
cat $D/$f.acceptance-1c.jsonl >> $D/$n.acceptance-1c.jsonl
nice -n 19 $HP $BB accept $n 8 >> logs/accept-$n.log 2>&1 || exit 1  # its json again from the log
python3 -c "
import json
rep = json.load(open('$D/$f.acceptance-1c.json'))
open('$D/$f.accepted.jsonl', 'w').writelines(l for l in open('$D/$f.manifest.jsonl') if rep[json.loads(l)['bundle']]['accepted'])"
nice -n 19 $HP $BB timing $f $D/$f.accepted.jsonl 8 >> logs/timing-$f.log 2>&1
cat $D/$f.accepted.jsonl.timing.jsonl >> $D/$n.accepted.jsonl.timing.jsonl
echo "exit recheck_fix $n"
