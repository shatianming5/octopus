#!/bin/bash
# drop_budget.sh OUT: docs/task-bundles.md "Dropping the old budget" on datasets/OUT-{train,val,test} (bundle dirs and
# manifests), then the check that no instruction and no manifest row still carries the budget
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL
W=/mnt/disk6/user/coupled-s1a
D=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
for s in train val test; do
  find $D/$1-$s -name instruction.md -exec grep -lZ "You have a budget of" {} + | xargs -0 -r dirname -z \
    | xargs -0 -r $HP -m octopus.envs.swe.data.build_bundles drop-budget
  ! grep -q '"budget":' $D/$1-$s.manifest.jsonl || $HP -m octopus.envs.swe.data.build_bundles drop-budget $D/$1-$s.manifest.jsonl
  echo "$s: instructions with a budget $(grep -rl --include=instruction.md "You have a budget of" $D/$1-$s | wc -l)," \
       "manifest rows with one $(grep -c '"budget":' $D/$1-$s.manifest.jsonl)"
done
