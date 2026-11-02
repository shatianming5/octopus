#!/bin/bash
# build.sh NAME PER_REPO KEY...: samefile bundles (sizes 2,2,3,4) of KEY... with the PR code (code/COMMIT); needs
# datasets/NAME.baseline/KEY.json for every key (build_bundles baseline, a container step)
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
name=$1; per=$2; shift 2
export PYTHONPATH=$W/code
cd $W && nice -n 19 /home/user/octo-env/areal/.venv/bin/python -m octopus.envs.swe.data.build_bundles build $W/swesmith-hf $name "$@" $per --sizes 2,2,3,4 --rule samefile --tag sf
echo "exit $?"
