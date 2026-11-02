#!/bin/bash
# base.sh NAME KEY...: build_bundles baseline (tests passing on the original commit) of KEY... into datasets/NAME.baseline
# (udocker trials, 8 at once, nice 19), with the deployed code snapshot (code/COMMIT)
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
export PYTHONPATH=$W/code
name=$1; shift
cd $W && nice -n 19 /home/user/harbor-udocker/uv-tools/harbor/bin/python -m octopus.envs.swe.data.build_bundles baseline $name "$@"
echo "exit $?"
