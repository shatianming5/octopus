#!/bin/bash
# judge.sh NAME [args]: statement_coverage judge of datasets/NAME with grok-4.6 straight to the user gateway (streamed
# chat/completions, custom User-Agent: octopus/envs/swe/data/acceptance_rewrite.ask); the gateway URL and key come from
# ~/.grok/xai.env into this process environment only (never argv, logs or files)
W=/mnt/disk6/user/coupled-s1a
set -a; . /home/user/.grok/xai.env; set +a
export REWRITE_MODEL=grok-4.6
name=$1; shift
cd $W && PYTHONPATH=$W/code /home/user/harbor-udocker/uv-tools/harbor/bin/python -m octopus.envs.swe.data.statement_coverage judge /home/user/harbor-udocker/datasets/$name $W/labels/$name.jsonl "$@"
echo "exit $?"
