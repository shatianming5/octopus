#!/bin/bash
# usage: scripts/remaining.sh <source dataset> <new dataset> <job> [<job> ...]
# New dataset = tasks of <source> with no 0/1 reward in any listed job.
source /home/user/harbor-udocker/scripts/env.sh
exec python3 -m octopus.envs.swe.data.mkremaining "$@"
