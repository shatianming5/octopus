#!/bin/bash
# usage: scripts/prebuild.sh <dataset under datasets/> [workers=4]  — build missing goldens up front
source /home/user/harbor-udocker/scripts/env.sh
exec "$N/uv-tools/harbor/bin/python" -m octopus.envs.swe.data.prebuild_goldens "$N/datasets/$1" "${2:-4}"
