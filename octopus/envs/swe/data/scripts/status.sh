#!/bin/bash
# usage: scripts/status.sh <job name>
source /home/user/harbor-udocker/scripts/env.sh
exec python3 -m octopus.envs.swe.data.status "$1"
