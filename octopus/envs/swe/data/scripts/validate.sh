#!/bin/bash
# usage: scripts/validate.sh <dataset> <new dataset>   — no model calls
# Runs the dataset through real Harbor with the built-in oracle and nop agents (same env, healthcheck
# and verifier as a Grok run), then keeps tasks where oracle=1 and nop=0.
set -euo pipefail
source /home/user/harbor-udocker/scripts/env.sh
DATA=$1; OUT=$2
cd /home/user
run() {
  harbor run -p "$N/datasets/$DATA" -a "$1" -e octopus.envs.swe.udocker:UdockerEnvironment \
    -n 48 -o "$N/jobs" --job-name "validate-$DATA-$1" --environment-build-timeout-multiplier 6 -y \
    > "$N/jobs/validate-$DATA-$1.console.log" 2>&1
}
rm -rf "$N/jobs/validate-$DATA-oracle" "$N/jobs/validate-$DATA-nop"
run oracle & run nop & wait
python3 -m octopus.envs.swe.data.select_valid "$DATA" "validate-$DATA-oracle" "validate-$DATA-nop" "$OUT"
