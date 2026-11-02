#!/bin/bash
# recheck.sh NAME: the bundles of datasets/NAME-r2 that passed accept, copied to NAME-r3 with the R1 healthcheck
# (build_bundles clean: copies of the original code outside /testbed deleted), then accept and timing again
# (accept_timing.sh, sharded)
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
W=/mnt/disk6/user/coupled-s1a
DS=/home/user/harbor-udocker/datasets
export PYTHONPATH=$W/code
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
src=$1-r2; dst=$1-r3
cd $W
if [ ! -f $DS/$dst.manifest.jsonl ]; then
  $HP - $DS $src $dst <<"PY" || exit 1
import json, shutil, sys
from pathlib import Path
ds, src, dst = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [l for l in open(ds / f"{src}.accepted.jsonl")]
for l in rows:
    shutil.copytree(ds / src / json.loads(l)["bundle"], ds / dst / json.loads(l)["bundle"])
(ds / f"{dst}.baseline").mkdir()
for f in (ds / f"{src}.baseline").glob("*.json"):
    shutil.copy(f, ds / f"{dst}.baseline" / f.name)
(ds / f"{dst}.manifest.jsonl").write_text("".join(rows))
PY
  $HP -m octopus.envs.swe.data.build_bundles clean $DS/$dst/* || exit 1
fi
scripts/accept_timing.sh $dst || exit 1
echo "exit recheck $1"
