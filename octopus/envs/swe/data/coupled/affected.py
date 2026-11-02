"""affected.py NAME: the accepted bundles of datasets/NAME whose healthcheck clean step (as in their task.toml now)
deletes other paths in their golden than the current clean_command would -> JSON list on stdout. Those are the ones
to accept and time again after `build_bundles clean` rewrites the step (recheck_fix.sh)."""
import glob, json, re, sys, tomllib
from pathlib import Path

import octopus.envs.swe.data.build_bundles as bb
from octopus.envs.swe.data.regen_tasks import N, golden_repo

DS = N / "datasets"
name = sys.argv[1]
out = []
for r in map(json.loads, open(DS / f"{name}.accepted.jsonl")):
    cmd = tomllib.loads((DS / name / r["bundle"] / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    m = re.search(r"if \[ -d /opt/miniconda3/envs/testbed \]; then rm -rf ([^;]*); fi", cmd)
    root = golden_repo(r["image"]).parent
    pkg = bb.package(r["tasks"][0]["files"][0])
    hits = lambda globs: {p for g in globs for p in glob.glob(f"{root}{g}")}
    if hits(m.group(1).split() if m else []) != hits(bb.clean_globs(pkg)):
        out.append(r["bundle"])
print(json.dumps(out))
