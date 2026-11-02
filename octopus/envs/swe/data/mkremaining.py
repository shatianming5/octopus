"""Tasks of a source dataset that still need a run: no 0/1 reward in any of the given jobs, or a 0
whose agent was cut short by a failed model request (e.g. an unretried 504), which says nothing
about whether the task is solvable.
usage: mkremaining.py <source dataset> <new dataset> <job> [<job> ...]  (names under datasets/ and jobs/)"""
import json, shutil, sys
from pathlib import Path

N = Path("/home/user/harbor-udocker")
source, out, jobs = N / "datasets" / sys.argv[1], N / "datasets" / sys.argv[2], sys.argv[3:]
scored = set()
for job in jobs:
    for rj in (N / "jobs" / job).glob("*/result.json"):
        d = json.loads(rj.read_text())
        rw = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        name = Path(d["config"]["task"]["path"]).name
        if rw in (1, 1.0):
            scored.add(name)
        elif rw in (0, 0.0):
            trajectory = rj.parent / "agent" / "trajectory.json"
            if not (trajectory.is_file() and "request failed" in trajectory.read_text()):
                scored.add(name)
if out.exists():
    raise SystemExit(f"{out} exists; pick a new name")
out.mkdir()
for task in sorted(source.iterdir()):
    if task.is_dir() and task.name not in scored:
        shutil.copytree(task, out / task.name, symlinks=True)
print(f"source={sum(1 for _ in source.iterdir())} scored={len(scored)} remaining={sum(1 for _ in out.iterdir())}")
