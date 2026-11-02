"""Keep the tasks that measure a fix, judged by real Harbor runs of the built-in agents:
the oracle (solution/solve.sh) must score 1 and nop must score 0.
usage: select_valid.py <dataset> <oracle job> <nop job> <new dataset>"""
import json, shutil, sys
from pathlib import Path

N = Path("/home/user/harbor-udocker")


def rewards(job: str) -> dict:
    out = {}
    for rj in (N / "jobs" / job).glob("*/result.json"):
        d = json.loads(rj.read_text())
        name = Path(d["config"]["task"]["path"]).name
        out[name] = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return out


src, oracle, nop, dest = N / "datasets" / sys.argv[1], rewards(sys.argv[2]), rewards(sys.argv[3]), N / "datasets" / sys.argv[4]
if dest.exists():
    raise SystemExit(f"{dest} exists; pick a new name")
tasks = sorted(t.name for t in src.iterdir() if (t / "task.toml").is_file())
valid = [t for t in tasks if oracle.get(t) in (1, 1.0) and nop.get(t) in (0, 0.0)]
report = N / "jobs" / f"{sys.argv[4]}.selection.json"
report.write_text(json.dumps({t: {"oracle": oracle.get(t), "nop": nop.get(t), "valid": t in valid} for t in tasks}, indent=1))
dest.mkdir()
for t in valid:
    shutil.copytree(src / t, dest / t)
print(f"{len(valid)}/{len(tasks)} valid -> {dest}; per-task report {report}")
