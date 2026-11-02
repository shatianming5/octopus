"""Pick new SWE-smith instances and write them as a Harbor source dataset for regen_tasks.py.

An instance qualifies when its kind matches (default lm_rewrite, like every task so far), its image is local, its branch
is in that image's golden repo, it has a problem statement, and no dataset under datasets/ uses it yet. Picks go
round-robin over repos (seeded shuffle), so the new tasks spread over every repo.

usage: select_swesmith.py PARQUET_DIR N NEW_DATASET [KIND]   (PARQUET_DIR: the SWE-bench/SWE-smith data/*.parquet)
then:  regen_tasks.py NEW_DATASET <graded dataset>, baseline_tasks.py, scripts/validate.sh
"""
import collections, glob, json, os, random, subprocess, sys
from pathlib import Path

import pandas as pd

from octopus.envs.swe.taskset import is_excluded, verifier_shell, write_task_tree

N = Path("/home/user/harbor-udocker")


def main(parquets: str, n: int, dest: Path, kind: str):
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick a new name")
    rows = pd.concat(pd.read_parquet(f, columns=["instance_id", "image_name", "repo", "problem_statement"])
                     for f in sorted(glob.glob(f"{parquets}/*.parquet")))
    local = {}  # repo part of the image name (HF says jyangballin/, the local copies swebench/) -> local image
    for line in subprocess.run([str(N / "bin/udocker"), "images"], capture_output=True, text=True).stdout.splitlines():
        if "swesmith" in line:
            image = line.split()[0]
            local[image.split(":")[0].split("/")[-1]] = image
    branches = set()
    for cid in os.listdir(N / "golden"):
        repo = N / "golden" / cid / "ROOT/testbed"
        if (repo / ".git").exists():
            refs = subprocess.run(["git", "-C", str(repo), "for-each-ref", "--format=%(refname:short)"],
                                  capture_output=True, text=True).stdout.split()
            branches.update(r.removeprefix("origin/") for r in refs)
    used = {name.split("--")[0] for d in (N / "datasets").iterdir() if d.is_dir() for name in os.listdir(d)}
    reasons, pool = collections.Counter(), collections.defaultdict(list)
    for r in rows.itertuples():
        if f".{kind}__" not in r.instance_id:
            continue
        image = r.image_name.split(":")[0].split("/")[-1]
        why = ("image not local" if image not in local else "branch not in golden" if r.instance_id not in branches
               else "used" if r.instance_id in used else "excluded" if is_excluded(r.instance_id, set())
               else "no problem statement" if not (r.problem_statement or "").strip() else "eligible")
        reasons[why] += 1
        if why == "eligible":
            pool[r.repo].append({"instance_id": r.instance_id, "image": local[image], "repo": r.repo,
                                 "instruction": r.problem_statement, "verifier_shell": verifier_shell(r.instance_id)})
    random.seed(0)
    for specs in pool.values():
        random.shuffle(specs)
    picked = [s for k in range(max(map(len, pool.values()), default=0)) for repo in sorted(pool) if k < len(pool[repo])
              for s in [pool[repo][k]]][:n]
    for spec in picked:
        write_task_tree(dest, spec)
    print(f"{kind}: {dict(reasons)}; {len(picked)} tasks from {len({s['repo'] for s in picked})} repos -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), N / "datasets" / sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "lm_rewrite")
