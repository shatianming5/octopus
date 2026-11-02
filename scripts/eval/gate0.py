"""Gate 0 summary: the same bundles under several eval arms (octopus.eval.dev_eval rows, one jsonl per arm; the first
is the reference), each episode read from its trial's tree.json and requests.jsonl.
The verdict: each other arm's episode y less the reference's, paired by bundle, with its 95% CI from the bootstrap
below; it passes when the CI's lower end is above 0. Every arm's episodes without a fork are counted by how main
ended (under OCTOPUS_AUTO_FORK every episode forks before main's first request: any such episode is a fault). Per arm:
  landed   = committed branch patches the join merged / (merged + taken back for a conflict), any depth (tree.json merge;
             committed branches only: an older run's tree may hold merged uncommitted patches)
  pass     = bundle tasks a branch named that its patch solved alone (row branch_credit) / such branches
  y, fork  = mean solved fraction, share of episodes that forked; y_forked = mean solved fraction of the forked ones
  commit   = branches that committed / branches, any depth
  reads    = read_thread calls per episode (main and branches); mounts = requests that carried a mount, per episode
             (voluntary and forced); forced = OCTOPUS_READ_EVENT mounts per episode (tree.json read_event)
  sibling / own / unclear = shares of the taken-back committed patches' files by what they conflict with (OCTOPUS_CONFLICT_BY,
             tree.json conflict_by; files of an episode run without it are left out)
Each metric is a ratio of sums over episodes; 95% CIs and the differences to the reference arm come from a bootstrap
over the bundles (task dir) every arm has episodes of (the others are named and left out), resampled jointly for all
arms (paired by bundle). Rows without y (infra errors) are left out.
usage: gate0.py NAME=RESULTS.jsonl NAME=RESULTS.jsonl ... [--boot 2000]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.runtime.core import tree_branches

METRICS = {  # name: (numerator, denominator) keys of an episode's counts
    "y": ("y", "ep"), "landed": ("merged", "taken"), "pass": ("solved", "named"), "commit": ("committed", "branches"),
    "fork": ("fork", "ep"), "y_forked": ("y_forked", "fork"),
    "reads": ("reads", "ep"), "mounts": ("mounts", "ep"), "forced": ("forced", "ep"),
    "sibling": ("sibling", "labelled"), "own": ("own", "labelled"), "unclear": ("unclear", "labelled")}


def episode(row: dict) -> collections.Counter:
    agent = Path(row["trial"]) / "agent"
    tree = json.loads((agent / "tree.json").read_text())
    fork = bool(tree.get("forks"))
    c = collections.Counter(ep=1, y=row["y"], fork=fork, y_forked=row["y"] if fork else 0)
    if not fork:
        c["unforked: " + str(tree.get("main_end"))] += 1
    histories = [tree["main"]]
    for _, _, b in tree_branches(tree.get("forks") or []):
        histories.append(b.get("messages") or [])
        c["branches"] += 1
        c["committed"] += b.get("ended_with") == "commit"
        if b.get("ended_with") != "commit":  # an uncommitted patch an older run merged is no landing
            continue
        merge = str(b.get("merge") or "")
        c["merged"] += merge.startswith("merged")
        c["taken"] += merge.startswith(("merged", "conflict"))
        for label in (b.get("conflict_by") or {}).values():
            c[label] += 1
            c["labelled"] += 1
    c["reads"] = sum((call.get("function") or {}).get("name") == "read_thread"
                     for h in histories for m in h for call in m.get("tool_calls") or [])
    requests = [json.loads(x) for x in (agent / "requests.jsonl").read_text().splitlines()]
    c["mounts"] = sum(bool(r.get("mount")) and r.get("id") is not None for r in requests)  # answered requests
    c["forced"] = sum(bool(e.get("mounted")) for e in tree.get("read_event") or [])
    credit = [b for b in row.get("branch_credit") or [] if b.get("task")]
    c["solved"], c["named"] = sum(bool(b.get("solved")) for b in credit), len(credit)
    return c


def value(c: collections.Counter, metric: str) -> float:
    num, den = METRICS[metric]
    return c[num] / c[den] if c[den] else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=RESULTS.jsonl, the reference arm first")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    names, per = [], {}  # arm -> bundle -> summed counts
    for arm in a.arms:
        name, path = arm.split("=", 1)
        names.append(name)
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        rows = list({(r["task_dir"], r["sample"]): r for r in rows}.values())  # an episode's last row: a retry's
        per[name] = collections.defaultdict(collections.Counter)
        for r in rows:
            if "y" in r:
                per[name][Path(r["task_dir"]).name] += episode(r)
        print(f"{name}: {sum(c['ep'] for c in per[name].values())} episodes, {len(per[name])} bundles, "
              f"{sum('y' not in r for r in rows)} infra-error rows left out")
    bundles = sorted(set.intersection(*(set(per[n]) for n in names)))  # every arm has episodes of them
    for n in names:
        if set(per[n]) - set(bundles):
            print(f"{n}: left out, not in every arm: {' '.join(sorted(set(per[n]) - set(bundles)))}")
    rng, boot = random.Random(0), collections.defaultdict(list)  # (arm, metric) -> resampled values; (arm, metric, "d")
    for _ in range(a.boot):
        pick = [rng.choice(bundles) for _ in bundles]
        tot = {n: sum((per[n].get(b, collections.Counter()) for b in pick), collections.Counter()) for n in names}
        for n in names:
            for m in METRICS:
                v = value(tot[n], m)
                boot[n, m].append(v)
                boot[n, m, "d"].append(v - value(tot[names[0]], m))

    def ci(xs):
        xs = sorted(x for x in xs if not math.isnan(x))
        return f"[{xs[int(0.025 * len(xs))]:+.3f}, {xs[int(0.975 * len(xs)) - 1]:+.3f}]" if xs else "[-]"
    ref = sum((per[names[0]][b] for b in bundles), collections.Counter())
    for n in names[1:]:
        tot = sum((per[n][b] for b in bundles), collections.Counter())
        lo = sorted(boot[n, "y", "d"])[int(0.025 * a.boot)]  # y is never nan: every bundle has episodes
        print(f"\nverdict {n} - {names[0]}: y {value(tot, 'y') - value(ref, 'y'):+.3f} {ci(boot[n, 'y', 'd'])} over "
              f"{len(bundles)} bundles: {'PASS' if lo > 0 else 'FAIL'} (CI lower end > 0)")
    for n in names:
        tot = sum((per[n][b] for b in bundles), collections.Counter())
        unforked = sorted((k.split(": ", 1)[1], v) for k, v in tot.items() if k.startswith("unforked: "))
        print(f"\n{n}: {tot['ep']} episodes, {tot['ep'] - tot['fork']} unforked"
              + (f" ({', '.join(f'{k} {v}' for k, v in unforked)})" if unforked else ""))
        print(f"{n}: " + "  ".join(f"{m} {value(tot, m):.3f} {ci(boot[n, m])}" for m in METRICS))
        if n != names[0]:
            print(f"  {n} - {names[0]}: " + "  ".join(f"{m} {value(tot, m) - value(ref, m):+.3f} {ci(boot[n, m, 'd'])}"
                                                     for m in METRICS))


if __name__ == "__main__":
    main()
