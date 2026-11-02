"""Shard a container stage (refresh / accept / timing) of a node-134 dataset by repo over node-134 and node-138.

shard.py plan STAGE NAME [OUT]   -> shards/STAGE-NAME/<host>.manifest.jsonl (+ <host>.go for the remote hosts): each
                                    repo to the host (with a golden for its image) that has the least work per slot
shard.py merge STAGE NAME [OUT]  -> the hosts' result lines folded into node-134's own stage log (one line per bundle),
                                    for accept also the accepted bundle dirs into datasets/NAME-valid
Rows: accept = datasets/NAME.manifest.jsonl, timing = datasets/NAME.accepted.jsonl, refresh = datasets/OUT rows with a
cut task. Remote hosts run shard_agent.sh, which pulls its job over the LAN and pushes <host>.<log> back.
"""
import collections, hashlib, json, os, shutil, sys
from pathlib import Path

W = Path("/mnt/disk6/user/coupled-s1a")
DS = Path("/home/user/harbor-udocker/datasets")
# node-68 left out: without an overlay each trial copies its 4 GB golden off a 96%-full HDD (80 of 208 bundles in 30 min)
SLOTS = {"g134": 8, "g138": 16}
HAVE = {h: {l.split()[1] for l in open(W / f"golden-{h}.txt")} for h in ("g138",)}  # repo keys with a golden there


def rows(stage, name, out):
    if stage == "refresh":
        return [r for r in map(json.loads, open(DS / f"{out}.manifest.jsonl"))
                if any(str(v).startswith("cut") for v in r["cut"].values())]
    return [json.loads(l) for l in open(DS / (f"{name}.accepted.jsonl" if stage == "timing" else f"{name}.manifest.jsonl"))]


def plan(stage, name, out=""):
    by = collections.defaultdict(list)
    for r in rows(stage, name, out):
        by[r["repo"]].append(r)
    load, part = collections.Counter(), collections.defaultdict(list)
    for repo in sorted(by, key=lambda k: -len(by[k])):
        hosts = ["g134"] + [h for h in HAVE if repo in HAVE[h]]
        h = min(hosts, key=lambda h: (load[h] + len(by[repo])) / SLOTS[h])
        load[h] += len(by[repo])
        part[h] += by[repo]
    d = W / "shards" / f"{stage}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    for h, rs in part.items():
        (d / f"{h}.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        if h != "g134":
            (d / f"{h}.job.json").write_text(json.dumps({"stage": stage, "name": name, "out": out, "conc": SLOTS[h]}))
            (d / f"{h}.go").touch()
    print(json.dumps({h: len(rs) for h, rs in part.items()}))


def merge(stage, name, out=""):
    d = W / "shards" / f"{stage}-{name}"
    own = {"accept": DS / f"{name}.acceptance-1c.jsonl", "timing": DS / f"{name}.accepted.jsonl.timing.jsonl",
           "refresh": DS / f"{out}.refresh.jsonl"}[stage]
    seen = {json.loads(l)["bundle"] for l in open(own)} if own.exists() else set()
    added = collections.Counter()
    with open(own, "a") as f:
        for part in sorted(d.glob(f"*.{stage}.jsonl")):
            for l in open(part):
                b = json.loads(l)["bundle"]
                if b not in seen:
                    seen.add(b)
                    f.write(l if l.endswith("\n") else l + "\n")
                    added[part.name.split(".")[0]] += 1
                    if stage == "accept" and json.loads(l)["report"]["accepted"]:
                        dest = DS / f"{name}-valid" / b
                        if not dest.exists():
                            shutil.copytree(DS / name / b, dest)
    want = {r["bundle"] for r in rows(stage, name, out)}
    print(json.dumps({"added": added, "missing": sorted(want - seen)[:20], "n_missing": len(want - seen)}))


if __name__ == "__main__":
    {"plan": plan, "merge": merge}[sys.argv[1]](*sys.argv[2:])
