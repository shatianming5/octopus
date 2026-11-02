"""Pass-2 drop rules of one screening batch (as for swesmith-n560-valid on 2026-10-06), over the two oracle/nop passes:
reference-fix pytest over 60 s in either pass, rewards flipped in pass 2, an F2P test deselected as known-failing,
a problem statement shared with another task of the pool (existing valid sets + finished batches + this batch).
Writes datasets/swesmith-NAME-final and pool-build/final-NAME.json.   usage: finalize_batch.py NAME"""
import collections, glob, json, os, re, shutil, sys
import pandas as pd

N = "/home/user/harbor-udocker"; B = "/home/user/pool-build"; NAME = sys.argv[1]
D = f"{N}/datasets"; VALID = f"{D}/swesmith-{NAME}-valid"; FINAL = f"{D}/swesmith-{NAME}-final"
POOL = ["swesmith-n560-valid", "swesmith-mb750", "swesmith-mb-spare"]


def dirs(ds):
    return sorted(t for t in os.listdir(f"{D}/{ds}") if os.path.isdir(f"{D}/{ds}/{t}")) if os.path.isdir(f"{D}/{ds}") else []


def job(name):
    out = {}
    for rj in glob.glob(f"{N}/jobs/{name}/*/result.json"):
        d = json.load(open(rj)); t = os.path.basename(d["config"]["task"]["path"])
        so = os.path.join(os.path.dirname(rj), "verifier/test-stdout.txt")
        m = re.findall(r" in ([0-9.]+)s", open(so, errors="replace").read()) if os.path.exists(so) else []
        out[t] = {"reward": ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                  "secs": float(m[-1]) if m else None, "exc": d.get("exception_info") is not None}
    return out


o1, n1, o2, n2 = (job(f"{p}-{NAME}-{a}") for p in ("v1", "v2") for a in ("oracle", "nop"))
df = pd.concat(pd.read_parquet(f, columns=["instance_id", "FAIL_TO_PASS", "problem_statement"])
               for f in sorted(glob.glob("/home/user/swesmith-hf/*.parquet")))
f2p = {i: set(x) for i, x in zip(df.instance_id, df.FAIL_TO_PASS)}
stmt = {i: (s or "").strip() for i, s in zip(df.instance_id, df.problem_statement)}
tasks = dirs(f"swesmith-{NAME}-valid")
pool = set()
for ds in POOL + [os.path.basename(p) for p in glob.glob(f"{D}/swesmith-*-final") if p != FINAL]:
    pool.update(dirs(ds))
count = collections.Counter(stmt.get(t, "") for t in pool | set(tasks))
drops = collections.defaultdict(list)
for t in tasks:
    kf = f"{VALID}/{t}/tests/known_failing.txt"
    known = set(open(kf).read().split()) if os.path.exists(kf) else set()
    secs = [j[t]["secs"] for j in (o1, o2) if t in j]
    if any(s is None for s in secs): drops["no pytest timing in oracle output"].append(t)
    if any(s is not None and s > 60 for s in secs): drops["oracle pytest > 60 s"].append(t)
    if (o2.get(t, {}).get("reward"), n2.get(t, {}).get("reward")) != (1, 0): drops["flipped in pass 2"].append(t)
    if f2p.get(t, set()) & known: drops["F2P test deselected as known-failing"].append(t)
    if count[stmt.get(t, "")] > 1: drops["shared problem statement"].append(t)
dropped = {t for v in drops.values() for t in v}
kept = [t for t in tasks if t not in dropped]
if os.path.exists(FINAL):
    raise SystemExit(f"{FINAL} exists")
os.mkdir(FINAL)
for t in kept:
    shutil.copytree(f"{VALID}/{t}", f"{FINAL}/{t}")
src = dirs(f"tasks-{NAME}")
report = {"name": NAME, "candidates": len(src) or len(dirs(f"swesmith-{NAME}")), "regenerated": len(dirs(f"swesmith-{NAME}")),
          "valid_pass1": len(tasks), "dropped": {k: len(v) for k, v in drops.items()}, "final": len(kept), "drops": drops,
          "trial_exceptions": {j: sum(x["exc"] for x in v.values()) for j, v in
                               zip(("v1-oracle", "v1-nop", "v2-oracle", "v2-nop"), (o1, n1, o2, n2))}}
json.dump(report, open(f"{B}/final-{NAME}.json", "w"), indent=1)
print(f"FINAL {NAME}: candidates {report['candidates']} regen {report['regenerated']} valid {len(tasks)} "
      f"dropped {report['dropped']} final {len(kept)} exceptions {report['trial_exceptions']}")
