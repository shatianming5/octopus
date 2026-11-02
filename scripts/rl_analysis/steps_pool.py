"""Pooled steps.py metrics over step ranges (y, fork rate, M2, check validity, commit-pass) from its cache.
usage: python steps_pool.py TRIAL CACHE.jsonl 1-10 11-20 ..."""
import json, re, statistics as st, sys
from datetime import datetime
trial, cache = sys.argv[1], sys.argv[2]
done = [datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp() for m in re.finditer(
    r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \S+ Step \d+/\d+ Train step",
    open(f"/home/user/areal-exp/logs/user/rq1-rl/{trial}/main.log").read())]
def w(t):
    for i, d in enumerate(done):
        if t <= d: return i + 1
    return len(done) + 1
R = [json.loads(l) for l in open(cache)]
for lo, hi in [(int(a), int(b)) for a, b in (x.split("-") for x in sys.argv[3:])]:
    ok = [r for r in R if "error" not in r and lo <= w(r["end"]) <= hi]
    m2 = [r["m2"] for r in ok if r["m2"] is not None]
    cv = [v for r in ok for v in r.get("cv", [])]; ov = [v for r in ok for v in r.get("ov", [])]
    cm = sum(r["commit"] for r in ok); cp = sum(r["cpass"] for r in ok)
    fk = sum(r["forked"] for r in ok) / len(ok)
    print(f"s{lo}-{hi} eps {len(ok)} y {st.mean(r['y'] for r in ok):.3f} fork {fk:.0%} "
          f"M2 {st.mean(m2):+.3f}+-{st.stdev(m2) / len(m2) ** .5:.3f} n{len(m2)} "
          f"cval {sum(cv) / max(1, len(cv)):.0%} n{len(cv)} oval {sum(ov) / max(1, len(ov)):.0%} n{len(ov)} commit-pass {cp / max(1, cm):.0%} n{cm}")
