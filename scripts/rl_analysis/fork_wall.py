"""Forked episodes vs their main-first control by step window: wall-clock C_wall ratio, share where the fork is faster,
and the y gain when faster / slower. usage: python fork_wall.py JOBS_DIR MAIN_LOG"""
import glob, json, os, re, statistics as st, sys
from datetime import datetime
from pathlib import Path
from octopus.envs.swe.reward import plan_outcome
J, LOG = sys.argv[1], sys.argv[2]
done = [datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp() for m in re.finditer(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \S+ Step \d+/\d+ Train step", open(LOG).read())]
w = lambda t: next((i + 1 for i, d in enumerate(done) if t <= d), len(done) + 1)
rows = []
for tf in glob.glob(f"{J}/*/*/agent/tree.json"):
    trial = Path(tf).parent.parent; rj = trial / "result.json"
    if not rj.exists() or json.loads(rj.read_text()).get("exception_info"): continue
    try:
        o = plan_outcome(trial, [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()])
    except Exception: continue
    c = o.get("control") or {}
    if o["forked"] and c.get("kind") == "main-first" and c.get("C_wall"):
        rows.append((w(rj.stat().st_mtime), o["y"], c["y"], o["C_wall"], c["C_wall"]))
def q(x, p): x = sorted(x); return x[int(p * (len(x) - 1))]
for lo, hi in [(1, 10), (11, 20), (21, 30), (31, 53)]:
    S = [r for r in rows if lo <= r[0] <= hi]
    if not S: continue
    ratio = [r[3] / r[4] for r in S]
    win = [r for r in S if r[1] >= r[2]]  # fork at least as good
    print(f"s{lo}-{hi}: pairs {len(S)}  fork/control wall median {q(ratio,.5):.2f}  fork faster {sum(x<1 for x in ratio)/len(S):.0%}  "
          f"fork wall med {q([r[3] for r in S],.5)/60:.1f} min vs control {q([r[4] for r in S],.5)/60:.1f}  "
          f"faster AND y>=control {sum(1 for r in S if r[3]<r[4] and r[1]>=r[2])/len(S):.0%}  y-gain when faster {st.mean([r[1]-r[2] for r in S if r[3]<r[4]]) if any(r[3]<r[4] for r in S) else float('nan'):+.3f}  when slower {st.mean([r[1]-r[2] for r in S if r[3]>=r[4]]):+.3f}")
