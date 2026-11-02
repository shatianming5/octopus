# Read-only (python3 - JOBS_DIR MAIN_LOG): episodes bucketed by the training step during which they finished
# (answers.json mtime between consecutive "Train step N ... done" times; a proxy: rollouts are off-policy by up to 2 versions).
# Per bucket: episodes, fork share, branches/ep, main accepted rewinds/ep and share of eps with one, rewind calls/ep (main),
# branch accepted rewinds per branch, accuracy. From main.log: the step's end time (log clock), minutes since the previous
# step's end (restarts included), ppo_actor/task_reward/avg (R = 2y-1 over the trained, non-filtered samples).
import json, os, sys, re, time, bisect, collections
from datetime import datetime
J, M = sys.argv[1], sys.argv[2]
ts = []; rew = {}; end = {}
for line in open(M, errors="replace"):
    m = re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \d+/\d+ Step (\d+)/\d+ Train step", line)
    if m:
        ts.append((datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp(), int(m.group(2)))); end[ts[-1][1]] = m.group(1)
        continue
    m = re.search(r"ppo_actor/task_reward/avg\s+│\s+(-?[\d.]+e[-+]\d+)", line)
    if m and ts: rew[ts[-1][1]] = float(m.group(1))
ts.sort(); T = [t for t, _ in ts]
B = collections.defaultdict(collections.Counter)
def calls(msgs, name):
    return sum(1 for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or []
               if (c.get("function") or {}).get("name") == name)
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not os.path.isdir(p): continue
    for t in os.listdir(p):
        q = os.path.join(p, t); f = os.path.join(q, "agent/tree.json"); a = os.path.join(q, "verifier/answers.json")
        if not (os.path.exists(f) and os.path.exists(a)): continue
        i = bisect.bisect_left(T, os.path.getmtime(a)); step = ts[i][1] if i < len(ts) else (ts[-1][1] + 1 if ts else 0)
        tr = json.load(open(f)); c = B[step]; c["n"] += 1
        c["y"] += bool(json.load(open(a))["main"].get("correct"))
        br = [b for fk in tr.get("forks") or [] for b in fk.get("branches") or []]
        c["forked"] += bool(br); c["br"] += len(br)
        rw = len(tr.get("rewinds") or []); c["mrw"] += rw; c["mrw_ep"] += bool(rw)
        msgs = list(tr.get("main") or []) + [m for r in tr.get("rewinds") or [] for m in r.get("messages") or []]
        c["mrwc"] += calls(msgs, "rewind")
        c["brw"] += sum(len(b.get("rewinds") or []) for b in br)
m = lambda a, b: a / b if b else float("nan")
dt = {s: (t - ts[i - 1][0]) / 60 for i, (t, s) in enumerate(ts) if i}
print("step  end(log)        min  task_reward | eps  fork%  br/ep  main_rw_eps%  main_rw/ep  main_rw_calls/ep  branch_rw/br  acc")
for s in sorted(B):
    c = B[s]; n = c["n"]
    print(f"{s:>4}  {end.get(s, '-'):<17} {dt.get(s, float('nan')):>4.0f}  {rew.get(s, float('nan')):+.3f} | {n:>5}  {m(c['forked'], n):.2f}  {m(c['br'], n):.2f}   {m(c['mrw_ep'], n):.3f}       {m(c['mrw'], n):.3f}      {m(c['mrwc'], n):.3f}         {m(c['brw'], c['br']):.3f}      {m(c['y'], n):.2f}")
