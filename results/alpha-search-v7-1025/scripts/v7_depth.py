# Read-only (python3 - JOBS_DIR MAIN_LOG [STEPS_PER_BUCKET, default 1]): unlimited-depth behaviour of v7 per step bucket
# (answers.json mtime between "Train step" times): episodes, fork share, branches/ep, max depth distribution, share of eps reaching depth>=2/>=3,
# window refusals, branch-cap refusals, request timeouts, main ending kinds, accuracy, and the accuracy of the episodes whose
# main ended with "no tool call" (graded from its last text before #306).
import json, os, sys, re, bisect, collections
from datetime import datetime
J, M = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ts = sorted((datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp(), int(m.group(2))) for line in open(M, errors="replace")
            for m in [re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \d+/\d+ Step (\d+)/\d+ Train step", line)] if m)
T = [t for t, _ in ts]
WIN, CAP = "not enough context window is left", "would pass the episode's limit of"
def walk(forks, d):
    for fk in forks or []:
        for b in fk.get("branches") or []:
            yield d, b
            yield from walk(b.get("forks"), d + 1)
def txt(msgs, s): return sum(1 for m in msgs if m.get("role") == "tool" and s in str(m.get("content")))
B = collections.defaultdict(collections.Counter); depth = collections.defaultdict(collections.Counter)
for dname in os.listdir(J):
    p = os.path.join(J, dname)
    if not os.path.isdir(p): continue
    for t in os.listdir(p):
        q = os.path.join(p, t); f = os.path.join(q, "agent/tree.json"); a = os.path.join(q, "verifier/answers.json")
        if not (os.path.exists(f) and os.path.exists(a)): continue
        try: tr = json.load(open(f)); y = bool(json.load(open(a))["main"].get("correct"))
        except Exception: continue
        i = bisect.bisect_left(T, os.path.getmtime(a)); s = ts[i][1] if i < len(ts) else ts[-1][1] + 1
        k = (s - 1) // K; c = B[k]; c["eps"] += 1; c["y"] += y
        bs = list(walk(tr.get("forks"), 1)); md = max([d for d, _ in bs], default=0)
        c["forked"] += bool(bs); c["branches"] += len(bs); depth[k][min(md, 5)] += 1
        msgs = list(tr.get("main") or []) + [m for _, b in bs for m in b.get("messages") or []]
        c["win_ref"] += txt(msgs, WIN); c["cap_ref"] += txt(msgs, CAP)
        try: c["timeout"] += "TimeoutError" in open(os.path.join(q, "agent/requests.jsonl")).read()
        except Exception: pass
        me = tr.get("main_end"); me = me.get("ended_with") if isinstance(me, dict) else me; c["end:" + str(me)] += 1
        if me == "no tool call": c["y_notool"] += y
print("steps   eps  fork%  br/ep  maxdepth dist(0..5+)                       d>=2  d>=3  win_ref cap_ref timeout%  acc  notool% acc(notool)  ends")
for k in sorted(B):
    c = B[k]; n = c["eps"]; dd = depth[k]
    ends = {e[4:]: v for e, v in c.items() if e.startswith("end:")}
    nt = c["end:no tool call"]
    print(f"{K*k+1:>2}-{K*k+K:<2} {n:>5}  {c['forked']/n:.2f}  {c['branches']/n:.2f}  {dict(sorted(dd.items()))}  "
          f"{sum(v for d, v in dd.items() if d >= 2)/n:.2f}  {sum(v for d, v in dd.items() if d >= 3)/n:.2f}  {c['win_ref']:>5} {c['cap_ref']:>5}  {c['timeout']/n:.3f}  {c['y']/n:.2f}  {nt/n:.2f}  {c['y_notool']/nt if nt else float('nan'):.2f}  {ends}")
