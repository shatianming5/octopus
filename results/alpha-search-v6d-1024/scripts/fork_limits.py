# Read-only (python3 - JOBS_DIR MAIN_LOG): fork limits hit in v6d training, per training-step bucket (answers.json mtime
# between "Train step N done" times). Episode cap (OCTOPUS_MAX_BRANCHES): main's fork refused "would pass the episode's
# limit"; depth (OCTOPUS_FORK_DEPTH=1): branch fork calls refused "Forking is not available at this depth"; thread cap:
# "threads are running". Counts episodes / branches / calls, rewound-away messages included.
import json, os, sys, re, bisect, collections
from datetime import datetime
J, M = sys.argv[1], sys.argv[2]
ts = sorted((datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp(), int(m.group(2))) for line in open(M, errors="replace")
            for m in [re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \d+/\d+ Step (\d+)/\d+ Train step", line)] if m)
T = [t for t, _ in ts]
CAP, DEPTH, THR = "would pass the episode's limit of", "Forking is not available at this depth", "threads are running and"
B = collections.defaultdict(collections.Counter)
def hits(msgs, s): return sum(1 for m in msgs if m.get("role") == "tool" and s in str(m.get("content")))
def forks(msgs): return sum(1 for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "fork")
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not os.path.isdir(p): continue
    for t in os.listdir(p):
        q = os.path.join(p, t); f = os.path.join(q, "agent/tree.json"); a = os.path.join(q, "verifier/answers.json")
        if not (os.path.exists(f) and os.path.exists(a)): continue
        try: tr = json.load(open(f))
        except Exception: continue
        i = bisect.bisect_left(T, os.path.getmtime(a)); s = ts[i][1] if i < len(ts) else ts[-1][1] + 1
        c = B[(s - 1) // 5]; c["eps"] += 1
        mm = list(tr.get("main") or []) + [x for r in tr.get("rewinds") or [] for x in r.get("messages") or []]
        c["main fork calls"] += forks(mm); k = hits(mm, CAP); c["cap refusals"] += k; c["eps capped"] += bool(k)
        k = hits(mm, THR); c["thread-cap refusals"] += k
        for fk in tr.get("forks") or []:
            pl = fk.get("prefix_len") or 0
            for b in fk.get("branches") or []:
                bm = list((b.get("messages") or [])[pl:]) + [x for r in b.get("rewinds") or [] for x in r.get("messages") or []]
                c["branches"] += 1; c["branch fork calls"] += forks(bm); k = hits(bm, DEPTH)
                c["depth refusals"] += k; c["branches depth-refused"] += bool(k)
print("steps   eps  main_fork_calls  eps_capped(%)  cap_refusals  thread_cap  branches  branch_fork_calls  branches_depth_refused(%)")
tot = collections.Counter()
for k in sorted(B):
    c = B[k]; tot.update(c)
    print(f"{5*k+1:>2}-{5*k+5:<2} {c['eps']:>6} {c['main fork calls']:>8} {c['eps capped']:>8} ({100*c['eps capped']/c['eps']:.1f}%) {c['cap refusals']:>8} {c['thread-cap refusals']:>8} {c['branches']:>8} {c['branch fork calls']:>10} {c['branches depth-refused']:>8} ({100*c['branches depth-refused']/max(c['branches'],1):.1f}%)")
c = tot
print(f"total  {c['eps']:>6} {c['main fork calls']:>8} {c['eps capped']:>8} ({100*c['eps capped']/c['eps']:.1f}%) {c['cap refusals']:>8} {c['thread-cap refusals']:>8} {c['branches']:>8} {c['branch fork calls']:>10} {c['branches depth-refused']:>8} ({100*c['branches depth-refused']/max(c['branches'],1):.1f}%)")
