# Read-only (TZ=America/New_York python3 - JOBS_DIR): why main's rewind rate fell. Per hour of finish (answers.json mtime):
# fork share, share of mains that reached the answer reserve (reserve refusal / clip / close note seen, rewound history
# included), P(main calls rewind | reserve reached) and | not reached, accepted main rewinds/ep, accuracy; reserve share
# by forked vs not; and the LOO advantage of samples whose main called rewind (what RL pushed on), per hour.
import json, os, sys, time, collections
J = sys.argv[1]
MARK = ("Refused: your context is nearly used up", "[clipped: context nearly used up]", "Your context is nearly used up. Close now")
def scan(msgs):
    res = any(m.get("role") in ("tool", "user") and any(k in str(m.get("content")) for k in MARK) for m in msgs)
    rw = sum(1 for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or []
             if (c.get("function") or {}).get("name") == "rewind")
    return res, rw
G = collections.defaultdict(list)
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not os.path.isdir(p): continue
    for t in os.listdir(p):
        q = os.path.join(p, t); f = os.path.join(q, "agent/tree.json"); a = os.path.join(q, "verifier/answers.json")
        if not (os.path.exists(f) and os.path.exists(a)): continue
        tr = json.load(open(f))
        msgs = list(tr.get("main") or [])
        for r in tr.get("rewinds") or []: msgs += r.get("messages") or []
        res, rwc = scan(msgs)
        G[d.split(":")[0]].append(dict(h=time.strftime("%H", time.localtime(os.path.getmtime(a))),
            y=bool(json.load(open(a))["main"].get("correct")), forked=bool(tr.get("forks")), res=res, rwc=rwc,
            rwa=len(tr.get("rewinds") or [])))
H = collections.defaultdict(lambda: collections.defaultdict(list))
for g in G.values():
    for i, e in enumerate(g):
        others = [x["y"] for j, x in enumerate(g) if j != i]
        e["A"] = (2 * e["y"] - 1) - (sum(2 * y - 1 for y in others) / len(others)) if others else 0.0
        e["mixed"] = 0 < sum(x["y"] for x in g) < len(g)
        H[e["h"]]["all"].append(e)
m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print("hour n fork% reserve% P(rw|res) P(rw|nores) rwAcc/ep acc | reserve% forked/unforked | meanA(main rw) n, mixed groups only")
for h in sorted(H):
    E = H[h]["all"]; R = [e for e in E if e["res"]]; N = [e for e in E if not e["res"]]
    F = [e for e in E if e["forked"]]; U = [e for e in E if not e["forked"]]
    W = [e for e in E if e["rwc"] and e["mixed"]]
    print(h, len(E), "%.2f" % m([e["forked"] for e in E]), "%.2f" % m([e["res"] for e in E]),
          "%.2f" % m([bool(e["rwc"]) for e in R]), "%.2f" % m([bool(e["rwc"]) for e in N]),
          "%.2f" % m([e["rwa"] for e in E]), "%.2f" % m([e["y"] for e in E]), "|",
          "%.2f/%.2f" % (m([e["res"] for e in F]), m([e["res"] for e in U])), "|", "%+.2f n%d" % (m([e["A"] for e in W]), len(W)))
print("hour | among mains that reached the reserve: acc rewound / closed, mean A (mixed groups) rewound / closed, n")
for h in sorted(H):
    R = [e for e in H[h]["all"] if e["res"]]; RW = [e for e in R if e["rwc"]]; RC = [e for e in R if not e["rwc"]]
    print(h, "| acc %.2f / %.2f" % (m([e["y"] for e in RW]), m([e["y"] for e in RC])),
          "| A %+.2f / %+.2f" % (m([e["A"] for e in RW if e["mixed"]]), m([e["A"] for e in RC if e["mixed"]])),
          "| n %d / %d" % (len(RW), len(RC)))
