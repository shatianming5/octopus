# Read-only (python3 - JOBS_DIR): does rewind work in v6d? Mechanics (accepted / refused calls, how threads that rewound end)
# and effect: among threads that reached the answer reserve, rewound vs closed, paired within the same question group
# (mean over groups having both kinds; 95% CI by a group bootstrap). Branch outcome: ended with done; main outcome: episode y.
import json, os, sys, random, collections
J = sys.argv[1]
MARK = ("Refused: your context is nearly used up", "[clipped: context nearly used up]", "Your context is nearly used up. Close now")
def own_msgs(msgs, rewinds): return list(msgs) + [m for r in rewinds or [] for m in r.get("messages") or []]
def reached(msgs): return any(m.get("role") in ("tool", "user") and any(k in str(m.get("content")) for k in MARK) for m in msgs)
def calls(msgs, name): return sum(1 for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or [] if (c.get("function") or {}).get("name") == name)
def results(msgs, text): return sum(1 for m in msgs if m.get("role") == "tool" and text in str(m.get("content")))
G = collections.defaultdict(lambda: {"b": [], "m": []}); C = collections.Counter(); ends = collections.Counter()
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not os.path.isdir(p): continue
    g = d.split(":")[0]
    for t in os.listdir(p):
        q = os.path.join(p, t); f = os.path.join(q, "agent/tree.json"); a = os.path.join(q, "verifier/answers.json")
        if not (os.path.exists(f) and os.path.exists(a)): continue
        tr = json.load(open(f)); y = bool(json.load(open(a))["main"].get("correct"))
        mm = own_msgs(tr.get("main") or [], tr.get("rewinds")); mrw = len(tr.get("rewinds") or [])
        C["main rewind calls"] += calls(mm, "rewind"); C["main rewinds accepted"] += mrw
        C["main refused: cap"] += results(mm, "No rewinds left"); C["main refused: bad to_step"] += results(mm, "rewind needs to_step")
        if reached(mm): G[g]["m"].append((bool(mrw), y))
        me = tr.get("main_end"); me = me.get("ended_with") if isinstance(me, dict) else me
        if mrw: ends["main rewound -> " + str(me)] += 1
        for fk in tr.get("forks") or []:
            pl = fk.get("prefix_len") or 0
            for b in fk.get("branches") or []:
                bm = own_msgs((b.get("messages") or [])[pl:], b.get("rewinds")); brw = len(b.get("rewinds") or [])
                C["branches"] += 1; C["branch rewind calls"] += calls(bm, "rewind"); C["branch rewinds accepted"] += brw
                C["branch refused: cap"] += results(bm, "No rewinds left"); C["branch refused: bad to_step"] += results(bm, "rewind needs to_step")
                done = b.get("ended_with") == "done"
                if brw: ends["branch rewound -> " + str(b.get("ended_with"))] += 1
                else: ends["branch no rewind -> " + str(b.get("ended_with"))] += 1
                if reached(bm): G[g]["b"].append((bool(brw), done, y))
for k in sorted(C): print(f"{k}: {C[k]}")
for k, v in sorted(ends.items(), key=lambda x: -x[1])[:12]: print(f"  {k}: {v}")
def paired(key, idx):
    diffs = []
    for g in G.values():
        rw = [x[idx] for x in g[key] if x[0]]; cl = [x[idx] for x in g[key] if not x[0]]
        if rw and cl: diffs.append(sum(rw) / len(rw) - sum(cl) / len(cl))
    if not diffs: return "n/a"
    mean = sum(diffs) / len(diffs); random.seed(0)
    bs = sorted(sum(random.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return f"{mean:+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}] over {len(diffs)} groups"
allb = [x for g in G.values() for x in g["b"]]; allm = [x for g in G.values() for x in g["m"]]
r = lambda xs, i: sum(x[i] for x in xs) / len(xs) if xs else float("nan")
print(f"reserve-reaching branches: rewound n={sum(x[0] for x in allb)} done {r([x for x in allb if x[0]],1):.2f} | closed n={sum(not x[0] for x in allb)} done {r([x for x in allb if not x[0]],1):.2f}")
print("  paired (same question) done-rate diff, rewound - closed:", paired("b", 1))
print("  paired episode accuracy diff (branch rewound - closed):", paired("b", 2))
print(f"reserve-reaching mains: rewound n={sum(x[0] for x in allm)} acc {r([x for x in allm if x[0]],1):.2f} | closed n={sum(not x[0] for x in allm)} acc {r([x for x in allm if not x[0]],1):.2f}")
print("  paired episode accuracy diff, main rewound - closed:", paired("m", 1))
