"""Passing commit branches: tool calls after their last edit/write (commit included), quantiles and CDF.
usage: python commit_tail.py JOBS_DIR"""
import glob, json, statistics, sys
J = sys.argv[1]
def tb(forks, parent="main"):
    for k, f in enumerate(forks):
        for b in f.get("branches") or []:
            name = f"fork{k}/{b['branch_id']}" if parent == "main" else f"{parent}/fork{k}/{b['branch_id']}"
            yield name, b
            yield from tb(b.get("forks") or [], name)
w = []
for tp in glob.glob(J + "/*/*/agent/tree.json"):
    d = tp.rsplit("/", 2)[0]
    try:
        tree = json.load(open(tp)); passed = {b["key"]: b.get("passed") for b in json.load(open(d + "/verifier/branches.json"))["branches"]}
    except Exception:
        continue
    for name, b in tb(tree.get("forks") or []):
        if b.get("ended_with") != "commit" or not passed.get(name.replace("/", "-")):
            continue
        calls = [c["function"]["name"] for m in b["messages"] if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
        last = max([i for i, c in enumerate(calls) if c in ("edit", "write")] or [-1])
        w.append(len(calls) - 1 - last)  # calls after its last edit, the commit included
w.sort()
print("passing commits", len(w), "calls after last edit: median", statistics.median(w), "p75", w[3*len(w)//4], "p90", w[9*len(w)//10], "p95", w[95*len(w)//100])
print({k: round(sum(x <= k for x in w) / len(w), 3) for k in (1, 2, 3, 4, 5, 6, 8, 10)})
