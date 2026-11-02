# python3 branch_fork_value.py RUN.jsonl (extract.py output, ../../alpha-search-v7-1025/scripts/extract.py): at fork depth 1 a
# branch's fork call is refused. Episodes whose branches called fork vs episodes with branches that did not, all groups and
# mixed groups (0 < correct < 8): n, accuracy, mean LOO advantage (R = 2y-1 over the group's graded episodes).
import json, sys, collections, statistics as st
rows = [json.loads(x) for x in open(sys.argv[1])][:-1]
G = collections.defaultdict(list)
for r in rows:
    G[r["g"].split(":")[0]].append(r)
for rs in G.values():
    R = [2.0 * r["y"] - 1 for r in rs]
    for r, x in zip(rs, R):
        r["A"] = x - (sum(R) - x) / (len(rs) - 1) if len(rs) > 1 else 0.0
        r["mixed"] = 0 < sum(z["y"] for z in rs) < len(rs)
br = [r for r in rows if r["nb"]]
print("episodes", len(rows), "with branches", len(br), "| branch fork calls per branch %.2f" % (sum(r["forkcalls"][1] for r in br) / sum(r["nb"] for r in br)))
for name, sel in (("all", br), ("mixed groups", [r for r in br if r["mixed"]])):
    for lab, f in (("branches called fork", lambda r: r["forkcalls"][1] > 0), ("branches did not", lambda r: r["forkcalls"][1] == 0)):
        s = [r for r in sel if f(r)]
        print(f"{name:<13} {lab:<21} n {len(s):>6}  acc {st.mean(r['y'] for r in s):.3f}  meanA {st.mean(r['A'] for r in s):+.3f}  branches/ep {st.mean(r['nb'] for r in s):.2f}")
