# python3 analyze3.py RUN.jsonl BUCKETS -- trained episodes (FIFO, as analyze2): branching process per step bucket
# (main fork share, depth-1 branches per forking main, branch-thread fork share, offspring per branch thread) and the
# net REINFORCE push on "call fork" per decision, sum_i w_i A_i (F_i - p D_i) / sum_i w_i T_i, per weighting.
import json, sys, collections

rows = [json.loads(x) for x in open(sys.argv[1])]
steps = rows.pop()["steps"]
groups = collections.defaultdict(list)
for r in rows:
    groups[r["g"].split(":")[0]].append(r)
mixed = sorted((max(r["mt"] for r in rs), g) for g, rs in groups.items() if len(rs) == 8 and 0 < sum(r["y"] for r in rs) < 8)
eps = []
for k in range(min(len(steps), len(mixed) // 16)):
    for _, g in mixed[16 * k:16 * k + 16]:
        rs = groups[g]
        R = [2.0 * r["y"] - 1 for r in rs]
        S = sum(R)
        for r, x in zip(rs, R):
            r.update(A=x - (S - x) / 7, step=steps[k][1], tok=r["T"]["main"] + r["T"]["br"])
            eps.append(r)
buckets = [tuple(map(int, b.split("-"))) for b in sys.argv[2].split(",")]
# per-token weight of an episode's main / branch tokens under each scheme (before the global normaliser)
W = {"tok": lambda e: (1.0, 1.0), "ep": lambda e: (1.0 / max(e["tok"], 1),) * 2,
     "brN": lambda e: (1.0, 1.0 / e["nb"] if e["nb"] else 1.0)}
print("steps  eps  fork  br/ep  nb=10 | br/forking-main  branch-fork-share  offspring/branch | push main (tok ep brN) | push branch (tok ep brN)")
for a, b in buckets:
    es = [e for e in eps if a <= e["step"] <= b]
    n = len(es)
    fm = [e for e in es if e["nb"]]
    n1 = sum(int(e.get("nd", {"1": e["nb"]}).get("1", 0)) for e in es)
    nbr = sum(e["nb"] for e in es)
    bthreads_forking = sum(sum(e.get("bf", [0])) for e in es)
    deeper = nbr - n1
    out = f"{a:>2}-{b:<2} {n:>4}  {len(fm)/n:.2f}  {nbr/n:.2f}  {sum(e['nb'] >= 10 for e in es)/n:.2f} |  {n1/max(len(fm),1):.2f}             {bthreads_forking/max(nbr,1):.2f}               {deeper/max(nbr,1):.2f}      |"
    for k in (0, 1):
        p = sum(e["forkcalls"][k] for e in es) / max(sum(e["turns"][k] for e in es), 1)
        vals = []
        for name, w in W.items():
            Z = sum(w(e)[0] * e["T"]["main"] + w(e)[1] * e["T"]["br"] for e in es)
            vals.append(sum(w(e)[k] * e["A"] * (e["forkcalls"][k] - p * e["turns"][k]) for e in es) / Z * 1e4)
        out += " " + " ".join(f"{v:+6.2f}" for v in vals) + " |"
    print(out)
print("(push x1e4 per unit loss weight; p = pooled fork calls per turn of that thread kind in the bucket)")
