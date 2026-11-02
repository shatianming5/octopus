# python3 analyze2.py RUN.jsonl BUCKETS  (BUCKETS like "1-3,4-6,7-9,10-11")
# Episode -> training step: first main.log "Train step" time at/after its group's last answers.json mtime (proxy).
# A_i: LOO over the group's graded episodes, R = 2y-1. Trained tokens T_i = main + branch completion tokens.
# Signed mass of a class under a weighting w: sum_{i in class, mixed groups} A_i w_i / sum_{i mixed} w_i
#   tok = T_i (what AReaL does: global token mean), ep = 1 (per-episode mean), brN = T_main + T_br / N_br.
import json, sys, bisect, collections, statistics as st

rows = [json.loads(x) for x in open(sys.argv[1])]
steps = rows.pop()["steps"]
groups = collections.defaultdict(list)
for r in rows:
    groups[r["g"].split(":")[0]].append(r)
# Trained groups: 8 graded episodes with mixed y (n_groups 128 = 16 x 8 every step; uniform groups are filtered).
# Step k takes the 16 such groups next in completion order (FIFO; v7 steps 1-5 match main.log n_valid_tokens to 3 tokens).
mixed = sorted((max(r["mt"] for r in rs), g) for g, rs in groups.items() if len(rs) == 8 and 0 < sum(r["y"] for r in rs) < 8)
eps = []
for k in range(min(len(steps), len(mixed) // 16)):
    for _, g in mixed[16 * k:16 * k + 16]:
        rs = groups[g]
        R = [2.0 * r["y"] - 1 for r in rs]
        S = sum(R)
        for r, x in zip(rs, R):
            r.update(A=x - (S - x) / (len(rs) - 1), step=steps[k][1], tok=r["T"]["main"] + r["T"]["br"], mixed=True)
            eps.append(r)
print("trained episodes placed", len(eps), "steps", len({e["step"] for e in eps}))

NB = [("0", lambda n: n == 0), ("1", lambda n: n == 1), ("2-3", lambda n: 2 <= n <= 3), ("4-9", lambda n: 4 <= n <= 9),
      ("10+", lambda n: n >= 10)]
DB = [("d0", lambda d: d == 0), ("d1", lambda d: d == 1), ("d2", lambda d: d == 2), ("d>=3", lambda d: d >= 3)]
W = {"tok": lambda e: e["tok"], "ep": lambda e: 1.0,
     "brN": lambda e: e["T"]["main"] + (e["T"]["br"] / e["nb"] if e["nb"] else 0)}
buckets = [tuple(map(int, b.split("-"))) for b in sys.argv[2].split(",")]


def sel(a, b):
    return [e for e in eps if a <= e["step"] <= b]


print("== (b) trained tokens per episode by branch count (trained episodes) ==")
print("steps   | " + " | ".join(f"nb={k}: n mean med" for k in ("0", "1", "10", ">10")) + " | mean(10)/mean(1)  med(10)/med(1)")
for a, b in buckets + [(buckets[0][0], buckets[-1][1])]:
    s = sel(a, b)
    cells, stat = [], {}
    for k, f in (("0", lambda n: n == 0), ("1", lambda n: n == 1), ("10", lambda n: n == 10), (">10", lambda n: n > 10)):
        xs = [e["tok"] for e in s if f(e["nb"])]
        stat[k] = (st.mean(xs), st.median(xs)) if xs else (float("nan"),) * 2
        cells.append(f"{len(xs):>4} {stat[k][0]:>6.0f} {stat[k][1]:>6.0f}")
    print(f"{a:>2}-{b:<2}   | " + " | ".join(cells) + f" | {stat['10'][0]/stat['1'][0]:>5.1f}x  {stat['10'][1]/stat['1'][1]:>5.1f}x")
mx = collections.Counter(min(e["nb"], 99) for e in eps)
print("nb max", max(e["nb"] for e in eps), "| nb>10 episodes", sum(v for k, v in mx.items() if k > 10))

for title, bins, key in (("branch count", NB, "nb"), ("max depth", DB, "md")):
    print(f"\n== (c) signed mass by {title}, mixed groups: tok (trainer) / ep (per-episode) / brN (branch tokens / N_br) ==")
    print("steps  " + "".join(f"| {n:>5}: share acc  meanA   tok    ep    brN " for n, _ in bins))
    for a, b in buckets + [(buckets[0][0], buckets[-1][1])]:
        m = [e for e in sel(a, b) if e["mixed"]]
        tot = {k: sum(f(e) for e in m) for k, f in W.items()}
        line = f"{a:>2}-{b:<2}  "
        for n, f in bins:
            c = [e for e in m if f(e[key])]
            if not c:
                line += f"| {n:>5}: {'-':>44}"
                continue
            ms = {k: sum(e["A"] * w(e) for e in c) / tot[k] for k, w in W.items()}
            line += (f"| {n:>5}: {len(c)/len(m):.2f} {st.mean(e['y'] for e in c):.2f} {st.mean(e['A'] for e in c):+.3f} "
                     f"{ms['tok']:+.3f} {ms['ep']:+.3f} {ms['brN']:+.3f} ")
        line += f"| sum tok {sum(e['A']*e['tok'] for e in m)/tot['tok']:+.3f}"
        print(line)

print("\n== per step: eps, fork share, br/ep, acc | main fork push sum A*(F - p*D) per scheme (tok==ep per decision) ==")
for s in sorted({e["step"] for e in eps}):
    es = sel(s, s)
    m = [e for e in es if e["mixed"]]
    D = sum(e["turns"][0] for e in m) or 1
    p = sum(e["forkcalls"][0] for e in m) / D
    net = sum(e["A"] * (e["forkcalls"][0] - p * e["turns"][0]) for e in m) / len(m)
    Db = sum(e["turns"][1] for e in m) or 1
    pb = sum(e["forkcalls"][1] for e in m) / Db
    netb = sum(e["A"] * (e["forkcalls"][1] - pb * e["turns"][1]) for e in m) / len(m)
    fk = [e for e in m if e["nb"]]
    nf = [e for e in m if not e["nb"]]
    print(f"step {s:>2}: eps {len(es):>4} fork {st.mean(bool(e['nb']) for e in es):.2f} br/ep {st.mean(e['nb'] for e in es):.2f} "
          f"d>=3 {st.mean(e['md'] >= 3 for e in es):.2f} acc {st.mean(e['y'] for e in es):.2f} | mixed {len(m):>3} "
          f"acc fork/nofork {st.mean(e['y'] for e in fk) if fk else float('nan'):.2f}/{st.mean(e['y'] for e in nf) if nf else float('nan'):.2f} "
          f"meanA fork {st.mean(e['A'] for e in fk) if fk else float('nan'):+.3f} | main push/ep {net:+.3f} (p {p:.3f}) branch push/ep {netb:+.3f}")
