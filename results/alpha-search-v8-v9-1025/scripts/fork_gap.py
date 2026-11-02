# Read-only (python3 fork_gap.py DIR TRIAL...): the 10-25 fork diagnosis near v0, from the 10-25 independent review's
# analysis. DIR holds TRIAL.jsonl (fork_extract.py) and TRIAL.steps ("STEP YYYYmmdd-HH:MM:SS.fff" step ends, EDT).
# Episodes are bucketed by tree.json mtime; steps 1-3 are used. y = graded answer; y8 = done-only grading (#306: a main
# that did not end with done scores 0). Per trial: forked share, text-ended ("no tool call") share among forked and
# unforked mains, how many text endings came right after the join note, their graded accuracy, and the within-group
# (same question group) forked-minus-unforked gap with its SE, under y, under y8, and under y8 with text-ended mains
# left out on both sides; the forked mains' mean leave-one-out advantage and the text-ended part of it.
# Pooled: inverse-variance mean over the trials.
import collections, datetime, json, statistics as S, sys, zoneinfo
D, TRIALS = sys.argv[1], sys.argv[2:]
NY = zoneinfo.ZoneInfo("America/New_York")


def load(t):
    T = [datetime.datetime.strptime(l.split()[1], "%Y%m%d-%H:%M:%S.%f").replace(tzinfo=NY).timestamp() for l in open(f"{D}/{t}.steps")]
    R = [json.loads(l) for l in open(f"{D}/{t}.jsonl")]
    R = [r for r in R if "terr" not in r and not r.get("noans")]
    for r in R:
        r["step"] = next((i + 1 for i, x in enumerate(T) if r["mt"] <= x), 999)
        r["forked"] = r["nforks"] > 0; r["g"] = r["gdir"].split(":")[0]; r["y"] = float(bool(r["y"]))
        r["y8"] = r["y"] if r["main_end"] == "done" else 0.0; r["txt"] = r["main_end"] == "no tool call"
    return [r for r in R if r["step"] <= 3]


def gap(rows, yk):
    g = collections.defaultdict(list)
    for r in rows: g[r["g"]].append(r)
    d = [S.mean(r[yk] for r in rs if r["forked"]) - S.mean(r[yk] for r in rs if not r["forked"])
         for rs in g.values() if any(r["forked"] for r in rs) and not all(r["forked"] for r in rs)]
    return S.mean(d), S.stdev(d) / len(d) ** .5, len(d)


def pool(est):
    w = [1 / se ** 2 for _, se, _ in est]
    return sum(wi * m for wi, (m, _, _) in zip(w, est)) / sum(w), 1 / sum(w) ** .5


P = collections.defaultdict(list)
for t in TRIALS:
    sub = load(t); fk = [r for r in sub if r["forked"]]; uf = [r for r in sub if not r["forked"]]
    tf = [r for r in fk if r["txt"]]; tu = [r for r in uf if r["txt"]]
    print(f"== {t} steps 1-3: n {len(sub)} forked {len(fk) / len(sub):.3f}")
    print(f"   text-ended: forked {len(tf)}/{len(fk)} = {len(tf) / len(fk):.3f} (right after the join note {sum(r['last_is_join'] for r in tf)}),"
          f" graded acc {S.mean(r['y'] for r in tf):.3f}; unforked {len(tu)}/{len(uf)} = {len(tu) / len(uf):.3f}, graded acc {S.mean(r['y'] for r in tu):.3f}")
    print(f"   done-ended acc: forked {S.mean(r['y'] for r in fk if r['main_end'] == 'done'):.3f} unforked {S.mean(r['y'] for r in uf if r['main_end'] == 'done'):.3f}")
    for lab, rows, yk in (("as graded (y)", sub, "y"), ("done-only (y8)", sub, "y8"), ("done-only, text-ended left out", [r for r in sub if not r["txt"]], "y8")):
        m, se, n = gap(rows, yk); P[lab].append((m, se, n))
        print(f"   within-group gap {lab}: {m:+.3f} ± {se:.3f} SE ({n} groups)")
    for s in (1, 2, 3):
        print(f"   step {s} gap as graded / done-only: " + " / ".join("{:+.3f} ± {:.3f}".format(*gap([r for r in sub if r["step"] == s], yk)[:2]) for yk in ("y", "y8")))
    A = []  # leave-one-out advantage of R = 2*y8-1 within each group, for forked mains
    for g in {r["g"] for r in sub}:
        rs = [r for r in sub if r["g"] == g]
        A += [(sum(2 * r["y8"] - 2 * u["y8"] for u in rs) / (len(rs) - 1), r["txt"]) for r in rs if r["forked"] and len(rs) > 1]
    print(f"   forked mains' mean leave-one-out A (done-only): {sum(a for a, _ in A) / len(A):+.3f}, of which text-ended "
          f"{sum(a for a, x in A if x) / len(A):+.3f}, rest {sum(a for a, x in A if not x) / len(A):+.3f}")
print("== pooled (inverse-variance) over", " ".join(TRIALS))
for lab, est in P.items():
    m, se = pool(est); print(f"   {lab}: {m:+.3f} ± {se:.3f} SE")
