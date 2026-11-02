"""search_ablate_table.py BASE.jsonl ARM.jsonl ...: one row per search_dev_eval output (an ablation arm, docs/ablations.md;
the first file is the baseline): episodes scored, infra failures (rows without "y", as search_dev_eval.summary: not
scored; over 2% of the episodes: NOT COMPARABLE, also said of the baseline), INCOMPLETE when it has fewer episodes than
the largest arm, accuracy, forked share, mean branches, forks_depth: the fork depth reached per episode (from the trial's
agent/tree.json where it is still on disk: Val's are, BCP's are removed; "-" else), mean calls, C and C_wall, and the
paired difference in accuracy vs the baseline over the qids both scored (per qid the mean over its samples; bootstrap 95%
CI, seed 0, 10,000 resamples, as paired.py). An episode's last row counts (a resumed run's retry supersedes)."""
import collections, json, os, random, statistics as st, sys


def load(f):
    return {(r["qid"], r["sample"]): r for r in map(json.loads, open(f))}


def depth(forks):  # tree.json: a branch's own forks sit in its "forks" (core.tree_branches)
    return max((1 + depth(b.get("forks") or []) for f in forks for b in f.get("branches") or []), default=0)


def per_qid(rows):
    by = collections.defaultdict(list)
    for (q, _), r in rows.items():
        if "y" in r:
            by[q].append(float(r["y"]))
    return {q: st.mean(v) for q, v in by.items()}


files = list(dict.fromkeys(sys.argv[1:]))  # a glob that repeats the baseline
if not files:
    sys.exit(__doc__)
arms = {f: load(f) for f in files}
base = per_qid(arms[files[0]])
most = max(map(len, arms.values()))
print("arm | n | infra | acc | forked | branches | forks_depth | calls | C | C_wall | diff vs base [95% CI] (qids)")
for f, rows in arms.items():
    ok = [r for r in rows.values() if "y" in r]
    infra = len(rows) - len(ok)
    m = lambda k: st.mean(float(r.get(k) or 0) for r in ok) if ok else float("nan")  # noqa: E731
    trees = [os.path.join(r["trial"], "agent/tree.json") for r in ok if r.get("trial")]
    ds = collections.Counter(depth(json.load(open(t)).get("forks") or []) for t in trees if os.path.exists(t))
    diff = "baseline"
    if f != files[0]:
        mine = per_qid(rows); qs = sorted(set(mine) & set(base)); d = [mine[q] - base[q] for q in qs]
        rng = random.Random(0); bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000)) if d else []
        diff = f"{st.mean(d):+.3f} [{bs[250]:+.3f}, {bs[9749]:+.3f}] ({len(qs)})" if d else "no common qids"
    flag = (" NOT COMPARABLE (infra > 2%" + (": the baseline)" if f == files[0] else ")")) * (infra > 0.02 * len(rows))
    flag += f" INCOMPLETE ({len(rows)} of {most} episodes)" * (len(rows) < most)
    print(" | ".join([os.path.basename(f).removesuffix(".jsonl"), str(len(ok)), f"{infra} ({infra / max(len(rows), 1):.1%})",
                      f"{m('y'):.3f}", f"{m('forked'):.0%}", f"{m('branches'):.2f}",
                      " ".join(f"d{k}:{v}" for k, v in sorted(ds.items())) or "-",
                      f"{m('calls'):.1f}", f"{m('C'):.0f}", f"{m('C_wall'):.0f}", diff]) + flag)
