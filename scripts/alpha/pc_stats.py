"""pc_stats.py A.jsonl B.jsonl: dev_eval rows run with rl.plan_control=true for two adapters on the same bundles. Per
adapter: episodes, fork points (rows with a no-fork control from the same fork call), fork_minus_control (y - control y)
by control kind, its mean paired by bundle with a bootstrap 95% CI (seed 0, 10,000 resamples), and how the branches
(at any depth) of those controlled episodes ended (tree.json ended_with); then B - A of the per-bundle fork_minus_control over bundles both have."""
import collections, json, random, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the checkout's octopus/, run as plain python3
from octopus.runtime.core import tree_branches


def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l); d[(r["task_dir"], r["sample"])] = r  # a rerun's last row counts
    return [r for r in d.values() if "error" not in r]


def boot(xs):
    rng = random.Random(0); bs = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(10000))
    return f"{st.mean(xs):+.3f} [{bs[250]:+.3f}, {bs[9749]:+.3f}]"


per = {}
for f in sys.argv[1:3]:
    rows = load(f); forked = [r for r in rows if r.get("control")]
    kinds = collections.defaultdict(list); ends = collections.Counter(); by = collections.defaultdict(list)
    for r in forked:
        d = r["y"] - r["control"]["y"]; kinds[r["control"]["kind"]].append(d); by[r["task_dir"]].append(d)
        try:
            forks = json.load(open(Path(r["trial"]) / "agent/tree.json")).get("forks") or []
            ends.update(b.get("ended_with") or "none" for _, _, b in tree_branches(forks))
        except OSError:
            ends["no tree"] += 1
    per[f] = {t: st.mean(v) for t, v in by.items()}
    print(Path(f).name, "episodes", len(rows), "fork points", len(forked), "| fork - control by kind:",
          {k: f"{st.mean(v):+.3f} (n={len(v)})" for k, v in sorted(kinds.items())})
    if per[f]:
        print("  per bundle (", len(per[f]), "bundles):", boot(list(per[f].values())), "| branch ends (every branch of controlled episodes):", dict(ends))
a, b = (per[f] for f in sys.argv[1:3])
both = sorted(set(a) & set(b))
if both:
    print(f"B - A per bundle over {len(both)} bundles:", boot([b[t] - a[t] for t in both]))
