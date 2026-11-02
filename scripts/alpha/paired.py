"""paired.py A.jsonl B.jsonl: dev_eval rows of two adapters on the same bundles: y, passes, forked, branches, errors
each, and B - A paired by bundle (mean over its samples) with a bootstrap 95% CI (seed 0, 10,000 resamples)."""
import collections, json, random, statistics as st, sys
def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l); d[(r["task_dir"], r["sample"])] = r  # a rerun's last row counts
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
for f, d in ((sys.argv[1], a), (sys.argv[2], b)):
    v = list(d.values()); ok = [r for r in v if "error" not in r]
    print(f.rsplit("/", 1)[-1], "n", len(v), "err", len(v) - len(ok), "y %.3f" % st.mean(r["y"] for r in ok),
          "passes %.3f" % st.mean(r["y"] == 1 for r in ok), "forked %.0f%%" % (100 * st.mean(bool(r.get("forked")) for r in ok)),
          "branches %.2f" % st.mean(r.get("branches") or 0 for r in ok))
by = collections.defaultdict(lambda: [[], []])
for i, d in enumerate((a, b)):
    for k, r in d.items():
        if "error" not in r: by[k[0]][i].append(r["y"])
ts = sorted(t for t in by if by[t][0] and by[t][1]); dd = [st.mean(by[t][1]) - st.mean(by[t][0]) for t in ts]
rng = random.Random(0); bs = sorted(st.mean(rng.choices(dd, k=len(dd))) for _ in range(10000))
print("bundles", len(ts), "paired diff %.3f" % st.mean(dd), "95%% CI [%.3f, %.3f]" % (bs[250], bs[9749]),
      "up/down/same", sum(x > 0 for x in dd), sum(x < 0 for x in dd), sum(x == 0 for x in dd))
