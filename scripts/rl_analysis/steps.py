"""Bundle RL run metrics per training step (y, fork rate, M2 main-first, main ends, branch closes, check validity): episodes binned by finish time (result.json mtime) into the windows between
successive "Train step N done" lines (window N = finished after step N-1, before step N). Per-episode outcomes are
cached in CACHE (jsonl) so reruns only score new episodes.
Run from a code checkout (imports octopus.envs.swe.reward).
usage: python steps.py JOBS_DIR MAIN_LOG CACHE.jsonl"""
import collections, json, re, statistics as st, sys
from datetime import datetime
from pathlib import Path
from octopus.envs.swe.reward import plan_outcome
from octopus.runtime.core import tree_branches

jobs, log, cache_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
done = [datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp()
        for m in re.finditer(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \S+ Step \d+/\d+ Train step", open(log).read())]
cache = {}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        r = json.loads(line); cache[r["trial"]] = r
new = []
for job in jobs.iterdir():
    t = [x for x in job.iterdir() if x.is_dir()] if job.is_dir() else []
    if len(t) != 1 or not (t[0] / "result.json").exists() or str(t[0]) in cache:
        continue
    trial = t[0]
    r = {"trial": str(trial), "end": (trial / "result.json").stat().st_mtime}
    try:
        res = json.loads((trial / "result.json").read_text())
    except json.JSONDecodeError:  # truncated: still being written, or the run was killed mid-write
        continue
    if res.get("exception_info"):
        r["error"] = res["exception_info"].get("exception_type")
    else:
        try:
            reqs = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
            o = plan_outcome(trial, reqs)
            tree = json.loads((trial / "agent/tree.json").read_text())
            br = o["branches"]
            ctrl = o["control"] or {}
            rcs = tree.get("red_checks") or []
            merges = [str(b.get("merge") or "-").split(":")[0] for _, _, b in tree_branches(tree["forks"])]
            r.update(y=o["y"], forked=bool(o["forked"]), end_main=tree.get("main_end"),
                     m2=o["y"] - ctrl["y"] if ctrl.get("kind") == "main-first" else None,
                     nbr=len(br), commit=sum(b["ended_with"] == "commit" for b in br),
                     cpass=sum(b["ended_with"] == "commit" and bool(b["passed"]) for b in br),
                     unclosed=sum(b["ended_with"] not in ("commit", "abort") for b in br),
                     checks=sum(len(rc["exits"]) for rc in rcs), weak=sum(len(rc.get("weak") or []) for rc in rcs),
                     merged=merges.count("merged"), conflict=merges.count("conflict"),
                     cv=[v for ls in (o.get("checks") or {}).values() for v in ls if v is not None],
                     ov=[float(v) for v in o.get("own_valid") or []],
                     gated=json.dumps(tree["forks"]).count("commit not accepted yet"))
        except Exception as exc:
            r["error"] = f"outcome {type(exc).__name__}"
    new.append(r); cache[r["trial"]] = r
with cache_path.open("a") as f:
    for r in new:
        f.write(json.dumps(r) + "\n")


def window(t):
    for i, d in enumerate(done):
        if t <= d:
            return i + 1
    return len(done) + 1  # still being collected


def pct(c, n):
    return f"{100 * c / n:3.0f}%" if n else "   -"


def row(name, rs):
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return f"{name:>7} {len(rs):4d}  (no scored episodes)"
    fk = [r["y"] for r in ok if r["forked"]]; uf = [r["y"] for r in ok if not r["forked"]]
    m2 = [r["m2"] for r in ok if r["m2"] is not None]
    ends = collections.Counter(r["end_main"] for r in ok)
    s = lambda k: sum(r[k] for r in ok)
    m2s = f"{st.mean(m2):+.3f}±{st.stdev(m2) / len(m2) ** .5:.3f} n{len(m2):3d}" if len(m2) > 1 else f"{'-':>19}"
    return (f"{name:>7} {len(rs):4d} {len(rs) - len(ok):3d} {st.mean(r['y'] for r in ok):6.3f} {pct(len(fk), len(ok))} "
            f"{(st.mean(fk) if fk else float('nan')):6.3f} {(st.mean(uf) if uf else float('nan')):6.3f}  {m2s}  "
            f"{pct(ends.get('done', 0), len(ok))} {pct(ends.get('step cap', 0), len(ok))} {pct(ends.get('budget', 0), len(ok))}  "
            f"{s('nbr'):4d} {pct(s('commit'), s('nbr'))} {pct(s('cpass'), s('commit'))} {pct(s('unclosed'), s('nbr'))}  "
            f"{pct(s('weak'), s('checks'))} {pct(s('conflict'), s('conflict') + s('merged'))}  "
            f"{pct(sum(sum(r.get('cv', [])) for r in ok), sum(len(r.get('cv', [])) for r in ok))} "
            f"{pct(sum(sum(r.get('ov', [])) for r in ok), sum(len(r.get('ov', [])) for r in ok))} {sum(r.get('gated', 0) for r in ok):4d}")


print(f"{'window':>7} {'eps':>4} {'err':>3} {'y':>6} {'fork':>4} {'y_fk':>6} {'y_uf':>6}  {'M2 main-first':>19}  "
      f"{'done':>4} {'cap':>4} {'bud':>4}  {'brs':>4} {'cmt':>4} {'pass':>4} {'uncl':>4}  {'weak':>4} {'cnfl':>4}  {'cval':>4} {'oval':>4} {'gate':>4}")
by = collections.defaultdict(list)
for r in cache.values():
    by[window(r["end"])].append(r)
for w in sorted(by):
    print(row(f"s{w}" + ("*" if w > len(done) else ""), by[w]))
print(row("all", list(cache.values())))
ends = collections.Counter(r.get("end_main") for r in cache.values() if "error" not in r)
print("main ends:", dict(ends.most_common()), "| errors:", dict(collections.Counter(r["error"] for r in cache.values() if "error" in r)))
