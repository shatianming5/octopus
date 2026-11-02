"""Held-out eval summary of one RL run per checkpoint (EVAL_DIR/results/step<N>.jsonl, rows of octopus.eval.dev_eval
with the control, rl.plan_control; infra-error rows without y are left out; * = step not done yet):
  y, fork rate; M1 = y - base y paired by (task, sample) vs --base; M2 = y - main-first control y over forked episodes;
  pooled late M2 over --late steps; M3' = fork rate and M2 by bundle size (tasks: row n, else solution/fix_*.patch);
  M4 = fork / control critical path (wall C_wall, tokens C) median and share where the fork is faster, and y per hour
  of wall (mean y / mean C_wall); rewinds per episode where rows have them (search_dev_eval); best checkpoint = max M1 among done steps whose M2 95% CI reaches 0 or above
  (upper bound >= 0: forking not significantly worse than its control), else max M1.
step<N>-small.jsonl (pool.py K_SMALL pass: the 1-3-task bundles at a higher k; base: --base-small, default
<base>-small.jsonl next to --base) is left out of the headline (y, fork, M1, M2, late M2: as without it), gets its own
lines first (M1 paired vs the base small pass) and is pooled into M3' 1-3: its episodes are independent samples of the
same checkpoint on the same bundles, extra to the step's own k. A small-only EVAL_DIR (no step<N>.jsonl) prints only
those lines and needs no --base.
usage: summary.py EVAL_DIR --base BASE_STEP0.jsonl [--base-small BASE_SMALL.jsonl] [--late 40,50,53 (default: last 3)]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

BUCKETS = [(1, 3), (4, 7), (8, 11), (12, 16)]
nan = float("nan")


def ci(x): return 1.96 * st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else nan
def mean(x): return st.mean(x) if x else nan
def fmt(x): return f"{mean(x):+.3f}±{ci(x):.3f} n={len(x)}" if x else "-"
def key(r): return r["task_dir"].rsplit("/", 1)[-1], r["sample"]
def load(f): return [r for r in map(json.loads, open(f)) if "y" in r]
def load_small(f): return load(f) if f.exists() else []
def pairs(R): return [r for r in R if r["forked"] and (r.get("control") or {}).get("kind") == "main-first"]
def m2(R): return [r["y"] - r["control"]["y"] for r in pairs(R)]


def size(r) -> int:
    return r.get("n") or len(r.get("tasks") or {}) or len(list(Path(r["task_dir"], "solution").glob("fix_*.patch")))


def ratio(R, k):
    x = [r[k] / r["control"][k] for r in pairs(R) if r.get(k) and r["control"].get(k)]
    return f"{st.median(x):5.2f} {sum(v < 1 for v in x) / len(x):4.0%}" if x else "    -    -"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--base", help="base main-pass rows (needed when EVAL_DIR has step<N>.jsonl)")
    ap.add_argument("--base-small", help="base K_SMALL-pass rows (default: <base>-small.jsonl next to --base)")
    ap.add_argument("--late", help="comma list of steps pooled for late M2 (default: the last 3)")
    a = ap.parse_args()
    res = Path(a.eval_dir, "results")
    def found(pat): return sorted(int(m[1]) for f in res.glob("step*.jsonl") if (m := re.fullmatch(pat, f.name)))
    steps = found(r"step(\d+)\.jsonl")
    bs = a.base_small or (a.base and Path(a.base).with_name(Path(a.base).stem + "-small.jsonl"))
    small = {"base": (load_small(Path(bs)) if bs else [], True)}
    small |= {f"s{n}": (load(res / f"step{n}-small.jsonl"), (res / f"step{n}-small.done").exists())
              for n in found(r"step(\d+)-small\.jsonl")}
    smap = {key(r): r["y"] for r in small["base"][0]}
    sm1 = {name: [r["y"] - smap[key(r)] for r in S if key(r) in smap] for name, (S, _) in list(small.items())[1:]}
    if any(S for S, _ in small.values()):
        print("small bundles (1-3 tasks), K_SMALL pass alone: n, y, fork, M1 paired vs base's K_SMALL pass, M2")
    for name, (S, done) in small.items():
        if S:
            print(f"{name + ('' if done else '*'):7s} {len(S):4d} {mean([r['y'] for r in S]):6.3f} "
                  f"{sum(r['forked'] for r in S) / len(S):5.0%}  {fmt(sm1.get(name)):24s} {fmt(m2(S)):24s}")
    if not steps:
        return
    if any(S for S, _ in small.values()):
        print()
    if not a.base:
        ap.error("--base is needed: EVAL_DIR has main-pass steps")
    base = load(a.base); bmap = {key(r): r["y"] for r in base}
    runs = [("base", base, True)] + [(f"s{n}", load(res / f"step{n}.jsonl"), (res / f"step{n}.done").exists()) for n in steps]
    m1 = {name: [r["y"] - bmap[key(r)] for r in R if key(r) in bmap] for name, R, _ in runs[1:]}

    print(f"{'ckpt':7s} {'n':>4s} {'y':>6s} {'fork':>5s}  {'M1 paired vs base':24s} {'M2 main-first':24s}")
    for name, R, done in runs:
        fk = sum(r["forked"] for r in R) / len(R) if R else nan
        print(f"{name + ('' if done else '*'):7s} {len(R):4d} {mean([r['y'] for r in R]):6.3f} {fk:5.0%}  "
              f"{fmt(m1.get(name)):24s} {fmt(m2(R)):24s}")
    late = [int(x) for x in a.late.split(",")] if a.late else steps[-3:]
    print(f"pooled late M2 (steps {','.join(map(str, late))}): {fmt([d for n in late for d in m2(load(res / f'step{n}.jsonl'))])}")

    print("\nM3' by bundle size (tasks): fork rate, M2 main-first")
    print(f"{'ckpt':7s} " + " ".join(f"{f'{lo}-{hi}':>27s}" for lo, hi in BUCKETS))
    for name, R, done in runs:
        cells = []
        for lo, hi in BUCKETS:
            B = [r for r in R + small.get(name, ([],))[0] if lo <= size(r) <= hi]
            cells.append(f"{sum(r['forked'] for r in B) / len(B):4.0%} {fmt(m2(B)):>22s}" if B else f"{'-':>27s}")
        print(f"{name + ('' if done else '*'):7s} " + " ".join(cells))

    print("\nM4 fork / control critical path over main-first pairs: median ratio, share fork faster; y per hour of wall")
    print(f"{'ckpt':7s} {'pairs':>5s} {'wall':>10s} {'tokens':>10s} {'y/h':>6s}")
    for name, R, done in runs:
        w = [r["C_wall"] for r in R if r.get("C_wall")]
        yh = mean([r["y"] for r in R]) / (mean(w) / 3600) if w else nan
        print(f"{name + ('' if done else '*'):7s} {len(pairs(R)):5d} {ratio(R, 'C_wall'):>10s} {ratio(R, 'C'):>10s} {yh:6.2f}")

    if any((r.get("rewinds") or {}).get("calls") for _, R, _ in runs for r in R):  # search rows (search_dev_eval)
        print("\nrewinds per episode (means): calls, steps removed, note characters")
        for name, R, done in runs:
            rw = [r.get("rewinds") or {} for r in R]
            print(f"{name + ('' if done else '*'):7s} " + " ".join(f"{mean([x.get(k, 0) for x in rw]):8.2f}"
                                                                for k in ("calls", "steps", "note_chars")))

    done = [(name, R) for name, R, d in runs[1:] if d and m1[name]]
    ok = [(name, R) for name, R in done if len(m2(R)) > 1 and mean(m2(R)) + ci(m2(R)) >= 0]
    pick = max(ok or done, key=lambda nr: mean(m1[nr[0]]), default=None)
    print(f"\nbest checkpoint: {pick[0] if pick else '-'} (rule: max M1 among done steps with M2 95% CI upper bound >= 0"
          f"{'' if ok else '; none qualifies, so max M1 overall'})")


if __name__ == "__main__":
    main()
