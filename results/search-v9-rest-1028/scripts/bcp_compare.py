"""bcp_compare.py [A.jsonl B.jsonl]: the BCP rows of two adapters (default out/bcp-v0-fork-k1.jsonl and
out/bcp-v45-fork-k1.jsonl), each summarised as octopus/eval/search_dev_eval.py's summary() counts them (an episode's
last row per (qid, sample): a rerun supersedes an infra failure; a row without "y" is an infra failure, not scored), with
the Wilson 95% interval, the endings and branches per episode, then the paired difference B - A over the qids scored in
both: mean, a bootstrap 95% interval over qids (10,000 resamples, seed 0) and the up / down / same counts.
Standard library only, so it runs without the eval's environment.
"""
import collections
import json
import math
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"


def last_rows(path):
    return list({(r["qid"], r["sample"]): r for r in map(json.loads, open(path))}.values())


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report(path):
    rows = last_rows(path)
    ok = [r for r in rows if "y" in r]
    k = sum(r["y"] for r in ok)
    lo, hi = wilson(k, len(ok))
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    print(f"{Path(path).name}: lora {ok[0]['lora']}, rows {sum(1 for _ in open(path))}, episodes {len(rows)}, "
          f"scored {len(ok)}, infra {len(rows) - len(ok)}")
    print(f"  accuracy {k / len(ok):.3f} [{lo:.3f}, {hi:.3f}] | forked {mean([r['forked'] for r in ok]):.0%} | "
          f"branches/episode {mean([r['branches'] for r in ok]):.2f} | recall {mean([r['recall'] for r in ok]):.3f}")
    ends = collections.Counter(r["ended_with"] for r in ok)
    print("  endings " + ", ".join(f"{e} {n} (acc {mean([r['y'] for r in ok if r['ended_with'] == e]):.3f})"
                                    for e, n in ends.most_common()))
    return {r["qid"]: int(r["y"]) for r in ok}


def main(a=OUT / "bcp-v0-fork-k1.jsonl", b=OUT / "bcp-v45-fork-k1.jsonl"):
    ya, yb = report(a), report(b)
    qs = sorted(set(ya) & set(yb))
    d = [yb[q] - ya[q] for q in qs]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
    print(f"paired {Path(b).stem} - {Path(a).stem} on {len(qs)} qids: {sum(d) / len(d):+.3f} "
          f"[{boots[249]:+.3f}, {boots[9749]:+.3f}] | up {d.count(1)}, down {d.count(-1)}, same {d.count(0)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
