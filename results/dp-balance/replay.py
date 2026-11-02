"""Replay one training step's data-parallel dispatch and micro-batch split from its spool group files (read-only, CPU):
plan episodes dispatched longest first by their tree tokens (octopus/rl/dp_balance.py), each rank then splitting its
rows as actor._allocate_by_prefix does.

A ppo_update micro-batch's time is modelled as b x tree tokens + a x attention query-key pairs (prefix_tree.cost's
terms), calibrated on mb-256190 (Qwen3.5-4B on a 4090D, chunk-major after #346; README.md). A call's time, for two FSDP
schedules:
  lockstep  every micro-batch index syncs the ranks (per-layer all-gathers, a reduce-scatter per backward): the sum
            over j of the slowest rank's j-th micro-batch; a rank with fewer runs AReaL's dummies, its shortest one
  no_sync   parameters stay gathered and gradients reduce once at the call's end: the slowest rank's total, its
            dummies included; the one reduce-scatter (the LoRA gradients) is left out
Each prints its time, the ideal (the mean rank total of the real micro-batches) and the idle share. With --check-log,
the replayed per-rank flat micro-batch lists must equal the step's logged ones.

usage: PYTHONPATH=<areal overlay>:<code> python results/dp-balance/replay.py [--check-log LOG_LINES] [--ranks 4] \\
           [--cap 65536] SPOOL_GROUP.pt ...
"""

import argparse
import re

import numpy as np
import torch

from octopus.rl import dp_balance
from octopus.rl.actor import batch_rows
from octopus.rl.prefix_tree import allocate_prefix_groups, build_tree, subset, tree_tokens

REF_TOKENS, REF_PAIRS = 256190, 15941267432  # mb-256190
# ppo_update 184.2 s; FlashAttention per layer 1.28 s a forward (816 calls / 8 layers / 33 per pass: 3.09 of them)
# and 5.65 s the backward
TOTAL, ATTENTION = 184.2, 8 * (816 / 8 / 33 * 1.28 + 5.65)


def features(seqs, mounts):
    tree = build_tree(np.concatenate(seqs), np.cumsum([0] + [len(s) for s in seqs]).tolist(), mounts or None)
    return len(tree.tokens), sum((n.end - n.start) * (len(n.mounted) + len(n.path) - (n.end - n.start))
                                 + (n.end - n.start) * (n.end - n.start + 1) // 2 for n in tree.nodes)


def rank_split(groups, items, cap):
    """A rank's micro-batches [(flat, tree tokens, pairs)] in run order, from its episodes [(group, rows)]."""
    seqs, mounts = [], {}
    for g, rows in items:
        s, m = subset(*groups[g], rows)
        mounts.update({r + len(seqs): [(p, src + len(seqs), a, b) for p, src, a, b in ms] for r, ms in m.items()})
        seqs += s
    return [(sum(len(seqs[i]) for i in mb), *features(*subset(seqs, mounts, mb)))
            for mb in allocate_prefix_groups(seqs, cap, 1, 1, mounts)]


def lockstep(times):
    n = max(len(x) for x in times)
    return sum(max(c) for c in zip(*[x + [min(x)] * (n - len(x)) for x in times]))


def no_sync(times):
    n = max(len(x) for x in times)
    return max(sum(x) + min(x) * (n - len(x)) for x in times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--cap", type=int, default=65536, help="MB_TOKENS")
    ap.add_argument("--check-log", help="the step's 'Microbatch #tokens (rank N): [...]' log lines")
    args = ap.parse_args()
    groups, episodes = [], []
    for f in args.groups:
        traj = torch.load(f, weights_only=False)
        seqs, mounts = batch_rows(traj)
        groups.append((seqs, mounts or {}))
        bounds = np.cumsum([0, *traj["rollout_group"].row_counts]).tolist()
        episodes += [(len(groups) - 1, list(range(a, b))) for a, b in zip(bounds[:-1], bounds[1:])]
    print(f"{len(groups)} groups, {len(episodes)} episodes, {args.ranks} ranks, cap {args.cap}", flush=True)

    weights = [tree_tokens([groups[g][0][r] for r in rows]) for g, rows in episodes]
    ranks = [rank_split(groups, [episodes[i] for i in p], args.cap)
             for p in dp_balance.longest_first(weights, args.ranks)]
    if args.check_log:
        logged = {int(m.group(1)): [int(x) for x in m.group(2).split(",")]
                  for m in re.finditer(r"\(rank (\d)\): \[([0-9, ]+)\]", open(args.check_log).read())}
        same = all(logged[r] == [f for f, _, _ in mbs] for r, mbs in enumerate(ranks))
        print(f"the dispatch replays the logged per-rank micro-batches exactly: {same}")
    a, b = ATTENTION / REF_PAIRS, (TOTAL - ATTENTION) / REF_TOKENS
    times = [[b * t + a * p for _, t, p in x] for x in ranks]
    ideal = sum(map(sum, times)) / len(times)
    calls = ", ".join(f"{f.__name__} {f(times) / 60:.1f} min idle {1 - ideal / f(times):.1%}"
                      for f in (lockstep, no_sync))
    print(f"micro-batches {[len(x) for x in ranks]}; ppo_update totals {[round(sum(x) / 60, 1) for x in times]} min, "
          f"ideal {ideal / 60:.1f}: {calls}", flush=True)

if __name__ == "__main__":
    main()
