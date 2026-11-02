"""Pick real micro-batches for the chunk-major GPU benchmark (bench.py) from a rollout spool, read-only, on the CPU.

Every group file is split into micro-batches as the actor splits it (actor.batch_rows, prefix_tree
allocate_prefix_groups at MB_TOKENS 262144, reads kept with their sources). For each target size, the micro-batch
whose tree is closest to it is saved as OUT_DIR/mb-<tree tokens>.pt: the group's [rows, L] tensors restricted to its
rows, the format bench.py reads. The spool's largest real micro-batch is 665k tree tokens; --cap 1000000 packs a
synthetic ~994k one the same way (block-checkpoint/bench.py's 994k unit of search-5271).

usage: PYTHONPATH=<areal overlay>:<octopus code> python results/chunk-major/select_mbs.py SPOOL_DIR OUT_DIR \
           [--targets 256000 516000 665000] [--cap 262144]
results (search-v9-rest spool, node-138): mb-256190 (12 rows of search-4394~d7b755f7), mb-515922 (17 rows of
search-5271~58e89d3d), mb-664738 (22 rows of search-5271~fd75528b); --cap 1000000 --targets 994000: see select.log
"""

import argparse
import glob
import os

import torch

from octopus.rl.actor import batch_rows
from octopus.rl.prefix_tree import allocate_prefix_groups, tree_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spool")
    ap.add_argument("out")
    ap.add_argument("--targets", type=int, nargs="+", default=[256000, 516000, 665000])
    ap.add_argument("--cap", type=int, default=262144, help="micro-batch cap (the actor's MB_TOKENS)")
    args = ap.parse_args()
    found = []  # (tree tokens, file, rows)
    for path in sorted(glob.glob(f"{args.spool}/*/*.pt")):
        group = torch.load(path, weights_only=False)
        seqs, mounts = batch_rows(group)
        for rows in allocate_prefix_groups(seqs, args.cap, 1, 1, mounts):
            found.append((tree_tokens([seqs[i] for i in rows]), path, rows))
        print(f"{os.path.basename(path)}: {len(seqs)} rows, largest micro-batch "
              f"{max(t for t, p, _ in found if p == path)} tree tokens", flush=True)
    os.makedirs(args.out, exist_ok=True)
    for target in args.targets:
        tokens, path, rows = min(found, key=lambda f: abs(f[0] - target))
        group = torch.load(path, weights_only=False)
        index = torch.tensor(sorted(rows))
        mb = {k: v[index] for k, v in group.items() if torch.is_tensor(v) and v.ndim == 2}
        torch.save(mb, f"{args.out}/mb-{tokens}.pt")
        print(f"target {target}: {tokens} tree tokens, {len(rows)} rows of {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
