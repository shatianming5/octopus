"""Build the tree SFT dataset from Harbor job dirs.

usage: python -m octopus.tools.tree_sft.build --jobs JOB_DIR [JOB_DIR ...] --out encoded.jsonl
           [--model PATH] [--max-len 16384] [--min-reward 1.0]
Writes one line per tree: {"task", "trial", "fork_no", "root_ids", "children": [{"kind", "ids", "labels"}]}
and prints how many trees/children survived each filter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from octopus.tools.tree_sft.encode import encode_tree
from octopus.tools.tree_sft.trees import collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/mnt/disk1/user/models/Qwen3.5-9B")
    ap.add_argument("--max-len", type=int, default=65536, help="root + one child")
    ap.add_argument("--max-tree-tokens", type=int, default=131072, help="root + all children; longer trees are dropped")
    ap.add_argument("--min-reward", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    trials = sorted(t for job in args.jobs for t in Path(job).iterdir() if t.is_dir())
    trees = collect(trials, args.min_reward)
    stats = Counter(trees=len(trees))
    with open(args.out, "w") as sink:
        for tree in trees:
            enc = encode_tree(tree, tok, args.max_len, max_tree_tokens=args.max_tree_tokens)
            if enc is None:
                stats["dropped"] += 1
                continue
            stats["kept"] += 1
            stats["children"] += len(enc["children"])
            stats["root_tokens"] += len(enc["root_ids"])
            stats["child_tokens"] += sum(len(c["ids"]) for c in enc["children"])
            stats["labelled_tokens"] += sum(sum(x != -100 for x in c["labels"]) for c in enc["children"])
            sink.write(json.dumps({"task": tree["task"], "trial": tree["trial"], "fork_no": tree["fork_no"], **enc}) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
