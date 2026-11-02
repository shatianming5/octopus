"""Method B: tree-aware SFT with LoRA. Custom loop: the root is forwarded once per tree, every child
continues from the root state, and the tree loss is the mean CE over all labelled tokens of the tree
(root and children; one tree = one unit of weight). Gradients accumulate over `--trees-per-step` trees.
Multi-GPU: `torchrun --nproc_per_node K`; each step's trees are split over the ranks and the LoRA
gradients summed, so the update equals the single-GPU one.

usage: python -m octopus.tools.tree_sft.train_tree --data encoded.jsonl --out DIR [--rank 16 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from octopus.tools.tree_sft.common import load_base, load_trees, offload
from octopus.tools.tree_sft.tree_forward import tree_loss, use_functional_linear_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/mnt/disk1/user/models/Qwen3.5-9B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--trees-per-step", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-trees", type=int, default=0, help="debug: use only the first N trees")
    ap.add_argument("--offload-activations", action="store_true", help="keep checkpointed layer inputs in pinned CPU memory (long trees)")
    args = ap.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed)  # same shuffle on every rank
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trees = load_trees(args.data)[: args.max_trees or None]

    use_functional_linear_cache()
    base = load_base(args.model)
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    causal = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(trees) / args.trees_per_step)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)

    log = (out / "train_log.jsonl").open("a") if rank == 0 else None
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(trees)))
        random.shuffle(order)
        for start in range(0, len(order), args.trees_per_step):
            window = order[start:start + args.trees_per_step]
            acc = torch.zeros(2, device="cuda", dtype=torch.float64)  # loss sum, labelled tokens
            for i in window[rank::world]:
                tree = trees[i]
                dev = "cuda"
                root = torch.tensor(tree["root_ids"], device=dev)
                children = [(torch.tensor(c["ids"], device=dev), torch.tensor(c["labels"], device=dev)) for c in tree["children"]]
                root_labels = torch.tensor(tree["root_labels"], device=dev) if tree.get("root_labels") else None
                try:
                    with offload(args.offload_activations):
                        loss_sum, n = tree_loss(causal, root, children, root_labels)
                    (loss_sum / max(n, 1) / args.trees_per_step).backward()
                except torch.cuda.OutOfMemoryError:
                    # a tree too long for this card: drop it (and this rank's partial gradients of this window), keep training
                    tokens = len(tree["root_ids"]) + sum(len(c["ids"]) for c in tree["children"])
                    print(json.dumps({"skipped_oom": tree.get("task"), "tree_tokens": tokens, "step": step, "rank": rank}), flush=True)
                    opt.zero_grad(set_to_none=True)
                    acc.zero_()
                    loss_sum = n = None
                    torch.cuda.empty_cache()
                    continue
                acc += torch.tensor([loss_sum.item(), n], device="cuda", dtype=torch.float64)
            if world > 1:
                for p in params:  # every rank joins every window's reduce, even with no trees or an OOM
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)
                dist.all_reduce(acc)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0:
                rec = {"step": step, "epoch": epoch, "loss": acc[0].item() / max(acc[1].item(), 1), "tokens": int(acc[1].item()), "grad_norm": round(grad_norm, 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
                       "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
        if rank == 0:
            model.save_pretrained(out / f"epoch{epoch + 1}")
    if rank == 0:
        model.save_pretrained(out / "final")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
