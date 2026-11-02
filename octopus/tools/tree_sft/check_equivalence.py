"""GPU check: tree_loss (shared root) equals the sum of flat_loss over root+child samples, in value
and in LoRA gradients. usage: python -m octopus.tools.tree_sft.check_equivalence [--model PATH]"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig, get_peft_model

from octopus.tools.tree_sft.common import load_base
from octopus.tools.tree_sft.tree_forward import IGNORE, flat_loss, tree_loss, use_functional_linear_cache


def grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/disk1/user/models/Qwen3.5-9B")
    ap.add_argument("--root", type=int, default=300)
    ap.add_argument("--child", type=int, default=120)
    ap.add_argument("--fp32", action="store_true", help="separate kernel/rounding noise from real mismatches")
    args = ap.parse_args()
    use_functional_linear_cache()
    model = get_peft_model(load_base(args.model, torch.float32 if args.fp32 else torch.bfloat16), LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():  # non-zero B so every LoRA matrix gets a gradient
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=1e-3)
    causal = model.base_model.model
    g = torch.Generator().manual_seed(0)
    root = torch.randint(1000, 50000, (args.root,), generator=g).cuda()
    children = []
    for _ in range(3):
        ids = torch.randint(1000, 50000, (args.child,), generator=g).cuda()
        labels = ids.clone()
        labels[: args.child // 3] = IGNORE
        children.append((ids, labels))
    root_labels = root.clone()  # pre-fork turns trained once: in the tree root, in the first flat row
    root_labels[: args.root // 2] = IGNORE

    model.zero_grad()
    t_loss, t_n = tree_loss(causal, root, children, root_labels)
    t_loss.backward()
    t_grads = grads(model)

    model.zero_grad()
    f_loss, f_n = 0.0, 0
    for i, (ids, labels) in enumerate(children):
        prefix = root_labels if i == 0 else torch.full_like(root, IGNORE)
        loss, n = flat_loss(causal, torch.cat([root, ids]), torch.cat([prefix, labels]))
        loss.backward()
        f_loss, f_n = f_loss + loss.item(), f_n + n
    f_grads = grads(model)

    print(f"tokens tree={t_n} flat={f_n}")
    print(f"loss   tree={t_loss.item():.4f} flat={f_loss:.4f} rel_diff={abs(t_loss.item() - f_loss) / f_loss:.2e}")
    num = sum((t_grads[k] - f_grads[k]).norm() ** 2 for k in f_grads) ** 0.5
    den = sum(f_grads[k].norm() ** 2 for k in f_grads) ** 0.5
    worst = max(f_grads, key=lambda k: ((t_grads[k] - f_grads[k]).norm() / (f_grads[k].norm() + 1e-12)).item())
    print(f"grads  params={len(f_grads)} global_rel_diff={(num / den).item():.2e} worst={worst} "
          f"{((t_grads[worst] - f_grads[worst]).norm() / (f_grads[worst].norm() + 1e-12)).item():.2e}")


if __name__ == "__main__":
    main()
