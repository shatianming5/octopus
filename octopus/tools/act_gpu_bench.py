"""One rank's actor forward + backward on a synthetic prefix-tree micro-batch at several ACT_GPU_GB values: GPU peak,
pinned host bytes of the saved activations, time, and the LoRA gradients' relative L2 distance to the first value's
(list it twice for the run-to-run noise floor). No FSDP: the full bf16
base sits on the card (~8 GB for 4B), so a sharded rank peaks lower by about that much minus FSDP's unshard buffers.

usage: python -m octopus.tools.act_gpu_bench --model TEXT_MODEL_DIR --tokens 65536 131072 --gb 0 4 8 16
(LoRA targets: octopus/rl/lora_targets.py, with OCTOPUS_SGLANG_PY set)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl import lm_head
from octopus.rl.lora_targets import resolve
from octopus.rl.prefix_tree import patch_attention, tree_inputs


def group(tree_tokens: int, n: int, vocab: int, rng) -> list[list[int]]:
    """n sequences sharing a prefix of a sixth of the tree (a GRPO group's root + task message)."""
    prefix = rng.integers(0, vocab, tree_tokens // 6).tolist()
    own = (tree_tokens - len(prefix)) // n
    return [prefix + rng.integers(0, vocab, own).tolist() for _ in range(n)]


def step(model, head, seqs, gb):
    flat = torch.tensor([sum(seqs, [])]).cuda()
    cu = torch.tensor([0] + np.cumsum([len(s) for s in seqs]).tolist(), dtype=torch.int32).cuda()
    head.octopus_tree, kwargs = tree_inputs(flat, cu)
    head.octopus_labels = torch.roll(flat, -1, -1).reshape(-1)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with lm_head.host_activations(model, gb * 2**30):  # as actor.py tree_offloaded_forward
        out = model(**kwargs).logits[0]
    host = lm_head._host.need
    (out[:, 0] * torch.linspace(-1, 1, out.shape[0], device=out.device)).sum().backward()
    torch.cuda.synchronize()
    grads = torch.cat([p.grad.flatten().float().cpu() for p in model.parameters() if p.grad is not None])
    return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, host / 2**30, kwargs["input_ids"].shape[1], grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--tokens", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--gb", type=float, nargs="+", default=[0, 4, 8, 16])
    ap.add_argument("--n-seqs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=resolve(model), init_lora_weights=False))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    patch_model(model.config)
    tile_model(model)
    checkpoint_blocks(model)  # as the actor (OCTOPUS_BLOCK_CHECKPOINT)
    patch_attention()
    head = lm_head.bind_head(model, 1.0)
    rng = np.random.default_rng(0)
    print("tokens  gb  tree  gpu_peak_GiB  host_pinned_GiB  step_s  grad_reldiff_vs_first", flush=True)
    for tokens in args.tokens:
        seqs = group(tokens, args.n_seqs, model.config.vocab_size, rng)
        step(model, head, seqs, args.gb[0])  # warm-up: compile, autotune, slab
        base = None
        for gb in args.gb:
            secs, peak, host, tree, grads = step(model, head, seqs, gb)
            base = grads if base is None else base
            diff = ((grads - base).norm() / base.norm()).item()
            print(f"{tokens} {gb:4g} {tree} {peak:6.1f} {host:6.1f} {secs:6.1f} {diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
