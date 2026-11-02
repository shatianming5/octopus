"""Is the shared-prefix (tree) forward the same computation as running each sequence alone?

Takes sequences that share a prefix (a real GRPO group from an AReaL rollout dump, or random ones),
scores every flat position's next token three ways — each sequence alone, the ordinary packed varlen
micro-batch, and the prefix tree — and compares logprobs and the gradients of one weighted loss. The
packed-vs-alone gap is the bf16 noise floor the tree has to match.

usage: python -m octopus.tools.tree_check --model TEXT_MODEL_DIR [--tiny] [--dump LOGS/<trial>/rollout/0]
       [--n-seqs 4] [--max-len 6000]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.lm_head import bind_head
from octopus.rl.prefix_tree import patch_attention, tree_inputs


def group_sequences(dump: str, tokenizer, n: int, max_len: int) -> list[list[int]]:
    """n sequences of one dumped group (records with the same group id), rebuilt from their segments."""
    import glob
    for f in sorted(glob.glob(f"{dump}/*.jsonl")):
        seqs = []
        for line in open(f):
            r = json.loads(line)
            ids = []
            for s in r.get("segments", []):
                t = tokenizer.encode(s["text"], add_special_tokens=False)
                if len(t) != s["len"]:
                    break
                ids += t
            else:
                if ids:
                    seqs.append(ids[:max_len])
        if len(seqs) >= n:
            return seqs[:n]
    raise SystemExit("no dumped group with enough rebuildable sequences")


def random_sequences(n: int, max_len: int, vocab: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    prefix = rng.integers(0, vocab, max_len // 2).tolist()
    return [prefix + rng.integers(0, vocab, int(rng.integers(max_len // 8, max_len // 2))).tolist() for _ in range(n)]


def scored(model, head, ids: list[list[int]], weights: list[torch.Tensor], mode: str):
    """Flat per-position logprobs and the gradients of sum(weights * logprobs) over all sequences."""
    model.zero_grad(set_to_none=True)
    flat = torch.tensor([sum(ids, [])]).cuda()
    w = torch.cat(weights).cuda()
    lens = [len(s) for s in ids]
    cu = torch.tensor([0] + np.cumsum(lens).tolist(), dtype=torch.int32).cuda()
    head.octopus_tree = None
    if mode == "alone":
        outs = []
        for s in ids:
            x = torch.tensor([s]).cuda()
            head.octopus_labels = torch.roll(x, -1, -1).reshape(-1)
            outs.append(model(input_ids=x, use_cache=False).logits[0])
        out = torch.cat(outs)
    else:
        kwargs = {"input_ids": flat, "position_ids": torch.tensor([[i for n in lens for i in range(n)]]).cuda(),
                  "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": max(lens), "max_length_k": max(lens),
                  "use_cache": False}
        if mode == "tree":
            head.octopus_tree, kwargs = tree_inputs(flat, cu)
        head.octopus_labels = torch.roll(flat, -1, -1).reshape(-1)
        out = model(**kwargs).logits[0]
    last = cu[1:] - 1  # nothing follows a sequence's last token
    keep = torch.ones(len(w), dtype=torch.bool, device=w.device)
    keep[last] = False
    (out[:, 0][keep] * w[keep]).sum().backward()
    grads = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    return out[:, 0][keep].detach(), out[:, 1][keep].detach(), grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiny", action="store_true", help="random weights, 4 layers (3 GDN + 1 attention)")
    ap.add_argument("--dump")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=6000)
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.tiny:
        cfg = AutoConfig.from_pretrained(args.model)
        cfg = getattr(cfg, "text_config", cfg)  # a vision-language checkpoint's language model
        cfg.num_hidden_layers, cfg.layer_types = 4, ["linear_attention"] * 3 + ["full_attention"]
        cfg.hidden_size, cfg.intermediate_size = 512, 1024
        cfg.num_attention_heads, cfg.num_key_value_heads = 2, 1
        cfg._attn_implementation = "flash_attention_2"
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
        model.requires_grad_(False)
        for name, p in model.named_parameters():  # gradients of the layers LoRA would train, enough to compare
            p.requires_grad_(name.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight")) and ".layers.3" in name)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF checkpoints layers only in training mode (Qwen3.5 has no dropout)
    patch_model(model.config)
    tile_model(model)  # as the trainer
    patch_attention()
    head = bind_head(model, 1.0)
    if args.dump:
        from transformers import AutoTokenizer
        ids = group_sequences(args.dump, AutoTokenizer.from_pretrained(args.model), args.n_seqs, args.max_len)
    else:
        ids = random_sequences(args.n_seqs, args.max_len, model.config.vocab_size)
    weights = [torch.randn(len(s)) for s in ids]
    res = {}
    for mode in ("alone", "packed", "tree"):
        times = []
        for _ in range(2):  # the second run is warm (kernel autotuning and compilation happen on the first)
            torch.cuda.synchronize(); t0 = time.time()
            res[mode] = scored(model, head, ids, weights, mode)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"{mode:6s}: cold {times[0]:5.1f}s, warm {times[1]:5.1f}s", flush=True)
    flat_n = sum(len(s) for s in ids)
    tree_n = len(tree_inputs(torch.tensor([sum(ids, [])]).cuda(),
                             torch.tensor([0] + np.cumsum([len(s) for s in ids]).tolist(), dtype=torch.int32).cuda())[1]["input_ids"][0])
    print(f"{len(ids)} sequences, {flat_n} flat tokens -> {tree_n} tree tokens ({flat_n / tree_n:.2f}x)")
    base = res["alone"]
    for mode in ("packed", "tree"):
        lp, ent, g = res[mode]
        d = (lp - base[0]).abs()
        cos = torch.nn.functional.cosine_similarity(g, base[2], dim=0).item()
        print(f"{mode:6s} vs alone: logprob |diff| mean {d.mean():.4f} max {d.max():.3f} | entropy |diff| mean "
              f"{(ent - base[1]).abs().mean():.4f} | grad cosine {cos:.5f}, norm ratio {(g.norm() / base[2].norm()).item():.4f}")


if __name__ == "__main__":
    main()
