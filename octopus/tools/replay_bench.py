"""Training-side bench on rollouts AReaL already dumped (no SGLang, no Harbor): rebuild each trajectory's
tokens from its dumped segments (each segment's token count must match), take the largest micro-batch the
actor's allocation makes (whole groups, sized in prefix-tree tokens), and run the actor's forward + backward
with the same patches (prefix tree, GDN varlen, chunked LM head, activations in pinned host memory, LoRA).
Reports peak memory and speed.

One GPU without FSDP sharding: ~9 GB stricter than the 2-GPU actor, so passing here means passing there.

usage: python -m octopus.tools.replay_bench --dump LOGS/<trial>/rollout/0 [--model TEXT_MODEL_DIR] [--rank 32]
       [--max-tokens 294912] [--no-warmup]
       python -m octopus.tools.replay_bench --synthetic 200000 [131072 ...] [--branch 100000:40000 ...] ...  (random
       tokens: one micro-batch per length, run one after another as the actor does: a thread of that length, plus one
       branch per --branch SHARED:OWN = its first SHARED tokens and OWN new ones)
Run it under a host-memory cap and with the actor's allocator setting, e.g. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
systemd-run --user --scope -p MemoryMax=120G -p MemorySwapMax=0 timeout 1800 ... (host memory per micro-batch token: docs/trainer-memory.md)
LoRA targets: octopus/rl/lora_targets.py, with OCTOPUS_SGLANG_PY set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.lm_head import bind_head, host_activations
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.prefix_tree import allocate_prefix_groups, patch_attention, tree_inputs
from octopus.rl.lora_targets import resolve

MODEL = "/mnt/disk1/user/models/Qwen3.5-9B-text"


def trajectories(dump: str, tok) -> tuple[list[dict], int]:
    """Each dumped record as tokens + loss mask; records whose segment token counts do not round-trip are skipped."""
    out, bad = [], 0
    for f in sorted(glob.glob(f"{dump}/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "segments" not in r:
                bad += 1
                continue
            ids, mask = [], []
            for s in r["segments"]:
                t = tok.encode(s["text"], add_special_tokens=False)
                if len(t) != s["len"]:
                    break
                ids += t
                mask += [s["role"] == "gen"] * len(t)
            else:
                out.append({"ids": ids, "mask": mask, "reward": r["reward"]})
                continue
            bad += 1
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--synthetic", type=int, nargs="+", help="tokens of a random main thread per micro-batch, not --dump")
    ap.add_argument("--branch", action="append", default=[], help="SHARED:OWN, a branch forked from the main thread")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=294912)
    ap.add_argument("--no-warmup", action="store_true", help="skip the autotune warm-up (reproduces the mid-step OOM)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.synthetic:
        batches = []
        for length in args.synthetic:
            main_ids = torch.randint(1000, 150000, (length,)).tolist()
            seqs = [main_ids] + [main_ids[:int(s)] + torch.randint(1000, 150000, (int(o),)).tolist()
                                 for s, o in (b.split(":") for b in args.branch)]
            batches.append([{"ids": t, "mask": [True] * len(t), "reward": 1.0} for t in seqs])
        trajs = batches[-1]
    else:
        trajs, bad = trajectories(args.dump, tok)
        mbs = allocate_prefix_groups([np.array(t["ids"]) for t in trajs], args.max_tokens)
        batches = [[trajs[i] for i in max(mbs, key=lambda m: sum(len(trajs[i]["ids"]) for i in m))]]
        print(f"{len(trajs)} trajectories rebuilt exactly, {bad} skipped", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF checkpoints layers only in training mode (Qwen3.5 has no dropout)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=resolve(model), lora_dropout=0.0))
    patch_model(model.config)
    tile_model(model)
    checkpoint_blocks(model)
    patch_attention()
    head = model.get_output_embeddings()
    if not args.no_warmup:  # what the actor's init check does
        ids = torch.tensor([trajs[-1]["ids"][:256]]).cuda()
        (-torch.log_softmax(model(input_ids=ids, use_cache=False).logits.float()[0, :-1], -1).mean()).backward()
        model.zero_grad(set_to_none=True)
    bind_head(model, 1.0).octopus_tree = None

    for batch in batches:
        run(model, head, batch)


def run(model, head, batch):
    """One micro-batch's forward + backward, as the actor runs it."""
    n = sum(len(t["ids"]) for t in batch)
    print(f"micro-batch: {len(batch)} seqs, {n} flat tokens", flush=True)
    ids = torch.tensor([sum((t["ids"] for t in batch), [])]).cuda()
    lens = [len(t["ids"]) for t in batch]
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32).cuda()
    pos = torch.tensor([[i for n_ in lens for i in range(n_)]]).cuda()
    mask = torch.tensor(sum((t["mask"][1:] + [False] for t in batch), [])).cuda()  # position t predicts token t+1
    adv = torch.tensor(sum(([t["reward"]] * len(t["ids"]) for t in batch), [])).cuda()
    head.octopus_labels = torch.roll(ids, -1, -1).reshape(-1)
    kwargs = {"input_ids": ids, "position_ids": pos, "cu_seq_lens_q": cu, "cu_seq_lens_k": cu,
              "max_length_q": max(lens), "max_length_k": max(lens), "use_cache": False}
    tree = tree_inputs(ids, cu)
    if tree is not None:
        head.octopus_tree, kwargs = tree
        print(f"prefix tree: {len(kwargs['input_ids'][0])} tree tokens", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with host_activations(model):
        out = model(**kwargs).logits[0]
    loss = -(out[:, 0] * adv * mask).sum() / mask.sum()  # policy-gradient-shaped loss on generated tokens
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"forward+backward: {dt:.0f}s, {n / dt:.0f} tokens/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB,"
          f" host RSS now {os.sysconf('SC_PAGE_SIZE') * int(open('/proc/self/statm').read().split()[1]) / 2**30:.1f} GiB,"
          f" peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB, loss {loss.item():.4f}, gen tokens {int(mask.sum())}", flush=True)


if __name__ == "__main__":
    main()
