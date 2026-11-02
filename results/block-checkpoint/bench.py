"""Block checkpointing (octopus/rl/block_checkpoint.py) on one real search-v9-rest micro-batch, as the actor trains it:
Qwen3.5-4B, LoRA (rank 64, the resolved targets), FSDP2 per decoder layer (bf16 params, fp32 reduce), prefix tree with
its reads, sequence tiles, chunked head, saved tensors in pinned host memory (lm_head.host_activations, ACT_GPU_GB=0).
Each --blocks entry runs the micro-batch's forward + backward with that block size at any micro-batch size (1 = HF
per-layer checkpointing; the actor's BLOCK_TOKENS threshold does not apply);
per run: host bytes the forward saved (the pinned buffer's size, lm_head._host.need) per tree token, GPU peak allocated,
forward + backward time, and the logprobs and LoRA gradients against the first run's (a second per-layer run gives the
run-to-run noise of the GPU kernels).

usage (results: bench-gpu138-1029.log, GPUs 3-4 of node-138, group 000041~search-1994~8a1dda3a):
       OCTOPUS_SGLANG_PY=... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       torchrun --nproc-per-node 2 results/block-checkpoint/bench.py GROUP.pt MODEL_DIR [--cap 200000] [--blocks 1 1 4]
GROUP.pt: a rollout-spool group (octopus/rl/spool.py); the micro-batch is the largest allocation unit of at most --cap
tree tokens (octopus/rl/prefix_tree.py allocate_prefix_groups, reads kept with their sources).
"""

import argparse
import time

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from octopus.rl import lm_head
from octopus.rl.actor import batch_rows
from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.lora_targets import resolve
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.prefix_tree import allocate_prefix_groups, patch_attention, tree_inputs, tree_tokens


def micro_batch(group: dict, cap: int):
    seqs, mounts = batch_rows(group)
    unit = max((u for u in allocate_prefix_groups(seqs, cap, 1, 1, mounts) if tree_tokens([seqs[i] for i in u]) <= cap),
               key=lambda u: tree_tokens([seqs[i] for i in u]))
    rows = torch.tensor(sorted(unit))
    sub = {k: v[rows] for k, v in group.items() if torch.is_tensor(v) and v.ndim == 2}
    seqs, mounts = batch_rows(sub)
    mask = sub["attention_mask"].bool()
    loss_mask = torch.cat([torch.cat([m[a][1:], m.new_zeros(1)]) for m, a in zip(sub["loss_mask"], mask)])
    ids = torch.from_numpy(np.concatenate(seqs))[None].cuda()
    cu = torch.tensor(np.cumsum([0] + [len(s) for s in seqs]), dtype=torch.int32).cuda()
    meta, kwargs = tree_inputs(ids, cu, mounts)
    return ids, loss_mask.cuda().bool(), meta, kwargs, len(unit), sum(len(v) for v in (mounts or {}).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("model")
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 1, 4])
    ap.add_argument("--pin-gib", type=float, default=0, help="pin the saved-activation buffer at this size up front "
                    "(else the first run takes torch's power-of-two pinned blocks, up to twice its size)")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, target_modules=resolve(model), lora_dropout=0.0))
    model.cuda()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    layers = model.get_decoder().layers
    for layer in layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    patch_model(model.config)
    tile_model(model)
    patch_attention()
    head = lm_head.bind_head(model, 1.0)
    hf = [layer._gradient_checkpointing_func for layer in layers]

    head.octopus_tree = None  # the actor's init check: every kernel's autotuning on an empty card
    ids = torch.randint(1000, 100000, (1, 256)).cuda()
    head.octopus_labels = torch.roll(ids, -1, -1).reshape(-1)
    model(input_ids=ids, use_cache=False).logits[0, :, 0].mean().backward()
    model.zero_grad(set_to_none=True)

    ids, loss_mask, meta, kwargs, n_rows, n_reads = micro_batch(torch.load(args.group, weights_only=False), args.cap)
    tokens = kwargs["input_ids"].shape[1]
    if rank == 0:
        print(f"micro-batch: {n_rows} rows, {n_reads} reads, {ids.shape[1]} flat tokens, {tokens} tree tokens",
              flush=True)
    lm_head._host.need = int(args.pin_gib * 2**30)
    first = None
    for k in args.blocks:
        for layer, f in zip(layers, hf):
            layer._gradient_checkpointing_func = f
        if k > 1:
            checkpoint_blocks(model, k, max_tokens=10**9)  # --blocks applies at any size (not BLOCK_TOKENS)
        model.zero_grad(set_to_none=True)
        head.octopus_tree = meta
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with lm_head.host_activations(model):
            logp = model(**kwargs).logits[0, :, 0]
        saved = lm_head._host.need
        (-(logp * loss_mask).sum() / loss_mask.sum()).backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        grads = {n: p.grad.to_local().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        lm_head.release_host_activations()
        line = (f"k={k}: host saved {saved / 2**30:.2f} GiB = {saved / tokens / 1024:.1f} KiB/tree token, "
                f"GPU peak {peak:.1f} GiB, forward+backward {dt:.1f} s")
        if first is None:
            first = logp.detach().clone(), grads
        else:
            rel = max(float((grads[n] - g).norm() / g.norm().clamp_min(1e-30)) for n, g in first[1].items())
            line += (f"; vs first run: logp max |diff| {float((logp.detach() - first[0]).abs().max()):.3g}, "
                     f"LoRA grads max rel L2 diff {rel:.3g}, identical {all(torch.equal(grads[n], g) for n, g in first[1].items())}")
        print(f"[rank {rank}] {line}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
