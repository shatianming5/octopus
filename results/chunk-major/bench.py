"""Chunk-major training (octopus/rl/chunk_major.py) against per-layer and block checkpointing on one real micro-batch,
as the actor trains it: Qwen3.5-4B bf16, LoRA r64 on the resolved targets, FSDP2 per decoder layer and root (bf16
params, fp32 reduce; 1 rank = one card), prefix tree with its reads, sequence tiles, chunked head.

Variants (--variants, the first is the reference):
  k1, k4       the actor's path: saved tensors in pinned host memory (lm_head.host_activations, ACT_GPU_GB 0), HF
               per-layer checkpointing (k1) or blocks of 4 layers (block_checkpoint.py, at any micro-batch size)
  k1s16        k1 with SEG 16k instead of 32k: the same arithmetic at other kernel-call bounds, whose bf16 rounding
               differs as chunk-major's does: the noise floor its differences compare to
  cm32-fp32    chunk-major, chunks of 32k tree tokens, kept tensors in pinned host memory, their gradient sums in fp32
               (cm64-bf16: 64k-token chunks, bf16 sums; ...); cm32-fp32@nested: its backward with the tiles' and
               kernel calls' checkpoints nested in each layer's, as before seq_tiles.no_inner
Per variant: compute_logp (no-grad forward) time and GPU peak; training forward + backward time, GPU peak and host
peak (process RSS above its level before the variant, sampled every 50 ms: a first micro-batch also holds torch's
power-of-two pinned blocks; chunk-major also prints its kept + sums bytes, predicted and carved: the steady state);
against the reference: logprob max |diff| on the loss positions and the LoRA gradients' cosine (all ranks' shards: one
vector) and their smallest per-tensor cosine; FlashAttention calls and MLP forwards per layer (rows over tree tokens)
of the reported training run.

usage (results: README.md):
  PYTHONPATH=<areal overlay>:<code> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  torchrun --nproc-per-node N results/chunk-major/bench.py MB.pt MODEL_DIR [--variants k1 k1 k4 cm32-fp32 ...]
MB.pt: one micro-batch, the [rows, L] tensors of a rollout-spool group's rows (select_mbs.py).
"""

import argparse
import contextlib
import os
import threading
import time

import numpy as np
import psutil
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from octopus.rl import chunk_major, lm_head, prefix_tree, seq_tiles
from octopus.rl.actor import batch_rows
from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.lora_targets import resolve
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.prefix_tree import build_tree, patch_attention, tree_inputs


class HostPeak:
    """Peak RSS of this process above its level at the start, sampled every 50 ms."""

    def __enter__(self):
        self.proc, self.base, self.peak, self.done = psutil.Process(), psutil.Process().memory_info().rss, 0, False
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.done:
            self.peak = max(self.peak, self.proc.memory_info().rss - self.base)
            time.sleep(0.05)

    def __exit__(self, *_):
        self.done = True
        self.thread.join()


def predicted_host(plan, model) -> float:
    """GiB of chunk-major's host stores: kept k, v (attention), layer-input rows (convolution) and fp32 final states
    (recurrence) per decoder layer, and their gradient sums (at least ACC_DTYPE)."""
    cfg, row = model.config, model.dtype.itemsize
    acc = max(chunk_major.ACC_DTYPE.itemsize, row)
    kinds = {  # per layer type: (kind, numbers per row, bytes per number kept, bytes per number of its sum)
        "full_attention": [(kind, cfg.num_key_value_heads * cfg.head_dim, row, acc) for kind in "kv"],
        "linear_attention": [("x", cfg.hidden_size, row, acc),
                             ("s", cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim,
                              4, 4)]}
    total = sum(plan.n_kept[kind] * numel * (kept + grad)
                for layer in cfg.layer_types for kind, numel, kept, grad in kinds[layer])
    return total / 2**30


def sdpa_attention_call(q, k, v, call, softmax_scale=None):
    """prefix_tree.attention_call for --fp32 (FlashAttention runs in bf16/fp16 only): SDPA node by node, causal from the
    bottom right as FlashAttention's."""
    from torch.nn.attention.bias import causal_lower_right
    _, keys, cu_q, cu_k, _, _ = call
    out = []
    for qa, qb, ka, kb in zip(cu_q[:-1].tolist(), cu_q[1:].tolist(), cu_k[:-1].tolist(), cu_k[1:].tolist()):
        q_ = q[qa:qb].transpose(0, 1)[None]
        k_, v_ = (t.index_select(0, keys[ka:kb]).repeat_interleave(q.shape[1] // k.shape[1], 1).transpose(0, 1)[None]
                  for t in (k, v))
        out.append(torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=causal_lower_right(qb - qa, kb - ka), scale=softmax_scale)[0].transpose(0, 1))
    return torch.cat(out)


def micro_batch(path: str):
    sub = torch.load(path, weights_only=False)
    seqs, mounts = batch_rows(sub)
    mask = sub["attention_mask"].bool()
    loss_mask = torch.cat([torch.cat([m[a][1:], m.new_zeros(1)]) for m, a in zip(sub["loss_mask"], mask)])
    flat = np.concatenate(seqs)
    cu = np.cumsum([0] + [len(s) for s in seqs]).tolist()
    ids, cu_t = torch.from_numpy(flat)[None].cuda(), torch.tensor(cu, dtype=torch.int32).cuda()
    meta, kwargs = tree_inputs(ids, cu_t, mounts)
    return build_tree(flat, cu, mounts), loss_mask.cuda().bool(), meta, kwargs, (ids, cu_t, mounts), len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mb")
    ap.add_argument("model")
    ap.add_argument("--variants", nargs="+", default=["k1", "cm32-fp32", "k1s16", "cm64-fp32", "k1", "k4",
                                                      "cm32-bf16", "cm64-bf16"])
    ap.add_argument("--no-logp", action="store_true", help="skip the compute_logp forward")
    ap.add_argument("--gpu-cap", type=float, default=40, help="GiB this process may allocate on its card")
    ap.add_argument("--seg", type=int, default=32, help="prefix_tree.SEG in k tokens (a chunk holds whole SEG "
                    "segments: smaller ones for smaller chunks)")
    ap.add_argument("--no-fsdp", action="store_true", help="one rank, unsharded (FSDP on one rank keeps two copies "
                    "of the parameters: 32 GiB in fp32)")
    ap.add_argument("--repeat", type=int, default=1, help="training runs per variant, the last reported")
    ap.add_argument("--ieee", action="store_true", help="with --fp32: no TF32 in cuBLAS/cuDNN (and run with "
                    "TRITON_F32_DEFAULT=ieee in a fresh TRITON_CACHE_DIR for the Triton kernels)")
    ap.add_argument("--fp32", action="store_true", help="fp32 parameters and activations, attention by SDPA "
                    "(sdpa_attention_call): the kernels' exactness below bf16's noise floor")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    # shared cards: this process runs out of memory at --gpu-cap GiB rather than squeezing the card's other users
    torch.cuda.set_per_process_memory_fraction(args.gpu_cap * 2**30 / torch.cuda.get_device_properties().total_memory)

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                                 attn_implementation="sdpa" if args.fp32 else "flash_attention_2")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    torch.manual_seed(0)  # before get_peft_model: LoRA A's init, the same in every process
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, target_modules=resolve(model), lora_dropout=0.0))
    for name, p in model.named_parameters():  # a trained adapter's B is not 0: the gradients of A count too
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=1e-3)
    model.cuda()
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    layers = model.get_decoder().layers
    if not args.no_fsdp:
        for layer in layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    patch_model(model.config)
    tile_model(model)
    patch_attention()
    if args.fp32:
        prefix_tree.attention_call = sdpa_attention_call
    head = lm_head.bind_head(model, 1.0)
    hf = [layer._gradient_checkpointing_func for layer in layers]

    head.octopus_tree = None  # the actor's init check: every kernel's autotuning on an empty card
    ids = torch.randint(1000, 100000, (1, 256)).cuda()
    head.octopus_labels = torch.roll(ids, -1, -1).reshape(-1)
    model(input_ids=ids, use_cache=False).logits[0, :, 0].mean().backward()
    model.zero_grad(set_to_none=True)
    if args.ieee:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if args.fp32:  # loaded as sdpa (FlashAttention refuses fp32); the tree path never calls it, but an sdpa model's
        # forward builds a [T, T] causal mask (9.6 GiB at 100k tokens), a flash_attention_2 one none
        for m in model.modules():
            if hasattr(m, "config"):
                m.config._attn_implementation = "flash_attention_2"

    prefix_tree.SEG = args.seg * 1024
    tree, loss_mask, meta, kwargs, (ids, cu, mounts), n_rows = micro_batch(args.mb)
    if rank == 0:
        print(f"micro-batch {args.mb}: {n_rows} rows, {ids.shape[1]} flat tokens, {len(tree.tokens)} tree tokens",
              flush=True)
    head_rows = lambda rows, y: lm_head._chunked_rows(head, rows, y)  # noqa: E731
    ref = None
    no_inner, calls, rows = seq_tiles.no_inner, [0], [0]
    attention_call = prefix_tree.attention_call

    def counted_call(*a, **k):
        calls[0] += 1
        return attention_call(*a, **k)
    prefix_tree.attention_call = counted_call
    for layer in layers:
        layer.mlp.gate_proj.register_forward_hook(
            lambda m, a, o: rows.__setitem__(0, rows[0] + a[0].numel() // a[0].shape[-1]))
    for name in args.variants:
        variant, _, nested = name.partition("@")
        seq_tiles.no_inner = contextlib.nullcontext if nested else no_inner
        for layer, f in zip(layers, hf):
            layer._gradient_checkpointing_func = f
        vars(model).pop("forward", None)
        line = name
        if variant.startswith("k"):
            k, _, seg = variant[1:].partition("s")
            if int(k) > 1:
                checkpoint_blocks(model, int(k), max_tokens=10**9)
            if seg:  # the same layers at other kernel-call bounds: the bf16 noise floor of the tree path
                prefix_tree.SEG = int(seg) * 1024
                head.octopus_tree, kwargs_k = tree_inputs(ids, cu, mounts)
                prefix_tree.SEG = args.seg * 1024
            else:
                head.octopus_tree, kwargs_k = meta, kwargs

            def run(train):
                if not train:
                    return model(**kwargs_k).logits[0, :, 0]
                with lm_head.host_activations(model):
                    return model(**kwargs_k).logits[0, :, 0]
        else:
            size, acc, *where = variant[2:].split("-")
            chunk_major.HOST = not where  # cm8-fp32-gpu: kept tensors and sums in GPU memory
            chunk_major.ACC_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[acc]
            t0 = time.time()
            plan = chunk_major.ChunkPlan(tree, int(size) * 1024, "cuda")
            line += (f": {len(plan.chunks)} chunks (plan {time.time() - t0:.1f} s), predicted host "
                     f"{predicted_host(plan, model):.1f} GiB")
            # as actor.py's forward replacement; the FSDP root's pre-forward wants an argument
            model.forward = lambda input_ids: chunk_major.chunk_major_output(plan, model, head_rows)

            def run(train):
                return model(kwargs["input_ids"])[0, :, 0]
        if not args.no_logp:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                run(False)
            torch.cuda.synchronize()
            line += (f"; compute_logp {time.time() - t0:.1f} s, GPU peak "
                     f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        for _ in range(args.repeat):  # the last run is reported: its pinned slab already sized and pinned
            model.zero_grad(set_to_none=True)
            calls[0] = rows[0] = 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with HostPeak() as host:
                t0 = time.time()
                logp = run(True)
                (-(logp * loss_mask).sum() / loss_mask.sum()).backward()
                torch.cuda.synchronize()
                dt = time.time() - t0
        line += (f"; train {dt:.1f} s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, host peak "
                 f"{host.peak / 2**30:.1f} GiB")
        # the micro-batch's pinned bytes, exactly (its slab: the steady state): saved activations or kept tensors + sums
        if not variant.endswith("-gpu"):  # (kept on the GPU: nothing pinned)
            line += f" (pinned {(chunk_major if variant.startswith('cm') else lm_head)._host.need / 2**30:.1f} GiB)"
        grads = {n: (p.grad if args.no_fsdp else p.grad.to_local()).float() for n, p in model.named_parameters()
                 if p.grad is not None}
        lm_head.release_host_activations()
        torch._C._host_emptyCache()
        if ref is None:
            ref = logp.detach().float(), {n: g.clone() for n, g in grads.items()}
        else:
            sums = torch.tensor([sum(float((grads[n] * g).sum()) for n, g in ref[1].items()),
                                 sum(float(g.square().sum()) for g in ref[1].values()),
                                 sum(float(grads[n].square().sum()) for n in ref[1])],
                                device="cuda", dtype=torch.float64)
            dist.all_reduce(sums)
            worst = min(float(torch.nn.functional.cosine_similarity(grads[n].flatten(), g.flatten(), 0))
                        for n, g in ref[1].items() if g.norm() > 0)
            diff = float((logp.detach().float() - ref[0])[loss_mask].abs().max())
            line += (f"; vs {args.variants[0]}: logp max |diff| {diff:.3g}, grad cosine "
                     f"{float(sums[0] / (sums[1] * sums[2]).sqrt()):.6f} (smallest per tensor {worst:.4f})")
        line += (f"; FlashAttention calls {calls[0]}, MLP forwards per layer "
                 f"{rows[0] / len(tree.tokens) / len(layers):.2f}")
        print(f"[rank {rank}] {line}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
