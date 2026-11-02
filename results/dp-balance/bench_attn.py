"""FlashAttention time of one full-attention layer of a real prefix tree, as the trainer calls it (FA2 varlen with
gathered keys, call by call): the attention term of prefix_tree.KAPPA (README.md). eval-attn's script, cut to that part.
usage: python bench_attn.py MB.pt"""
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from octopus.rl import prefix_tree
from octopus.rl.actor import batch_rows

torch.cuda.set_per_process_memory_fraction(0.75)
H, HKV, D = 16, 4, 256
dev = "cuda"
SCALE = D ** -0.5


def kernel_times(fn, reps=2):
    """Sum of CUDA kernel time by category over the last of reps runs."""
    for _ in range(reps - 1):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cats = {}
    for e in prof.key_averages():
        if e.device_time_total <= 0:
            continue
        n = e.key.lower()
        cat = ("flash_fwd" if "flash_fwd" in n else "flash_bwd" if "flash_bwd" in n else
               "flex/triton" if "triton" in n or "flex" in n else
               "index_select" if "index" in n and "add" not in n else "index_add/scatter" if "add" in n or "scatter" in n
               else "other")
        cats[cat] = cats.get(cat, 0.0) + e.device_time_total / 1e6
    return cats


def tflops(flops, s):
    return f"{flops / s / 1e12:.0f} TFLOP/s"


def main():
    path = sys.argv[1]
    mb = torch.load(path, weights_only=False)
    seqs, mounts = batch_rows(mb)
    flat = np.concatenate(seqs)
    cu = np.cumsum([0] + [len(s) for s in seqs]).tolist()
    tree = prefix_tree.build_tree(flat, cu, mounts)
    T = len(tree.tokens)
    nodes = tree.nodes
    calls = prefix_tree.attention_calls([n.end - n.start for n in nodes],
                                        [np.concatenate([n.mounted, n.path]) for n in nodes], dev)
    pairs = sum((n.end - n.start) * (len(n.mounted) + len(n.path) - (n.end - n.start)) +
                (n.end - n.start) * (n.end - n.start + 1) // 2 for n in nodes)
    flops = 4 * pairs * H * D
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
    k = torch.randn(T, HKV, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
    v = torch.randn(T, HKV, D, device=dev, dtype=torch.bfloat16, generator=g).requires_grad_()
    do = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, generator=g)

    def fa2():
        return prefix_tree.tree_attention(q[None], k[None], v[None], type("M", (), {"calls": lambda s, i: calls})(),
                                          SCALE, 3)[0]

    def fa2_both():  # call by call, as the trainer's per-call checkpoints free each call's gathered keys
        start = 0
        for call in calls:
            rows = slice(start, start + call[0])
            torch.autograd.backward(prefix_tree.attention_call(q[rows], k, v, call, SCALE), do[rows])
            start = rows.stop
    t0 = time.time()
    with torch.no_grad():
        f = kernel_times(fa2, 2)
    fb = kernel_times(fa2_both, 2)
    print(f"{path}: tree {T}, one layer, {len(calls)} FA2 calls, fwd FLOPs {flops:.3e}", flush=True)
    print("  FA2 forward kernels (s): " + ", ".join(f"{a} {b:.3f}" for a, b in f.items()) +
          f"; flash_fwd {tflops(flops, f['flash_fwd'])}")
    print("  FA2 fwd+bwd kernels (s): " + ", ".join(f"{a} {b:.3f}" for a, b in fb.items()) +
          f"; flash_bwd {tflops(2.5 * flops, fb['flash_bwd'])}, bwd/fwd {fb['flash_bwd'] / f['flash_fwd']:.2f}"
          f"  (wall {time.time() - t0:.0f} s)", flush=True)


main()
