"""Per decoder layer of Qwen3.5-4B (32, full attention every 4th), the norm of rank 0's merged LoRA gradients
(actor_ab.py --save DIR/grads-rank0.pt) of each run and their cosine to the first run's (logs/p3-mixed-paths.log).
usage: python grad_layers.py NAME=DIR [NAME=DIR ...]"""
import sys

import torch

runs = dict(a.split("=", 1) for a in sys.argv[1:])
g = {k: torch.load(v + "/grads-rank0.pt") for k, v in runs.items()}


def sq(d, layer):
    return sum(float(t.float().pow(2).sum()) for n, t in d.items() if f"layers.{layer}." in n)


def cos(a, b, layer):
    num = sum(float((a[n].float() * b[n].float()).sum()) for n in a if f"layers.{layer}." in n)
    return num / (sq(a, layer) * sq(b, layer)) ** .5


names = list(runs)
print("layer      " + "".join(f"{k:>12}" for k in names) + "".join(f"  cos({k},{names[0]})" for k in names[1:]))
for layer in range(32):
    tag = "attn" if layer % 4 == 3 else "gdn "
    print(f"{layer:5d} {tag} " + "".join(f"{sq(g[k], layer) ** .5:12.4g}" for k in names)
          + "".join(f"  {cos(g[k], g[names[0]], layer):12.3f}" for k in names[1:]))
