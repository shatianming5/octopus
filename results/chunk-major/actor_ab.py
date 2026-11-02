"""The real actor (octopus/rl/actor.py ChunkedHeadFSDPPPOActor on AReaL's FSDP engine) on real spool groups, with
OCTOPUS_CHUNK_MAJOR on or off (P3 acceptance, results/chunk-major/README.md "Use in training").

Each step is the controller's train calls on the rank's groups: compute_logp, compute_advantages, ppo_update (one
optimizer step). Per step it prints one JSON line per rank: call times, GPU peak (allocated; ppo_update's only after the
optimizer step, which resets it: the actor's own `step peak allocated` line has the step's), host peak (RSS above the
start of the call), chunk-major's slab: pin/unpin time and the bytes its last micro-batch carved, and AReaL's stats
(loss, grad norm), and per forward_backward_batch call the seconds of each micro-batch (from its forward to the next
one's, the GPU synchronized; the last one's to the call's end: the time a rank spends in it, waits for other ranks
included). Step 1 also saves the rank's compute_logp output and LoRA gradients (before the optimizer step) to
--save, for --compare A B. Every configuration starts from the same weights (OCTOPUS_INIT_LORA), so step 1 is like for
like.

usage: torchrun --nproc-per-node N results/chunk-major/actor_ab.py MODEL GROUP.pt[,GROUP.pt ...] [--steps 3]
           [--save DIR] [--seg K] [--gpu-cap 0.85] [--lockstep] [--rank-mb-tokens A,B,..] [--kept-on-gpu]
           [key=value config overrides]
       (env: OCTOPUS_CHUNK_MAJOR, OCTOPUS_CHUNK_TOKENS, OCTOPUS_BLOCK_CHECKPOINT, OCTOPUS_INIT_LORA)
       python results/chunk-major/actor_ab.py --compare DIR_A DIR_B
rank r trains the groups r, r + N, ...; PYTHONPATH = the patched AReaL overlay, then this code.
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil
import torch

CONFIG = Path(__file__).parents[2] / "octopus/rl/grpo.yaml"


class HostPeak:
    """Peak RSS of this process above its level at entry, sampled every 50 ms."""

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


def make_actor(model: str, overrides: list[str]):
    from omegaconf import OmegaConf
    from areal.api.cli_args import to_structured_cfg
    from areal.api.io_struct import FinetuneSpec
    from octopus.rl.actor import ChunkedHeadFSDPPPOActor
    from octopus.rl.config import RQ1RLConfig

    cfg = OmegaConf.merge(OmegaConf.load(CONFIG), OmegaConf.from_dotlist(overrides))
    cfg.actor.path, cfg.actor.backend = model, f"fsdp:d{int(os.environ.get('WORLD_SIZE', 1))}"
    a = ChunkedHeadFSDPPPOActor(OmegaConf.to_object(to_structured_cfg(cfg, RQ1RLConfig)).actor)
    a.create_process_group()
    a.initialize(None, FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1))
    return a


def deliver(engine, method, data):
    """AReaL's engine_blueprint for an RPC's batch (the actor's cpu-staged route)."""
    from areal.infra.platforms import current_platform
    from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target
    from areal.utils.data import broadcast_tensor_container, tensor_container_to

    group, device = resolve_broadcast_target(engine, current_platform.current_device(), method)
    return broadcast_tensor_container(tensor_container_to(data, device), src_rank=engine.current_data_parallel_head(),
                                      group=group)


def lora_grads(engine) -> dict:
    return {n: (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).float().cpu()
            for n, p in engine.model.named_parameters() if p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("groups")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--save", help="dir for step 1's logp and LoRA gradients (rank 0's gradients: merged by FSDP)")
    ap.add_argument("--seg", type=int, help="prefix_tree.SEG in k tokens (the bf16 noise floor: other kernel bounds)")
    ap.add_argument("--gpu-cap", type=float, default=0.85, help="fraction of the card this process may allocate")
    ap.add_argument("--fp32", action="store_true", help="fp32 model and activations, no TF32 (run with "
                    "TRITON_F32_DEFAULT=ieee and a fresh TRITON_CACHE_DIR), the tree's attention by SDPA: exactness "
                    "below bf16's noise floor, as bench.py --fp32 --ieee")
    ap.add_argument("--lockstep", action="store_true", help="AReaL's forward_backward_batch, not the actor's: FSDP "
                    "gathers and reduces per micro-batch under chunk-major too (before across_micro_batches)")
    ap.add_argument("--rank-mb-tokens", help="per rank, the micro-batch cap (tree tokens) instead of the config's: "
                    "unequal micro-batch lists")
    ap.add_argument("--kept-on-gpu", action="store_true", help="chunk-major's kept tensors and their gradient sums on "
                    "the GPU, not in the pinned slab (chunk_major.HOST)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_per_process_memory_fraction(args.gpu_cap)
    for k, v in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)

    from areal.utils.data import tensor_container_to
    from octopus.rl import chunk_major, prefix_tree
    from octopus.rl.dp_balance import split_episodes

    if args.seg:
        prefix_tree.SEG = args.seg * 1024
    chunk_major.HOST = not args.kept_on_gpu
    pins = []  # chunk-major's slab: seconds in start (pinning) and release (unpinning)

    def timed(f, what):
        def g(*a, **kw):
            t0 = time.time()
            r = f(*a, **kw)
            pins.append((what, time.time() - t0))
            return r
        return g
    chunk_major._host.start = timed(chunk_major._host.start, "start")
    chunk_major._host.release = timed(chunk_major._host.release, "release")

    if args.fp32:
        from bench import sdpa_attention_call  # results/chunk-major/bench.py, beside this script
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        prefix_tree.attention_call = sdpa_attention_call
        args.overrides += ["actor.dtype=float32", "actor.attn_impl=sdpa"]  # FlashAttention has no fp32
    if args.lockstep:
        from areal.engine import FSDPPPOActor
        from octopus.rl.actor import ChunkedHeadFSDPPPOActor
        ChunkedHeadFSDPPPOActor.forward_backward_batch = FSDPPPOActor.forward_backward_batch
    engine = make_actor(args.model, args.overrides)
    if args.rank_mb_tokens:
        engine.config.mb_spec.max_tokens_per_mb = int(args.rank_mb_tokens.split(",")[int(os.environ["RANK"])])
    marks, mb_s = [], []  # this call's micro-batch forward times; per forward_backward_batch call, each one's seconds
    engine.model.register_forward_pre_hook(lambda *_: torch.cuda.synchronize() or marks.append(time.time()))
    forward_backward_batch = engine.forward_backward_batch

    def timed_forward_backward_batch(*a, **kw):
        marks.clear()
        result = forward_backward_batch(*a, **kw)
        torch.cuda.synchronize()
        mb_s.append([round(y - x, 1) for x, y in zip(marks, marks[1:] + [time.time()])])
        return result
    engine.forward_backward_batch = timed_forward_backward_batch
    if args.fp32:  # the init check ran on sdpa; the tree path goes through the flash_attention patch (patch_attention)
        for m in engine.model.modules():
            if hasattr(m, "config"):
                m.config._attn_implementation = "flash_attention_2"
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    paths = args.groups.split(",")[rank::world]
    groups = [torch.load(p, weights_only=False) for p in paths]
    step_stats = {}
    original_step = engine.optimizer_step

    def optimizer_step():
        if step_stats["step"] == 1 and args.save:
            torch.save(lora_grads(engine), f"{args.save}/grads-rank{rank}.pt")
        return original_step()
    engine.optimizer_step = optimizer_step
    for step in range(1, args.steps + 1):
        step_stats.clear()
        step_stats["step"] = step
        items = split_episodes([{k: v.clone() if torch.is_tensor(v) else v for k, v in g.items()} for g in groups])
        line = {"rank": rank, "step": step, "config": {k: os.environ.get(k) for k in (
            "OCTOPUS_CHUNK_MAJOR", "OCTOPUS_CHUNK_TOKENS", "OCTOPUS_BLOCK_CHECKPOINT")}, "seg": prefix_tree.SEG,
            "groups": [os.path.basename(p) for p in paths]}
        for call in ("compute_logp", "ppo_update"):
            pins.clear()
            mb_s.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with HostPeak() as host:
                t0 = time.time()
                if call == "compute_logp":
                    logp = [lp.cpu() for lp in engine.compute_logp(deliver(engine, call, items))]
                    for item, lp in zip(items, logp, strict=True):
                        item["prox_logp"] = lp
                    adv = tensor_container_to(engine.compute_advantages(
                        deliver(engine, "compute_advantages", items)), "cpu")
                    if step == 1 and args.save:
                        torch.save(logp, f"{args.save}/logp-rank{rank}.pt")
                else:
                    engine.ppo_update(deliver(engine, call, adv))
                torch.cuda.synchronize()
                line[call] = {"s": round(time.time() - t0, 1),
                              "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                              "host_peak_gib": round(host.peak / 2**30, 1),
                              "slab_s": {w: round(sum(t for x, t in pins if x == w), 1) for w in ("start", "release")},
                              "slab_gib": round(chunk_major._host.need / 2**30, 1), "mb_s": list(mb_s)}
        stats = engine.export_stats()
        line["stats"] = {k: v for k, v in stats.items() if any(s in k for s in ("loss", "grad_norm", "importance"))}
        print("[actor_ab] " + json.dumps(line), flush=True)


def compare(a: str, b: str) -> None:
    """Step 1 of two runs: logp max and mean |diff| over all positions and the LoRA gradients' cosine, relative
    norm diff (rank 0's merged gradients)."""
    for r in range(8):
        if not Path(f"{a}/logp-rank{r}.pt").exists():
            break
        la, lb = torch.load(f"{a}/logp-rank{r}.pt"), torch.load(f"{b}/logp-rank{r}.pt")
        d = torch.cat([(x.float() - y.float()).abs().flatten() for x, y in zip(la, lb, strict=True)])
        print(f"rank {r}: logp max |diff| {d.max().item():.3g}, mean {d.mean().item():.3g}")
    ga, gb = torch.load(f"{a}/grads-rank0.pt"), torch.load(f"{b}/grads-rank0.pt")
    dot = sum(float((ga[k] * gb[k]).sum()) for k in ga)
    na, nb = (sum(float(g[k].pow(2).sum()) for k in g) ** 0.5 for g in (ga, gb))
    cos = {k: float(torch.nn.functional.cosine_similarity(ga[k].flatten(), gb[k].flatten(), 0)) for k in ga
           if ga[k].norm() > 0}
    print(f"LoRA grads: cosine {dot / na / nb:.5f} (smallest per tensor {min(cos.values()):.4f}), norm {na:.4g} / "
          f"{nb:.4g} (rel diff {abs(na - nb) / na:.2e})")


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        main()
