"""actor_ab.py (beside this script, same arguments) with, per micro-batch of compute_logp and ppo_update, the path it
took (chunk-major and its chunk count, or false: the flat HF layers) and, in ppo_update, which LoRA gradient shards are
non-finite after it: checked at the next micro-batch's forward, where actor_ab already synchronizes, after the last one
and before the optimizer step. Lines start with [nan_check] (README "Ranks on different paths",
logs/p3-mixed-paths.log). Launched as actor_ab.py is: torchrun ... results/chunk-major/nan_check.py MODEL GROUPS ..."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actor_ab  # noqa: E402

RANK = int(os.environ.get("RANK", 0))
st = {"call": None, "mb": 0, "cm": None}


def log(**kw):
    print("[nan_check] " + json.dumps({"rank": RANK, "call": st["call"], "mb": st["mb"], **kw}), flush=True)


def bad_shards(model) -> tuple[int, list[str]]:
    grads = [(n, getattr(p.grad, "_local_tensor", p.grad)) for n, p in model.named_parameters() if p.grad is not None]
    if not grads:
        return 0, []
    ok = torch.stack([torch.isfinite(g).all() for _, g in grads]).tolist()
    return len(grads), [n for (n, _), f in zip(grads, ok) if not f]


def install(engine):
    from octopus.rl import chunk_major
    model = engine.model

    def check(event):
        n, bad = bad_shards(model)
        log(event=event, chunk_major=st["cm"], grads=n, nonfinite=len(bad), names=bad)

    def pre(*_, **__):
        if st["call"] is not None:
            if st["mb"] and st["call"] == "ppo_update":
                check("after_mb")
            st["mb"] += 1
            st["cm"] = False
    model.register_forward_pre_hook(pre, with_kwargs=True)

    chunk_major_output = chunk_major.chunk_major_output

    def counted(plan, *a, **kw):
        st["cm"] = len(plan.chunks)
        return chunk_major_output(plan, *a, **kw)
    chunk_major.chunk_major_output = counted

    for name in ("compute_logp", "ppo_update"):
        def call(*a, _f=getattr(engine, name), _name=name, **kw):
            st["call"], st["mb"] = _name, 0
            try:
                return _f(*a, **kw)
            finally:
                st["call"] = None
        setattr(engine, name, call)

    forward_backward_batch = engine.forward_backward_batch

    def last(*a, **kw):
        result = forward_backward_batch(*a, **kw)
        torch.cuda.synchronize()
        if st["call"] == "ppo_update":
            check("after_last_mb")
        else:
            log(event="last_mb", chunk_major=st["cm"])
        return result
    engine.forward_backward_batch = last

    optimizer_step = engine.optimizer_step

    def step():
        check("before_step")
        return optimizer_step()
    engine.optimizer_step = step


def make_actor(*a, _make=actor_ab.make_actor, **kw):
    engine = _make(*a, **kw)
    install(engine)
    return engine


actor_ab.make_actor = make_actor

if __name__ == "__main__":
    actor_ab.main()
