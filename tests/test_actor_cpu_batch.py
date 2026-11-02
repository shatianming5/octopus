"""Host-staged actor batches (octopus/rl/actor.py cpu_staged_rpc_methods, patches/areal-cpu-batch.patch).

compute_logp and ppo_update on spooled groups delivered to the GPU (as before) and staged in host memory: the logprobs
and the LoRA gradients are bit-identical, and the staged batch leaves only the running micro-batch on the GPU. One
logged stat differs in its last digits: advantages/avg, which AReaL's ppo_update sums on the batch's device, now the
host. Backward kernels run in their deterministic modes here: by default flash-attn's backward and index_select's
accumulate in a run-dependent order (same-path runs: ~half the LoRA gradients differ, max 6e-5).

One GPU, the patched AReaL first on PYTHONPATH; OCTOPUS_TEST_MODEL (a text checkpoint, e.g. Qwen3.5-4B-text),
OCTOPUS_TEST_GROUPS (comma-separated spool files, octopus/rl/spool.py), optional OCTOPUS_TEST_CONFIG (a run's
config.yaml for the actor settings; default octopus/rl/grpo.yaml).
"""

import functools
import math
import os
from pathlib import Path

import pytest
import torch

MODEL, GROUPS = os.environ.get("OCTOPUS_TEST_MODEL"), os.environ.get("OCTOPUS_TEST_GROUPS")
CONFIG = os.environ.get("OCTOPUS_TEST_CONFIG", str(Path(__file__).parents[1] / "octopus/rl/grpo.yaml"))
pytestmark = pytest.mark.skipif(not (MODEL and GROUPS and torch.cuda.is_available()),
                                reason="needs a GPU, OCTOPUS_TEST_MODEL and OCTOPUS_TEST_GROUPS")


def make_actor():
    """A one-rank actor with deterministic backward kernels."""
    import flash_attn
    from omegaconf import OmegaConf
    from areal.api.cli_args import to_structured_cfg
    from areal.api.io_struct import FinetuneSpec
    from octopus.rl.actor import ChunkedHeadFSDPPPOActor
    from octopus.rl.config import RQ1RLConfig

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    flash_attn.flash_attn_varlen_func = functools.partial(flash_attn.flash_attn_varlen_func, deterministic=True)
    for k, v in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)
    cfg = OmegaConf.load(CONFIG)
    cfg.actor.path, cfg.actor.backend = MODEL, "fsdp:d1"
    a = ChunkedHeadFSDPPPOActor(OmegaConf.to_object(to_structured_cfg(cfg, RQ1RLConfig)).actor)
    a.create_process_group()
    a.initialize(None, FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1))
    return a


@pytest.fixture(scope="module")
def actor():
    return make_actor()


def deliver(engine, method, data, staged):
    """What AReaL's engine_blueprint does to an RPC's batch: move it to the broadcast device, broadcast it from the
    data-parallel head. staged: the actor's cpu_staged_rpc_methods route; else the GPU route every method took before."""
    from areal.infra.platforms import current_platform
    from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target
    from areal.utils.data import broadcast_tensor_container, tensor_container_to

    device = current_platform.current_device()
    group, device = (resolve_broadcast_target(engine, device, method) if staged
                     else (engine.context_and_model_parallel_group, device))
    return broadcast_tensor_container(tensor_container_to(data, device), src_rank=engine.current_data_parallel_head(),
                                      group=group)


def lora_grads(engine):
    grads = {}
    for name, p in engine.model.named_parameters():
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            grads[name] = g.detach().cpu().clone()
    return grads


def step(engine, items, staged):
    """compute_logp, compute_advantages (always on the GPU) and ppo_update on dispatch items (dp_balance.split_episodes),
    for comparing each staged call with the GPU-delivered one (grpo.yaml's reuse_train_logp skips compute_logp in
    training, and ppo_update's loss then ignores the prox_logp set here). The optimizer only records the gradients (the
    weights stay for the next call). Returns the logprobs, gradients, stats and, per call, the GPU bytes above the
    pre-call level at the first micro-batch's forward."""
    from areal.utils.data import tensor_container_to

    items = [dict(d) for d in items]
    resident, grads = {}, []

    def at_forward(module, args, kwargs):
        resident.setdefault(call, torch.cuda.memory_allocated() - base)

    hook = engine.model.register_forward_pre_hook(at_forward, with_kwargs=True)
    engine.optimizer_step = lambda: (grads.append(lora_grads(engine)), engine.optimizer_zero_grad(), {})[-1]
    try:
        torch.cuda.synchronize()
        base, call = torch.cuda.memory_allocated(), "compute_logp"
        logp = [lp.cpu() for lp in engine.compute_logp(deliver(engine, call, items, staged))]
        for item, lp in zip(items, logp, strict=True):
            item["prox_logp"] = lp
        adv = tensor_container_to(engine.compute_advantages(deliver(engine, "compute_advantages", items, False)), "cpu")
        base, call = torch.cuda.memory_allocated(), "ppo_update"
        engine.ppo_update(deliver(engine, call, adv, staged))
    finally:
        hook.remove()
        del engine.optimizer_step
    return logp, grads, engine.export_stats(), resident


def test_staged_batch_is_bit_identical_and_smaller_on_the_gpu(actor):
    from octopus.rl.dp_balance import split_episodes

    items = split_episodes([torch.load(f, weights_only=False) for f in GROUPS.split(",")])
    logp0, grads0, stats0, gpu0 = step(actor, items, staged=False)
    logp1, grads1, stats1, gpu1 = step(actor, items, staged=True)
    print(f"GPU bytes above the call's start at its first forward, GiB: delivered "
          f"{ {k: round(v / 2**30, 2) for k, v in gpu0.items()} }, staged { {k: round(v / 2**30, 2) for k, v in gpu1.items()} }")
    assert all(torch.equal(a, b) for a, b in zip(logp0, logp1, strict=True))
    assert len(grads0) == len(grads1) == 1 and grads0[0] and grads0[0].keys() == grads1[0].keys()
    assert all(torch.equal(grads0[0][k], grads1[0][k]) for k in grads0[0])
    differ = {k for k in stats0 if stats0[k] != stats1.get(k)}
    assert stats0.keys() == stats1.keys() and differ <= {"ppo_actor/advantages/avg"}
    assert all(math.isclose(stats0[k], stats1[k], rel_tol=1e-4) for k in differ)
    assert all(gpu1[k] < gpu0[k] for k in gpu0)
