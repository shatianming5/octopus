"""The actor's forward (octopus/rl/actor.py install_forward) with OCTOPUS_CHUNK_MAJOR on and off, on CPU in fp64 (the
tiny model and kernels of test_chunk_major.py): AReaL's packed micro-batches in, the [1, T_flat, 2] logits out, and the
compute_logp / ppo_update wrappers' end of call, and AReaL's loss with the proximal logp of the training forward.
Imports AReaL (octopus.rl.actor)."""

import contextlib
import functools
import types

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
actor = pytest.importorskip("octopus.rl.actor")
from areal.engine import FSDPPPOActor  # noqa: E402
from test_chunk_major import LORA_TARGETS, _flat, _mount, _random_dag, _tiny_model  # noqa: E402

from octopus.rl import chunk_major, lm_head, prefix_tree  # noqa: E402

TOKENS = 128  # several chunks per micro-batch (SEG 64)


def _columns64(logits, temperature, labels):
    """lm_head._head_columns without its fp32 softmax and compile: the head adds no rounding above fp64's."""
    logp = torch.log_softmax(logits / temperature, -1)
    return torch.stack([logp.gather(-1, labels[:, None])[:, 0], -(logp.exp() * logp).sum(-1).detach()], -1)


@pytest.fixture
def model(monkeypatch):
    model = _tiny_model(monkeypatch.setattr)
    monkeypatch.setattr(lm_head, "_head_columns", _columns64)
    monkeypatch.setattr(chunk_major, "TOKENS", TOKENS)
    monkeypatch.setattr(chunk_major, "_pending", False)
    actor.install_forward(model, 1.0)
    return model


def _mbs():
    """Two packed micro-batches as AReaL passes them (input_ids [1, N], cu_seq_lens_q), sequences sharing prefixes, and
    a fixed weight per flat position each (the loss is (logp * w).sum())."""
    out = []
    for seed, seqs in enumerate([_mount()[0], _random_dag(1)[0]]):
        flat, cu = _flat(seqs)
        w = torch.randn(1, len(flat), dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        out.append((dict(input_ids=torch.from_numpy(flat)[None], cu_seq_lens_q=torch.tensor(cu, dtype=torch.int32)), w))
    return out


def _train(model, mbs, dummy=False):
    """compute_logp's forward (no grad), then ppo_update's: each micro-batch forward and backward, the gradients summed;
    dummy: then AReaL's dummy micro-batch (the first again, loss x 0). Returns the logits of both and the gradients."""
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        logp = [model(**mb).logits for mb, _ in mbs]
    train = []
    for mb, w in mbs + ([(mbs[0][0], None)] if dummy else []):
        out = model(**mb).logits
        assert out.shape == (1, mb["input_ids"].shape[1], 2)
        (out.mean() * 0.0 if w is None else (out[..., 0] * w).sum()).backward()
        train.append(out.detach())
    return logp, train, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_chunk_major_matches_the_flag_off_forward(model, monkeypatch):
    """compute_logp's logits, ppo_update's logits and the gradients summed over two micro-batches and AReaL's dummy
    equal the flag-off path's (fp64, 1e-9), each micro-batch in several chunks."""
    mbs = _mbs()
    ref_logp, ref_train, ref = _train(model, mbs)
    plans = []
    plan = chunk_major.ChunkPlan
    monkeypatch.setattr(chunk_major, "ChunkPlan", lambda *a: plans.append(plan(*a)) or plans[-1])
    monkeypatch.setattr(chunk_major, "ENABLED", True)
    logp, train, grads = _train(model, mbs, dummy=True)
    assert [len(p.chunks) > 2 for p in plans] == [True] * 5  # 2 compute_logp, 2 train, the dummy
    assert not chunk_major._pending
    for a, b in zip(logp + train, ref_logp + ref_train + ref_train[:1]):
        assert (a - b).abs().max() <= 1e-9
    assert grads.keys() == ref.keys() and min(g.abs().max() for g in ref.values()) > 1e-6
    assert max((grads[n] - g).abs().max() for n, g in ref.items()) <= 1e-9


def _short_mb():
    """A short packed micro-batch whose sequences share nothing: no tree with the flag off (the HF layers run it
    flat)."""
    rng = np.random.default_rng(7)
    flat, cu = _flat([rng.integers(3, 1024, n) for n in (20, 25)])
    mb = dict(input_ids=torch.from_numpy(flat)[None], cu_seq_lens_q=torch.tensor(cu, dtype=torch.int32))
    assert prefix_tree.tree_inputs(mb["input_ids"], mb["cu_seq_lens_q"]) is None
    return mb, torch.randn(1, len(flat), dtype=torch.float64, generator=torch.Generator().manual_seed(2))


def _mixed_worker(rank: int, path: str) -> None:
    """One of 2 gloo ranks, the flag on: the tiny LoRA model sharded as AReaL's apply_fsdp2 does, the actor's forward on
    AReaL's packed kwargs; rank 0's micro-batch a prefix tree, rank 1's _short_mb. compute_logp's logits and the merged
    LoRA gradients equal an unsharded copy's flag-off tree path (fp64, 1e-9; SEG below the short one's length makes it a
    tree there: the flat path's HF kernels round in fp32)."""
    import copy
    import datetime
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))  # mispaired collectives may hang: fail fast
    model = get_peft_model(_tiny_model(setattr), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                            target_modules=LORA_TARGETS))
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:  # PEFT starts B at 0: A would get no gradient
            torch.nn.init.normal_(p, std=0.1)
    reference = copy.deepcopy(model)
    mesh = init_device_mesh("cpu", (2,))
    for m in model.modules():  # AReaL's apply_fsdp2: each decoder layer, the input embedding (untied), the root
        if type(m).__name__ == "Qwen3_5DecoderLayer" or isinstance(m, torch.nn.Embedding):
            fully_shard(m, mesh=mesh)
    fully_shard(model, mesh=mesh)
    lm_head._head_columns, chunk_major.TOKENS = _columns64, TOKENS
    for m in (model, reference):
        actor.install_forward(m, 1.0)
    mbs = [_mbs()[0], _short_mb()]

    chunk_major.ENABLED = True
    mb, w = mbs[rank]
    with torch.no_grad():
        logp = model(**mb).logits
    (model(**mb).logits[..., 0] * w).sum().backward()
    grads = {n: p.grad.full_tensor() for n, p in model.named_parameters() if p.requires_grad}

    chunk_major.ENABLED, prefix_tree.SEG = False, 32
    with torch.no_grad():
        ref_logp = reference(**mb).logits
    for mb_, w_ in mbs:  # FSDP: the mean over ranks
        (reference(**mb_).logits[..., 0] * w_ / 2).sum().backward()
    ref = {n: p.grad for n, p in reference.named_parameters() if p.requires_grad}
    assert (logp - ref_logp).abs().max() <= 1e-9
    assert grads.keys() == ref.keys() and min(g.abs().max() for g in ref.values()) > 1e-6
    assert max((grads[n] - g).abs().max() for n, g in ref.items()) <= 1e-9
    dist.destroy_process_group()


def test_ranks_with_a_tree_and_a_short_micro_batch_both_run_chunk_major(tmp_path):
    """With the flag every micro-batch runs chunk-major, also one whose sequences share nothing, so all FSDP ranks issue
    the same collectives (prefix_tree.tree_inputs with_tree): one rank's flat backward beside another's chunk-major one
    mispaired NCCL's gathers and reduce-scatters (NaN gradients, results/chunk-major/README.md "Ranks on different
    paths"). 2 gloo ranks, one on a tree, one on such a micro-batch, match the unsharded flag-off path
    (_mixed_worker); before, gloo aborted at a size mismatch."""
    pytest.importorskip("peft")
    torch.multiprocessing.spawn(_mixed_worker, args=(str(tmp_path / "pg"),), nprocs=2)


@pytest.mark.parametrize("chunked", [False, True])
def test_loss_mask_scores_only_the_trained_positions(model, monkeypatch, chunked):
    """With AReaL's loss_mask in the micro-batch (compute_logp's: the rollout's; ppo_update's: rolled, _mask_rolled)
    the logits equal the unmasked forward's at actor.scored_positions and are 0 elsewhere, flag off and chunk-major;
    the gradients of a loss over those positions equal the unmasked forward's (fp64, 1e-9)."""
    monkeypatch.setattr(chunk_major, "ENABLED", chunked)
    (mb, w), _ = _mbs()
    n = mb["input_ids"].shape[1]
    mask = (torch.rand(1, n, generator=torch.Generator().manual_seed(3)) < 0.2).long()
    extra = {"loss_mask": mask}
    for rolled in (False, True):
        monkeypatch.setattr(actor, "_mask_rolled", rolled)
        scored = actor.scored_positions(extra)
        runs = []
        for kwargs in ({}, extra):
            model.zero_grad(set_to_none=True)
            out = model(**mb, **kwargs).logits
            (out[..., 0] * w * scored).sum().backward()
            runs.append((out.detach()[0], {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}))
        (ref, ref_grads), (out, grads) = runs
        assert (out[scored] - ref[scored]).abs().max() <= 1e-9 and (out[~scored] == 0).all()
        assert grads.keys() == ref_grads.keys() and max((grads[k] - g).abs().max() for k, g in ref_grads.items()) <= 1e-9


@pytest.mark.parametrize("chunked", [False, True])
def test_reused_train_logp_is_the_recomputed_prox(model, monkeypatch, chunked):
    """grpo.yaml prox_logp_method reuse_train_logp: AReaL's decoupled loss with the proximal logp taken from the
    training forward (logprobs.detach()) gives the loss and gradients it gives with compute_logp's separate forward at
    the same weights (ppo_n_minibatches 1), rejection sampling included, flag off and chunk-major (fp64, exact)."""
    from areal.api.cli_args import RejectionSamplingConfig
    from areal.trainer.ppo import actor as ppo_actor
    monkeypatch.setattr(chunk_major, "ENABLED", chunked)
    noop = lambda *a, **k: None  # noqa: E731  (the stats want fp32 and registered denominators)
    monkeypatch.setattr(ppo_actor, "stats_tracker", types.SimpleNamespace(
        scope=lambda *a, **k: contextlib.nullcontext(), denominator=noop, stat=noop, stat_compact=noop, scalar=noop))
    loss_fn = functools.partial(ppo_actor.grpo_loss_fn, eps_clip=0.2, eps_clip_higher=0.28, c_clip=None,
                                rejection_sampling=RejectionSamplingConfig(metric="ratio", upper=5.0),
                                use_decoupled_loss=True)
    mbs, g = _mbs(), torch.Generator().manual_seed(0)
    with torch.no_grad():
        prox = [model(**mb).logits[..., 0] for mb, _ in mbs]  # compute_logp
    data = []
    for p in prox:
        mask = torch.rand(p.shape, generator=g) < 0.7
        old = (p + 0.8 * torch.randn(p.shape, generator=g)) * mask  # a stale behavior policy
        assert ((p - old).exp() > 5)[mask].any()  # some tokens rejected
        data.append(dict(loss_mask=mask, advantages=torch.randn(p.shape, generator=g), logprobs=old))
    runs = []
    for method, extra in (("recompute", [{"prox_logp": p} for p in prox]), ("reuse_train_logp", [{}, {}])):
        model.zero_grad(set_to_none=True)
        losses = []
        for (mb, _), d, e in zip(mbs, data, extra):  # every micro-batch at the step's weights
            logits = model(**mb).logits
            loss = loss_fn(logits[..., 0], logits[..., 1], {**d, **e}, prox_logp_method=method)
            loss.backward()
            losses.append(loss.detach())
        runs.append((torch.stack(losses), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (ref_loss, ref), (loss, grads) = runs
    assert torch.equal(loss, ref_loss) and grads.keys() == ref.keys()
    assert all(torch.equal(grads[n], g) for n, g in ref.items())


def _actor_calls(monkeypatch, forward):
    """A ChunkedHeadFSDPPPOActor whose AReaL compute_logp / ppo_update are forward(): the actor's own wrappers around
    them, nothing else of AReaL's engine."""
    monkeypatch.delenv("OCTOPUS_LIVE_ENV", raising=False)
    monkeypatch.setattr(FSDPPPOActor, "compute_logp", lambda self, data: forward(), raising=False)
    monkeypatch.setattr(FSDPPPOActor, "ppo_update", lambda self, data: forward(), raising=False)
    a = object.__new__(actor.ChunkedHeadFSDPPPOActor)
    a.config = types.SimpleNamespace(mb_spec=types.SimpleNamespace(max_tokens_per_mb=10**6))
    return a


def test_ppo_update_scores_its_rolled_loss_mask(monkeypatch):
    """compute_logp's batch has the rollout's loss_mask, ppo_update's the rolled one, with no `advantages` key in pure
    MOPD distillation (areal/trainer/ppo/actor.py _prepare_mopd_batch): both score the same positions."""
    mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 0]])
    masks = [mask, torch.roll(mask, -1, -1)]
    a = _actor_calls(monkeypatch, lambda: actor.scored_positions({"loss_mask": masks.pop(0)}).tolist())
    assert a.compute_logp([]) == a.ppo_update([]) == [0, 1, 1, 0, 0, 1, 0, 0]


def test_flag_off_runs_no_chunk_major_code(model, monkeypatch):
    def fail(*_, **__):
        raise AssertionError("chunk-major code ran with OCTOPUS_CHUNK_MAJOR off")
    for name in ("ChunkPlan", "chunk_major_output", "run_forward", "run_backward", "release_host"):
        monkeypatch.setattr(chunk_major, name, fail)
    a = _actor_calls(monkeypatch, lambda: _train(model, _mbs(), dummy=True))
    a.compute_logp([])
    a.ppo_update([])


def test_a_failed_call_does_not_block_the_next_one(model, monkeypatch):
    """A micro-batch whose backward raised, or never ran, leaves its kept tensors pending; the actor's call ends with
    release_host also when it raises (a call that returns with one pending raises), so the next call's forwards run."""
    monkeypatch.setattr(chunk_major, "ENABLED", True)
    mb, w = _mbs()[0]
    calls = []
    a = _actor_calls(monkeypatch, lambda: calls.pop(0)())
    run_backward = chunk_major.run_backward

    def failing_backward(*_):
        raise RuntimeError("backward failed")
    monkeypatch.setattr(chunk_major, "run_backward", failing_backward)
    calls.append(lambda: (model(**mb).logits[..., 0] * w).sum().backward())
    with pytest.raises(RuntimeError, match="backward failed"):
        a.ppo_update([])
    monkeypatch.setattr(chunk_major, "run_backward", run_backward)

    def no_backward():
        model(**mb)
        raise RuntimeError("loss failed")
    calls.append(no_backward)
    with pytest.raises(RuntimeError, match="loss failed"):
        a.ppo_update([])
    calls.append(lambda: model(**mb))  # returns without the backward: the micro-batch's gradient would be lost
    with pytest.raises(AssertionError, match="without backward"):
        a.ppo_update([])
    calls.append(lambda: model(**mb))
    with torch.no_grad():
        a.compute_logp([])
    calls.append(lambda: (model(**mb).logits[..., 0] * w).sum().backward())
    a.ppo_update([])
    assert not chunk_major._pending and not calls


def test_compute_logp_leaves_the_slab_to_ppo_update(model, monkeypatch):
    """compute_logp keeps chunk-major's pinned slab for the ppo_update after it; ppo_update hands it back to the OS."""
    monkeypatch.setattr(chunk_major, "ENABLED", True)
    released = []
    monkeypatch.setattr(chunk_major._host, "release", lambda: released.append(1))
    a = _actor_calls(monkeypatch, lambda: None)
    a.compute_logp([])
    assert not released
    a.ppo_update([])
    assert released == [1]


def test_chunk_major_needs_per_layer_checkpointing(monkeypatch):
    monkeypatch.setattr(chunk_major, "ENABLED", True)
    with pytest.raises(RuntimeError, match="OCTOPUS_BLOCK_CHECKPOINT=1"):
        actor.install_forward(torch.nn.Module(), 1.0, block_tokens=lambda tokens: None)

