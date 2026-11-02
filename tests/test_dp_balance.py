"""Plan episodes as data-parallel dispatch units, balanced by prefix-tree tokens (octopus/rl/dp_balance.py)."""

from types import SimpleNamespace

import numpy as np
import torch

import areal.infra.controller.train_controller as train_controller
from areal.experimental.openai.types import concat_tensor_interactions
from areal.trainer.ppo.stats import log_train_inference_stats
from areal.utils.data import batched_call, concat_batch

from octopus.rl import dp_balance

rng = np.random.default_rng(0)


def _interaction(ids, rollout_index, plan=1.0):
    t = {"input_ids": torch.tensor(ids)[None], "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
         "loss_mask": torch.ones(1, len(ids), dtype=torch.long), "logprobs": -torch.rand(1, len(ids)),
         "rewards": torch.tensor([float(rollout_index)]),
         "octopus_adv": torch.full((1, len(ids)), float(rollout_index))}
    return SimpleNamespace(to_tensor_dict=lambda: t, rollout_index=rollout_index, rollout_reward=None, octopus_plan=plan)


def _seq(n, prefix=()):
    return list(prefix) + rng.integers(1000, 2000, n - len(prefix)).tolist()


def _group(episodes, plan=1.0):
    """episodes: list of lists of sequences -> AReaL's group trajectory, through the rollout-side wrapper."""
    interactions = {f"{i}-{j}": _interaction(s, i, plan) for i, seqs in enumerate(episodes) for j, s in enumerate(seqs)}
    return dp_balance.with_episode_tokens(concat_tensor_interactions)(interactions)


def test_tree_tokens_counts_shared_prefix_once():
    p = _seq(100)
    assert dp_balance.tree_tokens([np.array(_seq(300, p)), np.array(_seq(250, p)), np.array(_seq(200, p))]) == 550
    assert dp_balance.tree_tokens([np.array([1, 2, 3])]) == 3


def test_episodes_dispatch_alone_balanced_and_rows_restored(monkeypatch):
    monkeypatch.setattr(train_controller, "_item_weight", train_controller._item_weight)
    monkeypatch.setattr(train_controller, "balanced_greedy_partition", train_controller.balanced_greedy_partition)
    p = _seq(100)
    big = _group([[_seq(300, p), _seq(250, p), _seq(200, p)], [_seq(120)], [_seq(80)], [_seq(60)]])
    even = _group([[_seq(150)] for _ in range(4)])
    other = _group([[_seq(100)], [_seq(100)]], plan=0.0)  # not all plan: AReaL normalizes it as a group
    assert big[dp_balance.EPISODE_TOKENS] == [550, 120, 80, 60]
    assert dp_balance.EPISODE_TOKENS not in other
    rows = {k: big[k].clone() for k in ("input_ids", "octopus_adv")}

    trained = []
    actor = SimpleNamespace(prepare_batch=lambda *a, **k: [big, even, other],
                            compute_advantages=lambda batch, **k: [{"advantages": None} for _ in batch])
    dp_balance.patch_controller(actor)
    items = actor.prepare_batch()
    assert len(items) == 4 + 4 + 1
    splits, _ = train_controller._dispatch_tensors(items, dp_size=3)
    loads = [sum(train_controller._item_weight(d) for d in s) for s in splits]
    # whole groups put the 810-token group on one rank; episodes reach the 550-token floor (the largest episode)
    assert max(loads) == 550 and sum(loads) == 810 + 600 + 200
    assert all(dp_balance.TOKENS in a for a, d in zip(actor.compute_advantages(items), items) if dp_balance.TOKENS in d)

    def engine_forward(batch):  # FSDPEngine.forward_batch output: padded to the rank batch's longest sequence
        return torch.zeros(batch["attention_mask"].shape[0], int(batch["attention_mask"].sum(-1).max()))

    for split in splits:  # recompute_logp on each rank (the RPC hands the worker its own copies)
        for d, logp in zip(split, batched_call(engine_forward, dp_balance.take_rows([dict(d) for d in split]))):
            d["prox_logp"] = logp  # the controller attaches it per item
    splits, _ = train_controller._dispatch_tensors(items, dp_size=3)
    for split in splits:  # compute_advantages
        local = dp_balance.take_rows([dict(d) for d in split])
        batch, meta = concat_batch(local)  # validates each item's rollout_group against its rows
        # raised in the live run when a cut item kept its group's width (prox_logp came back narrower)
        log_train_inference_stats(batch["prox_logp"], torch.roll(batch["logprobs"], -1, -1), batch["loss_mask"].bool())
        for i, (src, d) in enumerate(zip(split, local)):
            if src is not other:
                start, end = src[dp_balance.ROWS]
                width = int(src["attention_mask"][start:end].sum(-1).max())
                if src["input_ids"] is big["input_ids"]:
                    assert torch.equal(d["input_ids"], rows["input_ids"][start:end, :width])
                    assert torch.equal(d["octopus_adv"], rows["octopus_adv"][start:end, :width])
                assert meta.traj_group_sizes[i] == end - start and d["input_ids"].shape[1] == width
        trained += local
    assert sum(d["input_ids"].shape[0] for d in trained) == 6 + 4 + 2
