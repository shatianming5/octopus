"""FSDP data-parallel balance for plan episodes.

AReaL's controller sends each trajectory dict to one data-parallel rank whole
(areal/infra/controller/train_controller.py _dispatch_tensors). A trajectory dict is one GRPO group: every
sequence of its n_samples episodes. The weight is rows x padded length (_item_weight on an RTensor) and
seqpack.balanced_greedy_partition gives every rank the same number of groups, so with 8 groups on 5 ranks
the three largest-by-padding ranks take two groups each and set the step time (bundle-e8f step 14: rank 0
2.14M tokens, rank 3 0.60M).

A plan group needs nothing group-wide on the training side: its advantages arrive per token
(octopus/rl/workflow.py plan_group_rewards -> octopus_adv, which ChunkedHeadFSDPPPOActor.compute_advantages
puts over AReaL's group adv_norm) and the loss divides by the token count all-reduced over the ranks
(areal/engine/core/train_engine.py compute_total_loss_weight). So each episode of an all-plan group is its own
dispatch unit, weighted by its prefix-tree tokens (what the actor computes: on bundle-e8f a micro-batch slot
takes time in proportion to its prefix-tree tokens, flat tokens add nothing), and ranks are filled longest first. An episode's sequences stay together, so its shared prefixes still run once (octopus/rl/prefix_tree.py).
Other groups (branch/join, or a plan group with an episode that has no outcome) stay whole: AReaL normalizes
their advantages within the group.

rollout worker  with_episode_tokens: an all-plan group's per-episode tree tokens -> traj[EPISODE_TOKENS]
controller      split_episodes: one item per episode, a shallow copy sharing the group's RTensors, with
                its rows (ROWS) and weight (TOKENS); patch_controller installs it and the dispatch
train worker    take_rows: each item's group tensors cut to its rows before AReaL concatenates the batch
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

import areal.infra.controller.train_controller as train_controller
from areal.utils.data import RolloutGroup, split_and_unpad_tensor

from octopus.rl import spool
from octopus.rl.prefix_tree import tree_tokens

EPISODE_TOKENS = "octopus_episode_tokens"  # group trajectory: [tree tokens per episode], all-plan groups only
TOKENS = "octopus_tokens"  # dispatch item: its tree tokens
ROWS = "octopus_rows"  # dispatch item: (start, end) rows of its group's tensors


def _ids(t: dict) -> np.ndarray:
    return t["input_ids"][0][t["attention_mask"][0].bool()].numpy()


def with_episode_tokens(concat):
    """Wrap AReaL's concat_tensor_interactions (one rollout's rows contiguous, rollouts in first-seen order)."""

    def concat_with_tokens(interactions: dict) -> dict:
        result = concat(interactions)
        if interactions and all(getattr(v, "octopus_plan", 0.0) for v in interactions.values()):
            members: dict[int, list] = {}
            for v in interactions.values():
                members.setdefault(v.rollout_index if v.rollout_index is not None else 0, []).append(v)
            result[EPISODE_TOKENS] = [tree_tokens([_ids(v.to_tensor_dict()) for v in m]) for m in members.values()]
        return result

    return concat_with_tokens


def split_episodes(batch: list[dict]) -> list[dict]:
    out = []
    for d in batch:
        tokens = d.pop(EPISODE_TOKENS, None)
        if tokens is None:
            out.append(d)
            continue
        group, start = d["rollout_group"], 0
        for n, reward, t in zip(group.row_counts, group.rewards, tokens, strict=True):
            out.append({**d, "rollout_group": RolloutGroup((n,), (reward,)), ROWS: (start, start + n), TOKENS: t})
            start += n
    return out


def take_rows(data: list[dict]) -> list[dict]:
    """Train worker: drop the dispatch keys and cut each episode item's group-sized tensors to its rows (a tensor
    the controller attached per item, e.g. prox_logp, already has the item's rows), then trim the padding past
    its longest row as RTensor.remotize trims a whole trajectory. Every item then has its own sequence length
    as width, which is what the engine's forward output is cut back to (reorder_and_pad_outputs pads to the
    rank's longest sequence, batched_call trims to the item's width): kept at the group's width, prox_logp
    came back narrower than logprobs and log_train_inference_stats raised (PR #69's first deployment)."""
    out = []
    for d in data:
        d.pop(TOKENS, None)
        rows = d.pop(ROWS, None)
        if rows is not None:
            start, end = rows
            n = d["attention_mask"].shape[0]
            d = {k: v[start:end] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == n else v
                 for k, v in d.items()}
            d = split_and_unpad_tensor(d, n_trajs=1, traj_group_sizes=[end - start])[0]
        out.append(d)
    return out


def longest_first(nums: list[int], K: int) -> list[list[int]]:
    """Replaces seqpack.balanced_greedy_partition (equal item counts per rank): each item, largest first, goes
    to the least-loaded rank. The first K items open the K ranks, so none is empty when len(nums) >= K."""
    heap, parts = [(0, k) for k in range(K)], [[] for _ in range(K)]
    for i in sorted(range(len(nums)), key=lambda i: -nums[i]):
        load, k = heapq.heappop(heap)
        parts[k].append(i)
        heapq.heappush(heap, (load + nums[i], k))
    return parts


_item_weight = train_controller._item_weight


def patch_controller(actor) -> None:
    """Controller process: spooled groups a step takes are marked trained (octopus/rl/spool.py), episode items out of
    prepare_batch, their tree tokens as the dispatch weight (carried
    onto compute_advantages' results so ppo_update lands on the same ranks), longest-first partition."""
    train_controller._item_weight = lambda d: d[TOKENS] if TOKENS in d else _item_weight(d)
    train_controller.balanced_greedy_partition = longest_first
    prepare, advantages = actor.prepare_batch, actor.compute_advantages

    def prepare_batch(*args, **kwargs):
        batch = prepare(*args, **kwargs)
        spool.consume(batch)  # a step took these groups: gone once a recover checkpoint covers it
        return split_episodes(batch)

    def compute_advantages(batch, *args, **kwargs):
        result = advantages(batch, *args, **kwargs)
        for item, d in zip(result, batch, strict=True):
            if TOKENS in d:
                item[TOKENS] = d[TOKENS]
        return result

    actor.prepare_batch, actor.compute_advantages = prepare_batch, compute_advantages
