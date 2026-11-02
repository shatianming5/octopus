"""The AReaL workflow glue every environment's RL agent shares: the episode agent's base (EpisodeAgent: the model
endpoint and request settings, R = 2y−1), each finished episode's record (trial_outcome), the group step per row kind
(GROUP_STEPS, plan_group_rewards), the per-token export (to_tensor_dict with the advantage and reads), the rollout spool
and has_variance (train.py's dynamic_filter_fn). Importing it patches AReaL's workflow classes. The environments' agents:
octopus/envs/swe/rl.py (HarborEpisodeAgent), octopus/envs/search/rl.py (SearchEpisodeAgent).
"""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import os
import signal
from pathlib import Path

import torch
import areal.infra.workflow_executor as workflow_executor
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils.hf_utils import load_hf_tokenizer

from octopus.rl import spool
from octopus.rl.dp_balance import with_episode_tokens
from octopus.rl.prefix_tree import read_tensors
from octopus.rl.run_config import episode_env
from octopus.rl.group import assign_threads, plan_group_stats
from octopus.runtime.settings import cursor_mode

HARBOR = Path(os.environ.get("OCTOPUS_HARBOR", "/home/user/harbor-udocker"))
CODE = Path(os.environ.get("OCTOPUS_CODE", str(HARBOR / "code-v2")))
STOP_GRACE = 15  # seconds a cancelled episode's `harbor run` gets on SIGINT (it removes its container in <1 s, then waits on
# the agent's blocked LLM request thread) before SIGKILL


def trial_outcome(job_dir: Path, build) -> dict:
    """The episode record (group.py's contract) of a finished single-trial job, by its environment's builder
    (octopus/envs/swe/reward.plan_outcome; search: octopus/envs/search/reward.search_outcome); raises on infrastructure
    failure."""
    trial, requests = _trial_requests(job_dir)
    return build(trial, requests)


def _trial_requests(job_dir: Path) -> tuple[Path, list[dict]]:
    trials = [p for p in job_dir.iterdir() if p.is_dir()]
    if len(trials) != 1:
        raise RuntimeError(f"expected one trial in {job_dir}, found {len(trials)}")
    trial = trials[0]
    result = json.loads((trial / "result.json").read_text())
    if result.get("exception_info"):
        raise RuntimeError(f"harness exception: {result['exception_info'].get('exception_type')}")
    requests = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    if any(r.get("http_code") != 200 for r in requests):
        raise RuntimeError("a model request failed")
    return trial, requests


# A row kind's group step (plan_group_rewards), registered by the workflow module AReaL imports: kind=plan's by
# octopus/envs/swe/rl.py, kind=search's by octopus/envs/search/rl.py (rl imports no environment's RL glue)
GROUP_STEPS: dict = {}

# Each episode's outcome by proxy session id, until the workflow exports the episode and stamps it on its rows; the
# group step then sets each row's reward and per-token advantage. Per token ([1, L]), because AReaL splits and packs
# only [B, L] tensors into micro-batches; a [B] tensor would reach every micro-batch whole.
OUTCOME: dict[str, dict] = {}
_stamp = OpenAIProxyWorkflow._stamp_interaction_metadata
_to_tensor_dict = InteractionWithTokenLogpReward.to_tensor_dict
_group_arun_episode = GroupedRolloutWorkflow.arun_episode


def _stamp_outcome(interactions, metadata) -> None:
    _stamp(interactions, metadata)
    outcome = OUTCOME.pop(metadata.get("session_id"), None)
    for interaction in interactions.values():
        interaction.octopus_outcome = outcome


def _to_tensor_dict_with_adv(self) -> dict:
    result = _to_tensor_dict(self)  # cached per interaction: adding the key once keeps it
    # the leaf's reads (prefix_tree.record_reads), on every row of a cursor run: only it can mount, and the episode's
    # read tool is the episode's own setting, so the export never asks it (a reader recomputed without its mounts)
    if cursor_mode() and "octopus_node" not in result:
        result.update(read_tensors(self.interaction_id, self.metadata.get("octopus_reads", ()), result["input_ids"].shape))
    if getattr(self, "octopus_plan", 0.0):
        _apply_plan(result, self)
        return result
    for key in ("octopus_plan", "octopus_adv"):
        result.setdefault(key, torch.zeros(result["input_ids"].shape))
    return result


def _apply_plan(result: dict, interaction) -> None:
    """A plan leaf's reward and per-token advantage; loss mask 0 on the inherited prefix."""
    shape = result["input_ids"].shape
    result["rewards"] = torch.tensor([float(interaction.reward)])
    result["original_rewards"] = torch.tensor([float(interaction.reward)])
    for key in ("octopus_plan", "octopus_adv"):
        result[key] = torch.full(shape, float(getattr(interaction, key)))
    for start, end, value in getattr(interaction, "octopus_spans", None) or []:  # read credit, its own tokens
        result["octopus_adv"].view(-1)[start:end] += value
    result["loss_mask"][..., : interaction.octopus_own_start] = 0


@functools.lru_cache(maxsize=2)
def _token_text(tokenizer_path: str):
    """token id -> its text (the read_thread call spans of group.assign_threads)."""
    tokenizer = load_hf_tokenizer(tokenizer_path)
    return [tokenizer.decode([i]) for i in range(len(tokenizer))].__getitem__


def plan_group_rewards(merged: dict, data: dict) -> None:
    """Leave-one-out A_main (every branch on its episode's) and read credit for a plan (or search) GRPO group; bypasses
    group adv_norm later."""
    if not merged or data.get("kind") not in GROUP_STEPS:
        return
    by_rollout = collections.defaultdict(list)
    for interaction in merged.values():
        by_rollout[interaction.rollout_index].append(interaction)
    episodes, order = [], []
    for idx in sorted(by_rollout):
        rows = by_rollout[idx]
        outcome = next((x.octopus_outcome for x in rows if getattr(x, "octopus_outcome", None)), None)
        if not outcome:
            continue
        episodes.append(outcome)  # the episode's record (group.py's contract): the group step fills it in
        order.append(rows)
    GROUP_STEPS[data["kind"]](episodes, branch_credit=data.get("branch_credit", "outcome"))
    stats = plan_group_stats(episodes)
    if unparsed := sum(e.get("judge_unparsed") or 0 for e in episodes):  # search: answers graded without a verdict
        stats["judge_unparsed"] = unparsed
    print(f"[octopus] plan group {json.dumps(stats)}", flush=True)
    if cursor_mode():  # scripts/rl_analysis/read_audit.py: the group's exported leaves, and those that carry a read
        print(f"[octopus] read leaves {len(merged)} with reads "
              f"{sum(bool(x.metadata.get('octopus_reads')) for x in merged.values())}", flush=True)
    token_text = _token_text(data["tokenizer_path"]) if any(e.get("read_adv") for e in episodes) else None
    for rows, episode in zip(order, episodes):
        for interaction, row in zip(rows, assign_threads(rows, episode, token_text)):
            interaction.reward = row["reward"]
            interaction.octopus_plan = 1.0
            interaction.octopus_adv = row["adv"]
            interaction.octopus_own_start = row["own_start"]
            interaction.octopus_spans = row["spans"]
            _apply_plan(interaction._cache, interaction)  # deserialized interactions carry only the tensors


async def _group_arun_with_rewards(self, engine, data):
    # spool file I/O (a group's tensors) runs off the event loop that drives every episode of this worker
    if spool.DIR and (traj := await asyncio.to_thread(spool.take, engine.get_version(),
                                                       engine.config.max_head_offpolicyness,
                                                       Path(data["task_dir"]).name)) is not None:
        # a group the previous trainer process finished but never trained on, or {} -> None: AReaL rejects the call,
        # whose task's group is already out (octopus/rl/spool.py)
        return traj or None
    merged = await _group_arun_episode(self, engine, data)
    plan_group_rewards(merged, data)
    if spool.DIR and merged and has_variance(traj := _concat(merged)):  # else rl.filter_zero_variance drops it
        next(iter(merged.values())).octopus_spool = await asyncio.to_thread(spool.save, traj, Path(data["task_dir"]).name)
    return merged


def _concat_with_spool(interactions: dict) -> dict:
    """The group's tensor dict, carrying its spool file for the controller to mark once a step takes it."""
    result = _concat(interactions)
    if path := getattr(next(iter(interactions.values()), None), "octopus_spool", None):
        result[spool.KEY] = path
    return result


OpenAIProxyWorkflow._stamp_interaction_metadata = staticmethod(_stamp_outcome)
InteractionWithTokenLogpReward.to_tensor_dict = _to_tensor_dict_with_adv
GroupedRolloutWorkflow.arun_episode = _group_arun_with_rewards
# per-episode tree tokens on all-plan groups: the controller dispatches their episodes separately (dp_balance.py)
_concat = with_episode_tokens(workflow_executor.concat_tensor_interactions)
workflow_executor.concat_tensor_interactions = _concat_with_spool


def has_variance(group) -> bool:
    """dynamic_filter_fn: a group whose per-token advantages (plan_group_rewards: A and any read credit) are all 0
    carries no signal."""
    return bool(group["octopus_adv"].abs().amax() > 0)


class EpisodeAgent:
    """AReaL `rollout.agent.mode=inline` agent: run() runs the episode (an environment's run_episode) and returns its
    reward."""

    def __init__(self, jobs_dir: str | None = None, agent_timeout_mult: int = 72, parallel: int = 8, *,
                 temperature: float, max_new_tokens: int, chat_template_kwargs: dict, top_p: float | None = None, **_):
        self.jobs_dir = Path(jobs_dir or HARBOR / "rl/jobs")
        self.agent_timeout_mult = agent_timeout_mult
        self.parallel = parallel  # AReaL starts every episode of a batch (and the prefetched next one) at once
        self.temperature, self.max_new_tokens = temperature, max_new_tokens  # request_kwargs
        self.top_p = top_p  # an evaluation's (search_dev_eval); training's requests carry none
        self.chat_template_kwargs = chat_template_kwargs

    def model_env(self, base_url: str, api_key: str) -> dict:
        """The episode's model endpoint and per-request settings (run_config.request_kwargs). The model name is any:
        the proxy serves its engine's adapter version."""
        return {"OCTOPUS_CHAT_URL": base_url.rstrip("/") + "/chat/completions", "OCTOPUS_API_KEY": api_key,
                **episode_env("policy", self.temperature, self.max_new_tokens, self.chat_template_kwargs, self.top_p)}

    async def run(self, data: dict, **kwargs) -> float:
        outcome = await self.run_episode(data, kwargs["base_url"], kwargs["api_key"], kwargs.get("session_id"))
        if kwargs.get("session_id"):
            OUTCOME[kwargs["session_id"]] = outcome
        return 2.0 * float(outcome["y"]) - 1.0  # R = 2y−1


async def _run_group(command: str, **kw) -> int:
    """`bash -c command` in its own process group; its exit code. Cancelled (AReaL's group gather cancels the sibling
    episodes when one fails), it stops the group before re-raising, so the caller never removes the task copy under a
    `harbor run` that goes on (cancelling the coroutine leaves the child running). SIGINT first: Harbor cancels its
    trial and its environment removes the container; SIGKILL after STOP_GRACE seconds and for whatever is left."""
    proc = await asyncio.create_subprocess_exec("bash", "-c", command, start_new_session=True, **kw)
    try:
        return await proc.wait()
    except asyncio.CancelledError:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            await asyncio.wait_for(proc.wait(), STOP_GRACE)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
