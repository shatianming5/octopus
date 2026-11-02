"""AReaL integration of search episodes: the same trainer as code bundles (octopus/rl/train.py) with
workflow=octopus.envs.search.rl.SearchEpisodeAgent and rl.search_questions set.

One row per question (kind=search); each of the group's G = gconfig.n_samples episodes is one run of
octopus.envs.search.agent in the Harbor environment (a subprocess, like a Harbor trial: its HTTP calls stay off the
rollout worker's event loop). Its record is octopus.envs.search.reward.search_outcome's (octopus/rl/group.py's
contract); the group's advantages are octopus.envs.search.reward.search_group_step's (workflow.plan_group_rewards).
"""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from pathlib import Path

from datasets import Dataset

from octopus.rl.config import row_settings
from octopus.rl.workflow import CODE, GROUP_STEPS, HARBOR, EpisodeAgent, _run_group, trial_outcome
from octopus.envs.search.reward import search_group_step, search_outcome

GROUP_STEPS["search"] = search_group_step  # kind=search groups (workflow.plan_group_rewards)
MAX_NEW_TOKENS = 4096  # a search reply's cap, training and evaluation alike (CLM's), whatever gconfig.max_new_tokens
EVAL_SAMPLING = {"temperature": 0.7, "top_p": 0.95}  # evaluation only (search_dev_eval, CLM's); training samples at
# gconfig.temperature (1.0) and top_p 1.0, which the trainer's logprob recompute assumes


def search_row(config, q: dict) -> dict:
    """The kind=search row of question q ({qid, question, answer}), as training builds it."""
    return {"task_dir": f"search-{q['qid']}", "kind": "search", "qid": str(q["qid"]), "question": q["question"],
            "answer": str(q["answer"]), **row_settings(config)}


def search_dataset(config) -> Dataset:
    """rl.search_questions: a JSONL of {qid, question, answer}; rl.search_split (optional): a file of the qids to keep
    (whitespace-separated, e.g. part A's training split)."""
    cfg = config.rl
    rows = [json.loads(line) for line in Path(cfg.search_questions).read_text().splitlines() if line.strip()]
    if cfg.search_split:
        keep = set(Path(cfg.search_split).read_text().split())
        rows = [r for r in rows if str(r["qid"]) in keep]
    return Dataset.from_list([search_row(config, r) for r in rows])


class SearchEpisodeAgent(EpisodeAgent):
    """EpisodeAgent for kind=search rows: run() stores the outcome for the group step and returns R = 2y-1. Its replies
    are capped at MAX_NEW_TOKENS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "max_new_tokens": MAX_NEW_TOKENS})

    async def run_episode(self, data: dict, base_url: str, api_key: str, session_id: str | None = None) -> dict:
        job = f"search-{data['qid'][:48]}-{(session_id or '')[:12]}-{uuid.uuid4().hex[:8]}"  # unique per call
        job_dir = self.jobs_dir / job
        job_dir.mkdir(parents=True)
        question = self.jobs_dir / f"{job}.question.json"
        question.write_text(json.dumps({k: data[k] for k in ("qid", "question", "answer")}))
        env = {**self.model_env(base_url, api_key), "OCTOPUS_PLAN_CONTROL": "1" if data["plan_control"] else ""}
        command = " ".join([
            f"source {HARBOR}/scripts/env.sh && export PYTHONPATH={CODE}",
            *(f"&& export {k}={shlex.quote(v)}" for k, v in env.items()),
            # $OCTOPUS_SEARCH_PYTHON, else the python next to the Harbor environment's harbor CLI (not there when
            # harbor is a uv tool shim: set OCTOPUS_SEARCH_PYTHON to the interpreter of harbor's venv)
            "&& exec \"${OCTOPUS_SEARCH_PYTHON:-$(dirname \"$(command -v harbor)\")/python}\""
            f" -m octopus.envs.search.agent {question} {job_dir}",
        ])
        try:
            with open(self.jobs_dir / f"{job}.console.log", "w") as console:
                if (code := await _run_group(command, stdout=console, stderr=asyncio.subprocess.STDOUT)) != 0:
                    raise RuntimeError(f"search episode exited {code} for {job}")
        finally:
            question.unlink(missing_ok=True)
        return trial_outcome(job_dir, search_outcome)
