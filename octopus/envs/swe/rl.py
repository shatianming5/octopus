"""AReaL inline agent for RL on plan episodes: one episode is one Harbor trial of a free-fork tree (main + branches)
from the bare issue, whose model calls go to this episode's AReaL proxy session; scored in
octopus/envs/swe/reward.py (pure outcome: R = 2y−1 on the final tree, one advantage for the whole tree, plus the optional
read counterfactual). Infrastructure failures (a failed model request, a harness exception, missing logs) raise, so
AReaL rejects the episode instead of training on it. The workflow glue it shares with search: octopus/rl/workflow.py.

data: {"task_dir": abs path of one task dir, "kind": "plan", "plan_control": rl.plan_control, ...} (config.plan_row;
octopus/envs/search/rl.py: "search", which registers its group step in GROUP_STEPS)
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import uuid
from pathlib import Path

from octopus.envs.swe.reward import plan_group_step, plan_outcome
from octopus.envs.swe.taskset import HARBOR_ENV, bundle_verifier_shell, plan_verifier_shell
from octopus.rl.config import live_env
from octopus.rl.workflow import CODE, GROUP_STEPS, HARBOR, EpisodeAgent, _run_group, trial_outcome

EPISODE_MEM = os.environ.get("OCTOPUS_EPISODE_MEM", "16G")  # host-memory cap of one Harbor episode (container included)

GROUP_STEPS["plan"] = plan_group_step  # kind=plan groups (workflow.plan_group_rewards)


def _ensure_plan_verifier(src: Path, dst: Path, bundled: bool) -> Path:
    """A copy of the task with the plan verifier, or the bundle one for a bundle (removed after the episode)."""
    shutil.copytree(src, dst)
    if bundled:  # reference fixes for the verifier's check validity (/tests is never shown to the agent)
        (dst / "tests/fixes").mkdir(parents=True, exist_ok=True)
        for fix in (src / "solution").glob("fix_*.patch"):
            shutil.copy(fix, dst / "tests/fixes" / fix.name)
    out = dst / "tests/test.sh"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(bundle_verifier_shell() if bundled else plan_verifier_shell())
    out.chmod(0o755)
    return dst


class Slots:
    """asyncio.Semaphore whose size can change while episodes hold slots (live EPISODES_PER_SERVER)."""

    def __init__(self, n: int):
        self.n, self.used, self.cond = n, 0, asyncio.Condition()

    async def resize(self, n: int) -> None:
        if n != self.n:
            print(f"[octopus] live EPISODES_PER_SERVER {self.n} -> {n} ({self.used} running)", flush=True)
            async with self.cond:
                self.n = n
                self.cond.notify_all()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.used < self.n)
            self.used += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.used -= 1
            self.cond.notify_all()


class HarborEpisodeAgent(EpisodeAgent):
    """EpisodeAgent for kind=plan rows: run_episode() runs the Harbor trial."""

    _slots: Slots | None = None  # per rollout-worker process

    async def run_episode(self, data: dict, base_url: str, api_key: str, session_id: str | None = None) -> dict:
        # unique per call: AReaL restarts session ids at 1:0-0 after every resume, and `harbor run` with an existing
        # job name returns that old job's result without running (~10% of episodes were stale reruns, then rejected)
        job = f"rl-{Path(data['task_dir']).name[:48]}-{(session_id or '')[:12]}-{uuid.uuid4().hex[:8]}"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        bundled = (Path(data["task_dir"]) / "tests/tasks.json").exists()  # the task dir is the one source (build_bundles)
        env = {
            **self.model_env(base_url, api_key),
            "OCTOPUS_STRATEGY": "branch_fix",
            "OCTOPUS_PLAN_EPISODE": "1",
            "OCTOPUS_BUNDLE": "1" if bundled else "",  # settings.bundle(); "" clears an inherited one
            # one no-fork continuation per forking episode (rl.plan_control, the row's: an evaluation's); "" clears it
            "OCTOPUS_PLAN_CONTROL": "1" if data["plan_control"] else "",
        }
        task_dir = str(_ensure_plan_verifier(Path(data["task_dir"]), self.jobs_dir / f"{job}-task", bundled))
        agent = "octopus.envs.swe.agent:GrokHarborAgent"
        command = " ".join([
            f"source {HARBOR}/scripts/env.sh && export PYTHONPATH={CODE}",
            *(f"&& export {k}={shlex.quote(v)}" for k, v in env.items()),
            # each episode in its own memory cgroup: a runaway command the policy starts (one leaked 100+ GB) is
            # OOM-killed there instead of squeezing the trainer and the other episodes (no user systemd in a Slurm job:
            # OCTOPUS_EPISODE_MEM= runs without one, the job's cgroup is the cap)
            f"&& cd {HARBOR} && exec",
            *([f"systemd-run --user --scope -q -p MemoryMax={EPISODE_MEM} -p MemorySwapMax=0"] if EPISODE_MEM else []),
            f"harbor run -p {shlex.quote(task_dir)}",
            f"-a {agent} -e {HARBOR_ENV}",
            f"-m {env['OCTOPUS_MODEL']} -n 1 -k 1 -o {self.jobs_dir} --job-name {job}",  # model_env's, as the search env's
            f"--agent-timeout-multiplier {self.agent_timeout_mult} --environment-build-timeout-multiplier 6 -y",
            # a forking episode's control tree is graded too: a bundle grades two whole trees (up to 900 s each)
            *(["--verifier-timeout-multiplier 2"] if data["plan_control"] else []),
        ])
        if HarborEpisodeAgent._slots is None:
            HarborEpisodeAgent._slots = Slots(self.parallel)
        if (live := live_env().get("EPISODES_PER_SERVER")) is not None:
            # within the launch value: rollout.max_concurrent_rollouts was sized from it (rl_run.sh)
            await HarborEpisodeAgent._slots.resize(min(max(int(live), 1), self.parallel))
        try:
            async with HarborEpisodeAgent._slots:
                with open(self.jobs_dir / f"{job}.console.log", "w") as console:
                    if (code := await _run_group(command, stdout=console, stderr=asyncio.subprocess.STDOUT)) != 0:
                        raise RuntimeError(f"harbor run exited {code} for {job}")
        finally:
            shutil.rmtree(self.jobs_dir / f"{job}-task", ignore_errors=True)
        return trial_outcome(self.jobs_dir / job, plan_outcome)
