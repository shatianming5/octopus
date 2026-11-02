"""A cancelled RL episode stops its `harbor run` process group before removing the task copy."""

import asyncio
import os
import time

import pytest

import octopus.envs.swe.rl as swe_rl
import octopus.rl.workflow as workflow


def test_cancelled_episode_stops_its_harbor_run_then_removes_the_task_copy(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "scripts/env.sh").write_text(f"export PATH={tmp_path}/bin:$PATH\n")
    # `harbor run -p TASK ...` that keeps using its task copy, with a child in its process group (a background job
    # of a script ignores SIGINT: only the group SIGKILL ends it); SIGINT is recorded with whether the copy is still there
    harbor = tmp_path / "bin/harbor"
    harbor.write_text(f"""#!/bin/bash
trap 'test -e "$3/tests/test.sh" && echo copy-present > {tmp_path}/sigint; exit 130' INT
sleep 300 &
echo $$ > {tmp_path}/pids
wait
""")
    harbor.chmod(0o755)
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("fix it\n")
    monkeypatch.setattr(swe_rl, "HARBOR", tmp_path)
    monkeypatch.setattr(swe_rl, "EPISODE_MEM", "")
    monkeypatch.setattr(workflow, "STOP_GRACE", 5)
    monkeypatch.setattr(swe_rl.HarborEpisodeAgent, "_slots", None)

    async def cancel_running_episode():
        agent = swe_rl.HarborEpisodeAgent(jobs_dir=str(tmp_path / "jobs"), temperature=1.0, max_new_tokens=8192,
                                                chat_template_kwargs={})
        episode = asyncio.create_task(agent.run_episode({"task_dir": str(task), "kind": "plan", "plan_control": False},
                                                       "http://127.0.0.1:9/v1", "k"))
        while not (tmp_path / "pids").exists():
            await asyncio.sleep(0.05)
        (copy,) = (tmp_path / "jobs").glob("*-task")
        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode
        return copy

    copy = asyncio.run(cancel_running_episode())
    pgid = int((tmp_path / "pids").read_text())
    assert (tmp_path / "sigint").read_text().strip() == "copy-present"  # SIGINT first, while the copy still existed
    assert not copy.exists()
    deadline = time.time() + 5
    while time.time() < deadline:  # the SIGKILLed child is reaped by init
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
