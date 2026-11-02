"""One source per episode setting. The rollout worker exports a cursor run's reads whatever the episode's
OCTOPUS_READ_TOOL (the env file's, never search's DEFAULTS); every episode request carries the run config's
temperature, reply cap and chat-template switches; rl_run.sh routes
sessions over the servers it started."""

import ast
import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEAF, SOURCE = "chatcmpl-" + "a" * 29, "chatcmpl-" + "b" * 29


def test_cursor_run_exports_reads_without_the_read_tool(monkeypatch):
    """A leaf with a read exports it in a cursor run with OCTOPUS_READ_TOOL unset in the worker (the trainer would
    otherwise recompute the reader without its mount); outside cursor mode nothing can mount and nothing is added."""
    pytest.importorskip("areal")
    import torch

    import octopus.rl.workflow as workflow
    from octopus.rl.prefix_tree import MOUNT_KEYS, node_id

    leaf = SimpleNamespace(interaction_id=LEAF, metadata={"octopus_reads": [(3, SOURCE, 0, 2)]})
    monkeypatch.setattr(workflow, "_to_tensor_dict", lambda self: {"input_ids": torch.ones(1, 6, dtype=torch.long)})
    monkeypatch.delenv("OCTOPUS_READ_TOOL", raising=False)
    monkeypatch.setenv("OCTOPUS_CURSOR", "1")
    out = workflow._to_tensor_dict_with_adv(leaf)
    assert int(out["octopus_node"][0, 0]) == node_id(LEAF)
    assert [int(out[k][0, 3]) for k in MOUNT_KEYS[1:]] == [node_id(SOURCE), 0, 2]
    monkeypatch.delenv("OCTOPUS_CURSOR")
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    assert not set(MOUNT_KEYS) & set(workflow._to_tensor_dict_with_adv(leaf))


def test_search_defaults_leave_the_read_tool_to_the_env_file():
    tree = ast.parse((ROOT / "octopus/envs/search/defaults.py").read_text())
    defaults = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DEFAULTS")
    assert "OCTOPUS_READ_TOOL" not in defaults


def test_episode_requests_carry_the_run_config(monkeypatch, tmp_path):
    """The episode's temperature, reply cap and chat-template switches are the config's (train.py: request_kwargs),
    not literals; grpo.yaml's trainer recomputes logprobs at the temperature its requests carry."""
    pytest.importorskip("areal")
    import octopus.envs.swe.rl as swe_rl
    from octopus.rl.run_config import CONFIG, request_kwargs

    config = SimpleNamespace(gconfig=SimpleNamespace(temperature=0.7, max_new_tokens=123),
                             rollout=SimpleNamespace(agent=SimpleNamespace(chat_template_kwargs={"enable_thinking": True})))
    commands = []

    async def harbor_run(command, **kw):
        commands.append(command)
        return 0
    monkeypatch.setattr(swe_rl, "_run_group", harbor_run)
    monkeypatch.setattr(swe_rl, "trial_outcome", lambda *a: {"reward": 0.0})
    monkeypatch.setattr(swe_rl.HarborEpisodeAgent, "_slots", None)
    (tmp_path / "task").mkdir()
    agent = swe_rl.HarborEpisodeAgent(jobs_dir=str(tmp_path / "jobs"), **request_kwargs(config))
    asyncio.run(agent.run_episode({"task_dir": str(tmp_path / "task"), "kind": "plan", "plan_control": False}, "http://h/v1", "k"))
    env = dict(t.split("=", 1) for t in shlex.split(commands[0]) if t.startswith("OCTOPUS_"))
    assert json.loads(env["OCTOPUS_SAMPLING_JSON"]) == {"temperature": 0.7}
    assert env["OCTOPUS_MAX_TOKENS"] == "123" and json.loads(env["OCTOPUS_CHAT_TEMPLATE_KWARGS"]) == {"enable_thinking": True}

    # the trainer recomputes logprobs at the temperature its requests carry
    assert yaml.safe_load(CONFIG.read_text())["actor"]["temperature"] == "${gconfig.temperature}"


@pytest.mark.parametrize("bundled", [True, False])
def test_the_task_dir_decides_bundle_whatever_the_environment(monkeypatch, tmp_path, bundled):
    """A bundle task dir (tests/tasks.json) gets the bundle verifier and OCTOPUS_BUNDLE=1 with the variable unset here;
    a single task gets the plan verifier and OCTOPUS_BUNDLE cleared with it set (an old env file's export)."""
    pytest.importorskip("areal")
    import octopus.envs.swe.rl as swe_rl
    from octopus.envs.swe.taskset import bundle_verifier_shell, plan_verifier_shell

    seen = []

    async def harbor_run(command, **kw):
        task, = (tmp_path / "jobs").glob("*-task")
        seen.append((command, (task / "tests/test.sh").read_text()))
        return 0
    monkeypatch.setattr(swe_rl, "_run_group", harbor_run)
    monkeypatch.setattr(swe_rl, "trial_outcome", lambda *a: {"y": 0.0})
    monkeypatch.setattr(swe_rl.HarborEpisodeAgent, "_slots", None)
    if bundled:
        monkeypatch.delenv("OCTOPUS_BUNDLE", raising=False)
    else:
        monkeypatch.setenv("OCTOPUS_BUNDLE", "1")
    (tmp_path / "task/tests").mkdir(parents=True)
    if bundled:
        (tmp_path / "task/tests/tasks.json").write_text(json.dumps({"1": {"f2p": [], "p2p": []}}))
    agent = swe_rl.HarborEpisodeAgent(jobs_dir=str(tmp_path / "jobs"), temperature=1.0, max_new_tokens=8,
                                            chat_template_kwargs={})
    asyncio.run(agent.run_episode({"task_dir": str(tmp_path / "task"), "plan_control": False}, "http://h/v1", "k"))
    (command, verifier), = seen
    assert f"export OCTOPUS_BUNDLE={shlex.quote('1' if bundled else '')} " in command
    assert verifier == (bundle_verifier_shell() if bundled else plan_verifier_shell())


def test_route_servers_are_the_servers_rl_run_started():
    assert "export OCTOPUS_ROUTE_SERVERS=$SERVERS\n" in (ROOT / "scripts/rl_env/rl_run.sh").read_text()
    assert not any("OCTOPUS_ROUTE_SERVERS=" in p.read_text() for p in (ROOT / "scripts/rl_env").glob("*.env"))
