"""UdockerEnvironment.exec runs one process per command, here a local bash in place of `udocker run`: without
timeout_sec it waits for the command (the caller bounds it, as Harbor does for the verifier); an explicit timeout_sec
kills the command's whole process group, and concurrent commands do not wait for each other."""
import asyncio, threading, time

import pytest

pytest.importorskip("harbor")

from octopus.envs.swe import udocker as ue  # noqa: E402


def make_env(tmp_path):
    env = object.__new__(ue.UdockerEnvironment)
    env._ipc, env._procs, env._proot_env = tmp_path, set(), {}
    env._stopping, env._state_lock = threading.Event(), threading.Lock()
    env._command_argv = lambda cmd: ["/bin/bash", "-c", cmd.replace("/logs/udocker_ipc", str(tmp_path))]
    return env


def test_no_timeout_waits_for_the_command(tmp_path):
    r = asyncio.run(make_env(tmp_path).exec("sleep 1; echo out; echo err >&2; exit 3", cwd=str(tmp_path)))
    assert (r.return_code, r.stdout, r.stderr) == (3, "out\n", "err\n")
    assert not list(tmp_path.glob("c-*.sh"))  # the script is removed


def test_timeout_kills_the_process_group(tmp_path):
    mark = tmp_path / "late"
    r = asyncio.run(make_env(tmp_path).exec(f"(sleep 3; touch {mark}) & sleep 30", cwd=str(tmp_path), timeout_sec=1))
    assert r.return_code == 124
    time.sleep(3.5)
    assert not mark.exists()  # the background child died with the command


def test_concurrent_commands_overlap(tmp_path):
    env = make_env(tmp_path)

    async def main():
        return await asyncio.gather(*(env.exec(f"sleep 1; echo {i}", cwd=str(tmp_path)) for i in range(8)))
    t = time.time()
    results = asyncio.run(main())
    assert time.time() - t < 4
    assert [r.stdout for r in results] == [f"{i}\n" for i in range(8)]


def test_stop_kills_running_commands(tmp_path):
    env = make_env(tmp_path)
    env._cid, env._started = "", True

    async def main():
        task = asyncio.create_task(env.exec("sleep 30", cwd=str(tmp_path)))
        await asyncio.sleep(0.5)
        await env.stop(delete=False)
        return await task
    t = time.time()
    assert asyncio.run(main()).return_code == 137
    assert time.time() - t < 10


def test_stop_does_not_queue_behind_the_commands_holding_the_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(ue, "_POOL", ue.ThreadPoolExecutor(max_workers=1))
    env = make_env(tmp_path)
    env._cid, env._started = "", True

    async def main():
        task = asyncio.create_task(env.exec("sleep 30", cwd=str(tmp_path)))  # holds the only exec thread
        await asyncio.sleep(0.5)
        t = time.time()
        await env.stop(delete=False)
        return time.time() - t, await task
    took, result = asyncio.run(main())
    assert took < 10 and result.return_code == 137


def test_command_argv_is_udockers_proot():
    env = object.__new__(ue.UdockerEnvironment)
    env._proot = ["/x/proot", "-r", "/x/ROOT"]  # _capture_proot: what `udocker run` would exec
    assert env._command_argv("true") == ["/x/proot", "-r", "/x/ROOT", "/bin/bash", "-c", "true"]


def test_commands_get_udockers_environment_not_ours(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "secret")
    env = make_env(tmp_path)
    env._proot_env = {"PATH": "/usr/bin:/bin", "FROM_LOGIN": "1"}
    out = asyncio.run(env.exec("env", cwd=str(tmp_path))).stdout
    assert "FROM_LOGIN=1" in out and "XAI_API_KEY" not in out
