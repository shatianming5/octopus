"""A forked AReaL worker may answer /health only after 60 s (octopus/rl/train.py fork_ready_timeout)."""

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

local = pytest.importorskip("areal.infra.scheduler.local")
from octopus.rl.train import fork_ready_timeout


def test_fork_wait_outlasts_a_slow_import(monkeypatch):
    clock = [0.0]

    async def sleep(s):
        clock[0] += s

    class Resp:
        status = 200
        async def __aenter__(self):
            if clock[0] < 90:  # still importing: nothing listens yet
                raise aiohttp.ClientConnectionError()
            return self
        async def __aexit__(self, *a):
            return False

    session = SimpleNamespace(get=lambda url, timeout: Resp())
    monkeypatch.setattr(local, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(local, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(local.LocalScheduler, "_wait_for_fork_ready",
                        local.LocalScheduler.__dict__["_wait_for_fork_ready"])  # restored after the test

    sched = local.LocalScheduler.__new__(local.LocalScheduler)  # called on an instance, as fork_workers does
    assert not asyncio.run(sched._wait_for_fork_ready(session, "10.0.0.1", 1234))  # AReaL's 60 s
    clock[0] = 0.0
    fork_ready_timeout()
    assert asyncio.run(sched._wait_for_fork_ready(session, "10.0.0.1", 1234))
