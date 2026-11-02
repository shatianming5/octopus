"""Self-check of patches/areal-route-spread.patch: PYTHONPATH=<patched overlay>:<code dir> python check_areal_route.py"""
import asyncio
import contextlib
import contextvars
from types import SimpleNamespace

from areal.infra.remote_inf_engine import ROUTE_KEY, RemoteInfEngine
import areal.experimental.openai.proxy.proxy_rollout_server as proxy  # imports ROUTE_KEY without a cycle
from octopus.serving.route import Router

assert proxy.ROUTE_KEY is ROUTE_KEY
e = object.__new__(RemoteInfEngine)
e.route, e.rid_to_address, seen = Router(["a:1", "b:2"]), {}, []


async def fake(req, server_addr=None):
    seen.append((server_addr, dict(e.route.load)))


e._agenerate = fake
on = lambda s: contextvars.Context().run(lambda: (ROUTE_KEY.set(s), asyncio.run(e.agenerate(SimpleNamespace(rid="r", parent=None))))[1])  # noqa
e.route.turn, e.route.load["a:1"] = 0, 3  # a:1 busy: a new session goes to b:2, counted while its request runs
on("s1")
assert seen == [("b:2", {"a:1": 3, "b:2": 1})] and e.route.load == {"a:1": 3, "b:2": 0}
asyncio.run(e.agenerate(SimpleNamespace(rid="r", parent=None)))  # no session: the engine picks (server_addr None)
assert seen[-1] == (None, {"a:1": 3, "b:2": 0})


async def end():  # /rl/end_session ends the session's tree (then refuses it: not in the proxy's cache)
    e.route.loop = asyncio.get_running_loop()
    with contextlib.suppress(Exception):
        proxy.end_session("s1")
    await asyncio.sleep(0)


assert "s1" in e.route.session
asyncio.run(end())
assert "s1" not in e.route.session and "s1" in e.route.ended
proxy._capacity, e.route.ended["s-0"] = 1, None  # /rl/start_session clears a reused id's end
assert proxy.start_session(proxy.StartSessionRequest(task_id="s")).session_id == "s-0" and "s-0" not in e.route.ended
print("ok")
