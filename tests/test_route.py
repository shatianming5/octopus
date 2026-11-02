"""octopus/serving/route.py: where an AReaL engine sends a new episode (patches/areal-route-spread.patch)."""

import asyncio
import http.client
import http.server
import json
import os
import random
import shutil
import threading
import time
from pathlib import Path

import pytest

from octopus.serving import route
from octopus.serving.route import Router


def test_new_sessions_avoid_the_heavy_server_and_keep_their_own():
    async def go():
        r = Router(["a", "b", "c"])
        r.turn = 0
        assert await r.enter("heavy") == "a"
        for _ in range(39):  # its forks: 40 threads generating on a
            await r.enter("heavy", "a")
        got = [await r.enter(f"s{i}") for i in range(6)]  # six new episodes, each with one request in flight
        assert "a" not in got and sorted(got) == ["b", "b", "b", "c", "c", "c"]
        assert await r.enter("heavy") == "a" and await r.enter("s0") == got[0]  # a session keeps its server
        assert await r.enter("s1", "a") == "a"  # a request on a cursor goes to the cursor's server
        assert r.load == {"a": 42, "b": 4, "c": 3}
        for s in ("a", "b", "c"):
            r.leave(s)
        assert r.load == {"a": 41, "b": 3, "c": 2}

    asyncio.run(go())


def test_the_cap_holds_new_sessions_until_a_server_has_room():
    async def go():
        r = Router(["a", "b"], cap=2)
        r.turn = 0
        assert [await r.enter(f"s{i}") for i in range(4)] == ["a", "b", "a", "b"]  # both at the cap
        assert await r.enter("s0") == "a"  # a running session's request never waits: 3 on a
        waiting = asyncio.ensure_future(r.enter("new"))
        await asyncio.sleep(0.01)
        r.leave("a")  # a back at the cap: the new session still waits
        await asyncio.sleep(0.01)
        assert not waiting.done() and r.load == {"a": 2, "b": 2}
        r.leave("b")
        assert await asyncio.wait_for(waiting, 1) == "b" and r.load == {"a": 2, "b": 2}

    asyncio.run(go())


def test_one_slot_admits_one_of_many_waiting():
    async def go():
        r = Router(["a"], cap=1)
        await r.enter("s0")
        waiting = [asyncio.ensure_future(r.enter(f"n{i}")) for i in range(5)]
        await asyncio.sleep(0.01)
        r.leave("a")
        await asyncio.sleep(0.01)
        assert sum(w.done() for w in waiting) == 1 and r.load == {"a": 1}  # all woke, one took the slot
        for w in waiting:
            w.cancel()

    asyncio.run(go())


def test_first_requests_of_one_session_waiting_on_the_cap_share_its_server():
    async def go():
        r = Router(["a", "b"], cap=1)
        r.turn = 0
        assert [await r.enter("x"), await r.enter("y")] == ["a", "b"]  # both servers at the cap
        first = [asyncio.ensure_future(r.enter("s")) for _ in range(2)]  # two threads' first requests of session s
        await asyncio.sleep(0.01)
        r.leave("a")
        await asyncio.sleep(0.01)
        r.leave("b")
        assert await asyncio.wait_for(asyncio.gather(*first), 1) == ["a", "a"] and r.session["s"] == "a"

    asyncio.run(go())


def kv_router(monkeypatch, usage, **kw):
    """A router with a KV target over servers whose /v1/loads the dict `usage` scripts (None: unreadable)."""
    monkeypatch.setattr(route, "POLL", 0.01)
    monkeypatch.setattr(route, "server_load", lambda s: usage[s] and (usage[s] + (1000,))[:3])  # pool 1000 by default
    r = Router(list(usage), kv=0.75, **kw)
    r.turn = 0
    return r


def test_new_sessions_go_to_servers_below_the_kv_target(monkeypatch):
    async def go():
        usage = {"a": (0.9, 0), "b": (0.3, 0), "c": (0.5, 0), "d": (0.2, 3)}  # a over the target, d queues
        r = kv_router(monkeypatch, usage)
        got = [await r.enter(f"s{i}") for i in range(2)]
        assert sorted(got) == ["b", "c"]  # only those two are open, one new session each per poll
        assert await r.enter("s2") in ("b", "c")  # waited for the next poll
        assert await r.enter("s0") == got[0] and await r.enter("s9", "a") == "a"  # follow-ups go where they belong
        usage["c"] = (0.8, 0)
        await asyncio.sleep(0.05)
        assert {await r.enter(f"t{i}") for i in range(3)} == {"b"}

    asyncio.run(go())


def test_new_sessions_split_by_free_tokens_below_the_target(monkeypatch):
    async def go():  # an A6000's and a 4090D's pool (node-136, 10-27), both empty: free below 0.75, 202.5K and 405K
        r = kv_router(monkeypatch, {"small": (0.0, 0, 270_000), "big": (0.0, 0, 540_000)})
        monkeypatch.setattr(route, "random", random.Random(0))
        got = []
        for i in range(300):  # each as if the first of another router's poll (none sees the others' admissions)
            r.admitted.clear()
            got.append(await r.enter(f"s{i}"))
        assert 80 <= got.count("small") <= 120

    asyncio.run(go())


def test_all_servers_above_the_target_hold_new_sessions_but_not_follow_ups(monkeypatch):
    async def go():
        usage = {"a": (0.5, 0), "b": (0.5, 0)}
        r = kv_router(monkeypatch, usage)
        s0 = await r.enter("s0")
        usage.update(a=(0.9, 0), b=(0.8, 0))
        await asyncio.sleep(0.05)
        waiting = asyncio.ensure_future(r.enter("new"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        assert await asyncio.wait_for(r.enter("s0"), 0.01) == s0  # a running session never waits
        usage["b"] = (0.6, 0)
        assert await asyncio.wait_for(waiting, 1) == "b"

    asyncio.run(go())


def test_the_wait_is_bounded_then_the_lowest_kv_server_takes_it(monkeypatch):
    async def go():
        r = kv_router(monkeypatch, {"a": (0.95, 0), "b": (0.9, 2)}, wait=0.1)
        t = asyncio.get_running_loop().time()
        assert await r.enter("s0") == "b"
        assert 0.099 <= asyncio.get_running_loop().time() - t < 0.5  # uvloop (set by AReaL's tests) keeps ms clock ticks

    asyncio.run(go())


def test_unreadable_servers_fall_back_to_in_flight_counts(monkeypatch):
    async def go():
        r = kv_router(monkeypatch, {"a": None, "b": None}, cap=2)
        assert [await r.enter(f"s{i}") for i in range(4)] == ["a", "b", "a", "b"]  # no KV wait: the in-flight rule
        waiting = asyncio.ensure_future(r.enter("new"))  # the cap still holds
        await asyncio.sleep(0.05)
        assert not waiting.done()
        r.leave("b")
        assert await asyncio.wait_for(waiting, 1) == "b"

    asyncio.run(go())


def test_a_failing_poll_round_keeps_the_poller(monkeypatch):
    async def go():
        calls = []

        def load(s):
            calls.append(s)
            if len(calls) == 1:  # an error server_load's own except would not see (a patched-in raiser)
                raise http.client.IncompleteRead(b"")
            return {"a": (0.9, 0, 1000), "b": (0.3, 0, 1000)}[s]

        monkeypatch.setattr(route, "POLL", 0.01)
        monkeypatch.setattr(route, "server_load", load)
        r = Router(["a", "b"], kv=0.75)
        r.turn = 0
        assert [await r.enter("s0"), await r.enter("s1")] == ["a", "b"]  # the failed round: both unknown, one each
        got = [await asyncio.wait_for(r.enter(f"t{i}"), 1) for i in range(3)]  # later rounds read the servers
        assert got == ["b", "b", "b"] and not r.poller.done() and r.peak == {"a": (0.9, 0), "b": (0.3, 0)}

    asyncio.run(go())


def test_a_dead_poller_is_rebuilt_by_the_next_new_session(monkeypatch):
    async def go():
        r = kv_router(monkeypatch, {"a": (0.3, 0), "b": (0.3, 0)})
        s0 = await r.enter("s0")
        dead = r.poller
        dead.cancel()
        await asyncio.sleep(0.05)
        assert dead.done() and await r.enter("s1") == ({"a", "b"} - {s0}).pop()  # the other took none since the poll
        assert await asyncio.wait_for(r.enter("s2"), 1) in ("a", "b")  # both admitted: only a new poll opens them
        assert r.poller is not dead and not r.poller.done()

    asyncio.run(go())


def test_the_tree_budget_caps_each_servers_active_trees_until_one_ends(monkeypatch, capsys):
    async def go():
        monkeypatch.setattr(route, "LOG_EVERY", 1)
        usage = {"a": (0.0, 0, 1000), "b": (0.0, 0, 2000)}  # slots 0.75 x pool / 250: a 3, b 6
        r = kv_router(monkeypatch, usage, tree_tokens=250)
        weights = []
        draw = route.random.choices
        monkeypatch.setattr(route.random, "choices", lambda xs, ws: weights.append(dict(zip(xs, ws))) or draw(xs, ws))
        got = [await asyncio.wait_for(r.enter(f"s{i}"), 1) for i in range(9)]  # one per server per poll
        assert weights[0] == {"a": 3, "b": 6} and sorted(got) == ["a"] * 3 + ["b"] * 6
        assert await asyncio.wait_for(r.enter("s0"), 0.01) == got[0]  # a running tree's requests never wait
        waiting = asyncio.ensure_future(r.enter("new"))
        await asyncio.sleep(0.05)
        assert not waiting.done()  # every slot taken, KV still 0
        threading.Thread(target=route.end, args=(f"s{got.index('a')}",)).start()  # /rl/end_session's thread
        assert await asyncio.wait_for(waiting, 1) == "a" and r.trees() == {"a": 3, "b": 6}
        usage["b"] = (0.0, 1, 2000)  # b queues: the KV guard holds it although b has a slot
        await asyncio.sleep(0.05)
        route.end(f"s{got.index('b')}")
        waiting = asyncio.ensure_future(r.enter("new2"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        usage["b"] = (0.0, 0, 2000)
        assert await asyncio.wait_for(waiting, 1) == "b"
        out = capsys.readouterr().out
        assert "route: new session new2 -> b after " in out and "trees 6/6" in out
        assert "(trees, slots, KV, queue) per server {'a': (3, 3, 0.0, 0), 'b': (6, 6, 0.0, 0)}" in out

    asyncio.run(go())


def test_routers_share_active_trees_through_their_dir(monkeypatch, tmp_path):
    async def go():
        a, b = (kv_router(monkeypatch, {"s": (0.0, 0, 1000)}, tree_tokens=375, shared=str(tmp_path)) for _ in range(2))
        assert await a.enter("x") == "s"  # slots: 0.75 x 1000 / 375 = 2
        assert await asyncio.wait_for(b.enter("y"), 1) == "s"
        waiting = asyncio.ensure_future(b.enter("z"))
        await asyncio.sleep(0.05)
        assert not waiting.done() and a.trees() == b.trees() == {"s": 2}  # each sees the other's tree
        a._end("x")  # a's tree ends: b sees it at its next poll
        assert await asyncio.wait_for(waiting, 1) == "s"
        await asyncio.sleep(0.05)
        assert a.trees() == {"s": 2}  # b's two, read at a's poll
        os.utime(b.mine, (0, time.time() - route.STALE - 1))  # a router that stopped rewriting its file: dead
        assert a._read() == {}
        shutil.rmtree(tmp_path)  # removed under the routers: unreadable, then made again (0700) by the next write
        assert a._read() == {}
        a._publish()
        assert os.stat(tmp_path).st_mode & 0o777 == 0o700 and os.path.exists(a.mine)

    asyncio.run(go())


def test_a_session_that_ends_while_its_first_request_waits_never_takes_a_slot(monkeypatch):
    async def go():  # a group's other episodes are cancelled while their first requests wait on a full server
        r = kv_router(monkeypatch, {"a": (0.0, 0, 1000)}, tree_tokens=750)  # 1 slot
        assert await r.enter("s0") == "a"
        waiting = [asyncio.ensure_future(r.enter("w")) for _ in range(2)]  # two first requests of w
        await asyncio.sleep(0.05)
        route.end("w")  # the episode ends: its end finds nothing to free
        for w in waiting:
            with pytest.raises(route.Ended):
                await asyncio.wait_for(w, 1)
        route.end("s0")
        assert await asyncio.wait_for(r.enter("x"), 1) == "a" and r.session == {"x": "a"}
        route.end("x")
        with pytest.raises(route.Ended):  # a late first request of the ended w: still never admitted
            await asyncio.wait_for(r.enter("w"), 1)
        route.start("w")  # AReaL's /rl/start_session reuses the id: a new tree, admitted as any other
        assert await asyncio.wait_for(r.enter("w"), 1) == "a"

    asyncio.run(go())


def test_a_session_that_ends_before_its_first_request_arrives_never_takes_a_slot(monkeypatch):
    async def go():
        monkeypatch.setattr(route, "ENDED", 3)
        r = kv_router(monkeypatch, {"a": (0.0, 0, 1000)}, tree_tokens=750)
        for k in ("e", "f", "g", "h"):  # ended before any request of them reached the router; the oldest drops out
            threading.Thread(target=route.end, args=(k,)).start()
        await asyncio.sleep(0.05)
        assert list(r.ended) == ["f", "g", "h"]
        with pytest.raises(route.Ended):
            await asyncio.wait_for(r.enter("g"), 1)
        assert r.session == {} and await asyncio.wait_for(r.enter("e"), 1) == "a"

    asyncio.run(go())


def test_a_tree_without_requests_for_expire_seconds_frees_its_slot(monkeypatch, capsys):
    async def go():  # the backstop for an end that never came
        r = kv_router(monkeypatch, {"a": (0.0, 0, 1000)}, tree_tokens=750)
        assert await r.enter("lost") == "a"
        waiting = asyncio.ensure_future(r.enter("new"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        r.seen["lost"] -= route.EXPIRE  # its last request started an hour ago
        assert await asyncio.wait_for(waiting, 1) == "a" and r.session == {"new": "a"}
        assert "session lost expired" in capsys.readouterr().out

    asyncio.run(go())


def test_waiting_reads_the_shared_dir_only_at_polls_and_admissions(monkeypatch, tmp_path):
    async def go():
        r = kv_router(monkeypatch, {"a": (0.0, 0, 1000)}, tree_tokens=750, shared=str(tmp_path))
        assert await r.enter("s0") == "a"
        monkeypatch.setattr(route, "POLL", 100.0)  # no more polls after the one under way
        await asyncio.sleep(0.05)
        scans = []
        scandir = os.scandir
        monkeypatch.setattr(route.os, "scandir", lambda d: scans.append(d) or scandir(d))
        waiting = asyncio.ensure_future(r.enter("new"))
        for _ in range(50):  # leaving requests wake it: the server is still full, no read
            r.leave("a")
            await asyncio.sleep(0)
        assert not waiting.done() and scans == []
        waiting.cancel()

    asyncio.run(go())


def test_without_a_tree_budget_no_router_shares_its_trees(monkeypatch, tmp_path):
    monkeypatch.setattr(route, "DIR", str(tmp_path / "route"))
    monkeypatch.setenv("OCTOPUS_ROUTE_SERVERS", "a:1")
    monkeypatch.setenv("OCTOPUS_ROUTE_KV", "0.5")
    assert Router.from_env().shared is None
    monkeypatch.setenv("OCTOPUS_ROUTE_TREE_TOKENS", "40000")
    assert Router.from_env().shared == route.DIR


def test_server_load_reads_sglang_v1_loads():
    body = json.dumps({"version": "0.5.19", "loads": [  # a live Alpha rollout server's reply (10-25), trimmed
        {"dp_rank": 0, "num_running_reqs": 73, "num_waiting_reqs": 0, "num_used_tokens": 1292349,
         "max_total_num_tokens": 1937680, "token_usage": 0.667}]}).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == "/v1/loads?include=core"
            self.send_response(200 if ok else 404)
            self.end_headers()
            self.wfile.write(body if ok else b"")

        def log_message(self, *a):
            pass

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        assert route.server_load(f"127.0.0.1:{s.server_port}") == (0.667, 0, 1937680)
    finally:
        s.shutdown()
        s.server_close()
    assert route.server_load(f"127.0.0.1:{s.server_port}") is None  # down


def test_every_launcher_with_a_tree_budget_checks_the_overlays_end_hook():
    root = Path(__file__).parents[1]
    assert {"route.start(session_id)", "route.end(session_id)"} <= set((root / "patches/areal-route-spread.patch").read_text().split())
    for p in [*root.glob("results/search-v9-rest-1028/scripts/*.sh"), root / "scripts/rl_env/rl_run.sh"]:
        if "OCTOPUS_ROUTE_TREE_TOKENS" in (t := p.read_text()):
            assert "grep -qF 'route.start(session_id)'" in t, p  # an old overlay never frees a slot
