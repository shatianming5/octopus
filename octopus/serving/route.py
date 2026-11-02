"""Which rollout server an AReaL engine sends a request to (patches/areal-route-spread.patch; OCTOPUS_ROUTE_SERVERS, all
rollout servers, and OCTOPUS_ROUTE_CAP, both set by rl_run.sh; OCTOPUS_ROUTE_KV, OCTOPUS_ROUTE_TREE_TOKENS and
OCTOPUS_ROUTE_WAIT, host settings an env file exports, docs/search-env.md "Admission").

A proxy session is one episode tree: every thread of it (main, branches, controls) stays on one server, as its cursors,
reads (mounts) and prefix reuse need. A new session goes to the server with the fewest of this engine's requests in
flight, i.e. the threads of its episodes that are generating, not to the next server in turn: episodes differ in their
number of threads (forks), and a server that falls behind holds its episodes longer, so equal turns piled the forking
episodes onto one server (search-v6c-1024, step 23: one server 117 requests, KV full, requests past the 900 s timeout;
the other three nearly idle). With a cap (> 0) a new session waits until some server has fewer than `cap` requests in
flight; the requests of sessions already on a server never wait. Counts are this engine's (one per AReaL proxy worker):
the cap gates only this worker's new sessions; follow-up requests are uncapped and other workers' load is unseen, so
it bounds no server's total.

With a KV target (kv > 0) the router also sees every server's total load: one poller per router reads each server's
SGLang /v1/loads (token_usage, num_waiting_reqs, max_total_num_tokens) every POLL seconds, and a new session goes only
to a server below `kv` with no queued request that took no new session since its last poll (the poll cannot see what it
adds yet); else it waits for the next poll, a leaving request or an ending tree. An unreadable server (poll failed)
stays open to new sessions (one per poll still), taken only when no readable server is open, then by in-flight count;
with every server unreadable that is the rule without a target. After `wait` seconds a new session takes a server under
the cap anyway (the lowest KV when none is open): the wait spends its first request's client timeout
(OCTOPUS_HTTP_TIMEOUT, 900 s), so it stays well below.

Active trees: a session is an active tree from its admission until its proxy session ends (end(): AReaL's
/rl/end_session, which octopus/rl/proxy.py also calls) or, as a backstop for a lost end, until none of its requests
started for EXPIRE seconds. A session that ended (its id among the last ENDED ended ones, until AReaL's /rl/start_session
starts that id again: start()) is never admitted, as its first request may still wait for admission or not have reached
it yet (that request fails with Ended): its end found nothing to free, and no later end would come. With a tree budget (kv > 0
and OCTOPUS_ROUTE_TREE_TOKENS, tokens, > 0) a server takes a new tree only while it holds fewer than its slots, max(1, kv
x max_total_num_tokens / tree_tokens): KV shows a tree only as it grows (its forks never wait), so KV alone let a start
admit every waiting episode within seconds (search-v9-rest, 10-27: 4090D servers at KV 0.9+ with 10-40 queued); the KV
target stays a guard. A server's trees are the router's own plus every other router's on this host: each writes its
trees per server to its own file in DIR (one per user and host, mode 0700) at each change and each poll, and reads the
other files rewritten within STALE seconds at each poll and again before it admits a tree (a dead router's go stale).
Routers match servers by address string: a server one router reaches at its LAN address and another through a tunnel's
127.0.0.1 port counts as two, and routers on other hosts are not counted. Two routers that admit to one server at the
same instant can put it one tree over (no lock).

Of the open servers one is drawn at random in proportion to its free slots (without a budget: its free tokens below the
target, max_total_num_tokens x (kv - token_usage), so routers blind to each other's admissions split episodes by pool).
The wait bound (late) ignores the slots and the target.

Every new session prints one line (its wait, server, the server's trees and slots, KV and queue), and every LOG_EVERY
polls the router prints each server's trees, slots, KV and queue, the peak KV and queue so far and how many sessions the
wait bound admitted (token_usage leaves out evictable prefix cache, which admitted episodes reclaim).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import socket
import threading
import time
import urllib.request
import uuid
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

POLL = 2.0  # seconds between two load polls of all servers
LOG_EVERY = 30  # polls between two state lines (1 min)
STALE = 60.0  # seconds: a router file not rewritten for this long is a dead router's (a live one rewrites it each poll)
# seconds without a request start after which a tree counts as ended: AReaL's own default for dropping an inactive
# session (agent session_timeout_seconds, 3600), which this module cannot read
EXPIRE = 3600.0
ENDED = 4096  # ended session ids kept (insertion order, oldest out)
DIR = f"/dev/shm/octopus-route-{os.getuid()}"  # every router's active trees per server, on this host
ROUTERS: weakref.WeakSet[Router] = weakref.WeakSet()  # this process's, for end()


def server_load(addr: str) -> tuple[float, int, int] | None:
    """SGLang's load at host:port: (token_usage of its fullest DP rank, num_waiting_reqs and max_total_num_tokens of
    all); None: unreadable."""
    try:
        with urllib.request.urlopen(f"http://{addr}/v1/loads?include=core", timeout=POLL) as r:
            loads = json.load(r)["loads"]
        return (max(x["token_usage"] for x in loads), sum(x["num_waiting_reqs"] for x in loads),
                sum(x["max_total_num_tokens"] for x in loads))
    except Exception:  # any failure (http.client's errors are not all OSError): unknown, never a dead poller
        return None


class Ended(RuntimeError):
    """The session ended while a first request of it waited for admission."""


def end(key: str) -> None:
    """Proxy session `key` (one episode tree) ended: its server's tree slot is free. Any thread."""
    for r in list(ROUTERS):
        r.end(key)


def start(key: str) -> None:
    """Proxy session `key` started (again: AReaL reuses an id once its session is gone). Any thread."""
    for r in list(ROUTERS):
        r.start(key)


class Router:
    def __init__(self, servers: list[str], cap: int = 0, kv: float = 0.0, wait: float = 300.0, tree_tokens: int = 0,
                 shared: str | None = None):
        self.servers, self.cap, self.kv, self.wait, self.tree_tokens = servers, cap, kv, wait, tree_tokens
        self.load = dict.fromkeys(servers, 0)  # this engine's requests in flight, per route server
        self.session: dict[str, str] = {}  # active tree (session id) -> its server
        self.seen: dict[str, float] = {}  # active tree -> when its last request started (time.monotonic(), EXPIRE)
        self.ended: dict[str, None] = {}  # sessions ended and not started again: never admitted (ENDED kept)
        self.lock = threading.Lock()  # for ended: end() and start() come from AReaL's handler threads
        self.others: Counter = Counter()  # the other routers' active trees per server, as last read
        self.turn = random.randrange(len(servers))  # equal loads take the servers in turn, from a random one
        self.freed: asyncio.Event | None = None  # set when a request leaves, a tree ends or a poll lands
        self.usage: dict[str, tuple[float, int, int] | None] = {}  # last poll per server (none yet: no new session)
        self.size: dict[str, int] = {}  # max_total_num_tokens per server, from its last readable poll
        self.admitted: set[str] = set()  # servers that took a new session since the last poll
        self.poller: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None  # where new sessions run: end() comes from other threads
        # its own threads: asyncio's default pool is shared with the episodes' tool and judge calls
        self.pool = ThreadPoolExecutor(max_workers=len(servers), thread_name_prefix="octopus-route")
        self.peak: dict[str, tuple[float, int]] = {}  # highest token_usage and num_waiting_reqs polled, per server
        self.forced = 0  # new sessions the wait bound admitted
        self.shared = shared  # the dir of every router's active trees (None: this router's alone)
        self.mine = shared and os.path.join(shared, f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}.json")
        if shared:
            os.makedirs(shared, mode=0o700, exist_ok=True)
            if os.stat(shared).st_uid != os.getuid():  # another user's files would steer this router's admissions
                raise RuntimeError(f"route: {shared} belongs to another user")
        ROUTERS.add(self)

    @classmethod
    def from_env(cls) -> Router | None:
        servers = [a for a in os.getenv("OCTOPUS_ROUTE_SERVERS", "").split(",") if a]
        kv, tree = float(os.getenv("OCTOPUS_ROUTE_KV") or 0), int(os.getenv("OCTOPUS_ROUTE_TREE_TOKENS") or 0)
        return cls(servers, int(os.getenv("OCTOPUS_ROUTE_CAP") or 0), kv, float(os.getenv("OCTOPUS_ROUTE_WAIT") or 300),
                   tree, DIR if kv and tree else None) if servers else None

    def slots(self, s: str) -> int | None:
        """Active trees server s may hold; None: no tree budget, or its pool never read."""
        return max(1, int(self.kv * self.size[s] / self.tree_tokens)) if self.tree_tokens and s in self.size else None

    def trees(self) -> Counter:
        """Active trees per server: this router's and the other routers' as last read."""
        return Counter(self.session.values()) + self.others

    def _read(self) -> Counter:
        """The other live routers' active trees per server, from the shared dir."""
        n, now = Counter(), time.time()
        try:
            files = list(os.scandir(self.shared))
        except OSError:  # removed: the next _publish makes it again
            return n
        for f in files:
            if f.path == self.mine or not f.name.endswith(".json"):
                continue
            with contextlib.suppress(OSError, ValueError):  # removed meanwhile, or not a router's file
                if (age := now - f.stat().st_mtime) < STALE:
                    with open(f.path) as h:
                        n.update(json.load(h))
                elif age > 3600:  # an old run's
                    os.unlink(f.path)
        return n

    def open(self, s: str, late: bool, trees: Counter) -> bool:
        """May server s take a new session now (late: past the KV wait, the cap alone)."""
        if self.cap and self.load[s] >= self.cap:
            return False
        if not self.kv or late:
            return True
        if s not in self.usage or s in self.admitted or ((t := self.slots(s)) and trees[s] >= t):
            return False
        return (u := self.usage[s]) is None or (u[0] < self.kv and not u[1])

    def _open(self, late: bool) -> list[str]:
        trees = self.trees()
        return [s for s in self.servers if self.open(s, late, trees)]

    async def enter(self, key: str, server: str | None = None) -> str:
        """Count one request of session `key` in flight; its server: `server` (its parent cursor's or its rid's), else
        the session's, else (a new session) the best open one once there is one. Pair with leave()."""
        server = server or self.session.get(key)
        if server is None:
            loop = self.loop = asyncio.get_running_loop()
            if self.kv and (self.poller is None or self.poller.done() or self.poller.get_loop() is not loop):
                self.poller = loop.create_task(self._poll())
            start = loop.time()
            deadline = start + self.wait
            while (server := self.session.get(key)) is None:  # (another first request of it may get in meanwhile)
                if key in self.ended:
                    raise Ended(f"route: session {key} ended before its first request was admitted")
                late = loop.time() >= deadline
                if (ok := self._open(late)) and self.shared:  # the others' trees as of now before taking a slot
                    self.others = self._read()
                    ok = self._open(late)
                if ok:
                    break
                self.freed = self.freed or asyncio.Event()
                with contextlib.suppress(asyncio.TimeoutError):  # past the deadline only the cap holds: no timeout
                    await asyncio.wait_for(self.freed.wait(), max(deadline - loop.time(), 0) or None)
        if server is None:
            trees, free = self.trees(), {}  # no pool read (0): not drawn, the min rule below
            for s in ok:
                if ((u := self.usage.get(s)) and u[0] < self.kv and not u[1] and u[2]
                        and (w := self.slots(s) - trees[s] if self.tree_tokens else u[2] * (self.kv - u[0])) > 0):
                    free[s] = w
            k, self.turn = self.turn, (self.turn + 1) % len(self.servers)
            server = random.choices(list(free), list(free.values()))[0] if free else min(
                (s for s in self.servers[k:] + self.servers[:k] if s in ok),
                key=lambda s: ((self.usage.get(s) or (1.0,))[0], self.load[s]))
            if len(self.session) >= 1024:  # oldest session first (insertion order)
                self._drop(next(iter(self.session)))
            self.session[key] = server
            self.admitted.add(server)
            self.forced += self.kv > 0 and late
            self._publish()
            print(f"[octopus] route: new session {key} -> {server} after {loop.time() - start:.1f} s; trees "
                  f"{trees[server] + 1}/{self.slots(server) or '-'}, (KV, queue) {(self.usage.get(server) or ())[:2]}",
                  flush=True)
        if key in self.session:
            self.seen[key] = time.monotonic()
        if server in self.load:  # no await since the check above: the cap holds
            self.load[server] += 1
        return server

    def leave(self, server: str) -> None:
        if server in self.load:
            self.load[server] -= 1
            self._wake()

    def end(self, key: str) -> None:
        """Session `key` ended: its tree slot is free, and a first request of it not admitted yet never will be. Any
        thread."""
        with self.lock:
            self.ended[key] = None
            if len(self.ended) > ENDED:
                del self.ended[next(iter(self.ended))]
        if (loop := self.loop) and not loop.is_closed():
            loop.call_soon_threadsafe(self._end, key)

    def start(self, key: str) -> None:
        """Session `key` started: admitted as any other, though an earlier session of that id ended. Any thread."""
        with self.lock:
            self.ended.pop(key, None)

    def _end(self, key: str) -> None:
        if self._drop(key):
            self._publish()
        self._wake()  # a waiting first request of it fails

    def _drop(self, key: str) -> bool:
        self.seen.pop(key, None)
        return self.session.pop(key, None) is not None

    def _publish(self) -> None:
        """This router's active trees per server to its shared file (written whole, then renamed: never half-read)."""
        if self.shared:
            try:
                os.makedirs(self.shared, mode=0o700, exist_ok=True)  # again if removed
                with open(tmp := f"{self.mine}.tmp", "w") as f:
                    json.dump(Counter(self.session.values()), f)
                os.replace(tmp, self.mine)
            except OSError as e:  # the other routers miss this one's changes until a later write
                print(f"[octopus] route: cannot write {self.mine}: {e}", flush=True)

    def _wake(self) -> None:
        if self.freed:
            self.freed.set()
            self.freed = None

    async def _poll(self) -> None:
        loop, n = asyncio.get_running_loop(), 0
        while True:
            try:
                got = await asyncio.gather(*(loop.run_in_executor(self.pool, server_load, s) for s in self.servers))
            except Exception:  # a round that fails leaves every server unknown, the poller alive
                got = [None] * len(self.servers)
            self.usage, self.admitted = dict(zip(self.servers, got)), set()
            self.size.update((s, u[2]) for s, u in self.usage.items() if u and u[2])
            if self.shared:
                self.others = self._read()
            now = time.monotonic()
            for k in [k for k, t in self.seen.items() if now - t > EXPIRE]:
                self._drop(k)
                print(f"[octopus] route: session {k} expired: no request for {EXPIRE:g} s and no end", flush=True)
            self._wake()
            n += 1
            with contextlib.suppress(Exception):  # bookkeeping (a closed stdout, say) never ends the poller
                self._publish()  # also its heartbeat
                for s, u in self.usage.items():
                    if u:
                        p = self.peak.get(s, (0.0, 0))
                        self.peak[s] = (max(p[0], u[0]), max(p[1], u[1]))
                if n % LOG_EVERY == 0:
                    trees = self.trees()
                    now = {s: (trees[s], self.slots(s), u and round(u[0], 3), u and u[1]) for s, u in self.usage.items()}
                    print(f"[octopus] route: (trees, slots, KV, queue) per server {now}; peak (KV, queue) {self.peak}; "
                          f"{self.forced} new sessions admitted past the {self.wait:g} s wait", flush=True)
            await asyncio.sleep(POLL)
