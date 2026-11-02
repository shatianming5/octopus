"""The AReaL proxy of a rollout worker, in this process: the endpoint training's episodes call, for evaluation
(octopus/eval/dev_eval.py, octopus/eval/search_dev_eval.py) and the rollout load (octopus/tools/rollout_load.py).

serve() builds it as AReaL's rollout controller builds a proxy worker (proxy_rollout_server's /create_engine with the
run config's rollout section, /call initialize, set_version), so a request renders, parses and samples as in training
(_setup_openai_client: the ArealOpenAI client from rollout.agent) and reaches SGLang's /generate as the engine sends it.
The servers serve the policy under training's adapter names, <gconfig.lora_name>-v<version>, and must mount in this
process's OCTOPUS_MOUNT_LAYERS (config.check_mount_layers, as train.py checks its servers); OCTOPUS_ROUTE_SERVERS is
all of them: each episode's session stays on one server, as in training. episode() runs one episode as
OpenAIProxyWorkflow.arun_episode does (a session, the agent with its key and id), minus the export.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import socket

import uvicorn
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import StartSessionRequest

from octopus.serving import route
from octopus.serving.delta_render import REQUEST_FIELDS
from octopus.rl.config import check_mount_layers


@contextlib.asynccontextmanager
async def serve(config, servers: list[str], lora: str | None, port: int = 0):
    """The proxy on 127.0.0.1 (port 0: a free one) over servers (host:port, any host this one reaches), for the adapter
    lora = <name>-v<version> (None: rollout.use_lora=false, the base model); yields its base URL."""
    if bool(lora) != config.rollout.use_lora:  # either the adapter goes unrequested or a default version is asked
        raise SystemExit(f"--lora {lora!r} with rollout.use_lora={config.rollout.use_lora}: give both or neither")
    check_mount_layers(servers)
    os.environ["OCTOPUS_ROUTE_SERVERS"] = ",".join(servers)  # read by the engine
    if lora:
        config.rollout.lora_name, version = lora.rsplit("-v", 1)
    engine = RemoteSGLangEngine(config.rollout)
    engine.initialize(addr=servers)
    if lora:
        engine.set_version(int(version))
    srv._engine, srv._server_host = engine, "127.0.0.1"
    srv._setup_openai_client()
    consumed = inspect.signature(srv._openai_client.chat.completions.create).parameters
    if missing := [f for f in REQUEST_FIELDS if f not in consumed]:
        raise SystemExit(f"this AReaL's client does not consume {missing}: run with the overlay built from "
                         "patches/areal-cursor.patch on PYTHONPATH")
    srv._capacity = 1 << 30  # training grants one per episode (/grant_capacity)
    sock = socket.socket()
    sock.bind(("127.0.0.1", port))
    server = uvicorn.Server(uvicorn.Config(srv.app, log_level="warning"))
    # uvicorn's serve() takes SIGINT/SIGTERM for a graceful shutdown that waits on the episodes' requests; the caller's
    # handlers stay (search_dev_eval: SIGTERM stops the episodes and removes the tmpfs trials)
    server.capture_signals = contextlib.nullcontext
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving
        engine.destroy()


async def episode(agent, data: dict, base_url: str, task_id: str) -> dict:
    """agent.run_episode's record of one episode in its own proxy session (dropped after it: no export)."""
    session = srv.start_session(StartSessionRequest(task_id=task_id))
    try:
        return await agent.run_episode(data, base_url, session.api_key, session.session_id)
    finally:
        route.end(session.session_id)  # as AReaL's /rl/end_session (patches/areal-route-spread.patch): its tree slot frees
        with srv._lock:
            srv._session_cache.pop(session.session_id, None)
            srv._remove_api_keys_for_session(session.session_id)
