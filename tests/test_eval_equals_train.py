"""Evaluation runs training's episodes (contradiction 3): one scripted episode through octopus.eval.dev_eval's entry
point (--run: a run's config.yaml and episode_settings.json) and one through training's (AReaL's _resolve_workflow on
train.py's workflow and kwargs, OpenAIProxyWorkflow.arun_episode on train.plan_dataset's row, against a proxy set up
over HTTP as AReaL's rollout controller sets one up) send the same episode command and the same /generate bodies to
one fake SGLang server. A cursor-protocol field an endpoint does not consume fails loudly, and so does an episode
setting of the evaluation's environment that the run did not have. Harbor is stubbed (the scripted episode replaces
`harbor run`); runs where the patched AReaL is importable, with the Qwen3.5 tokenizer (test_delta_render.QWEN35).
"""

import asyncio
import http.server
import itertools
import json
import os
import re
import shlex
import sys
import threading
from pathlib import Path

import pytest
from test_delta_render import QWEN35

srv = pytest.importorskip("areal.experimental.openai.proxy.proxy_rollout_server")
import requests  # noqa: E402
from areal.api.cli_args import save_config  # noqa: E402
from areal.infra.rpc.serialization import serialize_value  # noqa: E402

from octopus.envs.swe import rl as swe_rl  # noqa: E402
from octopus.eval import dev_eval  # noqa: E402
from octopus.rl import config as rl_config, proxy  # noqa: E402

TOKENIZER = next((p for p in QWEN35 if p and Path(p).is_dir()), None)
pytestmark = pytest.mark.skipif(TOKENIZER is None, reason="no Qwen3.5 tokenizer here")
RUN_ENV = {"OCTOPUS_AUTO_MERGE": "1", "OCTOPUS_MAIN_MAX_CALLS": "7", "OCTOPUS_REWIND_TOOL": "1"}  # the run's episode knobs


class FakeSGLang(http.server.BaseHTTPRequestHandler):
    """/generate: the reply "ok" + eos, a cursor per reply (the engine sends only the tokens after its parent's)."""
    bodies, held, rids, eos, ok = [], {}, itertools.count(), None, None

    def do_GET(self):
        body = {"internal_states": [{"octopus_mount_layers": None}]} if self.path == "/get_server_info" else {}
        self.reply(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.bodies.append(body)
        ids = cls.held.get(body.get("cursor"), []) + body["input_ids"]
        out, rid = cls.ok + [cls.eos], f"r{next(cls.rids)}"
        cls.held[rid] = ids + out
        self.reply({"meta_info": {"id": rid, "finish_reason": {"type": "stop"}, "prompt_tokens": len(ids),
                                  "completion_tokens": len(out), "output_token_logprobs": [[-0.5, t, None] for t in out]}})

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def scripted_episodes(commands: list[str]):
    """swe_rl._run_group's stand-in: two chat turns of `main` from the episode command's environment (as the
    agent builds its requests: tree._complete), then the trial dir that trial_outcome reads."""
    async def run(command: str, **_) -> int:
        commands.append(command)
        words = shlex.split(command)
        env = dict(w.split("=", 1) for w, prev in zip(words[1:], words) if prev == "export")
        job = Path(words[words.index("-o") + 1], words[words.index("--job-name") + 1])
        messages, rows = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix f.py."}], []
        for turn in range(2):
            body = {"model": env["OCTOPUS_MODEL"], "messages": messages, "tool_choice": "auto",
                    "max_tokens": int(env["OCTOPUS_MAX_TOKENS"]), **json.loads(env.get("OCTOPUS_SAMPLING_JSON") or "{}"),
                    "chat_template_kwargs": json.loads(env["OCTOPUS_CHAT_TEMPLATE_KWARGS"])}
            r = await asyncio.to_thread(requests.post, env["OCTOPUS_CHAT_URL"], json=body, timeout=60,
                                        headers={"Authorization": "Bearer " + env["OCTOPUS_API_KEY"]})
            assert r.status_code == 200, r.text
            usage = r.json()["usage"]
            messages = [*messages, {"role": "assistant", "content": r.json()["choices"][0]["message"]["content"]},
                        {"role": "user", "content": f"Go on ({turn})."}]
            rows.append({"thread": "main", "http_code": 200, "prompt_tokens": usage["prompt_tokens"],
                         "completion_tokens": usage["completion_tokens"]})
        (job / "trial/agent").mkdir(parents=True)
        (job / "trial/agent/requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (job / "trial/agent/tree.json").write_text(json.dumps({"main": messages, "forks": []}))  # what gate0.py reads
        return 0
    return run


def same(command: str) -> str:
    """The command without what differs per episode: its job name, session key and proxy port."""
    command = re.sub(r"rl-\S+-[0-9a-f]{8}", "JOB", command)
    return re.sub(r"OCTOPUS_API_KEY=\S+", "KEY", re.sub(r"127\.0\.0\.1:\d+", "PROXY", command))


@pytest.fixture(autouse=True)
def environment():
    """The environment as before the test (dev_eval and set_derived_settings set episode settings in os.environ)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    FakeSGLang.bodies, FakeSGLang.held, FakeSGLang.rids = [], {}, itertools.count()
    FakeSGLang.ok, FakeSGLang.eos = tok.encode("ok", add_special_tokens=False), tok.eos_token_id
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLang)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    commands = []
    monkeypatch.setattr(swe_rl, "_run_group", scripted_episodes(commands))
    monkeypatch.setattr(swe_rl, "trial_outcome", lambda job, build: {
        "reward": 1.0, "passed": True, "y": 1.0, "tasks": None, "control": None, "trial": str(job / "trial")})
    monkeypatch.setattr(swe_rl.HarborEpisodeAgent, "_slots", None)
    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)
    monkeypatch.setenv("OCTOPUS_ROUTE_SERVERS", "")  # proxy.serve sets it: undone after the test
    monkeypatch.setattr(srv, "_server_host", "127.0.0.1")  # the proxy worker's --host (the default admin key: loopback)
    for key, value in RUN_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(srv, "_openai_client", None)
    (tmp_path / "tasks/t1").mkdir(parents=True)
    yield f"127.0.0.1:{server.server_port}", commands
    server.shutdown()


def train_episode(config, servers: list[str]) -> list:
    """One episode as training runs it: AReaL's proxy worker set up over HTTP (/create_engine with the rollout config,
    /call initialize, set_version), the workflow AReaL resolves from train.py's, on train.py's dataset row."""
    import uvicorn

    async def run():
        server = uvicorn.Server(uvicorn.Config(srv.app, host="127.0.0.1", port=0, log_level="warning"))
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"

        def call(path, payload):
            r = requests.post(url + path, json=payload, timeout=120)
            assert r.status_code == 200, r.text
        await asyncio.to_thread(call, "/create_engine", {"engine": "areal.engine.sglang_remote.RemoteSGLangEngine",
                                                         "init_kwargs": serialize_value({"config": config.rollout})})
        await asyncio.to_thread(call, "/call", {"method": "initialize", "kwargs": serialize_value({"addr": servers[0]})})
        await asyncio.to_thread(call, "/call", {"method": "set_version", "kwargs": serialize_value({"version": 3})})
        workflow = srv._engine._engine._resolve_workflow(config.workflow, rl_config.workflow_kwargs(config),
                                                         proxy_addr=url)
        try:
            return await workflow.arun_episode(srv._engine, rl_config.plan_row(config, f"{config.rl.plan_tasks}/t1"))
        finally:
            server.should_exit = True
            await serving
            srv._engine.destroy()
    return asyncio.run(run())


def test_eval_episode_is_the_training_episode(fake, monkeypatch, tmp_path):
    servers, commands = fake
    monkeypatch.setenv("OCTOPUS_ROUTE_SERVERS", servers)  # rl_run.sh: the servers it started
    config = rl_config.load(rl_config.CONFIG, [f"tokenizer_path={TOKENIZER}", f"rl.jobs_dir={tmp_path}/jobs",
                                               f"rl.plan_tasks={tmp_path}/tasks", "rl.plan_control=false"])
    run = tmp_path / "run"  # what training leaves in its log dir: AReaL's config.yaml, train.py's episode_settings.json
    save_config(config, str(run))
    rl_config.set_derived_settings(config)
    (run / rl_config.EPISODE_SETTINGS).write_text(json.dumps(rl_config.episode_settings()))
    interactions = train_episode(config, [servers])
    assert len(interactions) == 1  # concat: the two turns are one leaf
    trained, train_command = FakeSGLang.bodies, commands.pop()

    FakeSGLang.bodies, FakeSGLang.held, FakeSGLang.rids = [], {}, itertools.count()
    for key in RUN_ENV:  # the evaluation's environment lacks the run's knobs: the snapshot brings them
        monkeypatch.delenv(key)
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(sys, "argv", ["dev_eval", "--servers", servers, "--lora", "rq1-v3", "--run", str(run),
                                      "--tasks-dir", f"{tmp_path}/tasks", "--out", f"{tmp_path}/o.jsonl", "--k", "1",
                                      "--jobs-dir", f"{tmp_path}/jobs", "--parallel", "1"])
    dev_eval.main()
    (row,) = [json.loads(x) for x in (tmp_path / "o.jsonl").read_text().splitlines()]
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/eval"))  # gate0 reads the new rows
    import gate0
    assert gate0.episode(row)["ep"] == 1
    assert "error" not in row and row["calls"] == 2
    assert {k: v for k, v in rl_config.episode_settings().items()} == json.loads((run / rl_config.EPISODE_SETTINGS).read_text())
    assert same(commands.pop()) == same(train_command)
    assert len(trained) == 2 and trained[1]["cursor"] == "r0" and trained[0]["lora_path"] == "rq1-v3"
    assert FakeSGLang.bodies == trained


def test_eval_refuses_a_setting_the_run_did_not_have(monkeypatch):
    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)
    monkeypatch.setenv("OCTOPUS_READ_EVENT", "1")  # e.g. an exported leftover of an earlier eval arm
    with pytest.raises(SystemExit, match="OCTOPUS_READ_EVENT"):
        rl_config.run_episode_settings({"OCTOPUS_AUTO_MERGE": "1"}, [])
    rl_config.run_episode_settings({"OCTOPUS_AUTO_MERGE": "1"}, ["OCTOPUS_READ_EVENT=1"])  # varied on purpose: allowed


def test_arms_add_and_drop_settings_with_env(monkeypatch, tmp_path):
    """Gate 0's arms on one base's settings: --env adds keys the base lacks (not a stray: strays are this process's
    environment's), KEY= drops one; a settings dir written by `rl_run.sh settings` (config.py's main) is what
    dev_eval --run reads, a LoRA-less base included."""
    import runpy

    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)
    monkeypatch.setenv("OCTOPUS_AUTO_MERGE", "1")  # the env file's exports
    monkeypatch.setenv("OCTOPUS_CONFLICT_BY", "1")
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    monkeypatch.setattr(sys, "argv", ["config", str(tmp_path / "base"), f"tokenizer_path={TOKENIZER}",
                                      "rollout.use_lora=false"])
    runpy.run_module("octopus.rl.config", run_name="__main__")
    base = json.loads((tmp_path / "base" / rl_config.EPISODE_SETTINGS).read_text())
    assert {k: base[k] for k in ("OCTOPUS_AUTO_MERGE", "OCTOPUS_CONFLICT_BY", "OCTOPUS_READ_TOOL")} == \
        {"OCTOPUS_AUTO_MERGE": "1", "OCTOPUS_CONFLICT_BY": "1", "OCTOPUS_READ_TOOL": "1"}
    config = rl_config.load(tmp_path / "base/config.yaml")
    assert not config.rollout.use_lora and config.tokenizer_path == TOKENIZER
    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)  # the eval host's environment: no episode settings of its own
    arm1 = {k: v for k, v in base.items() if k != "OCTOPUS_READ_TOOL"}
    rl_config.run_episode_settings(arm1, [])
    assert "OCTOPUS_READ_TOOL" not in os.environ and os.environ["OCTOPUS_CONFLICT_BY"] == "1"
    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)
    rl_config.run_episode_settings(arm1, ["OCTOPUS_READ_TOOL=1", "OCTOPUS_READ_EVENT=1"])  # arm 3: keys added
    assert os.environ["OCTOPUS_READ_TOOL"] == os.environ["OCTOPUS_READ_EVENT"] == "1"
    for key in rl_config.episode_settings():
        monkeypatch.delenv(key)
    rl_config.run_episode_settings(base, ["OCTOPUS_READ_TOOL="])  # arm 1 from the base itself: KEY= drops it
    assert "OCTOPUS_READ_TOOL" not in os.environ and os.environ["OCTOPUS_AUTO_MERGE"] == "1"
    monkeypatch.setenv("OCTOPUS_CODE", "/here")
    rl_config.run_episode_settings({**base, "OCTOPUS_CODE": "/trainer/code"}, [])  # a host path in an old snapshot
    assert os.environ["OCTOPUS_CODE"] == "/here"
    # the proxy parser's switch is derived (set_derived_settings), never a snapshot entry an old file could lack
    assert "SGLANG_FORWARD_UNKNOWN_TOOLS" not in base and os.environ["SGLANG_FORWARD_UNKNOWN_TOOLS"] == "1"
    with pytest.raises(SystemExit, match="OCTOPUS_CODE"):  # a host path is no arm's switch
        rl_config.run_episode_settings(base, ["OCTOPUS_CODE=/elsewhere"])
    monkeypatch.setenv("OCTOPUS_API_KEY", "sk-test")  # credentials are no episode setting: never snapshotted or printed
    assert "OCTOPUS_API_KEY" not in rl_config.episode_settings() and "OCTOPUS_MAX_TOKENS" not in rl_config.HOST_ENV
    assert not rl_config.episode_key("OCTOPUS_ROUTE_GAP")  # removed (#351): a stale export never stops an eval


def test_unconsumed_cursor_fields_fail_loudly(fake, monkeypatch):
    """The proxy answers 400 to a cursor-protocol field its client does not consume (a CODE newer than the AReaL
    overlay), and proxy.serve refuses an AReaL whose client lacks one (an overlay without patches/areal-cursor.patch)."""
    servers, _ = fake
    config = rl_config.load(rl_config.CONFIG, [f"tokenizer_path={TOKENIZER}"])
    monkeypatch.setattr(srv, "REQUEST_FIELDS", (*srv.REQUEST_FIELDS, "octopus_future"))

    async def run():
        async with proxy.serve(config, [servers], "rq1-v0") as url:
            session = srv.start_session(srv.StartSessionRequest(task_id="t"))
            body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "octopus_future": 1}
            return await asyncio.to_thread(requests.post, url + "/chat/completions", json=body, timeout=60,
                                           headers={"Authorization": "Bearer " + session.api_key})
    r = asyncio.run(run())
    assert r.status_code == 400 and "octopus_future" in r.text and not FakeSGLang.bodies

    monkeypatch.setattr(proxy, "REQUEST_FIELDS", (*proxy.REQUEST_FIELDS, "octopus_future"))
    monkeypatch.setattr(srv, "_engine", None)

    async def refused():
        async with proxy.serve(config, [servers], "rq1-v0"):
            pass
    with pytest.raises(SystemExit, match="octopus_future"):
        asyncio.run(refused())


def test_lora_and_use_lora_agree(monkeypatch):
    """A trained run's config (use_lora) without --lora would request a default version; a LoRA-less config with
    --lora would ignore the adapter: proxy.serve refuses both before reaching a server."""
    config = rl_config.load(rl_config.CONFIG, ["tokenizer_path=unused"])

    async def serve(lora):
        async with proxy.serve(config, ["127.0.0.1:9"], lora):
            pass
    with pytest.raises(SystemExit, match="give both or neither"):
        asyncio.run(serve(None))
    config.rollout.use_lora = False
    with pytest.raises(SystemExit, match="give both or neither"):
        asyncio.run(serve("rq1-v3"))
