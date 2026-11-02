"""AReaL proxy cursor requests (patches/areal-cursor.patch, docs/cursor-protocol.md) against today's full mode.

Drives the proxy endpoint (proxy_rollout_server.chat_completions) with a deterministic stub engine, once with full message
lists and once with cursor + delta, and checks every prompt and every exported concat leaf bit for bit. Runs where the
patched AReaL is importable (node-94: AReaL venv, PYTHONPATH=<cursor overlay>:<this repo>); skips elsewhere.
`python tests/test_areal_cursor.py` prints the prompt-build timing at L = 8k / 32k / 128k.
"""

import asyncio
import hashlib
import json
import time

import pytest
from test_delta_render import EPISODE, QWEN35, SYNTHETIC, TAG, threads, tok  # noqa: F401  (tok is a fixture)

from octopus.serving.delta_render import parse_tool_arguments
from octopus.rl.prefix_tree import MOUNT_KEYS

srv = pytest.importorskip("areal.experimental.openai.proxy.proxy_rollout_server")
from areal.api import ModelResponse  # noqa: E402
from areal.experimental.openai.client import ArealOpenAI  # noqa: E402
from areal.experimental.openai.proxy.server import SessionData  # noqa: E402
from areal.utils.hf_utils import load_hf_processor_and_tokenizer  # noqa: E402
from fastapi import HTTPException  # noqa: E402

TOOLS, KWARGS = EPISODE["tools"], EPISODE["chat_template_kwargs"]


class StubSGLangEngine:
    """Replies with metadata["reply"]; logprobs and versions follow metadata["key"]. "SGLang" in the name: the client
    aligns tools for it as for the real engine."""

    def __init__(self, tok):
        self.tok, self.prompts, self.hold = tok, {}, None

    async def agenerate(self, req):
        md = req.metadata
        self.prompts[md["key"]] = list(req.input_ids)
        await asyncio.sleep(0)
        if md.get("hold"):
            await self.hold.wait()
        out = self.tok.encode(md["reply"], add_special_tokens=False) + ([] if md.get("length") else [self.tok.eos_token_id])
        seed = int(hashlib.sha1(md["key"].encode()).hexdigest()[:8], 16)
        return ModelResponse(input_tokens=list(req.input_ids), output_tokens=out,
                             output_logprobs=[-((seed + i) % 97) / 10 for i in range(len(out))],
                             output_versions=[seed % 5] * len(out), stop_reason="length" if md.get("length") else "stop",
                             tokenizer=self.tok)


def start(tok, sessions=("s",)):
    """The proxy's client as _setup_openai_client builds it (Qwen3.5 loads a VL processor, kept), on a stub engine."""
    engine, processor = StubSGLangEngine(tok), load_hf_processor_and_tokenizer(tok.name_or_path)[0]
    srv._openai_client = ArealOpenAI(engine=engine, tokenizer=tok, processor=processor, tool_call_parser="qwen3_coder",
                                     reasoning_parser="qwen3", engine_max_tokens=262144, chat_template_type="concat",
                                     require_multimodal_processor=True)
    srv._session_cache.clear()
    srv._session_cache.update({s: SessionData(session_id=s) for s in sessions})
    return engine


def reply_text(tok, reply: dict) -> str:
    """The text a model generates for a reply: its turn as the template renders it, after the generation prompt."""
    text = tok.apply_chat_template(parse_tool_arguments([{"role": "user", "content": "x"}, reply]), tools=TOOLS,
                                   tokenize=False, **KWARGS)
    return text.rsplit("<|im_start|>assistant\n", 1)[1].rsplit("<|im_end|>", 1)[0].removeprefix("<think>\n\n</think>\n\n")


def body(messages, key, reply, **md):
    return {"messages": messages, "tools": TOOLS, "chat_template_kwargs": KWARGS, "max_completion_tokens": 4096,
            "temperature": 1.0, "top_p": 1.0, "metadata": {"key": key, "reply": reply, **md}}


async def run(tok, mode: str, waves: list[list[tuple]]):
    """Send each wave's requests (key, parent key, delta, reply, metadata) concurrently, a request once its parent has
    replied: full mode sends the parent's history + delta, cursor mode the parent's id + delta. Returns the engine (its
    prompts by key), {key: parent key as the proxy linked it} and the exported concat leaves {key: tensors}."""
    engine, session = start(tok), srv._session_cache["s"]
    done, ids, hist = {}, {}, {}

    async def send(key, parent, delta, reply, md):
        if parent:
            await done.setdefault(parent, asyncio.Event()).wait()
        req = body(hist[parent] + delta if mode == "full" and parent else delta, key, reply_text(tok, reply), **md)
        if mode == "cursor" and parent:
            req["cursor"] = ids[parent]
        r = await srv.chat_completions(req, session_id="s")
        ids[key] = r.id
        hist[key] = (hist[parent] if parent else []) + delta + [r.choices[0].message.model_dump(exclude_none=True)]
        done.setdefault(key, asyncio.Event()).set()

    for wave in waves:
        await asyncio.gather(*(send(*step) for step in wave))
    key_of = {v: k for k, v in ids.items()}
    parents = {key_of[i]: x.parent and key_of[x.parent.interaction_id] for i, x in session.completions.items()}
    leaves = session.export_interactions(discount=1.0, style="concat", drop_retry_orphans=True)
    return engine, parents, {key_of[i]: x.to_tensor_dict() for i, x in leaves.items()}


def synthetic_waves():
    """Tool loops, plain-text reply, nudge, wrapped user messages, a 3-way fork sent at once, a reply cut by length, a
    rewind to an older cursor and a control rerun of an earlier prompt with another reply."""
    S, note = SYNTHETIC, lambda k: {"role": "user", "content": f"You are branch-{k}, one of 3 parallel branches."}
    return [[("r0", None, S[0:2], S[2], {})], [("r1", "r0", S[3:5], S[5], {})], [("r2", "r1", S[6:7], S[7], {})],
            [("r3", "r2", S[8:9], S[9], {})], [("r4", "r3", S[10:12], S[12], {})],
            [(f"f{k}", "r4", S[13:14] + [note(k)], S[15] if k else S[21], {}) for k in range(3)],
            [("f1a", "f1", S[16:17], {**S[7], "content": "long " * 300}, {"length": True}),
             ("f2a", "f2", S[16:17], S[17], {})],
            [("f1b", "f1a", [{"role": "user", "content": "Continue."}], S[17], {}), ("f2b", "f2a", S[18:20], S[21], {})],
            [("w", "r1", [{"role": "tool", "tool_call_id": "call_edit", "content": TAG + "edit failed"}], S[9], {})],
            [("c", "r1", S[6:7], S[17], {})], [("c1", "c", S[8:9], S[9], {})]]


def episode_waves():
    """The recorded bundle-e11 episode (main + 4 branches, one nested fork): each reply once, all threads at once."""
    owner, steps = {}, []
    for name, thread in threads("main", EPISODE["main"], EPISODE["forks"]):
        prev, start_at = None, 0
        for i in [i for i, m in enumerate(thread) if m["role"] == "assistant"]:
            p = json.dumps(thread[:i + 1])
            if p not in owner:
                owner[p] = f"{name}:{i}"
                steps.append((owner[p], prev, thread[start_at:i], thread[i], {}))
            prev, start_at = owner[p], i + 1
    return [steps]


def assert_same(full, cursor):
    (e_full, p_full, l_full), (e_cur, p_cur, l_cur) = full, cursor
    assert e_cur.prompts == e_full.prompts
    assert p_cur == p_full
    assert l_cur.keys() == l_full.keys()
    for key, t in l_full.items():
        # a cursor run's leaves carry their reads (workflow's export, once imported): here none, all zero
        cur = {name: v for name, v in l_cur[key].items() if name not in MOUNT_KEYS or name in t}
        assert all(not l_cur[key][k].any() for k in MOUNT_KEYS[1:] if k in l_cur[key] and k not in t), key
        assert t.keys() == cur.keys(), key  # mixed groups concatenate leaves of both kinds
        for name, v in t.items():
            if hasattr(v, "dtype"):
                assert v.dtype == cur[name].dtype and v.equal(cur[name]), (key, name)


@pytest.mark.parametrize("waves", [synthetic_waves, episode_waves])
def test_cursor_equals_full(tok, waves):
    full, cursor = (asyncio.run(run(tok, mode, waves())) for mode in ("full", "cursor"))
    assert_same(full, cursor)
    if waves is episode_waves:
        assert len(full[0].prompts) == 119, len(full[2])


def test_greedy_rerun_parent_differs(tok):
    """A rerun of r2's prompt that returns r2's reply (greedy or chance): full mode's longest-prefix match links the
    next request to r2, the first-inserted equal candidate, while its cursor names the rerun g. Prompts are equal;
    the leaf's logprobs (and versions) come from different replies."""
    S, waves = SYNTHETIC, synthetic_waves()[:3]
    waves += [[("g", "r1", S[6:7], S[7], {})], [("g1", "g", S[8:9], S[9], {})]]
    (e_full, p_full, l_full), (e_cur, p_cur, l_cur) = (asyncio.run(run(tok, m, waves)) for m in ("full", "cursor"))
    assert e_cur.prompts == e_full.prompts
    assert (p_full["g1"], p_cur["g1"]) == ("r2", "g")
    assert l_full.keys() == l_cur.keys() == {"g1"}
    assert l_full["g1"]["input_ids"].equal(l_cur["g1"]["input_ids"])
    assert not l_full["g1"]["logprobs"].equal(l_cur["g1"]["logprobs"])


def test_cursor_errors(tok):
    """Unknown cursor, a cursor of another session and a cursor whose reply is still generating: 404 naming it."""

    async def go():
        engine = start(tok, ("s", "t"))
        engine.hold = asyncio.Event()
        root = body(SYNTHETIC[0:2], "r0", reply_text(tok, SYNTHETIC[2]), hold=True)
        pending = asyncio.create_task(srv.chat_completions(root, session_id="s"))
        while not srv._session_cache["s"].completions:
            await asyncio.sleep(0)
        (inflight,) = srv._session_cache["s"].completions
        for cursor, session in [("chatcmpl-nope", "s"), (inflight, "s")]:
            with pytest.raises(HTTPException) as e:
                await srv.chat_completions({**body(SYNTHETIC[3:5], "x", ""), "cursor": cursor}, session_id=session)
            assert e.value.status_code == 404 and cursor in e.value.detail
        engine.hold.set()
        r = await pending
        with pytest.raises(HTTPException) as e:
            await srv.chat_completions({**body(SYNTHETIC[3:5], "x", ""), "cursor": r.id}, session_id="t")
        assert e.value.status_code == 404 and r.id in e.value.detail
        r1 = await srv.chat_completions({**body(SYNTHETIC[3:5], "r1", "ok"), "cursor": r.id}, session_id="s")
        assert srv._session_cache["s"].completions[r1.id].parent.interaction_id == r.id

    asyncio.run(go())


@pytest.mark.parametrize("control", ["fork0", "fork0/branch-1/fork0"])
def test_agent_cursor_equals_full(tok, monkeypatch, tmp_path, control):
    """The Harbor agent (tests/test_cursor_client.py's scripts on a fake container: a tree with a rewind, a fork with a
    nested fork, joins and a no-fork control) against this proxy on the stub engine, with and without OCTOPUS_CURSOR:
    every prompt, parent link and exported concat leaf is the same."""
    import threading

    from test_cursor_client import _episode

    loop = asyncio.new_event_loop()  # the proxy's, as in its server; the agent runs its own
    threading.Thread(target=loop.run_forever, daemon=True).start()
    monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", json.dumps(KWARGS))
    out = {}
    for mode in ("0", "1"):
        engine, key = start(tok), {}
        generate = engine.agenerate

        async def agenerate(req):  # the logprobs patch's rows, empty: each token's own logprob
            response = await generate(req)
            response.output_top_logprobs = [[] for _ in response.output_tokens]
            return response
        engine.agenerate = agenerate

        def serve(rid, req, calls):
            reply = reply_text(tok, {"role": "assistant", "content": "", "tool_calls": calls})
            r = asyncio.run_coroutine_threadsafe(srv.chat_completions(
                {**req, "metadata": {"key": rid, "reply": reply}}, session_id="s"), loop).result()
            key[r.id] = rid
            return r.model_dump(exclude_none=True)
        (tmp_path / mode).mkdir()
        _episode(monkeypatch, tmp_path / mode, mode, control, serve)
        session = srv._session_cache["s"]
        parents = {key[i]: x.parent and key[x.parent.interaction_id] for i, x in session.completions.items()}
        leaves = session.export_interactions(discount=1.0, style="concat", drop_retry_orphans=True)
        out[mode] = engine, parents, {key[i]: x.to_tensor_dict() for i, x in leaves.items()}
    assert_same(out["0"], out["1"])
    assert len(out["0"][0].prompts) > 10


def bench(tok, targets=(8_000, 32_000, 128_000)):
    """Prompt build (_prepare_prompt) and whole proxy request (stub engine) for the same next request after L history
    tokens: a tool loop whose results are ~1.5k tokens each, in full and in cursor mode (median of 3 requests)."""
    import statistics

    import areal.experimental.openai.client as client
    timings, prepare = {}, client._prepare_prompt

    async def timed(**kw):
        t = time.perf_counter()
        r = await prepare(**kw)
        timings["prepare"] = time.perf_counter() - t
        return r

    client._prepare_prompt = timed
    out = "\n".join(f"line {i}: value = compute({i}, {i * 7 % 13})  # ok" for i in range(120))
    reply = SYNTHETIC[17]
    for mode in ("full", "cursor"):
        engine, ids, hist, rows = start(tok), {}, [], {}
        msgs, k, target = SYNTHETIC[0:2], 0, list(targets)
        samples = []
        while target:
            req = body(hist + msgs if mode == "full" else msgs, f"b{k}", reply_text(tok, reply))
            if k:
                req["cursor"] = ids[k - 1]
            if mode == "full":
                req.pop("cursor", None)
            t = time.perf_counter()
            r = asyncio.run(srv.chat_completions(req, session_id="s"))
            total = time.perf_counter() - t
            ids[k], L = r.id, len(engine.prompts[f"b{k}"])
            hist = hist + msgs + [r.choices[0].message.model_dump(exclude_none=True)]
            msgs = [{"role": "tool", "tool_call_id": "call_test", "content": TAG + out}]
            if L >= target[0]:
                samples.append((L, timings["prepare"], total))
                if len(samples) == 3:
                    rows[target.pop(0)] = (samples[0][0], statistics.median(s[1] for s in samples),
                                           statistics.median(s[2] for s in samples))
                    samples = []
            k += 1
        timings[mode] = rows
    client._prepare_prompt = prepare
    print("| L (prompt tokens) | full prompt build ms | cursor prompt build ms | full request ms | cursor request ms |")
    print("|---|---|---|---|---|")
    for t in targets:
        (L, fp, ft), (_, cp, ct) = timings["full"][t], timings["cursor"][t]
        print(f"| {L:,} | {fp * 1e3:.1f} | {cp * 1e3:.2f} | {ft * 1e3:.1f} | {ct * 1e3:.2f} |")


if __name__ == "__main__":
    import os

    import transformers

    bench(transformers.AutoTokenizer.from_pretrained(next(p for p in QWEN35 if p and os.path.isdir(p))))
