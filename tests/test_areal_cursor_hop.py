"""AReaL engine → SGLang cursor hop (patches/areal-cursor.patch, docs/cursor-protocol.md "AReaL engine hop").

The rollout engine sends /generate a cursor + the new token ids instead of the whole prompt when the prompt extends a
reply whose tokens the same server holds. The CPU tests drive RemoteInfEngine.agenerate against a fake SGLang that
keeps slice 4's cursor store; they run where the patched AReaL is importable (PYTHONPATH=<areal overlay>:<this repo>)
and skip elsewhere. The live test (OCTOPUS_SGLANG_URL: one server with the slice-4 overlay, --disable-radix-cache)
runs the proxy on the real engine twice, with and without cursor hops, and compares everything a trainer sees.
`python tests/test_areal_cursor_hop.py` prints /generate request bytes and the hop's build cost at L = 8k/32k/128k.
"""

import asyncio
import base64
import contextvars
import dataclasses
import json
import os
import random
import statistics
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

rie = pytest.importorskip("areal.infra.remote_inf_engine")
import aiohttp  # noqa: E402
from areal.api import ModelRequest, ModelResponse  # noqa: E402
from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig  # noqa: E402
from areal.engine.sglang_remote import RemoteSGLangEngine, SGLangBackend  # noqa: E402

from octopus.serving.cursor_store import CursorStore  # noqa: E402

EOS, PAD = 9, 0  # Qwen3.5: <|im_end|>, <|endoftext|>
TOK = SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD)


class FakeSGLang:
    """/generate with slice 4's store, per server: a cursor request's input is the cursor's tokens + its input_ids (an
    unknown cursor is the 404 arequest_with_retry raises after its retries), and every reply is stored as its whole
    input + output. Outputs come from `script`: [(output ids, finish reason)]; on_abort runs when one is aborted.
    unpatched: SGLang without slice 4, which drops "cursor" and generates from input_ids alone. A mount (#132) puts M
    placeholder ids (negative, one per mount request) between the cursor's tokens and input_ids, as SGLang does; an
    unknown mount cursor is a 404 too; no_mount: SGLang without the mount patch, which drops "mount"."""

    def __init__(self, script, unpatched=False, no_mount=False):
        self.script, self.stores, self.sent, self.inputs = list(script), defaultdict(lambda: CursorStore(1 << 30)), [], []
        self.on_abort, self.unpatched, self.no_mount, self.mounts = lambda: None, unpatched, no_mount, []

    async def __call__(self, session, addr, endpoint, payload, method, max_retries, timeout):
        self.sent.append((addr, payload.get("cursor"), payload["input_ids"]))
        ids = payload["input_ids"]
        if "cursor" in payload and not self.unpatched:
            held = self.stores[addr].get(payload["cursor"])
            mount = payload.get("mount") and not self.no_mount and payload["mount"][0]
            self.mounts.append(payload.get("mount"))
            if held is None or mount and self.stores[addr].get(mount["cursor"]) is None:
                raise RuntimeError("Failed after 3 retries") from aiohttp.ClientResponseError(None, (), status=404)
            ids = held.tolist() + ([-len(self.sent)] * (mount["end"] - mount["start"]) if mount else []) + ids
        out, finish = self.script.pop(0)
        if finish == "abort":
            self.on_abort()
        rid = f"g{len(self.sent)}"
        self.stores[addr].put(rid, ids + out)
        self.inputs.append(ids)
        return {"output_ids": out, "meta_info": {"id": rid, "finish_reason": {"type": finish}, "prompt_tokens": len(ids),
                                                  "output_token_logprobs": [[-t / 10, t, None] for t in out]}}


def engine(monkeypatch, fake):
    e = rie.RemoteInfEngine(InferenceEngineConfig(), SGLangBackend())
    e.addresses, e.logger = ["a:1", "b:1"], rie.logger
    e.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    monkeypatch.setattr(rie, "arequest_with_retry", fake)

    async def no_session():
        return None
    monkeypatch.setattr(rie.workflow_context, "get_aiohttp_session", no_session)
    return e


def ask(e, ids, parent=None, **kw) -> ModelResponse:
    gconfig = GenerationHyperparameters(max_new_tokens=64, stop_token_ids=[EOS, PAD])
    return asyncio.run(e.agenerate(ModelRequest(input_ids=list(ids), gconfig=gconfig, tokenizer=TOK, image_data=None,
                                                parent=parent, **kw)))


def child(r: ModelResponse, delta: list[int]) -> list[int]:
    """AReaL concat's next prompt after reply r: its tokens with eos for the stop token, + the new tokens."""
    return r.input_tokens + r.output_tokens_without_stop + [EOS] + delta


def test_children_send_deltas_to_the_parents_server(monkeypatch):
    """The root goes whole. Its children (after a reply stopped on eos, after one cut by length) and a second child of
    the root (a fork) send the parent's id + the tokens after its input + output, to the parent's server (round robin
    would pick b next), and the server sees each whole prompt."""
    fake = FakeSGLang([([5, 6, EOS], "stop"), ([7, 8], "length"), ([3, EOS], "stop"), ([4, EOS], "stop")])
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    r1 = ask(e, child(r0, [11, 12]), r0)
    r2 = ask(e, child(r1, [13]), r1)
    r3 = ask(e, child(r0, [14]), r0)
    assert fake.sent == [("a:1", None, [1, 2, 3]), ("a:1", "g1", [11, 12]), ("a:1", "g2", [EOS, 13]),
                         ("a:1", "g1", [14])]
    assert fake.inputs == [r.input_tokens for r in (r0, r1, r2, r3)]
    assert [r.cursor for r in (r0, r1, r2, r3)] == [("a:1", f"g{k}") for k in range(1, 5)]


def test_endoftext_stop_parent_sends_full(monkeypatch):
    """A reply that stopped on <|endoftext|> (Qwen3.5's pad): concat puts eos in its place, so the stored tokens are
    not a prefix of the next prompt."""
    fake = FakeSGLang([([5, 6, PAD], "stop"), ([7, EOS], "stop")])
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    r1 = ask(e, child(r0, [11]), r0)
    assert fake.sent[1] == ("a:1", None, [1, 2, 3, 5, 6, EOS, 11]) and r1.cursor == ("a:1", "g2")


def test_other_server_parent_sends_full(monkeypatch):
    """A request whose rid is already pinned to another server than the parent's cursor."""
    fake = FakeSGLang([([5, EOS], "stop"), ([7, EOS], "stop")])
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    e.rid_to_address["x"] = "b:1"
    ask(e, child(r0, [11]), r0, rid="x")
    assert fake.sent[1] == ("b:1", None, [1, 2, 3, 5, EOS, 11])


def test_evicted_cursor_resends_full(monkeypatch):
    """SGLang answers 404 (LRU eviction, restart): the request goes again with all input ids, and its reply is a cursor
    again for the next one."""
    fake = FakeSGLang([([5, EOS], "stop"), ([7, EOS], "stop"), ([8, EOS], "stop")])
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    fake.stores["a:1"].seqs.clear()
    r1 = ask(e, child(r0, [11]), r0)
    ask(e, child(r1, [12]), r1)
    assert fake.sent[1:] == [("a:1", "g1", [11]), ("a:1", None, [1, 2, 3, 5, EOS, 11]), ("a:1", "g3", [12])]
    assert r1.output_tokens == [7, EOS] and r1.cursor == ("a:1", "g3")


def test_unpatched_server_fails_loudly(monkeypatch):
    """A server without the cursor patch answers 200 for the delta alone; its prompt_tokens give it away."""
    fake = FakeSGLang([([5, EOS], "stop"), ([7, EOS], "stop")], unpatched=True)
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    with pytest.raises(RuntimeError, match="a:1 ignored cursor g1: it needs patches/apply_sglang_cursor.py"):
        ask(e, child(r0, [11]), r0)


def test_resume_continues_from_the_aborted_request(monkeypatch):
    """Interrupted by a weight update: the resumed segment sends the aborted request's id and no tokens; versions per
    segment; the reply's cursor is the last segment's."""
    fake = FakeSGLang([([5, 6], "abort"), ([7, EOS], "stop"), ([8, EOS], "stop")])
    e = engine(monkeypatch, fake)
    fake.on_abort = lambda: e.set_version(1)
    r = ask(e, [1, 2, 3])
    ask(e, child(r, [11]), r)
    assert fake.sent == [("a:1", None, [1, 2, 3]), ("a:1", "g1", []), ("a:1", "g2", [11])]
    assert (r.input_tokens, r.output_tokens, r.output_versions) == ([1, 2, 3], [5, 6, 7, EOS], [0, 0, 1, 1])
    assert r.cursor == ("a:1", "g2") and fake.inputs[1] == [1, 2, 3, 5, 6]


def test_proxy_passes_the_parent(monkeypatch):
    """Through proxy_rollout_server.chat_completions: a cursor request and a full message list whose prefix match finds
    the parent both hop from the parent's cursor."""
    import transformers
    from areal.experimental.openai.client import ArealOpenAI
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData
    from test_delta_render import QWEN35

    path = next((p for p in QWEN35 if p and os.path.isdir(p)), None) or pytest.skip("no Qwen3.5 tokenizer here")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    fake = FakeSGLang([(tok.encode(f"Reply {k}.") + [tok.eos_token_id], "stop") for k in range(3)])
    srv._openai_client = ArealOpenAI(engine=engine(monkeypatch, fake), tokenizer=tok, chat_template_type="concat")
    srv._session_cache.clear()
    srv._session_cache["s"] = SessionData(session_id="s")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."}]
    ask_ = lambda body: asyncio.run(srv.chat_completions({"max_completion_tokens": 16, **body}, session_id="s"))
    r0 = ask_({"messages": msgs})
    r1 = ask_({"messages": [{"role": "user", "content": "Again."}], "cursor": r0.id})
    hist = msgs + [{"role": "assistant", "content": "Reply 0."}, {"role": "user", "content": "Again."},
                   {"role": "assistant", "content": "Reply 1."}, {"role": "user", "content": "More."}]
    ask_({"messages": hist})
    assert [c for _, c, _ in fake.sent] == [None, "g1", "g2"]
    assert all(len(ids) < 20 for _, c, ids in fake.sent if c) and r1.id in srv._session_cache["s"].completions


# Read (docs/cross-branch-read.md): a request that mounts another reply's tokens ---------------------------------------
def test_read_mounts_on_the_parents_cursor_and_counts_placeholders(monkeypatch):
    """B's request continues its parent's cursor with a mount of A's tokens [5, 7) on the same server; the server
    puts M = 2 placeholders between B's tokens and the new ones, which prompt_tokens counts on the mount request and on
    every later request from B's cursors. The reply records them (at = B's tokens the server held); its own tokens are
    the real ones. A mount of a reply that holds a mount itself is counted in the server's tokens, its placeholders
    inside the range too (a transitive read)."""
    fake = FakeSGLang([([5, EOS], "stop")] * 7)
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    a1 = ask(e, child(r0, [11, 12]), r0)
    b1 = ask(e, child(r0, [21]), r0)
    b2 = ask(e, child(b1, [22]), b1, mount=(a1, 5, 7))
    assert fake.sent[-1] == ("a:1", "g3", [22]) and fake.mounts[-1] == [{"cursor": "g2", "start": 5, "end": 7}]
    assert fake.inputs[-1] == [1, 2, 3, 5, EOS, 21, 5, EOS, -4, -4, 22]
    assert b2.input_tokens == child(b1, [22]) and b2.placeholders == ((8, 2),) and b2.cursor == ("a:1", "g4")
    b3 = ask(e, child(b2, [23]), b2)
    assert fake.sent[-1] == ("a:1", "g4", [23]) and fake.mounts[-1] is None and b3.placeholders == ((8, 2),)
    ask(e, child(a1, [13]), a1, mount=(b3, 9, 11))  # b3's tokens 9, 10 sit after its 2 placeholders on the server
    assert fake.mounts[-1] == [{"cursor": "g5", "start": 11, "end": 13}]
    ask(e, child(a1, [14]), a1, mount=(b3, 7, 10))  # b3's 2 placeholders sit before its token 8: inside the range
    assert fake.mounts[-1] == [{"cursor": "g5", "start": 7, "end": 12}]


def test_server_without_the_mount_patch_fails_loudly(monkeypatch):
    """It drops "mount" and answers for B's tokens + the new ones; prompt_tokens lacks the M placeholders."""
    fake = FakeSGLang([([5, EOS], "stop")] * 3, no_mount=True)
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    a1 = ask(e, child(r0, [11, 12]), r0)
    with pytest.raises(RuntimeError, match="a:1 ignored the mount of g2: it needs patches/apply_sglang_mount.py"):
        ask(e, child(r0, [21]), r0, mount=(a1, 5, 7))


def test_a_404_never_drops_a_read(monkeypatch):
    """A mount request whose mounted cursor is gone, and a later request from a mounted reply whose cursor is gone,
    fail with an explicit error instead of the resend of all ids (it would run without the mount). So does a request
    that cannot continue its mounted parent's cursor (a rid pinned to another server) and a mount of a cursor on
    another server than the reader's."""
    fake = FakeSGLang([([5, EOS], "stop")] * 3)
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    a1 = ask(e, child(r0, [11, 12]), r0)
    b1 = ask(e, child(r0, [21]), r0, mount=(a1, 5, 7))
    del fake.stores["a:1"].seqs["g2"]
    with pytest.raises(RuntimeError, match="cursor g1 or a cursor it mounts is gone from a:1: a send of all 6 input"):
        ask(e, child(r0, [22]), r0, mount=(a1, 5, 7))
    del fake.stores["a:1"].seqs["g3"]
    with pytest.raises(RuntimeError, match="cursor g3 or a cursor it mounts is gone from a:1"):
        ask(e, child(b1, [23]), b1)
    assert all(c is not None for _, c, _ in fake.sent[1:])  # after the root, never all ids
    e.rid_to_address["x"] = "b:1"
    with pytest.raises(RuntimeError, match="cannot continue its parent's cursor on b:1"):
        ask(e, child(b1, [24]), b1, rid="x")
    other = ModelResponse(input_tokens=[1, 2], output_tokens=[3, EOS], tokenizer=TOK, cursor=("b:1", "z"))
    with pytest.raises(RuntimeError, match=r"mounts cursor \('b:1', 'z'\): a read needs both threads on one server"):
        ask(e, child(r0, [25]), r0, mount=(other, 0, 2))


def test_a_read_cut_before_its_first_token_goes_again_with_its_mount(monkeypatch):
    """Aborted before any output (a weight-update pause): the next segment is the same mount request from the parent's
    cursor, not a continuation of the aborted one; the placeholders are recorded once."""
    fake = FakeSGLang([([5, EOS], "stop"), ([6, EOS], "stop"), ([], "abort"), ([7, EOS], "stop")])
    e = engine(monkeypatch, fake)
    r0 = ask(e, [1, 2, 3])
    a1 = ask(e, child(r0, [11, 12]), r0)
    b = ask(e, child(r0, [21]), r0, mount=(a1, 5, 7))
    mount = [{"cursor": "g2", "start": 5, "end": 7}]
    assert fake.sent[2:] == [("a:1", "g1", [21])] * 2 and fake.mounts[-2:] == [mount, mount]
    assert b.placeholders == ((5, 2),) and b.output_tokens == [7, EOS] and b.cursor == ("a:1", "g4")


def test_proxy_turns_a_read_into_a_token_range(monkeypatch):
    """Through proxy_rollout_server.chat_completions: B's cursor request with mount {from: A's reply 1, to: A's reply
    3} mounts A's tokens from reply 1's turn (its <|im_start|>assistant) to reply 3's, on reply 3's cursor. B's
    interaction records it for the trainer (source, range in the source's tokens, at = B's tokens before the new
    ones); B's tokens and the exported leaves hold no placeholder."""
    import transformers
    from areal.experimental.openai.client import ArealOpenAI
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData
    from test_delta_render import QWEN35

    path = next((p for p in QWEN35 if p and os.path.isdir(p)), None) or pytest.skip("no Qwen3.5 tokenizer here")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    fake = FakeSGLang([(tok.encode(f"Reply {k}.") + [tok.eos_token_id], "stop") for k in range(7)])
    srv._openai_client = ArealOpenAI(engine=engine(monkeypatch, fake), tokenizer=tok, chat_template_type="concat")
    srv._session_cache.clear()
    srv._session_cache["s"] = session = SessionData(session_id="s")
    ask_ = lambda body: asyncio.run(srv.chat_completions({"max_completion_tokens": 16, **body}, session_id="s"))  # noqa
    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    a = [ask_({"messages": [{"role": "system", "content": "Be brief."}] + user("Hi.")})]
    for k in (1, 2, 3):
        a.append(ask_({"messages": user(f"A{k}."), "cursor": a[-1].id}))
    b1 = ask_({"messages": user("B1."), "cursor": a[0].id})
    b2 = ask_({"messages": user("B2."), "cursor": b1.id, "mount": {"thread": "A", "from": a[1].id, "to": a[3].id}})
    src, b1r, b2r = (session.completions[r.id].model_response for r in (a[3], b1, b2))
    m = session.completions[b2.id].metadata["mount"]
    assert m == {"thread": "A", "source": a[3].id, "start": m["start"], "end": m["end"],
                 "at": len(b1r.input_tokens) + len(b1r.output_tokens)}
    assert fake.mounts[-1] == [{"cursor": src.cursor[1], "start": m["start"], "end": m["end"]}]
    text = tok.decode(src.input_tokens[m["start"]:m["end"]])
    assert text.startswith("<|im_start|>assistant\n") and text.endswith("<|im_end|>\n"), text
    assert all(x in text for x in ("Reply 1.", "A2.", "Reply 2.", "A3."))
    assert "A1." not in text and "Reply 3" not in text
    assert min(b2r.input_tokens) >= 0 and b2r.input_tokens[:m["at"]] == b1r.input_tokens + b1r.output_tokens
    assert b2r.placeholders == ((m["at"], m["end"] - m["start"]),)
    # A has ended (delta_render.read_range): its last reply is read through its end, "from" may be "to"
    b3 = ask_({"messages": user("B3."), "cursor": b2.id, "mount": {"thread": "A", "from": a[3].id, "to": a[3].id,
                                                                   "ended": True}})
    m3 = session.completions[b3.id].metadata["mount"]
    assert m3["end"] == len(src.input_tokens) + len(src.output_tokens) and m3["start"] == m["end"]
    assert fake.mounts[-1] == [{"cursor": src.cursor[1], "start": m3["start"], "end": m3["end"]}]
    text = tok.decode((src.input_tokens + src.output_tokens)[m3["start"]:])
    assert text.startswith("<|im_start|>assistant\n") and text.endswith("Reply 3.<|im_end|>") and "A3." not in text
    leaves = session.export_interactions(discount=1.0, style="concat", drop_retry_orphans=True)
    assert all(int(t["input_ids"].min()) >= 0 for t in (x.to_tensor_dict() for x in leaves.values()))
    # the trainer replays [start, end) of the source's leaf (prefix_tree.record_reads): A's last reply is that leaf
    assert leaves[a[3].id].to_tensor_dict()["input_ids"].shape[-1] == m3["end"]


def test_route_servers_keep_each_episode_tree_on_one_server(monkeypatch):
    """OCTOPUS_ROUTE_SERVERS (patches/areal-route-spread.patch, after the cursor patch). (c) New sessions go to the least
    loaded route server (octopus/serving/route.py), in turn here: each request ends before the next. (a) A request on
    an existing cursor goes to the server holding it, also when its session's route entry is gone (the next free
    request of that session is a new session's). (b) A session's requests without
    a parent cursor stay on its server, so an episode tree (one proxy session) stays on one server and reads its own
    threads there; a source on another server is an error."""
    monkeypatch.setenv("OCTOPUS_ROUTE_SERVERS", "a:1,b:1,c:1")
    fake = FakeSGLang([([5, EOS], "stop")] * 10)
    e = engine(monkeypatch, fake)
    e.route.turn = 0
    # one request of proxy session s: the proxy sets ROUTE_KEY in that request's context
    on = lambda s, *a, **kw: contextvars.Context().run(lambda: (rie.ROUTE_KEY.set(s), ask(e, *a, **kw))[1])  # noqa
    roots = [on(s, [1, 2, k]) for k, s in enumerate(("s1", "s2", "s3", "s4"))]
    assert [r.cursor[0] for r in roots] == ["a:1", "b:1", "c:1", "a:1"]  # (c)
    s1, s2 = roots[:2]
    assert on("s2", child(s2, [11]), s2).cursor[0] == "b:1" and fake.sent[-1] == ("b:1", "g2", [11])
    e.route.session.clear()
    a1 = on("s1", child(s1, [12, 13]), s1)  # (a): the cursor's server, not the next server in turn
    assert a1.cursor[0] == "a:1" and fake.sent[-1] == ("a:1", "g1", [12, 13])
    assert on("s1", [1, 2, 9]).cursor[0] == "b:1"  # a free request after the entry was dropped: a new session
    e.route.session["s1"] = "a:1"
    assert on("s1", [1, 2, 8]).cursor[0] == "a:1"  # (b): a free request of a known session keeps its server
    b1 = on("s1", child(s1, [21]), s1)
    on("s1", child(b1, [22]), b1, mount=(a1, 5, 7))  # a read of the session's own thread, on its server
    assert fake.sent[-1] == ("a:1", b1.cursor[1], [22]) and fake.mounts[-1][0]["cursor"] == a1.cursor[1]
    with pytest.raises(RuntimeError, match="a read needs both threads on one server"):  # (b): source on b:1
        on("s1", child(b1, [23]), b1, mount=(s2, 0, 2))


# Live: the proxy on the real engine -----------------------------------------------------------------------------------
URL = os.environ.get("OCTOPUS_SGLANG_URL")
live = pytest.mark.skipif(not URL, reason="needs a SGLang server with the slice-4 overlay: OCTOPUS_SGLANG_URL")
NOTE = "Do not call any tool in this reply. Write the integers from 1 to 400, one per line, and nothing else."
# key, parent, client request (full message list / cursor), user note, extra body; one request at a time
STEPS = [("r0", None, "full", None, {}), ("r1", "r0", "cursor", None, {}),
         ("r2", "r1", "full", None, {}),  # a full message list: the proxy's prefix match finds r1
         ("cut", "r2", "cursor", None, {"max_completion_tokens": 8}), ("cut1", "cut", "full", None, {}),
         ("fa", "r2", "cursor", "You are branch-0 of 2. Your task: Task 1.", {}),
         ("fb", "r2", "cursor", "You are branch-1 of 2. Your task: Task 2.", {}), ("fa1", "fa", "full", None, {}),
         ("rw", "r1", "cursor", "Rewound to checkpoint 1: the edit after it is undone.", {}),
         ("long", "rw", "cursor", NOTE, {"max_completion_tokens": 1500}),  # interrupted by a weight update
         ("long1", "long", "cursor", None, {}), ("ev", "long1", "cursor", None, {})]  # ev: long1's cursor evicted


def results(msg: dict, note=None) -> list[dict]:
    calls = msg.get("tool_calls") or []
    out = [{"role": "tool", "tool_call_id": t["id"], "content": f"{t['function']['name']}: exit 0\nok {k}\n"}
           for k, t in enumerate(calls)] or [{"role": "user", "content": "That was not a tool call. Call a tool."}]
    return out + ([{"role": "user", "content": note}] if note else [])


async def run(tok, mode: str, split: int | None = None, steps=STEPS):
    """`steps` (STEPS) through proxy_rollout_server.chat_completions on RemoteSGLangEngine (the proxy's client as
    _setup_openai_client builds it, Qwen3.5 VL processor). mode "full": SGLang's request ids are dropped, so no reply
    has a cursor and every /generate carries all ids (today). "cursor": the patch. "long" is interrupted as AReaL
    does for a weight update (executor paused, /pause_generation aborts it, version + 1, /continue_generation); the
    full run, given the cursor run's abort point `split`, cuts its first segment there instead, so both runs see the
    same segments. Returns {key: (prompt, output, logprobs, versions)}, parent links, exported concat leaves and the
    /generate requests (key, cursor, ids sent, bytes)."""
    from areal.experimental.openai.client import ArealOpenAI
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData
    from areal.utils.hf_utils import load_hf_processor_and_tokenizer
    from test_delta_render import EPISODE

    eng = RemoteSGLangEngine(InferenceEngineConfig(request_retries=3))
    inner, paused, real, sent, now = eng._engine, threading.Event(), rie.arequest_with_retry, [], {}
    inner.addresses, inner.logger = [URL.removeprefix("http://")], rie.logger
    inner.workflow_executor = SimpleNamespace(is_paused=paused.is_set)
    if mode == "full":
        parse = inner.backend.parse_generation_response
        inner.backend.parse_generation_response = lambda r: SimpleNamespace(**{**vars(parse(r)), "rid": None})

    async def record(**kw):
        p = kw["payload"]
        if kw["endpoint"] == "/generate":
            sent.append((now["key"], p.get("cursor"), len(p["input_ids"]), len(json.dumps(p))))
            if now.get("split"):  # the full run's interrupted segment
                kw["payload"] = {**p, "sampling_params": {**p["sampling_params"], "max_new_tokens": now.pop("split")}}
                r = await real(**kw)
                r["meta_info"]["finish_reason"] = {"type": "abort"}
                inner.set_version(inner.get_version() + 1)
                return r
        return await real(**kw)
    rie.arequest_with_retry = record

    async def interrupt():
        await asyncio.sleep(2)
        paused.set()
        await asyncio.to_thread(inner.pause_generation)
        inner.set_version(inner.get_version() + 1)
        await asyncio.to_thread(inner.continue_generation)
        paused.clear()

    srv._openai_client = ArealOpenAI(
        engine=eng, tokenizer=tok, processor=load_hf_processor_and_tokenizer(tok.name_or_path)[0],
        tool_call_parser="qwen3_coder", reasoning_parser="qwen3", engine_max_tokens=262144,
        chat_template_type="concat", require_multimodal_processor=True)
    srv._session_cache.clear()
    srv._session_cache["s"] = session = SessionData(session_id="s")
    ids, hist, out = {}, {}, {}
    try:
        for key, parent, client, note, extra in steps:
            now["key"] = key
            delta = EPISODE["main"][:2] if parent is None else results(hist[parent][-1], note)
            body = {"messages": hist[parent] + delta if client == "full" and parent else delta,
                    "tools": EPISODE["tools"], "chat_template_kwargs": EPISODE["chat_template_kwargs"],
                    "temperature": 0, "max_completion_tokens": 400, **extra}
            if client == "cursor" and parent:
                body["cursor"] = ids[parent]
            if key == "ev" and mode == "cursor":  # SGLang lost long1's tokens: the hop gets a 404
                pr = session.completions[ids[parent]].model_response
                pr.cursor = (pr.cursor[0], "evicted")
            if key == "long" and mode == "full":
                now["split"] = split
            tasks = [srv.chat_completions(body, session_id="s")] + ([interrupt()] if key == "long" and
                                                                    mode == "cursor" else [])
            r = (await asyncio.gather(*tasks))[0]
            ids[key] = r.id
            hist[key] = (hist[parent] if parent else []) + delta + [r.choices[0].message.model_dump(exclude_none=True)]
            m = session.completions[r.id].model_response
            out[key] = (m.input_tokens, m.output_tokens, m.output_logprobs, m.output_versions)
    finally:
        rie.arequest_with_retry = real
    key_of = {v: k for k, v in ids.items()}
    parents = {key_of[i]: x.parent and key_of[x.parent.interaction_id] for i, x in session.completions.items()}
    leaves = session.export_interactions(discount=1.0, style="concat", drop_retry_orphans=True)
    return out, parents, {key_of[i]: x.to_tensor_dict() for i, x in leaves.items()}, sent


@live
def test_live_cursor_hops_equal_full_ids():
    import transformers
    from test_delta_render import QWEN35

    tok = transformers.AutoTokenizer.from_pretrained(next(p for p in QWEN35 if p and os.path.isdir(p)))
    cur = asyncio.run(run(tok, "cursor"))
    segments = [s for s in cur[3] if s[0] == "long"]  # from rw's cursor, then from the aborted request's, no ids
    assert len(segments) == 2 and segments[0][1] and segments[1][1] and segments[1][2] == 0, segments
    versions = cur[0]["long"][3]
    split = versions.count(versions[0])
    assert 0 < split < len(versions), "the interrupt missed the generation"
    full = asyncio.run(run(tok, "full", split))
    assert cur[0] == full[0]  # prompt, output tokens, logprobs, versions of every reply
    assert cur[1] == full[1]
    assert cur[2].keys() == full[2].keys()
    for key, t in full[2].items():
        for name, v in t.items():
            if hasattr(v, "dtype"):
                assert v.dtype == cur[2][key][name].dtype and v.equal(cur[2][key][name]), (key, name)
    assert [s[0] for s in cur[3] if s[1] is None] == ["r0", "ev"]  # the root, and the resend after the 404
    assert all(s[1] is None for s in full[3])
    by_key = defaultdict(list)
    for s in full[3]:
        by_key[s[0]].append(s)
    print("\n| request | prompt tokens | full ids: ids sent, bytes | cursor hop: ids sent, bytes |\n|---|---|---|---|")
    for key, *_ in STEPS:
        print(f"| {key} | {len(cur[0][key][0]):,} | " + "; ".join(f"{n:,}, {b:,}" for _, _, n, b in by_key[key])
              + " | " + "; ".join(f"{'cursor ' if c else 'full '}{n:,}, {b:,}" for k, c, n, b in cur[3] if k == key)
              + " |")


@live
def test_live_history_past_128k(tmp_path, monkeypatch):
    """One thread grown past 150k tokens (each turn: the reply's results + a ~10k-token log as a user note) through the
    proxy and real cursor hops, with OCTOPUS_CURSOR_AUDIT on: every cursor prompt equals AReaL's full-mode concat, and
    the client's messages, the /generate body and the proxy's prompt build stay flat while full mode's grow with L."""
    import transformers
    from test_delta_render import QWEN35

    from octopus.serving import cursor_audit

    monkeypatch.setenv("OCTOPUS_CURSOR_AUDIT", str(tmp_path))
    monkeypatch.setattr(cursor_audit, "DIR", str(tmp_path))
    tok = transformers.AutoTokenizer.from_pretrained(next(p for p in QWEN35 if p and os.path.isdir(p)))
    rng = random.Random(0)
    words = "error warning retry timeout socket parser config cache worker thread queue token request".split()
    log = lambda i: f"Part {i} of the build log; reply with one short sentence.\n" + "\n".join(  # noqa: E731
        f"[{rng.randrange(10**6):06d}] {rng.choice(words)} {rng.choice(words)} in module_{rng.randrange(500)}.py:"
        f"{rng.randrange(2000)} ({rng.random():.4f})" for _ in range(340))
    steps = [("r0", None, "full", None, {})] + [(f"l{i}", f"l{i - 1}" if i > 1 else "r0", "cursor", log(i),
                                                 {"max_completion_tokens": 32}) for i in range(1, 17)]
    _, _, _, sent = asyncio.run(run(tok, "cursor", steps=steps))
    recs = [json.loads(x) for f in tmp_path.glob("*.jsonl") for x in f.read_text().splitlines()]
    prompts = [r for r in recs if r["kind"] == "prompt" and r["cursor"]]
    hops = {r["rid"]: r for r in recs if r["kind"] == "hop"}
    assert len(prompts) == len(steps) - 1 and all(r["match"] for r in prompts)
    assert max(r["L"] for r in prompts) > 150_000
    assert [s[1] is None for s in sent] == [True] + [False] * (len(steps) - 1)  # the root, then cursor hops only
    print("\n| L | Δ tokens | client messages bytes, cursor | full | prompt build ms, cursor | full render+tokenize "
          "| /generate bytes, cursor | all ids |\n|---|---|---|---|---|---|---|---|")
    for r in prompts:
        h = hops[r["rid"]]
        print(f"| {r['L']:,} | {r['delta_tokens']:,} | {r['msg_bytes']:,} | {r['full_msg_bytes']:,} | {r['build_ms']:.2f} "
              f"| {r['oracle_ms']:.1f} | {h['bytes']:,} | {h['full_bytes']:,} |")


@live
def test_live_forced_read_episode(monkeypatch, tmp_path):
    """The read tool live: the Harbor agent (tests/test_rl_plan.py's fake container, cursor mode, OCTOPUS_READ_TOOL=1)
    against the proxy on RemoteSGLangEngine and one SGLang with the mount overlay (radix cache; any mamba strategy,
    overlap on or off (#172); the first recorded live run used no_buffer and overlap off; every request greedy: the
    agent sends no temperature, the proxy's default is 1.0). Replies are forced, except the free ones: the scripted
    call's text as its canonical tokens, sent as input, plus the one token the model generates after it (eos), so each
    is a server reply with its cursor. main calls and forks: branch-0 makes two calls that find a key and commits;
    branch-1 first asks for main's step 1 (in its own history: refused) and branch-0's step 3, repeats the read until it
    mounts, replies freely once and commits. After the join main calls read_thread(branch-0, 3, 4), replies freely and
    is done. Each free reply's /generate body carries its mount on branch-0's cursor from the reader's parent cursor; the reader's interaction
    keeps the record; the next request from a mounted reply passes the prompt_tokens check; the free replies'
    log-probs equal the offline K2 computation (transformers + flash-attn 2 + fla, kv_read_probe.Probe on the visible
    GPU: prefill the reader's tokens before the mount and branch-0's up to its end, append the mounted K/V to the
    full-attention layers, continue at max(at, end)) within #132's bars (mean |d| <= 0.03, argmax agreement >= 0.98);
    nothing stays locked."""
    import torch
    import transformers
    from areal.experimental.openai.client import ArealOpenAI
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData
    from test_areal_cursor import reply_text
    from test_delta_render import EPISODE, QWEN35
    from test_rl_plan import DONE, _call, _commit, _fork_call, _run_tree

    from octopus.tools.kv_read_probe import Probe

    path = next(p for p in QWEN35 if p and os.path.isdir(p))
    tok = transformers.AutoTokenizer.from_pretrained(path)
    eng = RemoteSGLangEngine(InferenceEngineConfig(request_retries=3))
    inner, real, sent = eng._engine, rie.arequest_with_retry, []
    inner.addresses, inner.logger = [URL.removeprefix("http://")], rie.logger
    inner.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    generate = eng.agenerate

    async def agenerate(req):  # metadata["force"]: the reply's text, exactly its canonical tokens, then the model's
        if not (text := req.metadata.get("force")):  # own next token (a grammar left the tokenization and the stop
            return await generate(req)  # token, often <|endoftext|>, which ends a cursor chain, to chance)
        forced = tok.encode(text, add_special_tokens=False)
        pre = req.copy()
        pre.input_ids, pre.gconfig.max_new_tokens = req.input_ids + forced, 1
        r = await generate(pre)  # its cursor holds the prompt + the text + that token
        assert r.output_tokens == [tok.eos_token_id], f"no eos after the forced text {text!r}: {r.output_tokens}"
        n = len(forced)
        return dataclasses.replace(r, input_tokens=req.input_ids, output_tokens=forced + r.output_tokens,
                                   output_logprobs=[0.0] * n + r.output_logprobs,
                                   output_versions=r.output_versions[:1] * n + r.output_versions,
                                   output_top_logprobs=None if r.output_top_logprobs is None else
                                   [[] for _ in range(n)] + r.output_top_logprobs)

    async def record(**kw):
        if kw["endpoint"] == "/generate":
            sent.append(kw["payload"])
        return await real(**kw)
    eng.agenerate = agenerate
    monkeypatch.setattr(rie, "arequest_with_retry", record)
    monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", json.dumps(EPISODE["chat_template_kwargs"]))
    monkeypatch.setenv("OCTOPUS_MAX_TOKENS", "400")
    srv._openai_client = ArealOpenAI(engine=eng, tokenizer=tok, tool_call_parser="qwen3_coder", reasoning_parser="qwen3",
                                     engine_max_tokens=131072, chat_template_type="concat")
    srv._session_cache.clear()
    srv._session_cache["s"] = session = SessionData(session_id="s")
    loop = asyncio.new_event_loop()  # the proxy's, as in its server; the agent runs its own
    threading.Thread(target=loop.run_forever, daemon=True).start()
    shell = lambda c: _call("shell", {"command": c})  # noqa: E731
    read = lambda t, a, b: _call("read_thread", {"thread": t, "from_step": a, "to_step": b})  # noqa: E731
    b0, b1, state = "fork0/branch-0", "fork0/branch-1", {}

    def reply(thread, turn):  # None: a free reply; branch-1's are chosen in serve, from what it has seen
        if thread == "main":
            return {1: shell("ls"), 2: _fork_call(5, 6), 3: read("branch-0", 3, 4), 4: None}.get(turn, DONE)
        if thread == b0:
            return {1: shell("grep -rn DEPLOY_KEY ."), 2: shell("cat deploy.cfg")}.get(turn, _commit(thread))
        return []

    def branch_1(turn, body):
        if state.get(b1) == "free" or turn > 8:
            return _commit(b1)
        if "Mounted branch-0's steps 3 to 3" in json.dumps(body["messages"]):
            state[b1] = "free"
            return None
        return read("main", 1, 1) + read("branch-0", 3, 3) if turn == 1 else read("branch-0", 3, 3)
    key, bodies, free = {}, {}, set()

    def serve(rid, body, calls):
        if rid.startswith(b1 + "#"):
            calls = branch_1(int(rid.split("#")[1]), body)
        md = {"force": reply_text(tok, {"role": "assistant", "content": "", "tool_calls": calls})} if calls else {}
        r = asyncio.run_coroutine_threadsafe(srv.chat_completions({**body, "temperature": 0, "metadata": md}, session_id="s"),
                                             loop).result()
        key[r.id], bodies[rid] = rid, body
        if not calls:
            free.add(r.id)
        return r.model_dump(exclude_none=True)

    def exec_out(command):  # branch-0's two calls find a key (the tool runner's stdout: exit code, then output)
        if command.startswith("python3 /tmp/harbor_tools.py "):
            args = json.loads(base64.b64decode(command.split()[2]))["args"]
            if "deploy" in str(args.get("command")).lower():
                return "exit 0\ndeploy.cfg:1:DEPLOY_KEY=QX-4471-ZEBRA\n"
        return None
    _run_tree(monkeypatch, tmp_path, reply, serve=serve, exec_out=exec_out, OCTOPUS_HTTP_WORKERS="1",
              OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1")
    ids = {rid: i for i, rid in key.items()}
    results = lambda rid: [m["content"].split("\n", 1)[1] for m in bodies[rid]["messages"] if m["role"] == "tool"]
    assert results(f"{b1}#2")[0] == "main's step 1 is in your own history already (main has no readable step yet). " \
                                    "Nothing was read."
    reads = {rid: b["mount"] for rid, b in bodies.items() if "mount" in b}
    b1_read = next(rid for rid in reads if rid.startswith(b1))
    assert reads == {"main#4": {"thread": "branch-0", "from": ids[f"{b0}#1"], "to": ids[f"{b0}#3"]},
                     b1_read: {"thread": "branch-0", "from": ids[f"{b0}#1"], "to": ids[f"{b0}#2"]}}
    assert results("main#4")[0].startswith("Mounted branch-0's steps 3 to 4 (")
    assert len([p for p in sent if "mount" in p]) == 2
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2").cuda().eval()
    p, d, agree, report = Probe(model), [], [], {}
    for rid in reads:
        me = session.completions[ids[rid]]
        m = me.metadata["mount"]
        at, s, e = m["at"], m["start"], m["end"]
        b, src, parent = me.model_response, session.completions[m["source"]].model_response, me.parent.model_response
        assert m == {"thread": "branch-0", "source": reads[rid]["to"], "start": s, "end": e,
                     "at": len(parent.input_tokens) + len(parent.output_tokens)}
        hop = next(x for x in sent if x.get("mount") and x["cursor"] == parent.cursor[1])
        assert hop["mount"] == [{"cursor": src.cursor[1], "start": s, "end": e}]
        assert b.placeholders == ((at, e - s),) and min(b.input_tokens) >= 0
        assert "QX-4471-ZEBRA" in tok.decode(src.input_tokens[s:e])
        later = [x.model_response for x in session.completions.values() if x.parent is me]  # from the mounted reply
        assert all(r.placeholders == b.placeholders for r in later)  # empty if the free reply called no tool
        assert ids[rid] in free
        cont = b.input_tokens[at:] + b.output_tokens  # the new tokens, then the free reply
        with torch.no_grad():
            _, cache_b = p.run(b.input_tokens[:at], 0)
            _, cache_a = p.run(src.input_tokens[:e], 0)
            kv = p.kv(cache_a, s, e)
            del cache_a
            h, _ = p.run(cont, max(at, e), p.mounted(cache_b, kv, 0))
            lp = p.logprobs(h[len(b.input_tokens) - at - 1:-1])
        target = torch.tensor(b.output_tokens, device=lp.device)
        dk = [abs(x - y) for x, y in zip(b.output_logprobs, lp.gather(1, target[:, None])[:, 0].tolist())]
        d, agree = d + dk, agree + (lp.argmax(-1) == target).tolist()
        report[rid] = {"at": at, "mount": [s, e], "M": e - s, "positions_from": max(at, e), "reply_tokens": len(dk),
                       "mean_abs": statistics.mean(dk), "max_abs": max(dk), "later": len(later),
                       "read_result": next(x for x in results(rid) if x.startswith("Mounted")),
                       "free_reply": tok.decode(b.output_tokens)[:200]}
        del cache_b, kv, h, lp
    code, info = asyncio.run(_get(URL + "/get_server_info"))
    locked = info["internal_states"][0]["octopus_full_protected"]
    print(json.dumps({"reads": report, "mean_abs": statistics.mean(d), "argmax_agree": statistics.mean(agree),
                      "requests": len(sent), "b1_results": {r: results(r) for r in sorted(bodies) if r.startswith(b1)},
                      "locked_after": locked}, indent=1))
    assert statistics.mean(d) <= 0.03 and statistics.mean(agree) >= 0.98 and locked == 0


async def _get(url: str):
    async with aiohttp.ClientSession() as s, s.get(url) as resp:
        return resp.status, await resp.json()


# Cost -------------------------------------------------------------------------------------------------------------------
def bench(targets=(8_000, 32_000, 128_000)):
    """The /generate body and the engine's cost to build it (prefix check, payload, JSON encoding as aiohttp sends it)
    for a next request after about L tokens: a 400-token reply, then one ~1.5k-token tool result; full vs cursor,
    median of 5."""
    rng, backend = random.Random(0), SGLangBackend()
    print("| L (prompt tokens) | full request bytes | cursor request bytes | full build ms | cursor build ms "
          "| of which prefix check ms |")
    print("|---|---|---|---|---|---|")
    for L in targets:
        prompt = [rng.randrange(248_000) for _ in range(L - 1_900)]
        parent = ModelResponse(input_tokens=prompt, output_tokens=[rng.randrange(248_000) for _ in range(399)] + [EOS],
                               tokenizer=TOK, cursor=("a:1", "c"))
        req = ModelRequest(input_ids=child(parent, [rng.randrange(248_000) for _ in range(1_500)]), tokenizer=TOK,
                           gconfig=GenerationHyperparameters(max_new_tokens=4096), image_data=None, parent=parent)
        rows = []
        for mode in ("full", "cursor"):
            times, checks = [], []
            for _ in range(5):
                t = time.perf_counter()
                cursor = rie._parent_cursor(req, "a:1") if mode == "cursor" else None
                checks.append(time.perf_counter() - t)
                payload = backend.build_generation_request(req, with_lora=False, version=0).payload
                if cursor:
                    payload = {**payload, "cursor": cursor[0], "input_ids": req.input_ids[cursor[1]:]}
                body = json.dumps(payload)
                times.append(time.perf_counter() - t)
            rows.append((len(body), statistics.median(times), statistics.median(checks)))
        (fb, ft, _), (cb, ct, cc) = rows
        print(f"| {len(req.input_ids):,} | {fb:,} | {cb:,} | {ft * 1e3:.2f} | {ct * 1e3:.2f} | {cc * 1e3:.2f} |")


if __name__ == "__main__":
    bench()
