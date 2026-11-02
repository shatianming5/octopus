"""SGLang cursor store (patches/apply_sglang_cursor.py, docs/cursor-protocol.md).

Store semantics run anywhere. The patch test needs an unpatched SGLang 0.5.19 importable, the prompt test the patched
one (node-94: sglang-4090d python, PYTHONPATH=<cursor overlay>:<this repo>:<pytest shim>). The live test drives a
patched server (OCTOPUS_SGLANG_URL=http://127.0.0.1:<port>, greedy, --disable-radix-cache so equal prompts give equal
outputs). `python tests/test_sglang_cursor.py` prints request bytes and server-side prompt build time at L = 8k/32k/128k.
"""

import asyncio
import json
import os
import shutil
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_delta_render import EPISODE, QWEN35, SYNTHETIC, TAG, concat_tails, tok  # noqa: F401  (tok is a fixture)

from octopus.serving.cursor_store import CursorStore
from octopus.serving.delta_render import parse_tool_arguments, render_delta

TOOLS, KWARGS = EPISODE["tools"], EPISODE["chat_template_kwargs"]
EOS, PAD = 9, 0


def reply(prompt, output, finish="stop", rid="r"):
    return {"output_ids": output, "prompt_token_ids": prompt, "meta_info": {"id": rid, "finish_reason": {"type": finish}}}


def test_store_budget_and_lru_eviction():
    s = CursorStore(budget_bytes=40)  # 10 tokens
    s.put("a", [1, 2, 3, 4])
    s.put("b", [5, 6, 7])
    assert s.used == 28 and s.get("a").tolist() == [1, 2, 3, 4]  # a is now the most recently used
    s.put("c", [8, 9, 10])  # 40 bytes: still in
    s.put("d", [11])  # over budget: b (least recently used) goes, then nothing more is needed
    assert s.get("b") is None and list(s.seqs) == ["a", "c", "d"] and s.used == 32
    s.put("a", [1])  # a replaced: its old bytes are freed
    assert s.used == 20 and s.get("a").tolist() == [1]
    s.put("e", list(range(11)))  # larger than the whole budget: everything goes, itself last
    assert not s.seqs and s.used == 0
    assert s.get("unknown") is None


def test_put_reply_is_the_concat_parent():
    s = CursorStore(budget_bytes=1 << 20)
    s.put_reply(reply([1, 2], [3, 4, EOS]), EOS, PAD, [EOS])
    assert s.get("r").tolist() == [1, 2, 3, 4, EOS]
    s.put_reply(reply([1, 2], [3, 4, EOS, PAD]), EOS, PAD, [EOS])  # every trailing stop token goes, then one eos
    assert s.get("r").tolist() == [1, 2, 3, 4, EOS]
    s.put_reply(reply([1, 2], [3, EOS], "length"), EOS, PAD, [EOS])  # cut by length or aborted: the output stays
    assert s.get("r").tolist() == [1, 2, 3, EOS, EOS]
    s.put_reply(reply([1, 2], [], "abort"), EOS, None, [EOS])  # aborted before prefill
    assert s.get("r").tolist() == [1, 2, EOS]
    ctx = [{"role": "user", "content": "stub"}, {"role": "assistant", "content": "x"}]
    s.put_reply(reply([1], [3, EOS]), EOS, PAD, [EOS], ctx)  # the stub kept for templates that read the history
    assert s.ctx["r"] is ctx
    s.put_reply(reply([1], [3, EOS]), EOS, PAD, [EOS])  # replaced without one: gone
    assert "r" not in s.ctx
    glm = reply([1], [3, 7])  # stopped on a stop of the model's own (GLM's <|observation|>), whose terminator is none
    glm["meta_info"]["finish_reason"]["matched"] = 7
    s.put_reply(glm, EOS, PAD, [])
    assert s.get("r").tolist() == [1, 3]
    assert s.prompt["r"] == 1  # the prompt's length: where a read finds the reply's turn


def test_chat_mount_finds_turns_in_the_prompt(tok, monkeypatch):  # noqa: F811  (the fixture)
    """kv_mount.prepare_chat takes a read's range as the AReaL proxy does, from each reply's prompt: a reply that wrote
    its template's generation-prompt token (<|im_start|>) does not move the start of its turn."""
    from octopus.serving import kv_mount
    from octopus.serving.delta_render import turn_start
    monkeypatch.setattr(kv_mount, "prepare", lambda obj, cursors, tm: None)
    s, im_start = CursorStore(budget_bytes=1 << 20), tok.convert_tokens_to_ids("<|im_start|>")
    a = tok.encode(tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                           add_generation_prompt=True), add_special_tokens=False)
    s.put_reply(reply(a, [1, im_start, 2, tok.eos_token_id], rid="a"), tok.eos_token_id, None, [tok.eos_token_id])
    b = s.get("a").tolist() + tok.encode("\n<|im_start|>user\ny<|im_end|>\n<|im_start|>assistant\n",
                                         add_special_tokens=False)
    s.put_reply(reply(b, [3, tok.eos_token_id], rid="b"), tok.eos_token_id, None, [tok.eos_token_id])
    obj = SimpleNamespace()
    kv_mount.prepare_chat(obj, SimpleNamespace(cursor="b", mount={"from": "a", "to": "b", "thread": "t"}), s,
                          SimpleNamespace(tokenizer=tok))
    assert obj.mount == [{"cursor": "b", "start": turn_start(tok, a), "end": turn_start(tok, b)}]
    assert turn_start(tok, a) != turn_start(tok, s.get("a"))  # the whole cursor: at the reply's own <|im_start|>
    for frm in ("a", "b"):  # an ended source: through "to"'s stored tokens (read_range); "from" may be "to"
        kv_mount.prepare_chat(obj, SimpleNamespace(cursor="b", mount={"from": frm, "to": "b", "ended": True}), s,
                              SimpleNamespace(tokenizer=tok))
        assert obj.mount == [{"cursor": "b", "start": turn_start(tok, {"a": a, "b": b}[frm]), "end": len(s.get("b"))}]
    s.put("g", s.get("a").tolist())  # a /generate reply keeps no prompt length: refused as not a chat reply
    with pytest.raises(Exception, match="does not name stored chat replies") as e:
        kv_mount.prepare_chat(obj, SimpleNamespace(cursor="b", mount={"from": "g", "to": "b"}), s,
                              SimpleNamespace(tokenizer=tok))
    assert e.value.status_code == 404


def sglang_root() -> Path:
    return Path(pytest.importorskip("sglang").__file__).parent


def test_patch_applies_once_alongside_the_lora_patch(tmp_path):
    """On a copy of the importable SGLang (already lora-patched on node-94), in either order: the second run of each is
    a no-op and the result compiles."""
    import py_compile

    import patches.apply_sglang_cursor as cursor
    import patches.apply_sglang_lora_abort_release as lora
    src = sglang_root()
    if cursor.MARK in (src / "srt/entrypoints/openai/serving_chat.py").read_text():
        pytest.skip("the importable SGLang is already cursor-patched: run without the overlay")
    for order in ((cursor, lora), (lora, cursor)):
        root = tmp_path / order[0].__name__ / "sglang"
        for rel in [*cursor.EDITS, "srt/managers/tokenizer_manager.py"]:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, root / rel)
        first = [m.patch(root) for m in order]
        assert [m.patch(root) for m in order] == [False, False]
        assert first[order.index(cursor)]
        texts = {rel: (root / rel).read_text() for rel in cursor.EDITS}
        for rel, edits in cursor.EDITS.items():
            assert all(texts[rel].count(new) == 1 for _, new in edits), rel
        for path in [*map(root.joinpath, cursor.EDITS), root / "srt/octopus_cursor_store.py",
                     root / "srt/octopus_delta_render.py"]:
            py_compile.compile(str(path), doraise=True)


def serving(tok):
    """OpenAIServingChat of the patched SGLang on a stub tokenizer manager: the real template and tokenizer code."""
    chat = pytest.importorskip("sglang.srt.entrypoints.openai.serving_chat")
    if not hasattr(chat, "CURSORS"):
        pytest.skip("SGLang is not cursor-patched: PYTHONPATH=<cursor overlay>")
    tm = MagicMock()
    tm.tokenizer = tok
    tm.config_value = {"tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}.get
    tm.server_args.default_chat_template_kwargs = None
    tm.server_args.context_length = 262144
    tm.request_logger.log_requests = False
    tm.model_config.is_multimodal = False
    tm.model_config.get_default_sampling_params.return_value = {}
    tm.model_config.hf_config = SimpleNamespace(model_type="qwen3_5", architectures=["Qwen3_5ForCausalLM"])
    templates = SimpleNamespace(chat_template_name=None, jinja_template_content_format="string", reasoning_config=None,
                                force_reasoning=False)
    return chat, chat.OpenAIServingChat(tm, templates)


def request(**body):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    return ChatCompletionRequest.model_validate_json(json.dumps({"model": "m", **body}))


@pytest.mark.parametrize("thinking", [False, True])
def test_cursor_prompt_is_the_cursor_tokens_and_the_concat_tail(tok, thinking):
    """Every (parent reply, request) pair of the synthetic thread, sent as JSON through SGLang's request model with the
    tools: the prompt is the stored tokens + AReaL concat's tail."""
    chat, s = serving(tok)
    kwargs = {"enable_thinking": thinking}
    chat.CURSORS.put("c", [7, 8, 9])
    for j, i, tail in concat_tails(tok, SYNTHETIC, TOOLS, kwargs, every_parent=True):
        adapted, _ = s._convert_to_internal_request(request(
            messages=SYNTHETIC[j + 1:i], tools=TOOLS, chat_template_kwargs=kwargs, cursor="c"))
        assert adapted.input_ids == [7, 8, 9] + tail, (j, i)


def test_unknown_cursor_is_a_404(tok):
    _, s = serving(tok)
    r = asyncio.run(s.handle_request(request(messages=SYNTHETIC[3:5], cursor="nope"), None))
    assert r.status_code == 404 and b"cursor nope is unknown or evicted" in r.body


def engine(rets: list[dict], seen: list):
    """TokenizerManager.generate_request's stand-in: records each request and returns the next output."""
    async def generate(obj, raw):
        seen.append(obj)
        yield rets[len(seen) - 1]
    return generate


def test_chat_reply_is_stored_as_the_concat_parent(tok):
    """A non-streaming reply is stored under its response id (prompt + output without stop + eos); the response does
    not show prompt ids it was not asked for; a request from it continues those tokens."""
    chat, s = serving(tok)
    eos, seen = tok.eos_token_id, []
    s.tokenizer_manager.generate_request = engine([{
        "text": "Hi", "output_ids": [13, 14, eos], "prompt_token_ids": [1, 2, 3], "meta_info": {
            "id": "r1", "finish_reason": {"type": "stop", "matched": eos}, "prompt_tokens": 3, "completion_tokens": 3,
            "cached_tokens": 0, "weight_version": "default"}}], seen)
    adapted, req = s._convert_to_internal_request(request(messages=SYNTHETIC[:2], tools=TOOLS, chat_template_kwargs=KWARGS))
    r = asyncio.run(s._handle_non_streaming_request(adapted, req, None))
    assert r.id == "r1" and r.choices[0].prompt_token_ids is None and seen[0].return_prompt_token_ids
    assert chat.CURSORS.get("r1").tolist() == [1, 2, 3, 13, 14, eos]
    assert chat.CURSORS.ctx["r1"][0] == SYNTHETIC[0] and chat.CURSORS.ctx["r1"][-1]["role"] == "assistant"
    delta = [{"role": "user", "content": "Go on."}]
    adapted, _ = s._convert_to_internal_request(request(messages=delta, tools=TOOLS, chat_template_kwargs=KWARGS,
                                                        cursor="r1"))
    assert adapted.input_ids == [1, 2, 3, 13, 14, eos] + render_delta(tok, delta, TOOLS, KWARGS)


def test_chat_reply_aborted_in_the_queue_is_not_stored(tok):
    """TokenizerManager._handle_abort_req's reply has no prompt ids: SGLang's 200 with finish_reason abort, no cursor."""
    chat, s = serving(tok)
    s.tokenizer_manager.generate_request = engine([{"text": "", "output_ids": [], "meta_info": {
        "id": "a1", "finish_reason": {"type": "abort", "message": "Abort in waiting queue"},
        "weight_version": "default", "completion_tokens": 0}}], [])
    r = asyncio.run(s.handle_request(request(messages=SYNTHETIC[:2], tools=TOOLS, chat_template_kwargs=KWARGS), None))
    assert r.id == "a1" and r.choices[0].finish_reason == "abort" and chat.CURSORS.get("a1") is None


def test_generate_cursor(monkeypatch):
    """/generate: input_ids + output_ids as generated are stored under the request id; a request from it gets them
    before its own input_ids; an unknown cursor is a 404."""
    hs = pytest.importorskip("sglang.srt.entrypoints.http_server")
    if not hasattr(hs, "CURSORS"):
        pytest.skip("SGLang is not cursor-patched: PYTHONPATH=<cursor overlay>")
    from fastapi import HTTPException
    from sglang.srt.managers.io_struct import GenerateReqInput
    seen = []
    tm = SimpleNamespace(generate_request=engine([{"output_ids": [7, 8], "meta_info": {"id": "g0"}},
                                                  {"output_ids": [5, 6], "meta_info": {"id": "g1"}}], seen))
    monkeypatch.setattr(hs, "_global_state", SimpleNamespace(tokenizer_manager=tm))
    asyncio.run(hs.generate_request(GenerateReqInput(input_ids=[1, 2, 3]), MagicMock()))
    asyncio.run(hs.generate_request(GenerateReqInput(cursor="g0", input_ids=[9]), MagicMock()))
    assert seen[1].input_ids == [1, 2, 3, 7, 8, 9] and hs.CURSORS.get("g1").tolist() == [1, 2, 3, 7, 8, 9, 5, 6]
    with pytest.raises(HTTPException) as e:
        asyncio.run(hs.generate_request(GenerateReqInput(cursor="nope", input_ids=[9]), MagicMock()))
    assert e.value.status_code == 404 and e.value.detail == "cursor nope is unknown or evicted"


# Live server -------------------------------------------------------------------------------------------------------
URL = os.environ.get("OCTOPUS_SGLANG_URL")
live = pytest.mark.skipif(not URL, reason="needs a patched SGLang server: OCTOPUS_SGLANG_URL")


def post(path: str, body: dict):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rendered(tok, messages):
    """The full render as SGLang's chat path does it (its tools are Tool.model_dump()s: description before name)."""
    from sglang.srt.entrypoints.openai.protocol import Tool
    return list(tok.apply_chat_template(parse_tool_arguments(messages), add_generation_prompt=True, **KWARGS,
                                        tools=[Tool.model_validate(t).model_dump() for t in TOOLS])["input_ids"])


def concat_oracle(tok, parent, delta):
    """AReaL concat (_concat_prompt_token_ids_with_parent) from the full message list: the parent's tokens (prompt +
    output without stop + eos) + the full render after their eos count."""
    if parent is None:
        return rendered(tok, delta)
    out, eos = parent["output"], tok.eos_token_id
    n = len(out)
    if parent["finish"] != "length":
        while n and out[n - 1] in (eos, tok.pad_token_id):
            n -= 1
    tokens = parent["prompt"] + out[:n] + [eos]
    whole = rendered(tok, parent["messages"] + [parent["reply"]] + delta)
    return tokens + whole[[p for p, t in enumerate(whole) if t == eos][tokens.count(eos) - 1] + 1:]


def ask(tok, parent, delta, max_tokens=200):
    """A cursor request (the first one: full); checks its prompt against the oracle and its output against /generate
    on the oracle's input_ids."""
    body = {"model": "m", "messages": delta, "tools": TOOLS, "chat_template_kwargs": KWARGS, "temperature": 0,
            "max_tokens": max_tokens, "return_prompt_token_ids": True, "return_token_ids": True}
    if parent:
        body["cursor"] = parent["id"]
    code, r = post("/v1/chat/completions", body)
    assert code == 200, r
    c, m = r["choices"][0], r["choices"][0]["message"]
    node = {"id": r["id"], "messages": (parent["messages"] + [parent["reply"]] if parent else []) + delta,
            "reply": {"role": "assistant", "content": m["content"] or "", **({"tool_calls": [
                {"id": t["id"], "type": "function", "function": t["function"]} for t in m["tool_calls"]]}
                if m.get("tool_calls") else {})},
            "prompt": c["prompt_token_ids"], "output": c["response_token_ids"], "finish": c["finish_reason"],
            "bytes": len(json.dumps(body))}
    assert node["prompt"] == concat_oracle(tok, parent, delta)
    print(f"cursor={parent['id'][:8] if parent else None} id={node['id'][:8]} sent {node['bytes']} B, prompt "
          f"{len(node['prompt'])} tokens, {len(node['output'])} out, {node['finish']}")
    code, g = post("/generate", {"input_ids": node["prompt"], "sampling_params": {
        "temperature": 0, "max_new_tokens": max_tokens, "skip_special_tokens": False}})
    assert code == 200 and g["output_ids"] == node["output"], (g["output_ids"], node["output"])
    return node


def results(node, note=None):
    """The messages after a reply: a result per tool call (or a nudge), then an optional user note."""
    calls = node["reply"].get("tool_calls", [])
    out = [{"role": "tool", "tool_call_id": t["id"], "content": TAG + f"{t['function']['name']}: exit 0\nok {k}\n"}
           for k, t in enumerate(calls)] or [{"role": "user", "content": "That was not a tool call. Call a tool."}]
    return out + ([{"role": "user", "content": note}] if note else [])


@live
def test_live_chat_cursor_tree(tok):
    """A tool loop, a reply cut by length and its continuation, a fork (two children of one cursor, one continued) and
    a rewind (a request from an older cursor): every prompt is the concat oracle's, every output /generate's."""
    r0 = ask(tok, None, EPISODE["main"][:2])
    r1 = ask(tok, r0, results(r0))
    r2 = ask(tok, r1, results(r1))
    cut = ask(tok, r2, results(r2), max_tokens=8)
    assert cut["finish"] == "length"
    ask(tok, cut, results(cut))
    fa = ask(tok, r2, results(r2, "You are branch-0 of 2. Your task: Task 1."))
    fb = ask(tok, r2, results(r2, "You are branch-1 of 2. Your task: Task 2."))
    assert fa["prompt"][:len(fb["prompt"]) - 30] == fb["prompt"][:len(fb["prompt"]) - 30]
    ask(tok, fa, results(fa))
    rw = ask(tok, r1, results(r1, "Rewound to checkpoint 1: the edit after it is undone."))
    after_r2 = r2["prompt"] + r2["output"][:-1]  # what a request from r2 would continue
    assert rw["prompt"][:len(after_r2)] != after_r2
    for path, body in (("/v1/chat/completions", {"model": "m", "messages": results(r0), "cursor": "nope"}),
                       ("/generate", {"input_ids": [1], "cursor": "nope"})):
        code, r = post(path, body)
        assert code == 404 and "cursor nope is unknown or evicted" in json.dumps(r), (path, r)


@live
def test_live_generate_cursor_equals_full(tok):
    """/generate from a cursor (its input_ids + output_ids as generated) + delta = /generate on the whole input_ids:
    same output ids and logprobs. The stored tokens are a prefix of AReaL concat's (output ended by eos: its eos is
    concat's; cut by length: concat appends eos), so the delta is concat's tokens after them."""
    params = lambda n: {"sampling_params": {"temperature": 0, "max_new_tokens": n}, "return_logprob": True}
    eos = tok.eos_token_id
    for n in (300, 8):
        prompt = rendered(tok, EPISODE["main"][:2])
        code, g0 = post("/generate", {"input_ids": prompt, **params(n)})
        assert code == 200
        stored, out = prompt + g0["output_ids"], g0["output_ids"]
        assert (out[-1] == eos) == (g0["meta_info"]["finish_reason"]["type"] == "stop"), g0["meta_info"]
        concat = prompt + (out[:-1] if out[-1] == eos else out) + [eos] + render_delta(tok, [
            {"role": "user", "content": "That was not a tool call. Call a tool."}], TOOLS, KWARGS)
        assert concat[:len(stored)] == stored
        cursor_id = g0["meta_info"]["id"]
        for _ in range(2):  # and again from the cursor request's own id
            _, c = post("/generate", {"cursor": cursor_id, "input_ids": concat[len(stored):], **params(64)})
            _, f = post("/generate", {"input_ids": concat, **params(64)})
            assert c["output_ids"] == f["output_ids"] and c["meta_info"]["prompt_tokens"] == len(concat)
            assert c["meta_info"]["output_token_logprobs"] == f["meta_info"]["output_token_logprobs"]
            print(f"/generate first {g0['meta_info']['finish_reason']['type']}: cursor sent {len(concat) - len(stored)} "
                  f"of {len(concat)} ids, {len(c['output_ids'])} out ids and logprobs equal")
            cursor_id, stored = c["meta_info"]["id"], concat + c["output_ids"]
            concat = stored + render_delta(tok, [{"role": "user", "content": "Go on."}], TOOLS, KWARGS)


# Cost --------------------------------------------------------------------------------------------------------------
def bench(tok, targets=(8_000, 32_000, 128_000)):
    """Request bytes and the server's prompt build (request JSON parse + _convert_to_internal_request: template,
    tokenizer, sampling params) for the same next request, one turn's tool results, after L history tokens: the recorded
    main thread's turns repeated, full and from a cursor (median of 5)."""
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    chat, s = serving(tok)
    turns, main = [], EPISODE["main"]
    for m in main[2:]:
        (turns.append([m]) if m["role"] == "assistant" else turns[-1].append(m))
    per_message = (len(rendered(tok, main)) - len(rendered(tok, main[:2]))) / (len(main) - 2)
    print("| L (prompt tokens) | full request bytes | cursor request bytes | full prompt build ms | cursor prompt build ms |")
    print("|---|---|---|---|---|")
    for target in targets:
        history, k = list(main[:2]), 0
        while len(history) * per_message < target:
            history += turns[k % (len(turns) - 1)]
            k += 1
        reply, delta = turns[k % (len(turns) - 1)][0], turns[k % (len(turns) - 1)][1:]
        chat.CURSORS.put("c", rendered(tok, history + [reply]))  # about L tokens, as a cursor would hold
        rows = []
        for body in ({"messages": history + [reply] + delta}, {"messages": delta, "cursor": "c"}):
            raw = json.dumps({"model": "m", "tools": TOOLS, "chat_template_kwargs": KWARGS, **body})
            times = []
            for _ in range(5):
                t = time.perf_counter()
                adapted, _ = s._convert_to_internal_request(ChatCompletionRequest.model_validate_json(raw))
                times.append(time.perf_counter() - t)
            rows.append((len(raw), statistics.median(times), len(adapted.input_ids)))
        (fb, ft, L), (cb, ct, _) = rows
        print(f"| {L:,} | {fb:,} | {cb:,} | {ft * 1e3:.1f} | {ct * 1e3:.2f} |")


if __name__ == "__main__":
    import transformers

    bench(transformers.AutoTokenizer.from_pretrained(next(p for p in QWEN35 if p and os.path.isdir(p))))
