import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from octopus.serving.delta_render import concat_start, render_delta, template_form, template_messages, terminator

QWEN35 = [os.environ.get("QWEN35_TOKENIZER", ""), "/mnt/disk1/user/models/Qwen3.5-4B-text",
          "/home/ubuntu/models/Qwen3.5-4B-text"]
EPISODE = json.loads(gzip.decompress((Path(__file__).parent / "fixtures/delta_render_episode.json.gz").read_bytes()))
TAG = "[context left: 41234 | calls left: 37]\n"


def call(name: str, **args) -> dict:
    return {"id": "call_" + name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# Every message shape a thread sends after a reply (core.run_branch_continuation, the agent's budget tag).
SYNTHETIC = [
    {"role": "system", "content": "You fix bugs in /testbed."},
    {"role": "user", "content": "The repository in /testbed has 2 independent bugs."},
    {"role": "assistant", "content": "", "tool_calls": [call("view", path="a.py"), call("shell", command="pytest -x\ncat log")]},
    {"role": "tool", "tool_call_id": "call_view", "content": TAG + "def f():\n    return 1\n"},
    {"role": "tool", "tool_call_id": "call_shell", "content": TAG + "exit 1\n  FAILED test_a.py::test_f  \n"},
    {"role": "assistant", "content": "The bug is in f.", "reasoning_content": "f returns 1.",
     "tool_calls": [call("edit", path="a.py", old="return 1", new="return 2", opts={"count": 1}, lines=[1, 2])]},
    {"role": "tool", "tool_call_id": "call_edit", "content": TAG + "edited a.py"},
    {"role": "assistant", "content": "I believe it is fixed."},
    {"role": "user", "content": "That was not a tool call. Invoke a function with:\n```tool_code\nview(path=\"a.py\")\n```"},
    {"role": "assistant", "content": "<think>\nview it\n</think>\n\n```tool_code\nview(path=\"a.py\")\n```"},
    {"role": "user", "content": "```tool_output\nexit 0\ndef f():\n    return 2\n```"},
    {"role": "user", "content": "<tool_response>\nexit 0\n</tool_response>"},
    {"role": "assistant", "content": "", "tool_calls": [call("fork", tasks=[{"task": "fix b", "acceptance": "pytest b"}])]},
    {"role": "tool", "tool_call_id": "call_fork", "content": TAG + "fork recorded; branches branch-0"},
    {"role": "user", "content": "Started in the background: branch-0 (task 2). Call join to collect it."},
    {"role": "assistant", "content": "", "reasoning_content": "wait for it", "tool_calls": [call("join")]},
    {"role": "tool", "tool_call_id": "call_join", "content": TAG + "branches finished:\n- branch-0 done"},
    {"role": "assistant", "content": "", "tool_calls": [call("test")]},
    {"role": "tool", "tool_call_id": "call_test", "content": TAG + "2 passed"},
    {"role": "user", "content": "branches finished:\n- branch-0.0 (Task 2) committed"},
    {"role": "user", "content": "You are branch-1, one of 2 parallel branches cloned from the context above.\nYour task: Task 2"},
    {"role": "assistant", "content": "Done.", "tool_calls": [call("done", summary="fixed f")]},
]


@pytest.fixture(scope="module")
def tok():
    path = next((p for p in QWEN35 if p and Path(p).is_dir()), None)
    transformers = pytest.importorskip("transformers")
    if path is None:
        pytest.skip("no Qwen3.5 tokenizer here")
    return transformers.AutoTokenizer.from_pretrained(path)


def concat_tails(tok, thread: list[dict], tools, kwargs, every_parent: bool = False):
    """AReaL concat (client._concat_prompt_token_ids_with_parent) for each request thread[:i] (thread[i] its reply):
    (j, i, the full render's tokens after the parent reply's terminator, concat_start) with the previous reply thread[j]
    as the parent, as in training, or every earlier one. tool_call arguments in the template's form; a request the
    template rejects (Mistral's role alternation) is none."""
    parents = []
    for i in [i for i, m in enumerate(thread) if m["role"] == "assistant"]:
        try:
            ids = list(tok.apply_chat_template(template_messages(tok, thread[:i], kwargs), tools=tools,
                                               add_generation_prompt=True, **kwargs)["input_ids"])
        except jinja2.TemplateError:
            continue
        for j in parents[0 if every_parent else -1:]:
            yield j, i, ids[concat_start(tok, thread[:j + 1], ids, tools, kwargs):]
        parents.append(i)


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("with_tools", [True, False])
def test_delta_matches_concat_for_every_parent(tok, thinking, with_tools):
    """Deltas of one message to most of the thread (replies, real and wrapped user messages inside) all match."""
    tools, kwargs = EPISODE["tools"] if with_tools else None, {"enable_thinking": thinking}
    assert {tuple(terminator(tok, m, kwargs)) for m in SYNTHETIC if m["role"] == "assistant"} == {(tok.eos_token_id,)}
    tails = list(concat_tails(tok, SYNTHETIC, tools, kwargs, every_parent=True))
    assert len(tails) == 28
    for j, i, tail in tails:
        assert render_delta(tok, SYNTHETIC[j + 1:i], tools, kwargs) == tail, (j, i)


def threads(name: str, messages: list[dict], forks: list[dict]):
    """Each thread's full message list: a branch resumes from its forking thread's first prefix_len messages."""
    yield name, messages
    for k, fork in enumerate(forks):
        for b in fork["branches"]:
            yield from threads(("" if name == "main" else name + "/") + f"fork{k}/{b['branch_id']}",
                               messages[:fork["prefix_len"]] + b["messages"], b["forks"])


def test_delta_matches_concat_on_recorded_episode(tok):
    """bundle-e11, main + 4 branches (one nested fork, fork-task / background / join messages): every request's delta
    is the concat tail, and its length is the recorded prompt's growth (parent prompt + completion + delta)."""
    tools, kwargs = EPISODE["tools"], EPISODE["chat_template_kwargs"]
    recorded, checked = {}, 0
    for name, thread in threads("main", EPISODE["main"], EPISODE["forks"]):
        replies = [i for i, m in enumerate(thread) if m["role"] == "assistant"]
        own = [i for i in replies if json.dumps(thread[:i + 1]) not in recorded]
        recorded.update({json.dumps(thread[:i + 1]): r for i, r in zip(own, EPISODE["requests"][name])})
        for j, i, tail in concat_tails(tok, thread, tools, kwargs):
            delta = render_delta(tok, thread[j + 1:i], tools, kwargs)
            assert delta == tail, (name, j, i)
            (prompt, completion), (now, _) = recorded[json.dumps(thread[:j + 1])], recorded[json.dumps(thread[:i + 1])]
            assert now == prompt + completion + len(delta), (name, j, i)
            checked += 1
    assert checked == 198


# Other templates: a tokenizer dir from the env var, else the hub id from the local HF cache (no download), else skip.
# (env var, hub id, thread shape the template takes, synthetic and episode checks per (thinking, tools) setting).
# Llama-3.x raises on a reply with parallel tool calls, so its threads keep each reply's first call; Mistral also
# needs 9-character tool call ids and rejects a user message after another (adjacent ones are joined; ones with tool
# results between them leave the request out).
OTHERS = {"Qwen3": ("QWEN3_TOKENIZER", "Qwen/Qwen3-0.6B", "parallel", 28, 198),
          "Llama-3.2": ("LLAMA32_TOKENIZER", "unsloth/Llama-3.2-1B-Instruct", "first", 28, 198),
          "Mistral-v0.3": ("MISTRAL_TOKENIZER", "unsloth/mistral-7b-instruct-v0.3", "mistral", 10, 102),
          "DeepSeek-V3": ("DEEPSEEK_V3_TOKENIZER", "deepseek-ai/DeepSeek-V3", "parallel", 28, 198),
          "DeepSeek-R1": ("DEEPSEEK_R1_TOKENIZER", "deepseek-ai/DeepSeek-R1", "parallel", 28, 198),
          "DeepSeek-V3.1": ("DEEPSEEK_V31_TOKENIZER", "deepseek-ai/DeepSeek-V3.1", "parallel", 28, 198),
          "GLM-4.5": ("GLM45_TOKENIZER", "zai-org/GLM-4.5", "parallel", 28, 198),
          "GLM-4.7": ("GLM47_TOKENIZER", "zai-org/GLM-4.7", "parallel", 28, 198),
          "gpt-oss": ("GPT_OSS_TOKENIZER", "openai/gpt-oss-20b", "parallel", 28, 198)}
# A reply's terminator (after a plain reply, after a tool call) where it is not eos.
TERMINATORS = {"GLM-4.5": ([], []), "GLM-4.7": ([], []), "gpt-oss": (["<|end|>"], ["<|call|>"])}


def load(env: str, hub: str, name: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(os.environ.get(env) or hub, local_files_only=True)
    except OSError:
        pytest.skip(f"no {name} tokenizer here")


def first_calls(thread: list[dict]) -> list[dict]:
    """Each reply with its first tool call only, and without the other calls' results."""
    dropped = {c["id"] for m in thread for c in (m.get("tool_calls") or [])[1:]}
    return [{**m, "tool_calls": m["tool_calls"][:1]} if m.get("tool_calls") else m
            for m in thread if m.get("tool_call_id") not in dropped]


def mistral(thread: list[dict]) -> list[dict]:
    """First calls, each tool call id replaced by a 9-character one, adjacent user messages joined."""
    ids, out = {}, []

    def short(i: str) -> str:
        return ids.setdefault(i, f"call{len(ids):05d}")

    for m in first_calls(thread):
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**c, "id": short(c["id"])} for c in m["tool_calls"]]}
        elif m.get("tool_call_id"):
            m = {**m, "tool_call_id": short(m["tool_call_id"])}
        elif m["role"] == "user" and out and out[-1]["role"] == "user":
            prev = out.pop()
            m = {**prev, "content": prev["content"] + "\n" + m["content"]}
        out.append(m)
    return out


SHAPES = {"parallel": lambda thread: thread, "first": first_calls, "mistral": mistral}


@pytest.mark.parametrize("name", OTHERS)
def test_delta_matches_concat_on_other_templates(name):
    """The synthetic every-parent checks and the recorded episode's checks with another template, the cursor's thread
    as history (Qwen3: <|im_end|> with its own think-block rule; Llama-3.2: <|eot_id|>, ipython tool role, tools in
    the first user message; Mistral: system prompt in the last user message, role alternation; DeepSeek-V3/R1: the
    conversation's first tool call / result rendered differently, string arguments; DeepSeek-V3.1: string arguments;
    GLM-4.5/4.7: no terminator, the next message's role token marks the cut; gpt-oss: <|end|> / <|call|>, a tool
    result names the call before it, earlier analysis channels dropped once a plain reply follows)."""
    env, hub, shape, n_synthetic, n_episode = OTHERS[name]
    tok, serve = load(env, hub, name), SHAPES[shape]
    plain, call = TERMINATORS.get(name, ([tok.eos_token], [tok.eos_token]))
    assert [terminator(tok, SYNTHETIC[7]), terminator(tok, SYNTHETIC[2])] == \
        [tok.convert_tokens_to_ids(plain), tok.convert_tokens_to_ids(call)]
    synthetic = serve(SYNTHETIC)
    for kwargs in ({"enable_thinking": True}, {"enable_thinking": False}):
        for tools in (EPISODE["tools"], None):
            tails = list(concat_tails(tok, synthetic, tools, kwargs, every_parent=True))
            assert len(tails) == n_synthetic
            for j, i, tail in tails:
                assert render_delta(tok, synthetic[j + 1:i], tools, kwargs, synthetic[:j + 1]) == tail, \
                    (kwargs, tools is None, j, i)
    tools, kwargs, checked = EPISODE["tools"], EPISODE["chat_template_kwargs"], 0
    for thread_name, thread in threads("main", EPISODE["main"], EPISODE["forks"]):
        thread = serve(thread)
        for j, i, tail in concat_tails(tok, thread, tools, kwargs):
            assert render_delta(tok, thread[j + 1:i], tools, kwargs, thread[:j + 1]) == tail, (thread_name, j, i)
            checked += 1
    assert checked == n_episode


GPT_OSS_CALL = call("view", path="a.py")


@pytest.mark.parametrize("reply, output, delta, tail", [
    ({"role": "assistant", "content": "", "tool_calls": [GPT_OSS_CALL], "reasoning_content": "Need a.py."},
     "<|channel|>analysis<|message|>Need a.py.<|end|><|start|>assistant to=functions.view<|channel|>commentary json"
     '<|message|>{"path": "a.py"}<|call|>', [{"role": "tool", "tool_call_id": "call_view", "content": "RESULT"}],
     '<|start|>functions.view to=assistant<|channel|>commentary<|message|>"RESULT"<|end|><|start|>assistant'),
    ({"role": "assistant", "content": "Done.", "reasoning_content": "All good."},
     "<|channel|>analysis<|message|>All good.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>",
     [{"role": "user", "content": "Next."}], "<|start|>user<|message|>Next.<|end|><|start|>assistant")])
def test_gpt_oss_parent_from_a_model_output(reply, output, delta, tail):
    """A gpt-oss parent as AReaL holds it: the model's output (an analysis channel ending on the marker <|end|>, then
    the call or the final answer) and the parsed message (reasoning in reasoning_content, which the template does not
    render). Full mode (_concat_prompt_token_ids_with_parent: the parent's tokens + the re-render after concat_start)
    and cursor mode (+ render_delta) give the same prompt, ending on the new message and the generation prompt; with
    AReaL importable (the cursor overlay), its patched function gives it too."""
    tok = load("GPT_OSS_TOKENIZER", "openai/gpt-oss-20b", "gpt-oss")
    tools, history = EPISODE["tools"], [SYNTHETIC[1], reply]
    encode = lambda messages: list(tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True)[
        "input_ids"])
    prompt, out = encode(history[:1]), tok.encode(output, add_special_tokens=False)[:-1]  # without the stop
    parent = prompt + out + terminator(tok, reply)
    full = encode(history + delta)
    full = parent + full[concat_start(tok, history, full, tools):]
    assert full == parent + render_delta(tok, delta, tools, None, history)
    assert tok.decode(full[len(parent):]) == tail
    try:
        from areal.experimental.openai.client import _concat_prompt_token_ids_with_parent
    except ImportError:
        return
    p = SimpleNamespace(model_response=SimpleNamespace(input_tokens=prompt, output_tokens_without_stop=out),
                        messages=history[:1], output_message_list=[reply])
    assert _concat_prompt_token_ids_with_parent(delta, p, tok, tools)[0] == full


def test_template_without_a_concat_cut_fails_at_the_probe():
    """Devstral (MistralCommonBackend): encode() does not parse special tokens, and a rendering may not end on a reply."""
    with pytest.raises(ValueError, match="MistralCommonBackend: encode.. does not turn .* refuses a conversation that "
                                         "ends on an assistant reply"):
        template_form(load("DEVSTRAL_TOKENIZER", "mistralai/Devstral-Small-2507", "Devstral"))


def test_probe_is_per_chat_template_kwargs():
    """A template that reads the history only under a kwarg: the probe takes the kwargs (Qwen3's tokens, a template
    that numbers user turns when asked)."""
    tok = load("QWEN3_TOKENIZER", "Qwen/Qwen3-0.6B", "Qwen3")
    tok.chat_template = (
        "{%- for m in messages %}<|im_start|>{{ m.role }}{% if number_turns and m.role == 'user' %} {{ loop.index }}"
        "{% endif %}\n{{ m.content }}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n"
        "{%- endif %}")
    assert template_form(tok) == template_form(tok, {"number_turns": False}) == (True, False)
    with pytest.raises(ValueError, match="no stub renders"):
        template_form(tok, {"number_turns": True})


@pytest.mark.parametrize("name, env, hub", [
    ("Qwen3.5", "QWEN35_TOKENIZER", ""), ("granite-4.0-h", "GRANITE4H_TOKENIZER", "ibm-granite/granite-4.0-h-350m"),
    ("Llama-3.2", "LLAMA32_TOKENIZER", "unsloth/Llama-3.2-1B-Instruct"), ("GLM-4.5", "GLM45_TOKENIZER", "zai-org/GLM-4.5"),
    ("gpt-oss", "GPT_OSS_TOKENIZER", "openai/gpt-oss-20b")])
def test_read_card_and_turn_start_follow_the_chat_template(monkeypatch, tmp_path, name, env, hub):
    """The read path's template facts from the template itself: read_thread's card is the paragraph holding the tool in
    transformers' rendering of a user message with it as the only tool (Qwen3.5: main's hand-written <tools> card byte
    for byte, from the tokenizer dir), and turn_start finds a reply's generation prompt in its prompt and in its cursor
    (prompt + reply + terminator), where Qwen3.5's last <|im_start|> was."""
    from octopus.serving.delta_render import turn_start
    from octopus.runtime.core import READ_TOOL, read_card
    if name == "Qwen3.5":
        path = next((p for p in QWEN35 if p and Path(p).is_dir()), None) or pytest.skip("no Qwen3.5 tokenizer here")
        tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(path)
    else:
        tok, path = load(env, hub, name), tmp_path
        (tmp_path / "chat_template.jinja").write_text(tok.chat_template)
    monkeypatch.setenv("OCTOPUS_TOKENIZER", str(path))
    tool = {"type": "function", "function": {**{k: READ_TOOL["function"][k] for k in ("description", "name",
                                                                                         "parameters")}, "strict": False}}
    text = tok.apply_chat_template([{"role": "user", "content": "stub"}], tools=[tool], tokenize=False)
    (block,) = [p for p in text.split("\n\n") if "read_thread" in p]
    assert read_card() == "read_thread is available from here on:\n" + block
    if name == "Qwen3.5":
        assert read_card() == "read_thread is available from here on:\n<tools>\n" + json.dumps(
            tool, ensure_ascii=False) + "\n</tools>"
    thread = [{"role": "user", "content": "Fix f1."}, {"role": "assistant", "content": "Fixed."},
              {"role": "user", "content": "Now f2."}]
    prompt = tok.encode(tok.apply_chat_template(thread, tokenize=False, add_generation_prompt=True),
                        add_special_tokens=False)
    turn = len(tok.encode(tok.apply_chat_template(thread, tokenize=False), add_special_tokens=False))
    cursor = prompt + tok.encode("Fixed f2.", add_special_tokens=False) + terminator(tok, {"role": "assistant"})
    assert turn_start(tok, prompt) == turn_start(tok, cursor) == turn < len(prompt)
    if name == "Qwen3.5":
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        assert turn == len(cursor) - 1 - cursor[::-1].index(im_start)
