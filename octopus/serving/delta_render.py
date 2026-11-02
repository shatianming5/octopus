"""Prompt tokens of the messages sent after a cursor (docs/cursor-protocol.md), at a cost independent of the history.

AReaL's concat mode builds a request's prompt as the parent's tokens (its prompt + output without stop + the reply's
terminator) followed by the full conversation re-rendered and re-tokenized, cut after the parent reply's terminator
(concat_start: the markers up to it, counted where the template renders the parent's thread).
render_delta returns that tail from the new messages alone: it renders them after a stub that ends on an assistant
reply, as every cursor does, and tokenizes only the text after the stub reply's terminator. Special tokens split
tokenization, so the tail's ids do not depend on what precedes it (tests/test_delta_render.py checks it bit for bit
against the concat oracle).

A reply's terminator is what the chat template renders after it in the history, in place of the token the model
stopped on (terminator, probed once per template): eos for Qwen, Llama-3.x, DeepSeek and Mistral; <|end|> after a plain
reply and <|call|> after a tool call for gpt-oss; nothing for GLM-4.5/4.7, whose next message starts with its role
token. The markers concat counts are those terminators, and for an empty one the tokens that start the next message.

The stub is constant where the template renders the messages after a reply independently of the history (Qwen,
Llama-3.x, DeepSeek-V3.1). Other templates carry state across turns; the stub then holds the few facts they read,
taken from the cursor's history (context_stub): the thread's system message, whether a tool result came before, and
the cursor's own reply. template_form probes the template once per (tokenizer, chat_template_kwargs) on PROBE
against the concat oracle and picks the first stub that matches, or raises naming the check that fails.
"""

import functools
import json
import re

# The fields the cursor protocol adds to a chat request (tree._complete sends them): the AReaL proxy answers
# 400 to one it does not consume (patches/areal-cursor.patch), and octopus/rl/proxy.py refuses an AReaL without them.
REQUEST_FIELDS = ("cursor", "mount")

# A real user query, so the template's last query (Qwen3.5 renders an assistant turn's think block only after it) is
# the stub's or one in the delta, never one the stub hides; then an assistant reply, the role every cursor ends on.
STUB = [{"role": "user", "content": "stub"}, {"role": "assistant", "content": "stub"}]
# A tool exchange, for templates that render a conversation's first tool call or result differently (DeepSeek-V3/R1).
# Ids of 9 alphanumerics, as Mistral requires.
STUB_TOOL = [{"role": "assistant", "content": "", "tool_calls": [
    {"id": "stub00000", "type": "function", "function": {"name": "stub", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "stub00000", "content": "stub"}]
# The message rendered after a stub to count the markers up to its reply's terminator.
SENTINEL = "octopus-cursor-sentinel"


def _call(n: int, content: str | None = "", path: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [{"id": f"probe000{n}", "type": "function",
            "function": {"name": "view", "arguments": json.dumps({"path": path or f"{n}.py", "lines": [1, n]})}}]}


def _result(n: int) -> dict:
    return {"role": "tool", "tool_call_id": f"probe000{n}", "content": f"def f{n}(): pass"}


# Shapes a thread sends after a reply: tool results, a user message, a plain reply, after a system prompt. One call per
# reply (Llama-3.x raises on parallel calls), user and plain assistant turns alternating (Mistral raises otherwise).
PROBE = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix f1."}, _call(1), _result(1),
         {"role": "assistant", "content": "Fixed f1."}, {"role": "user", "content": "Now f2."}, _call(2), _result(2),
         _call(3), _result(3), {"role": "assistant", "content": "Fixed f2."}, {"role": "user", "content": "Thanks."},
         {"role": "assistant", "content": "Done."}]
PROBE_TOOLS = [{"type": "function", "function": {"name": "view", "description": "Show a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "array"}}, "required": ["path"]}}}]


def render_delta(tokenizer, messages: list[dict], tools: list[dict] | None = None,
                 chat_template_kwargs: dict | None = None, history: list[dict] | None = None) -> list[int]:
    """Prompt token ids that extend a cursor by messages (OpenAI chat format, tool_call arguments as JSON strings or
    dicts). history: the cursor's thread up to its reply, or its context_stub; read only by templates that need it."""
    dict_args, needs_history = template_form(tokenizer, chat_template_kwargs)
    if needs_history and history is None:
        raise ValueError(f"{_name(tokenizer)}: this chat template needs the cursor's history")
    return _tail(tokenizer, dict_args, context_stub(history) if needs_history else STUB, messages, tools,
                 chat_template_kwargs or {})


def terminator(tokenizer, reply: dict, chat_template_kwargs: dict | None = None) -> list[int]:
    """Token ids the chat template renders after reply (an assistant message) before the next message, in place of
    the token the model stopped on: what follows the reply's output without its stop in a cursor's tokens."""
    return list(_ends(tokenizer, _key(chat_template_kwargs))[0][bool(reply.get("tool_calls"))])


def concat_start(tokenizer, history: list[dict], all_tokens: list[int], tools: list[dict] | None = None,
                 chat_template_kwargs: dict | None = None) -> int:
    """Where AReaL concat's new tokens start in all_tokens, the render of history (the parent's thread, ending on its
    reply) + the new messages with the generation prompt: after the reply's terminator (_cut)."""
    kwargs = chat_template_kwargs or {}
    markers = set(_ends(tokenizer, _key(kwargs))[1].values())
    return _cut(tokenizer, parse_tool_arguments(history, _dict_args(tokenizer, _key(kwargs))), tools, kwargs,
                [(p, p + 1) for p, t in enumerate(all_tokens) if t in markers])


def turn_start(tokenizer, tokens) -> int:
    """Where the last turn of tokens (a request's prompt, or a cursor: prompt + reply + terminator) starts: at its
    generation prompt, the last occurrence of the control token the chat template's generation prompt starts with
    (Qwen <|im_start|>, granite <|start_of_role|>, Llama-3.x <|start_header_id|>, GLM <|assistant|>)."""
    return len(tokens) - 1 - tokens[::-1].index(_generation_start(tokenizer))


def read_range(tokenizer, mount: dict, from_prompt, to_prompt, to_len: int) -> tuple[int, int]:
    """The token range [start, end) a read's mount {"from", "to", "ended"?} names in "to"'s cursor tokens (to_len of
    them; from_prompt / to_prompt: each reply's prompt): from the start of "from"'s turn (its generation prompt,
    turn_start) to the start of "to"'s, so the step before "to" ends with its tool results; with "ended" (the source
    thread has ended and "to" is its last reply, whose tool results no prompt holds) through the end of "to"'s tokens.
    The one rule of the AReaL proxy (_read_range, patches/areal-cursor.patch) and SGLang chat (kv_mount.prepare_chat)."""
    return turn_start(tokenizer, from_prompt), to_len if mount.get("ended") else turn_start(tokenizer, to_prompt)


@functools.cache
def _generation_start(tokenizer) -> int:
    bare = tokenizer.apply_chat_template(PROBE[1:2], tokenize=False)
    return tokenizer.encode(tokenizer.apply_chat_template(PROBE[1:2], tokenize=False, add_generation_prompt=True)
                            [len(bare):], add_special_tokens=False)[0]


def context_stub(history: list[dict]) -> list[dict]:
    """The stub for a cursor whose thread is history (ending on its reply): the thread's system message, a user query,
    a tool exchange if a tool result came before, the reply. What a cursor store keeps instead of the thread: the stub
    of stub + later messages + their reply is the thread's."""
    system = history[:1] if history[0]["role"] == "system" else []
    return system + STUB[:1] + (STUB_TOOL if any(m["role"] == "tool" for m in history[:-1]) else []) + history[-1:]


def template_messages(tokenizer, messages: list[dict], chat_template_kwargs: dict | None = None) -> list[dict]:
    """messages with tool_call arguments in the form the template takes: dicts, as AReaL's concat path passes them,
    unless the template only takes JSON strings (DeepSeek-R1/V3/V3.1 concatenate them into the text)."""
    return parse_tool_arguments(messages, _dict_args(tokenizer, _key(chat_template_kwargs)))


def parse_tool_arguments(messages: list[dict], as_dicts: bool = True) -> list[dict]:
    """tool_call arguments as dicts, as AReaL's concat path hands them to the template (_parse_tool_call_arguments),
    or as JSON strings."""
    def form(a):
        if as_dicts:
            return json.loads(a) if isinstance(a, str) else a
        return a if isinstance(a, str) else json.dumps(a)

    return [{**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": form(c["function"]["arguments"])}}
                                 for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in messages]


def _cut(tokenizer, history: list[dict], tools, kwargs, at: list[tuple[int, int]]) -> int:
    """Where the messages after history (ending on a reply, tool_call arguments in the template's form) start in a
    render of history + them, given that render's markers as (start, end) offsets: after the k-th marker, k the
    markers up to the reply's terminator where the template renders history before one more message (a tool result
    after a tool call, else a user message); at the next marker when the terminator is empty. Counted on the
    template's own rendering of history, not on the parent's tokens: those keep what the model generated, which a
    re-render may not show (gpt-oss's analysis channel, ending on the marker <|end|>, from a message that has no
    `thinking`)."""
    ends, markers, pattern = _ends(tokenizer, _key(kwargs))
    calls, end = history[-1].get("tool_calls"), ends[bool(history[-1].get("tool_calls"))]
    after = {"role": "tool", "tool_call_id": calls[0]["id"], "content": SENTINEL} if calls else \
        {"role": "user", "content": SENTINEL}
    text = tokenizer.apply_chat_template(history + [after], tools=tools, tokenize=False, add_generation_prompt=True,
                                         **kwargs)
    found = [markers[m.group()] for m in pattern.finditer(text, 0, text.rindex(SENTINEL))]
    k = len(found) - found[::-1].index(end[0]) if end else len(found) - 1
    if not k < len(at) + bool(end):
        raise ValueError(f"{_name(tokenizer)}: the render holds {len(at)} of the markers {sorted(markers)}, the "
                         f"parent's reply ends after {k}: no new messages to cut")
    return at[k - 1][1] if end else at[k][0]


def _tail(tokenizer, dict_args: bool, stub, messages, tools, kwargs) -> list[int]:
    stub, messages = parse_tool_arguments(stub, dict_args), parse_tool_arguments(messages, dict_args)
    text = tokenizer.apply_chat_template(stub + messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                         **kwargs)
    at = [m.span() for m in _ends(tokenizer, _key(kwargs))[2].finditer(text)]
    return tokenizer.encode(text[_cut(tokenizer, stub, tools, kwargs, at):], add_special_tokens=False)


def template_form(tokenizer, chat_template_kwargs: dict | None = None) -> tuple[bool, bool]:
    """(tool_call arguments as dicts, the stub needs the cursor's history), probed once per (tokenizer, kwargs):
    render_delta of every (parent reply, request) pair of PROBE, with and without tools, against AReaL concat's tail."""
    return _probe(tokenizer, _key(chat_template_kwargs))


def _key(chat_template_kwargs: dict | None) -> str:
    return json.dumps(chat_template_kwargs or {}, sort_keys=True)


def _name(tokenizer) -> str:
    return tokenizer.name_or_path or type(tokenizer).__name__


@functools.cache
def _ends(tokenizer, kwargs_json: str) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], dict[str, int], re.Pattern]:
    """(terminator after a plain reply, after a tool-call reply), the markers {text: id} and their pattern. A reply's
    terminator is the first control token (an added token marked special, which a model never writes as content,
    unlike Qwen's <tool_call> or DeepSeek's tool markup) the template renders after the reply, when it is the same
    before every next message (a user message, a tool result, the generation prompt); else there is none (GLM: the
    next message's role token). The markers are all these first control tokens."""
    kwargs, controls = json.loads(kwargs_json), {i for i, t in tokenizer.added_tokens_decoder.items() if t.special}
    dict_args = _dict_args(tokenizer, kwargs_json)

    def render(messages):
        try:
            return tokenizer.apply_chat_template(parse_tool_arguments(messages, dict_args), tokenize=False,
                                                 add_generation_prompt=True, **kwargs)
        except Exception:  # a template that rejects this order (gpt-oss: a tool result after a plain reply)
            return None

    def common(a: str, b: str) -> int:
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

    head, ends, found = PROBE[1:2], [], set()
    prompt = render(head)
    for reply, other in (({"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}),
                         (_call(1), _call(1, path="2.py"))):  # two replies of one kind that differ in their content
        first = set()
        for after in (PROBE[5:6], [_result(1)], []):
            x, y = render(head + [reply] + after), render(head + [other] + after)
            if x is not None and y is not None:  # after the reply's start and its last character that varies
                i = max(common(prompt, x), len(x) - common(x[::-1], y[::-1]))
                first.add(next((t for t in tokenizer.encode(x[i:], add_special_tokens=False) if t in controls), None))
        ends.append(tuple(first) if len(first) == 1 and None not in first else ())
        found |= first - {None}
    if not found:
        raise ValueError(f"{_name(tokenizer)}: the chat template renders no control token after a reply, so nothing "
                         "marks where a cursor's tokens end in a re-render")
    markers = {tokenizer.added_tokens_decoder[i].content: i for i in found}
    return tuple(ends), markers, re.compile("|".join(map(re.escape, sorted(markers, key=len, reverse=True))))


@functools.cache
def _dict_args(tokenizer, kwargs_json: str) -> bool:
    """tool_call arguments as dicts (AReaL's form), unless the template raises on dicts and renders JSON strings
    (DeepSeek-R1/V3/V3.1 concatenate them into the text)."""
    def renders(as_dicts: bool) -> bool:
        try:
            tokenizer.apply_chat_template(parse_tool_arguments(PROBE[1:2] + [_call(1, None)], as_dicts),
                                          tokenize=False, **json.loads(kwargs_json))
            return True
        except Exception:
            return False

    return renders(True) or not renders(False)


@functools.cache
def _probe(tokenizer, kwargs_json: str) -> tuple[bool, bool]:
    name, eos, kwargs = _name(tokenizer), tokenizer.eos_token, json.loads(kwargs_json)
    failed = []  # what the cut needs: eos text tokenizes to the eos id, a rendered thread may end on a reply
    if tokenizer.encode(eos, add_special_tokens=False) != [tokenizer.eos_token_id]:
        failed.append(f"encode() does not turn {eos!r} into its id {tokenizer.eos_token_id} (special tokens in text "
                      "are not parsed, as in MistralCommonBackend), so a rendered tail cannot be tokenized")
    try:
        tokenizer.apply_chat_template(STUB, tokenize=False, **kwargs)
    except Exception as e:
        failed.append(f"the chat template refuses a conversation that ends on an assistant reply ({e})")
    if failed:
        raise ValueError(f"{name}: " + "; ".join(failed))
    dict_args = _dict_args(tokenizer, kwargs_json)

    def render(messages, tools):
        try:
            return tokenizer.apply_chat_template(parse_tool_arguments(messages, dict_args), tools=tools,
                                                 tokenize=False, add_generation_prompt=True, **kwargs)
        except Exception as e:  # Gemma-3: no tool role
            raise ValueError(f"{name}: the chat template cannot render a tool-calling thread: {e}") from e

    pairs = []  # (parent reply j, request i, tools, concat tail)
    replies = [i for i, m in enumerate(PROBE) if m["role"] == "assistant"]
    for tools in (None, PROBE_TOOLS):
        for i in replies:
            ids = tokenizer.encode(render(PROBE[:i], tools), add_special_tokens=False)
            for j in [j for j in replies if j < i]:
                pairs.append((j, i, tools, ids[concat_start(tokenizer, PROBE[:j + 1], ids, tools, kwargs):]))

    def mismatch(j, i, tools, tail, needs_history) -> str | None:
        try:
            if tail == _tail(tokenizer, dict_args, context_stub(PROBE[:j + 1]) if needs_history else STUB,
                             PROBE[j + 1:i], tools, kwargs):
                return None
            return f"parent PROBE[{j}], request PROBE[:{i}], tools {tools is not None}: the tail differs"
        except Exception as e:  # the stub breaks a rule the template checks (Mistral's role alternation)
            return f"parent PROBE[{j}], request PROBE[:{i}], tools {tools is not None}: the template raised {e!r}"

    first = {}
    for needs_history in (False, True):
        first[needs_history] = next((m for pair in pairs if (m := mismatch(*pair, needs_history))), None)
        if first[needs_history] is None:
            return dict_args, needs_history
    raise ValueError(f"{name}: no stub renders the messages after a cursor as AReaL concat does with this template "
                     f"(constant stub: {first[False]}; history stub: {first[True]})")
