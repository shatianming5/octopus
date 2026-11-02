"""The generic core of the Octopus runtime, the same in every environment: how a model writes tool calls
(TOOL_FORMATS), how a reply goes back into its thread's history (OCTOPUS_CURSOR, docs/cursor-protocol.md) and the chat
request (post_chat), thread names in the fork tree, the runtime tools (fork / commit / abort against a RuntimePort that
only records them: tools name branches, KV stays the serving engine's; commit is a semantic delta, never merge_kv;
rewind, read_thread and its card), one thread's loop (run_branch_continuation), and what a branch and the joining
thread are told and a fork's tree.json record. Whatever depends on the world the threads work in is a
WorkspaceProvider's (octopus/runtime/tree.py), passed in. Switches: octopus/runtime/settings.py.
"""

from __future__ import annotations

import bisect
import collections
import datetime
import functools
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from octopus.runtime.settings import answer_reserve, read_tool, rewind_max, rewind_tool, thinking, window_room


# How a model writes a tool call, by SGLang's --tool-call-parser name (sglang 0.5.19 srt/function_call/*_detector.py; a
# run's is rollout.agent.tool_call_parser, see tool_format). Which calls a reply made is the server's parse (the
# reply's tool_calls, reply_tools); "call" only locates them in a completion's text (octopus/rl/group.call_spans): it matches
# one call, with the tool name in group `name`, or for a JSON call the object's opening brace in group `json` (the name
# is the object's "name").
TOOL_FORMATS = {
    "qwen3_coder": {"call": r"<function=(?P<name>[^>\n]+)>.*?</function>"},  # Qwen3.5
    "qwen25": {"call": r"<tool_call>\s*(?P<json>\{)"},  # Qwen2.5/Qwen3, hermes
    "llama3": {"call": r"(?:^|<\|python_tag\|>|<\|end_header_id\|>|;)\s*(?P<json>\{)"},
    "mistral": {"call": r"\[TOOL_CALLS\](?P<name>[^\[\s]+)(?:\[CALL_ID\][^\[]*)?\[ARGS\]\{.*?\}"  # Small 3.2 and later
                        r"(?=\s*(?:\[TOOL_CALLS\]|</s>|$))|(?:\[TOOL_CALLS\]\s*\[|(?<=\}),)\s*(?P<json>\{)"},  # v0.3 array
    "deepseekv3": {"call": r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n.*?<｜tool▁call▁end｜>"},
    "deepseekv31": {"call": r"<｜tool▁call▁begin｜>(?P<name>[^<]+)<｜tool▁sep｜>.*?<｜tool▁call▁end｜>"},
    "glm45": {"call": r"<tool_call>(?P<name>[^\n<]+).*?</tool_call>"},  # GLM-4.5 to 4.7
    "kimi_k2": {"call": r"<\|tool_call_begin\|>\s*(?:functions\.)?(?P<name>[\w.\-]+):\d+\s*"
                        r"<\|tool_call_argument_begin\|>.*?<\|tool_call_end\|>"},
    "gpt-oss": {"call": r"to=functions\.(?P<name>[^\s<]+).*?<\|call\|>"},
}
TOOL_FORMATS |= {"qwen": TOOL_FORMATS["qwen25"], "hermes": TOOL_FORMATS["qwen25"], "step3p5": TOOL_FORMATS["qwen3_coder"],
                 "glm": TOOL_FORMATS["glm45"], "glm47": TOOL_FORMATS["glm45"]}  # SGLang's other names for them


def tool_format(name: str = "") -> dict:
    """TOOL_FORMATS[name], by default the run's: OCTOPUS_TOOL_FORMAT, which octopus/rl/config.set_derived_settings sets
    from rollout.agent.tool_call_parser for training and dev_eval (qwen3_coder when unset)."""
    name = name or os.environ.get("OCTOPUS_TOOL_FORMAT") or "qwen3_coder"
    if name not in TOOL_FORMATS:
        raise ValueError(f"tool-call format {name!r} is none of {sorted(TOOL_FORMATS)}")
    return TOOL_FORMATS[name]




def token_index(pieces: list[str]):
    """A completion's token texts -> (their joined text, offset -> the index of the token covering that character)."""
    ends = list(itertools.accumulate(len(p) for p in pieces))
    return "".join(pieces), lambda offset: bisect.bisect_right(ends, offset)


def reply_tools(choice: dict) -> tuple[list, str]:
    """A reply's tool calls as the server parsed them, and its text."""
    return choice.get("tool_calls") or [], choice.get("content") or ""


def reply_message(choice: dict, content, tool_calls: list | None = None) -> dict:
    """A reply as it goes back into its thread's history, with the reasoning_content the server split off (SGLang's
    qwen3 parser on a stray </think>): AReaL links a thread's requests by exact message prefix against its own record
    of the reply, and a reply re-sent without it split the thread (run e1c, fork0/branch-5)."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if choice.get("reasoning_content"):
        message["reasoning_content"] = choice["reasoning_content"]
    return message


def cursor_delta(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OCTOPUS_CURSOR: the response id of the last reply in a thread's history and the messages after it, which is
    what a request sends (None and every message: none, a full send). Scans back from the end only to that reply."""
    for i in range(len(messages) - 1, -1, -1):
        if "cursor" in messages[i]:
            return messages[i]["cursor"], messages[i + 1:]
    return None, messages


def with_cursor(reply: dict, payload: dict) -> dict:
    """The reply as its thread's history keeps it, with the response's cursor (OCTOPUS_CURSOR: payload["cursor"], set by
    the transport while the reply is still the server's) only if it holds the server's message as is. A rewritten reply
    (calls dropped) stays unmarked: the next request resends it in its delta from the previous cursor, so the server
    renders what full mode sends."""
    if "cursor" not in payload:  # also a reply without choices (handled as an empty reply)
        return reply
    choice = payload["choices"][0]["message"]
    if (reply.get("content") or "") == (choice.get("content") or "") \
            and reply.get("tool_calls") == (choice.get("tool_calls") or None):
        reply["cursor"] = payload["cursor"]
    return reply


def without_cursors(value):
    """A copy of messages (or records holding them, e.g. tree.json forks) without the replies' cursors: files read the
    history as in full mode, and a run that re-sends it never names another run's cursor."""
    if isinstance(value, dict):
        return {k: without_cursors(v) for k, v in value.items() if k != "cursor"}
    return [without_cursors(v) for v in value] if isinstance(value, list) else value


def decode_tool_call(call: dict, default_id: str = "") -> tuple[str, dict, str]:
    """Turn one OpenAI-style tool_call into name, arguments, id."""
    function = call.get("function") or {}
    name = function.get("name") or "shell"
    raw_args = function.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        args = {"command": str(raw_args)}
    if not isinstance(args, dict):
        args = {"command": str(raw_args)}
    return name, args, str(call.get("id") or default_id or name)


RETRYABLE = {403, 429, 500, 502, 503, 504}


def post_chat(
    url: str,
    body: dict,
    headers: dict,
    timeout: int | None = None,
    attempts: int | None = None,
    log=None,
) -> dict:
    """POST one chat completion. Retry proxy 403/429/5xx and dropped connections with backoff
    (2, 4, ... up to 60 s; OCTOPUS_HTTP_ATTEMPTS tries, default 6), then stop this request."""
    attempts = attempts or int(os.environ.get("OCTOPUS_HTTP_ATTEMPTS", "6"))
    timeout = timeout or int(os.environ.get("OCTOPUS_HTTP_TIMEOUT", "180"))
    payload = json.dumps(body).encode()
    delay = 2
    last = "grok request failed"
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=payload, headers=headers)
        started = time.time()
        event = {
            "t": started,
            "attempt": attempt,
            "payload_bytes": len(payload),
        }
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
            data = json.loads(raw.decode())
            usage = data.get("usage") or {}
            event.update({
                "http_code": status,
                "id": data.get("id"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "elapsed_s": round(time.time() - started, 3),
            })
            if log:
                log(event)
            return data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {exc.code} {detail}".strip()
            event.update({
                "http_code": exc.code,
                "error": detail,
                "elapsed_s": round(time.time() - started, 3),
            })
            if log:
                log(event)
            if exc.code not in RETRYABLE or attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            event.update({
                "error": last,
                "elapsed_s": round(time.time() - started, 3),
            })
            if log:
                log(event)
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(last)


def branch_ids(owner: str | None, start: int, n: int) -> list[str]:
    """The ids a fork's n branches are told: branch-<i> under main (every fork restarts at 0), <owner>.<i> under branch
    <owner>, numbered on across its forks (branch-1.5: its sixth sub-branch), so no branch shares an id with a
    shallower one or with another branch's children."""
    if owner is None:
        return [f"branch-{i}" for i in range(n)]
    return [f"{owner}.{start + i}" for i in range(n)]


def thread_name(parent: str, k: int, branch_id: str) -> str:
    """A branch's thread name, from its path in the fork tree: fork<k>/<branch id> for the k-th fork of main,
    <parent>/fork<k>/<branch id> for the k-th fork of branch <parent> (requests.jsonl `thread`)."""
    return f"fork{k}/{branch_id}" if parent == "main" else f"{parent}/fork{k}/{branch_id}"


def branch_turns() -> int:
    """A branch's turn cap (a branch control and a read control start from what it had left): OCTOPUS_BRANCH_MAX_TURNS,
    100 when unset; main's is settings.call_limit."""
    return int(os.environ.get("OCTOPUS_BRANCH_MAX_TURNS", "100"))


def limits_note(shared: bool, budget: int = 0, calls: bool = True) -> str:
    """What each tool result's tag shows and what running out of it does: one text for every thread of both
    environments (shared: a run with branches; budget: a shared completion-token budget stated here, search's
    OCTOPUS_SEARCH_BUDGET; SWE has none; calls: main's calls are capped)."""
    left = [f"the tokens left of the session's budget of {budget} completion tokens"
            + (" (shared by you and your branches)" if shared else "")] if budget else []
    left += ["your context left (the tokens you may still add to your conversation, counted before the tool results "
             "that follow; what you started from does not count)"]
    left += (["your calls left" + (" (a branch: its turns left)" if shared else "")] if calls
             else ["for a branch, its turns left"] if shared else [])
    return (f"Every tool result's tag shows {' and '.join(left)}; when "
            f"{'any' if len(left) > 2 else 'either' if len(left) > 1 else 'it'} runs out you are stopped where you stand.")


def branch_key(thread: str) -> str:
    """The same path as one file-name-safe token: the key in plan.tsv and branches.json, the branch's workdir and
    result names (fork0/branch-1/fork0/branch-0 -> fork0-branch-1-fork0-branch-0)."""
    return thread.replace("/", "-")


def tree_branches(forks: list, parent: str = "main"):
    """(thread name, fork record, branch record) for every branch of a tree.json fork list at any depth, depth first
    (a branch's own forks sit in its "forks")."""
    for k, fork in enumerate(forks):
        for branch in fork.get("branches") or []:
            name = thread_name(parent, k, branch["branch_id"])
            yield name, fork, branch
            yield from tree_branches(branch.get("forks") or [], name)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        },
    }


# FORK_ALONE as SWE's fork descriptions state it (search's names its own closing call)
FORK_RULE = " A reply whose fork starts cannot also call done, commit or abort: call them in a later reply."
RUNTIME_TOOLS = [
    _tool(
        "fork",
        "Clone this conversation into N private branches. Need at least two "
        "tasks. One task does not unlock edit/shell. Each task is that "
        "branch's next user turn." + FORK_RULE,
        {
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
        },
        ["tasks"],
    ),
    _tool(
        "commit",
        "Keep one branch's semantic summary. Does not merge KV. Names the branch. Available only after a successful fork.",
        {
            "branch_id": {"type": "string"},
            "summary": {"type": "string"},
        },
        ["branch_id", "summary"],
    ),
    _tool(
        "abort",
        "Drop one branch's private suffix. The shared prefix stays. Available only after a successful fork.",
        {"branch_id": {"type": "string"}},
        ["branch_id"],
    ),
]


RUNTIME_NAMES = tuple(item["function"]["name"] for item in RUNTIME_TOOLS)
FORK_STARTED = "fork started: branches "  # an accepted fork's result (started)


class RuntimePort:
    """Host runtime. Override to bind SGLang/Octopus; default only records."""

    def __init__(self, owner: str | None = None) -> None:
        self.owner = owner  # the branch whose runtime this is (None: main), for its children's ids (branch_ids)
        self.forked = 0  # branches this port has forked so far
        self.calls: list[tuple[str, dict]] = []
        self.last_fork: dict | None = None

    def fork(self, tasks: list[str]) -> str:
        self.calls.append(("fork", {"tasks": tasks}))
        branches = [{"branch_id": bid, "task": task}
                    for bid, task in zip(branch_ids(self.owner, self.forked, len(tasks)), tasks)]
        self.forked += len(tasks)
        self.last_fork = {"branches": branches}
        return FORK_STARTED + ", ".join(item["branch_id"] for item in branches)

    def commit(self, branch_id: str, summary: str) -> str:
        self.calls.append(("commit", {"branch_id": branch_id, "summary": summary}))
        return f"commit recorded: {branch_id}"

    def abort(self, branch_id: str) -> str:
        self.calls.append(("abort", {"branch_id": branch_id}))
        return f"abort recorded: {branch_id}"


def fork_task_list(tasks) -> list | None:
    """A fork call's tasks as a list (None: not a list). The tool-call parser leaves the array as the model wrote it,
    a string, when that is not strict JSON: read here with raw newlines or tabs inside strings, a backslash before a
    character JSON does not escape (\\| \\' \\.) kept as written, and anything after the array ignored (e11-e12: 19 of
    146 such refusals). Raises json.JSONDecodeError on a string that still does not parse (e11-e12: mostly an
    unescaped " inside a task's shell command)."""
    if isinstance(tasks, str):
        text = tasks.strip()
        bad = [m.start() for m in re.finditer(r"\\(.)", text, re.S) if m[1] not in '"\\/bfnrtu']
        fixed = re.sub(r"\\(.)", lambda m: m[0] if m[1] in '"\\/bfnrtu' else "\\" + m[0], text, flags=re.S)
        try:
            tasks = json.JSONDecoder(strict=False).raw_decode(fixed)[0]
        except json.JSONDecodeError as e:  # where in the string as written (the k-th doubled backslash sits at p + k)
            raise json.JSONDecodeError(e.msg, text, e.pos - sum(p + k < e.pos for k, p in enumerate(bad))) from None
    return tasks if isinstance(tasks, list) else None


def unparsed_tasks(e: json.JSONDecodeError, shape: str = '["...", ...]') -> str:
    """apply_runtime's refusal of a fork whose tasks do not parse (fork_task_list): where, and the shape expected (e11:
    of 120 such refusals saying "needs at least two tasks", 41 were followed by another such string, 37 by no fork)."""
    return (f"fork refused: tasks is not a JSON array ({e.msg} at char {e.pos}: ...{e.doc[max(0, e.pos - 80):e.pos]}"
            f"<-- here). Send {shape}; inside a string escape every \" as \\\" and every \\ as \\\\.")


def apply_runtime(name: str, args: dict | None, port: RuntimePort | None = None, need: str = "tasks",
                  needs=None, fewest: int = 2) -> tuple[str, bool]:
    """Run one host runtime tool. Returns observation text and stop=False. A fork's tasks are the texts its branches
    are told (a provider turns its own task shape into them first: WorkspaceProvider.runtime); need: what a refused
    fork is told it needs; fewest: the fewest tasks a fork takes (WorkspaceProvider.fork_min); needs(name, args): what
    a commit or abort with its branch_id (and summary) still lacks, or None, worded "<name> needs ..." so that
    rejected() sees the call refused (the thread does not close)."""
    port = port or RuntimePort()
    args = dict(args or {})
    if name == "fork":
        try:
            tasks = fork_task_list(args.get("tasks"))
        except json.JSONDecodeError as e:
            return unparsed_tasks(e), False
        tasks = [t if isinstance(t, str) and t.strip() else None for t in tasks] if isinstance(tasks, list) else None
        if not tasks or len(tasks) < fewest or any(item is None for item in tasks):
            return (f"fork needs at least two {need}" if fewest > 1 else
                    f"fork needs {need}, each a non-empty string"), False
        return port.fork(list(tasks)), False
    if name == "commit":
        branch_id, summary = args.get("branch_id"), args.get("summary")
        if not isinstance(branch_id, str) or not branch_id.strip() or not isinstance(summary, str) or not summary.strip():
            return "commit needs branch_id and summary", False
        lacks = needs and needs(name, args)
        return (lacks, False) if lacks else (port.commit(branch_id, summary), False)
    if name == "abort":
        branch_id = args.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id.strip():
            return "abort needs branch_id", False
        lacks = needs and needs(name, args)
        return (lacks, False) if lacks else (port.abort(branch_id), False)
    return f"unknown runtime tool {name}", False


CLOSING = {"done", "commit", "abort"}
_REJECTED = ("commit needs", "abort needs")


def rejected(name: str | None, observed) -> bool:
    """A commit/abort the runtime refused (missing arguments): the branch has not closed."""
    return name in {"commit", "abort"} and str(observed or "").startswith(_REJECTED)


def closing_step(steps: list[dict]) -> dict | None:
    """A thread's last accepted done/commit/abort step (refused ones do not count), None without one."""
    return next((s for s in reversed(steps) if s.get("name") in CLOSING and not s.get("rejected")), None)


def handed(closing: dict | None) -> str | None:
    """What a closing step hands its forking thread: a commit's summary, a done's answer (search), else None."""
    args = (closing or {}).get("arguments") or {}
    return args.get("summary", args.get("answer"))


def closing_name(steps: list[dict]) -> str | None:
    """How a branch ended: its last accepted tool call name (refused commit/aborts do not count)."""
    return next((s["name"] for s in reversed(steps) if s.get("name") and not s.get("rejected")), None)


def add_note(messages: list, text: str, pending: bool = True) -> None:
    """A runtime note to a thread before its next request: a user message, or with thinking() on a tool result. The
    Qwen3.5 template keeps an assistant turn's <think> block only after the last real user query, so a user note would
    re-render the history without its reasoning (no longer an extension of the previous prompt). pending: the history
    ends with tool results this thread has not sent yet, and the note joins the last of them (a new dict: the old one
    may be shared); else (a rewind back to sent ones) it is a tool message of its own, thinking or not: tagged like a
    tool result, so the thread's latest tag is its calls and context left now, not those of the step it went back to."""
    if pending and not thinking():
        messages.append({"role": "user", "content": text})
    elif pending and messages[-1].get("role") == "tool":
        messages[-1] = dict(messages[-1], content=f"{messages[-1]['content']}\n\n{text}")
    else:
        messages.append({"role": "tool", "content": text})


REWIND_FAILS, REWIND_REPEATS = 6, 3
# OCTOPUS_ANSWER_RESERVE: the result of every non-closing call once a thread is inside its reserve (a rewind runs
# there too under OCTOPUS_REWIND_TOOL, and the notes say so: REWIND_FREES)
# OCTOPUS_READ_TOOL: each tag names the thread first ([main | step 3], [branch-1 | token budget left: ...]); a result
# with any thread's tag (one a branch inherited from its parent too) is tagged already
THREAD_TAG = re.compile(r"\[[^\]\n|]+ \| (?:token budget left: |context left: |step \d)")
RESERVE_NOTE = ("Refused: your context is nearly used up ({left} tokens left). Close now: call {calls} with what you "
                "have so far{rewind}. Until you do, every other call is refused.")
# and the end of the results that would have passed into it, clipped (clip_results)
CLIPPED = "[clipped: context nearly used up]"
CLOSE_NOTE = ("Your context is nearly used up. Close now: call {calls} with what you have so far{rewind}; any other "
              "call is refused.")
REWIND_FREES = ", or rewind to an earlier step to free context"  # while the thread has rewinds left
# a thread's first reply without a tool call, while it has room for another turn: it goes on, reminded (once per thread)
REMIND_NOTE = ("You replied without a tool call. A thread's result counts only through its closing call: close with "
               "{calls} now, or continue with a tool call.")
# OCTOPUS_REWIND_MAX: a rewind call past the thread's cap
REWIND_SPENT = "No rewinds left (at most {} per thread); nothing was rewound."
# a reply whose fork started: its branches extend it, so its thread must go on from it too (a reply that also closed
# or rewound the thread would be in none of its thread's training leaves)
FORK_ALONE = "{0} refused: a reply whose fork started cannot also call {0}. Call it in a later reply."


def started(name: str, observed) -> bool:
    """A fork call the runtime accepted: its branches start (RuntimePort.fork's result, FORK_STARTED)."""
    return name == "fork" and str(observed).startswith(FORK_STARTED)
MESSAGE_TOKENS = 40  # a tool message's own tokens beside its result (role, id, its tag)


@functools.cache
def policy_tokenizer(path: str):
    """The policy's tokenizer (the tokenizers library's tokenizer.json in the dir OCTOPUS_TOKENIZER names), loaded
    once."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


def clip_results(results: list[str], room: int, note: str) -> list[str] | None:
    """OCTOPUS_ANSWER_RESERVE: a reply's tool results cut to `room` tokens of the policy's tokenizer (MESSAGE_TOKENS a
    message beside), each one past it cut where its first characters hold the tokens left (a binary search on the
    character cut by token count) and ending with CLIPPED, the last with note; None when they fit.
    settings.answer_reserve makes sure OCTOPUS_TOKENIZER is set."""
    tok = policy_tokenizer(os.environ["OCTOPUS_TOKENIZER"])
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)  # noqa: E731
    sizes = [count(text) for text in results]
    left = room - MESSAGE_TOKENS * len(results)
    if sum(sizes) <= left:
        return None
    left -= count("\n\n" + note) + count("\n" + CLIPPED) * len(results)  # what a clip adds
    out = []
    for text, size in zip(results, sizes):
        keep = max(0, left)
        cut = bisect.bisect_right(range(len(text) + 1), keep, key=lambda c: count(text[:c])) - 1
        out.append(text if size <= keep else text[:cut] + "\n" + CLIPPED)
        left -= size
    out[-1] += "\n\n" + note
    return out


def step_number(raw) -> int | None:
    """A step argument (rewind's to_step, read's from_step/to_step): an integer, or a string of digits."""
    return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) or isinstance(raw, str) and raw.strip().isdigit() else None


def rewind_target(args: dict, first: int, last: int) -> int | str:
    """OCTOPUS_REWIND_TOOL: the step a rewind call names (one of this thread's steps first..last), or why not."""
    step = step_number(args.get("to_step"))
    if step is None or not first <= step <= last:
        return (f"rewind needs to_step, one of this thread's steps: an integer from {first} (where it started) to "
                f"{last}. Nothing was rewound.")
    return step


def rewind_due(failures: list) -> bool:
    return bool(failures) and (len(failures) >= REWIND_FAILS or max(collections.Counter(failures).values()) >= REWIND_REPEATS)


async def run_branch_continuation(start: dict, complete, dispatch, max_steps: int | None = 8, closing=CLOSING,
                                  budget: int | None = None, join=None, *, refused=rejected,
                                  max_calls: int | None = None, before_turn=None, snapshot=None, restore=None,
                                  default_id=None, log=None, rewound=None, ws=None) -> dict:
    """One thread of a fork tree until it closes: a branch with the defaults and budget=settings.thread_budget(), the main
    thread with the root's arguments (CursorRuntime.run). Ends on a closing call (its reply's later calls are dropped),
    max_steps turns, max_calls (tool calls + rewinds; None = no cap), a second reply without a tool call (the first,
    while another turn fits in max_steps, the budget and its reserve and the reply did not use up the shared token
    budget, payload["budget_out"], is answered with REMIND_NOTE: once per thread, start["reminded"]), a failed request,
    or `budget` tokens of growth (None: no cap; after each reply's calls at most the room the serving window leaves past
    its first request and its mounts, start["mounted"]: settings.window_room; its first request's tag still shows the
    budget as given): the first reply whose prompt grew more than that past the first request is dropped and the
    thread ends with a "context" step. OCTOPUS_ANSWER_RESERVE (settings.answer_reserve, R): once a
    reply leaves the thread less than R of that growth, only its closing calls and rewind (OCTOPUS_REWIND_TOOL) run;
    every other call, that reply's too, is refused with RESERVE_NOTE (the step "rejected"), so the thread closes,
    rewinds or reaches the cap; and tool results
    that would take it past budget - R are clipped to the room left above R (clip_results, counted with the policy's
    tokenizer: CLIPPED, CLOSE_NOTE), the thread then in its reserve as well, so its closing turn fits in R (a join note
    is not clipped; the cap reads a request's prompt, so a closing reply, which ends the thread, is never dropped; nor
    are a rewinding reply's results, which the rewind removes). A reply whose fork started (started: the runtime
    accepted it) has its closing calls after the fork and its rewind refused (FORK_ALONE, not counted as calls): its
    branches extend it, so the thread goes on from it too. A rewind sets the growth back to its step's (the prompt
    after that step, marks' "prompt"), so its note's tag shows the context left there, and the thread leaves its
    reserve (the clipped one too) when that growth is below budget - R.
    With a budget the loop also sets start["context_left"] before each request (`budget` less the growth
    through the last reply: its prompt and completion, before the tool results that answer it), which complete shows
    in the thread's tool-result tag.

    start: branch_id, session_id, task, messages; before each request the
    loop sets start["calls_left"] (turns or calls left under max_steps / max_calls, None uncapped). complete and dispatch may
    be sync or async. closing: the tool names that end it (main and a replayed join: done; a CLOSING call outside it is
    a "rejected" step, never the thread's close); refused(name, observed): a closing call that did not close. ws: the environment (a WorkspaceProvider, octopus/runtime/tree.py): its
    check_close sees each commit first, its result goes into the call's step, and its refusal (once per thread,
    the step "gated"; a later commit "after_gate") is the call's result instead; rewinds take what the dropped steps
    did (ws.dropped), the note on them (ws.rewind_note) and which working tools edit (ws.edits) or never change the
    workspace (ws.unchanged) from it. join: async (messages) -> note or None, awaited
    after each turn's tool results: a fork the thread made in that turn runs there, and its join note is added
    (add_note). before_turn(turns) -> a reason to stop before the next request, or None. snapshot() / restore(snap):
    the thread's workspace, for rewinds. default_id(steps, index): the id of a call without one.
    log(kind, **fields): "call" / "rewind" / "remind" / "end" events (the main thread's trajectory). Under
    OCTOPUS_CURSOR a reply kept as the server sent it carries its cursor (with_cursor, cursor_delta).

    OCTOPUS_REWIND: after REWIND_FAILS failed calls, or one error REWIND_REPEATS times (ws.dropped), since the thread's
    first edit, the runtime undoes the attempt (restores the workspace taken before that edit, at most OCTOPUS_REWIND
    times).
    OCTOPUS_REWIND_TOOL: a step is a reply with tool calls (or the reminded one and its REMIND_NOTE), numbered on from
    start["step"] (the steps of the history the thread starts from; the loop sets it to the last step before each
    request, for the tag of its tool results). The
    tree is snapshot before a step's first call that may change it (not ws.unchanged), so each step's end state is known;
    rewind(to_step, note) then restores it after the turn, cuts the history back to that step's end and adds the note
    (rewound(end) -> what else it undid, e.g. the thread's forks and merges past that history length, or ""; an
    OCTOPUS_REWIND rewind calls it too, for its effects only). Each rewind's record (the result's "rewinds") keeps
    the messages it removed and the note that replaced them. OCTOPUS_REWIND_MAX (settings.rewind_max): the rewinds
    the thread ran, start["rewinds_used"] (a control starts from its thread's), are at most that many; a further
    rewind call is refused (REWIND_SPENT, "rejected") and the reserve notes no longer offer rewind (REWIND_FREES).
    With OCTOPUS_REWIND too, each rewind re-bases the other: a rule rewind drops the steps past its checkpoint, a tool
    rewind the rule's checkpoint if past its step, and both clear the rule's failures and edits. OCTOPUS_READ_TOOL numbers
    and tags the steps the same way and publishes start["marks"] = (base, marks) for other threads' reads
    (CursorRuntime._read)."""
    messages = start["messages"]
    session_id = start["session_id"]
    branch_id = start["branch_id"]
    steps: list[dict] = []
    first_prompt, grown = None, 0
    rewinds_left = int(os.environ.get("OCTOPUS_REWIND", "0") or 0)  # inside the thread's own tree
    checkpoint, since, rewinds = None, 0, []  # since: the first step of OCTOPUS_REWIND's current attempt
    edits, unchanged = (ws.edits, ws.unchanged) if ws else ((), ())
    tool_rewind, base = rewind_tool(), start.get("step") or 0
    rewind_cap, start["rewinds_used"] = rewind_max(), start.get("rewinds_used", 0)  # the rewinds it ran: never undone
    start["reminded"] = start.get("reminded") or 0  # REMIND_NOTEs it got (a control's from its thread's): never undone
    # per step from base: the history's length and steps' after it, and its tree ({"snap"} once taken; shared by the
    # steps after it while no call could change the tree: none taken = unchanged since)
    marks, fresh = [{"end": len(messages), "steps": 0, "tree": {}}], True
    if read_tool():  # where each step of this thread ends in its history, for other threads' reads (_read)
        start["marks"] = (base, marks)
    turns = calls = 0
    gated = False  # a closing commit ws.check_close refused (once)
    reserve = answer_reserve() if budget else 0
    clipped = False  # a reply's results were clipped at the reserve: the thread is in it from then on (or a rewind)
    allowed = {"calls": " or ".join(sorted(closing))}
    log = log or (lambda kind, **fields: None)

    async def run(name: str, args: dict):  # a working-tool call in the thread's workspace, not shown to the model
        ran = dispatch(name, args, branch_id)
        return (await ran if hasattr(ran, "__await__") else ran)[0]

    def end(text: str) -> None:
        steps.append({"name": None, "text": text, "session_id": session_id, "arguments": {}})
        log("end", text=text)
    while (max_steps is None or turns < max_steps) and (max_calls is None or calls < max_calls):
        reason = before_turn(turns) if before_turn else None
        if reason:
            end(f"stopped: {reason}")
            break
        # what this thread may still call, from this reply on (complete shows it next to the budget tag)
        left = [cap - used for cap, used in ((max_steps, turns), (max_calls, calls)) if cap is not None]
        start["calls_left"] = min(left) if left else None
        if budget:  # its growth left, for its tool-result tag
            start["context_left"] = max(0, budget - grown)
        if tool_rewind or read_tool():
            start["step"] = base + len(marks) - 1
        turns += 1
        try:
            payload = complete(messages, session_id)
            if hasattr(payload, "__await__"):
                payload = await payload
        except Exception as exc:
            # One thread's failed model call (e.g. upstream timeout) ends that thread, not the trial.
            end(f"request failed: {exc}")
            break
        prompt = (payload.get("usage") or {}).get("prompt_tokens")
        first_prompt = first_prompt or prompt
        if prompt:  # the prompt after the last step: a rewind back to that step grows back to it
            marks[-1]["prompt"] = prompt
        if budget and prompt and first_prompt and prompt - first_prompt > budget:
            steps.append({"name": "context", "arguments": {}, "session_id": session_id,
                          "observed": f"stopped: grew {prompt - first_prompt} tokens past its start (budget {budget})"})
            break
        if prompt and first_prompt:  # its growth through this reply, before the tool results that answer it
            grown = prompt - first_prompt + int((payload.get("usage") or {}).get("completion_tokens") or 0)
        reserved = clipped or reserve and budget - grown < reserve  # only closing calls and rewind run from here on
        spent = rewind_cap and start["rewinds_used"] >= rewind_cap
        allowed["rewind"] = REWIND_FREES if tool_rewind and not spent else ""
        choice = (payload.get("choices") or [{}])[0].get("message") or {}
        tool_calls, text = reply_tools(choice)
        if not tool_calls:
            if budget and first_prompt:  # as after a reply's calls
                budget = max(1, min(budget, window_room(first_prompt + start.get("mounted", 0))))
            # room for another turn (calls: a request is made only with one left, and this reply spent none)
            room = ((max_steps is None or turns < max_steps) and not clipped
                    and not (budget and budget - grown < max(reserve, 1)) and not payload.get("budget_out"))
            if start["reminded"] or not room:
                end(text)
                break
            start["reminded"] += 1  # its reply and the reminder are a step of their own (a rewind keeps or drops both)
            note = REMIND_NOTE.format(**allowed)
            messages.append(with_cursor(reply_message(choice, text), payload))
            add_note(messages, note)
            marks.append({"end": len(messages), "steps": len(steps), "tree": marks[-1]["tree"]})
            log("remind", text=text, note=note)
            continue
        stop, executed, back, forked, alone_n = False, [], None, False, 0
        for index, call in enumerate(tool_calls):
            name, args, call_id = decode_tool_call(call, default_id=default_id(len(steps), index) if default_id
                                                   else f"{branch_id}-{len(steps)}-{index}")
            alone = forked and (name in closing or tool_rewind and name == "rewind")  # FORK_ALONE: refused
            held = reserved and name not in closing and not (tool_rewind and name == "rewind") or alone  # refused
            if tool_rewind and not held and name not in unchanged and name not in ("rewind", "read_thread") and fresh:
                if "snap" not in marks[-1]["tree"]:  # the tree at the last step, before it changes
                    marks[-1]["tree"]["snap"] = await snapshot()
                fresh = False
            if rewinds_left and not held and checkpoint is None and name in edits:  # before the thread's first edit
                snap = await snapshot()
                if snap is not None:
                    checkpoint, since = {"messages": len(messages), "snapshot": snap}, len(steps)
            fields, gate = {}, None
            closes = name == "commit" and name in closing  # main's commit closes nothing: not checked or refused
            if ws and name == "commit" and not held:
                gate, fields = await ws.check_close(start, args, closes, run)
                gate = gate if closes and not gated else None  # a closing call is refused once
            if alone:
                observed, call_stop, refusal, alone_n = FORK_ALONE.format(name), False, True, alone_n + 1
            elif held:
                observed = RESERVE_NOTE.format(left=max(0, budget - grown), **allowed)
                call_stop, refusal = False, True
            elif gate:
                gated = True
                observed, call_stop, refusal = gate, False, True
            elif tool_rewind and name == "rewind":  # applied after the turn; an invalid one is only answered
                target = REWIND_SPENT.format(rewind_cap) if spent else rewind_target(args, base, base + len(marks) - 1)
                observed, call_stop, refusal = target, False, bool(spent)
                if isinstance(target, int):
                    back, observed = (target, args.get("note"), len(steps)), f"rewound to step {target}"
            else:
                result = dispatch(name, args, branch_id)
                observed, call_stop = await result if hasattr(result, "__await__") else result
                # a closing tool this thread does not close with (search: commit / abort) closes nothing: rejected
                refusal = refused(name, observed) or name in CLOSING and name not in closing
            steps.append({
                "name": name,
                "arguments": args,
                "tool_call_id": call_id,
                "session_id": session_id,
                "branch_id": branch_id,
                "observed": observed,
                **({"rejected": True} if refusal else {}),
                **fields,
                **({"gated": True} if gate else {"after_gate": True} if name == "commit" and gated and not refusal else {}),
            })
            executed.append((name, args, call_id, observed))
            forked = forked or started(name, observed)
            if call_stop or (name in closing and not refusal):  # the reply's later calls never run: dropped
                stop = True
                break
        if back and forked:  # a rewind before the fork that started in its reply: FORK_ALONE too
            i, back = back[2], None
            steps[i].update(observed=FORK_ALONE.format("rewind"), rejected=True)
            executed[i - len(steps)] = (*executed[i - len(steps)][:3], steps[i]["observed"])
            alone_n += 1
        if budget and first_prompt:  # at most the room the serving window leaves it, its mounts included
            budget = max(1, min(budget, window_room(first_prompt + start.get("mounted", 0))))
        close_now = CLOSE_NOTE.format(**allowed)
        if reserve and not reserved and not stop and not back and (  # a rewind removes the results and gives back room
                cut := clip_results([str(o) for *_, o in executed], budget - reserve - grown, close_now)):
            executed, clipped = [(*e[:3], o) for e, o in zip(executed, cut)], True
        messages.append(with_cursor(reply_message(choice, text, tool_calls[:len(executed)]), payload))
        messages += [{"role": "tool", "tool_call_id": call_id, "content": observed}
                     for _, _, call_id, observed in executed]
        calls += len(executed) - alone_n  # FORK_ALONE's refusals are not counted
        for name, args, call_id, observed in executed:
            log("call", name=name, args=args, call_id=call_id, observed=observed, text=text)
        # a fork this turn made starts first, from the history with its forking reply (fork_record); a rewind then
        # drops that reply with the rest of the attempt, the fork runs on (or was collected here) as recorded
        note = await join(messages) if join else None
        if note:
            add_note(messages, note)
        marks.append({"end": len(messages), "steps": len(steps), "tree": marks[-1]["tree"] if fresh else {}})
        fresh = True
        attempt = ws.dropped(steps[since:]) if rewinds_left and checkpoint is not None and not stop else None
        if back and not stop:
            to, mark = back[0], marks[back[0] - base]
            if mark["tree"].get("snap"):
                await restore(mark["tree"]["snap"])
            edited, failed = ws.dropped(steps[mark["steps"]:])
            note = ws.rewind_note(edited, failed, to, back[1], rewound(mark["end"]) if rewound else "")
            rewinds.append({"step": len(steps), "dropped_messages": len(messages) - mark["end"], "to_step": to,
                            "failures": len(failed), "edited": edited, "messages": messages[mark["end"]:],
                            "note": note})
            del messages[mark["end"]:]
            add_note(messages, note, pending=False)
            # a later rewind to this step keeps the note
            marks[to - base + 1:] = []
            marks[-1] = dict(mark, end=len(messages), steps=len(steps))
            grown = mark["prompt"] - first_prompt if mark.get("prompt") and first_prompt else grown  # grown back
            clipped = clipped and budget - grown < reserve  # out of the reserve once back below it
            if checkpoint is not None and checkpoint["messages"] > mark["end"]:
                checkpoint = None
            elif checkpoint is not None and checkpoint["messages"] == mark["end"]:  # a rule rewind keeps the note too
                checkpoint["messages"] = len(messages)
            since, start["rewinds_used"] = len(steps), start["rewinds_used"] + 1
            log("rewind", **rewinds[-1])
        elif attempt and rewind_due(attempt[1]):
            edited, failed = attempt
            await restore(checkpoint["snapshot"])
            note = ws.rewind_note(edited, failed)
            rewinds.append({"step": len(steps), "dropped_messages": len(messages) - checkpoint["messages"],
                            "failures": len(failed), "edited": edited, "messages": messages[checkpoint["messages"]:],
                            "note": note})
            if rewound:  # what it undid with the turn's reply (a read it mounts), not told: the note stays the rule's
                rewound(checkpoint["messages"])
            del messages[checkpoint["messages"]:]
            add_note(messages, note, pending=False)
            marks[:] = [m for m in marks if m["end"] <= checkpoint["messages"]]  # the tool's steps up to the checkpoint
            marks[-1] = dict(marks[-1], end=len(messages), steps=len(steps), tree={"snap": checkpoint["snapshot"]})
            grown = marks[-1]["prompt"] - first_prompt if marks[-1].get("prompt") and first_prompt else grown
            clipped = clipped and budget - grown < reserve
            checkpoint["messages"] = len(messages)  # a later rewind keeps this note: every failed attempt stays known
            since, rewinds_left, calls = len(steps), rewinds_left - 1, calls + 1
            log("rewind", **rewinds[-1])
        if stop:
            break
    return {
        "rewinds": rewinds,
        "reminded": start["reminded"],
        "branch_id": branch_id,
        "session_id": session_id,
        "task": start["task"],
        "steps": steps,
        "messages": messages,
    }


# What a fork's branches are told, what the joining thread is told, and a fork's tree.json record: one text for every
# environment, which hands in its own parts (WorkspaceProvider, octopus/runtime/tree.py)

# OCTOPUS_REWIND_TOOL: handled by the thread's own loop (run_branch_continuation), not RUNTIME_NAMES
REWIND_TOOL = _tool(
    "rewind",
    "Abandon a dead end: go back to the end of an earlier step of this thread, the step N in each tool result's tag. "
    "The repository is restored to how it was after that step, every later step (this call too) is "
    "removed from your history, and your note stays in their place with a runtime line naming the files they edited "
    "and their failures. Use it after repeated failures or when the approach is wrong. Calls and budget tokens already "
    "spent stay spent; your context left grows back by the removed steps.",
    {"to_step": {"type": "integer", "description": "The step to go back to: its tool results stay, all after it go."},
     "note": {"type": "string", "description": "What was tried since that step, why it failed, what to do instead."}},
    ["to_step", "note"],
)


# OCTOPUS_READ_TOOL: handled by CursorRuntime._read (the next request carries the mount), not dispatched. Not in the
# request's tool list (a list that changes mid-episode breaks the shared prefix): its definition is appended where a
# read can succeed (read_card: a branch's task message, main's first fork result)
READ_TOOL = _tool(
    "read_thread",
    "Read another thread of this episode, never your own (a branch), without copying "
    "its text: the readable steps among from_step to to_step are mounted into your attention from your next reply on "
    "and stay there. A step is a reply and the tool results after it, numbered as the step N in that thread's tool "
    "result tags; a branch's steps continue the numbering of the thread it forked from. A step is readable once that "
    "thread's next reply is back or it has ended, unless your own history holds it "
    "already; without from_step and to_step, its latest readable steps are mounted. One read_thread per reply, not "
    "with fork or rewind. For files use view.",
    {"thread": {"type": "string", "description": "The other thread: a branch id such as branch-1."},
     "from_step": {"type": "integer", "description": "The first step to read (omitted: from its first step)."},
     "to_step": {"type": "integer", "description": "The last step to read, from_step or later (omitted: to its latest "
                                                   "readable step)."}},
    ["thread"],
)


def read_card(description: str = READ_TOOL["function"]["description"],
              parameters: dict = READ_TOOL["function"]["parameters"]) -> str:
    """read_thread's definition as the system prompt shows a declared tool: the paragraph holding it in the policy's
    chat template rendered for one user message with it as the only tool, in the shape the servers hand the template
    (the OpenAI function model: description, name, parameters, strict false). OCTOPUS_TOKENIZER: the policy's
    tokenizer dir (octopus/rl/config.set_derived_settings: the run config's tokenizer_path)."""
    tool = {"type": "function", "function": {"description": description, "name": "read_thread",
                                             "parameters": parameters, "strict": False}}
    text = chat_template(os.environ["OCTOPUS_TOKENIZER"]).render(messages=[{"role": "user", "content": "stub"}],
                                                                 tools=[tool], add_generation_prompt=False)
    (block,) = [p for p in text.split("\n\n") if "read_thread" in p]
    return "read_thread is available from here on:\n" + block


@functools.cache
def chat_template(path: str):
    """The chat template of the tokenizer dir path, in the jinja2 environment of transformers' apply_chat_template
    (the Harbor environment has no transformers): its render(messages=..., tools=...) is apply_chat_template's text."""
    import jinja2.ext
    import jinja2.sandbox

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals.update(raise_exception=raise_exception, strftime_now=lambda f: datetime.datetime.now().strftime(f))
    file = Path(path) / "chat_template.jinja"
    return env.from_string(file.read_text() if file.exists() else
                           json.loads((Path(path) / "tokenizer_config.json").read_text())["chat_template"])


def runtime_tools() -> list[dict]:
    """The runtime tools every environment offers: fork / commit / abort, and rewind under OCTOPUS_REWIND_TOOL (an
    environment may offer its own fork / commit / abort in their place: SWE's strategies)."""
    return with_rewind_rule(RUNTIME_TOOLS) + [rewind_capped(REWIND_TOOL)] if rewind_tool() else RUNTIME_TOOLS


def rewind_capped(tool: dict) -> dict:
    """A rewind tool's definition, naming OCTOPUS_REWIND_MAX's cap when set (one text for every thread)."""
    cap = rewind_max()
    return dict(tool, function=dict(tool["function"], description=tool["function"]["description"] + (
        f" You may rewind at most {cap} times per thread; further rewinds are refused."))) if cap else tool


def with_rewind_rule(tools: list[dict]) -> list[dict]:
    """Under OCTOPUS_REWIND_TOOL: the tools with their fork's FORK_RULE naming rewind too."""
    old, new = FORK_RULE, FORK_RULE.replace("commit or abort", "commit, abort or rewind")
    return [{**t, "function": {**t["function"], "description": t["function"]["description"].replace(old, new)}}
            if t["function"]["name"] == "fork" else t for t in tools]


LINEAR_DROP = {"fork", "commit", "abort", "read_thread"}


LOCK_NOTICE = " Available only after a successful fork."


def _without_lock_notice(item: dict) -> dict:
    fn = item["function"]
    return {**item, "function": {**fn, "description": fn["description"].replace(LOCK_NOTICE, "")}}


def unlocked_tools(tools: list[dict], linear: bool) -> list[dict]:
    """An environment's working tools and the runtime tools with nothing locked: each from the start, without its lock
    notice; linear: one thread, no forking (the runtime tools a linear run keeps carry no lock notice)."""
    catalog = [_without_lock_notice(item) for item in tools + runtime_tools()]
    return [item for item in catalog if not linear or item["function"]["name"] not in LINEAR_DROP]


BRANCH_REPORT = "findings, files, changes"  # what a commit summary carries (the search environment has its own)


def branch_task_message(branch_id: str, task: str, siblings: int, workdir: str | None = None,
                        statement: str | None = None, report: str = BRANCH_REPORT, roster: list | None = None,
                        card: str = None, close: str = "", reason: bool = False, rules: str = "",
                        contract: str | None = None, names: bool = False) -> str:
    """What a cloned branch is told: who it is, its one task, where it works, and how to close.
    Without its id a branch cannot call commit/abort, and without the contract it never stops. statement: the bundle
    task's own /tasks/<k>.md block, after the forking thread's task text (which may be wrong or a placeholder).
    roster (OCTOPUS_READ_TOOL): the threads it may read at fork time and card (read_offer). The environment's parts
    (WorkspaceProvider): close, a sentence after the commit/abort contract (SWE's under OCTOPUS_AUTO_MERGE says what
    its done does in a branch); reason, an abort carries a reason; rules, a paragraph after them (SWE branch_fix: one fix
    attempt, verified by its acceptance check); contract, how the branch closes in place of the commit/abort contract
    (search: done); names: branch_id and the roster are read_thread's thread names (search: fork1/branch-0), not
    branch ids."""
    text = (
        (f"You are {branch_id}, one of {siblings} parallel branches cloned from the context above.\n" if siblings > 1 else
         f"You are {branch_id}, the one branch cloned from the context above.\n") +
        f"Your task: {task}\n" + (f"{statement}\n" if statement else "") +
        "Work only on this task; "
        + ("the other branches cover" if siblings > 1 else "the thread that forked you covers") + " the rest. "
        + (contract if contract is not None else "When it is done, call commit with "
           f'branch_id="{branch_id}" and a summary the thread that forked you can act on ({report}). '
           f'If it cannot be done, call abort with branch_id="{branch_id}"' + (" and a reason." if reason else "."))
    )
    if close:
        text += " " + close
    text += rules
    if workdir:
        text += (
            f"\nYou work in a private copy of the repo at {workdir}: every tool runs there and /testbed paths "
            "map to it. Other branches and the main repo do not see your edits; when you close, they are saved "
            "as a patch the thread that forked you may apply."
        )
    if roster:
        text += "\n" + read_offer(roster, card, names)
    return text


def task_line(task: str) -> str:
    """A branch's task as a roster names it: its first line, 80 characters at most."""
    return task.strip().split("\n", 1)[0][:80]


def read_offer(roster: list, card: str | None = None, names: bool = False, now: bool = False) -> str:
    """OCTOPUS_READ_TOOL: read_thread offered to a branch: the threads it may read (CursorRuntime._others), as
    (read_thread id, or with names its thread name; a branch's task line or None), then read_thread's card (default
    read_card(); the search environment's has its own wording). In its task message (at its fork), or (now) a note
    before a later request, once a thread it may read has a readable step (WorkspaceProvider.read_when_readable)."""
    return (f"Other threads {'now' if now else 'at this fork'} (read_thread {'names' if names else 'ids'}): "
            + "; ".join(f"{tid} (task: {line})" if line else tid for tid, line in roster) + ".\n"
            + (card or read_card()))


def branch_starts(port: RuntimePort, history: list, workdirs: dict | None = None, report: str = BRANCH_REPORT,
                  card: str = None, close: str = "", reason: bool = False, rules: str = "",
                  contract: str | None = None, read: bool | None = None, named: dict | None = None,
                  others: list = ()) -> list[dict]:
    """The port's last fork as N independent continuations of the forking thread's history (card, close, reason,
    rules, contract: the read card and the environment's parts of the task message, branch_task_message; read: whether
    the task message offers read_thread, its roster and card, None: read_tool(); named: the roster's read_thread names by
    branch id, None: the branch ids). The roster names the branch's siblings, then others: (read_thread name, task) of
    every other thread the forking thread may read (CursorRuntime._others); main and a nested fork's forking
    branches wait for this fork, and their steps before it are the branch's own history, so no read of them can mount.
    Each branch's task goes after the history (add_note: with thinking on, into its own copy of the forking reply's last
    tool result)."""
    info = port.last_fork or {}
    frozen = [dict(item) for item in history]
    starts = []
    branches = info.get("branches") or []
    for branch in branches:
        roster = [((named or {}).get(b["branch_id"], b["branch_id"]), task_line(b["task"]))
                  for b in branches if b is not branch] + [(k, task_line(task)) for k, task in others]
        messages = list(frozen)
        add_note(messages, branch_task_message(
            (named or {}).get(branch["branch_id"], branch["branch_id"]), branch["task"], len(branches),
            (workdirs or {}).get(branch["branch_id"]),
            branch.get("statement"), report, roster if (read_tool() if read is None else read) else None, card, close,
            reason, rules, contract, bool(named),
        ))
        starts.append({"branch_id": branch["branch_id"], "task": branch["task"], "messages": messages})
    return starts


def fork_record(main: list, reports: list[dict], fork_tasks=None) -> dict:
    """Where a fork sits in the main thread and what each branch said after the shared prefix.

    ``prefix_len`` counts the messages of ``main`` (the forking thread's history at the fork), which every branch
    starts from;
    ``fork_index`` is the main assistant message that called fork; ``join_index`` is where the
    main thread continues (the "branches finished" note is appended there). fork_tasks(message): the (check, task
    id) of each task of the fork call in the forking reply, as the environment records them
    (WorkspaceProvider.fork_tasks; none: "" and None).
    """
    fork_index = max((
        i for i, m in enumerate(main)
        if m.get("role") == "assistant"
        and any((c.get("function") or {}).get("name") == "fork" for c in m.get("tool_calls") or [])
    ), default=None)  # None: the environment's fork before main's first reply (OCTOPUS_AUTO_FORK)
    tasks = fork_tasks(main[fork_index]) if fork_tasks and fork_index is not None else []
    branches = []
    for i, report in enumerate(reports):
        own = report["messages"]
        if own[: len(main)] == main:
            own = own[len(main):]
        closing = closing_step(report["steps"]) or {}
        branches.append({
            "branch_id": report["branch_id"],
            "task": report["task"],
            "messages": own,
            "ended_with": closing_name(report["steps"]),
            # why an unclosed branch stopped: its last text step (e.g. "stopped: deadline", a read control's)
            "stop": None if closing else next((s.get("text") for s in reversed(report["steps"]) if s.get("name") is None), None),
            "accepted": closing.get("acceptance_verified"),
            # what it hands the forking thread: its commit's summary, its done's answer (search grades it: read control,
            # search_gate)
            "summary": handed(closing),
            "check": tasks[i][0] if i < len(tasks) else "",  # as the forking thread wrote it (plan: its validity)
            "acceptance_own": closing.get("acceptance_own"),
            "task_id": tasks[i][1] if i < len(tasks) else None,
            "rewinds": report.get("rewinds") or [],
            "reminded": report.get("reminded", 0),  # REMIND_NOTEs in its history
        })
    return {"prefix_len": len(main), "fork_index": fork_index, "join_index": len(main), "branches": branches}


def join_note(reports: list[dict], patches: dict | None = None, merged: dict | None = None,
              close: str = "commit") -> str:
    """The message the main thread gets once every branch has closed: each branch's outcome, its
    commit summary (a done's answer and evidence), and where its file changes were saved, so the join can build on (and
    merge) the branches instead of redoing them. patches: {branch_id: (patch_path, diffstat)}. merged
    (OCTOPUS_AUTO_MERGE): {branch_id: "merged" | "conflict: <files>"} for each committed patch the runtime already
    applied. close: the call an unclosed branch did not make (search: done). An environment may add its own guidance
    (WorkspaceProvider.join_note)."""
    lines = ["branches finished:"]
    for report in reports:
        closing = closing_step(report["steps"])
        if closing and closing["name"] == "commit":
            checked = closing.get("acceptance_verified")
            passed = closing["arguments"].get("acceptance_passed") if checked is None else checked
            how = ("as reported" if checked is None else
                   "checked by the runtime with the branch's own check, which fails on the fork point" if
                   closing.get("acceptance_own") else "checked by the runtime") + (
                "; committed again after the runtime refused it" if closing.get("after_gate") else "")
            said = "" if passed is None else f" (acceptance {'passed' if passed else 'FAILED'}, {how})"
            outcome = f"committed{said}: {closing['arguments'].get('summary', '')}"
        elif closing and closing["name"] == "abort" and closing["arguments"].get("reason"):
            outcome = f"aborted: {closing['arguments']['reason']}"
        elif closing and closing["name"] == "done" and closing["arguments"].get("answer"):
            evidence = closing["arguments"].get("evidence")
            outcome = f"done: {closing['arguments']['answer']}" + (f"\n  evidence: {evidence}" if evidence else "")
        elif closing:
            outcome = closing["name"]
        else:  # unclosed: it hands nothing (handed), neither a final reply nor a last result
            outcome = f"stopped after {len(report['steps'])} steps without {close}"
        line = f"- {report['branch_id']} ({report['task']}): {outcome}"
        patch = (patches or {}).get(report["branch_id"])
        if patch:
            path, stat = patch
            done = (merged or {}).get(report["branch_id"])
            if done and done.startswith("conflict: "):
                done = f"not applied: it conflicts in {done[10:]}; your files are unchanged"
            line += (
                "\n  no file changes" if not stat
                else f"\n  changes saved to {path} ({stat}); {done}" if done
                else f"\n  changes saved to {path} ({stat}); merge into /testbed with: git apply --3way {path}"
            )
        lines.append(line)
    return "\n".join(lines)
