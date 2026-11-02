"""With thinking on, a runtime note must not re-render a thread's history without its reasoning (core.add_note).

fixtures/qwen35_chat_template.jinja is node-94's /mnt/disk1/user/models/Qwen3.5-4B-text/chat_template.jinja: it
keeps an assistant turn's <think> block only after the last real user query, so every injected user message dropped
the reasoning of all earlier replies and the next prompt no longer extended the previous one.
"""

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from octopus.serving.delta_render import parse_tool_arguments
from octopus.runtime.core import branch_starts
from octopus.runtime.settings import thinking
from octopus.runtime.core import add_note, run_branch_continuation, RuntimePort

TEMPLATE = (Path(__file__).parent / "fixtures/qwen35_chat_template.jinja").read_text()
TOOLS = [{"type": "function", "function": {"name": "view", "parameters": {"type": "object", "properties": {}}}}]
ON = json.dumps({"enable_thinking": True})


def render(messages: list[dict], generation: bool = True) -> str:
    return render_jinja_template([parse_tool_arguments(messages)], tools=TOOLS, chat_template=TEMPLATE,
                                 add_generation_prompt=generation, enable_thinking=True)[0][0]


def reply(i: int, name: str = "view", **args) -> dict:
    return {"id": f"r{i}", "choices": [{"message": {
        "content": f"step {i}", "reasoning_content": f"reasoning {i}",
        "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }}]}


def test_main_prompts_extend_across_join_and_runtime_notes(monkeypatch):
    monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", ON)
    start = {"branch_id": "main", "session_id": "s", "task": "",
             "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix the bug"}]}
    sent = []

    def complete(messages, _sid):
        sent.append(copy.deepcopy(messages))
        return reply(len(sent), path="a.py")

    async def join(_messages):  # a fork's join note after the first turn
        return "branches finished" if len(sent) == 1 else None

    def before_turn(turns):  # a runtime note before a turn (add_note)
        if turns == 2:
            add_note(start["messages"], "The main thread has finished.")
    asyncio.run(run_branch_continuation(start, complete, lambda *_: ("ok", False), max_steps=4, join=join,
                                        before_turn=before_turn))
    prompts = [render(m) for m in sent]
    assert all(b.startswith(a) for a, b in zip(prompts, prompts[1:]))
    assert all(f"reasoning {i}" in prompts[-1] for i in range(1, len(sent)))
    assert sent[1][-1]["role"] == "tool" and sent[1][-1]["content"].endswith("\n\nbranches finished")

    # a rewind cuts back to results already sent: its note is a tool message of its own, every kept reasoning stays
    cut = sent[2][:5]
    add_note(cut, "rewound to step 1", pending=False)
    assert cut[4] == sent[2][4] and cut[-1] == {"role": "tool", "content": "rewound to step 1"}
    assert render(cut).startswith(render(cut[:4], generation=False)) and "reasoning 1" in render(cut)


@pytest.mark.parametrize("thinking", [True, False])
def test_branch_first_prompt_extends_main_through_forking_reply(monkeypatch, thinking):
    monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", json.dumps({"enable_thinking": thinking}))
    port = RuntimePort()
    observed = port.fork(["fix a", "fix b"])
    forking = {"role": "assistant", "content": "", "reasoning_content": "two bugs",
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "fork", "arguments": "{}"}}]}
    main = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix the bugs"}, forking,
            {"role": "tool", "tool_call_id": "c1", "content": observed}]
    starts = branch_starts(port, main)
    if not thinking:  # unchanged: the task is a user message after the history
        assert [s["messages"][:-1] for s in starts] == [main, main]
        assert all(s["messages"][-1]["role"] == "user" and "Your task: " in s["messages"][-1]["content"] for s in starts)
        return
    through_fork = render(main[:3], generation=False)
    for start, task in zip(starts, ("fix a", "fix b")):
        last = start["messages"][-1]
        assert len(start["messages"]) == len(main) and last["role"] == "tool"
        assert last["content"].startswith(observed + "\n\n") and f"Your task: {task}" in last["content"]
        assert render(start["messages"]).startswith(through_fork)
    assert main[-1]["content"] == observed  # main's own result is not touched


@pytest.mark.parametrize("kwargs,on", [(None, False), ("", False), ('{"enable_thinking": false}', False),
                                       ('{"enable_thinking": true}', True), ("{}", True), ('{"foo": 1}', True),
                                       ('{"enable_thinking": "false"}', True)])
def test_thinking_reads_the_kwargs_as_the_template_does(monkeypatch, kwargs, on):
    """The template thinks unless enable_thinking is defined and false; unset (the Grok API) is off."""
    if kwargs is None:
        monkeypatch.delenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", raising=False)
    else:
        monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", kwargs)
    assert thinking() is on


def test_read_control_copy_keeps_a_note_after_the_step_listing(monkeypatch):
    from test_rl_plan import _import_agent
    _import_agent(monkeypatch)
    without_step_listing = sys.modules["octopus.runtime.tree"].without_step_listing
    read = "Mounted branch-1's steps 1 to 2 (90 tokens; ...): ...\nstep 1: view(a.py) -> x\nstep 2: shell(ls) -> y"
    assert without_step_listing(read) == read.split("\nstep ", 1)[0]
    assert without_step_listing(read + "\n\nbranches finished:\n- b") == (
        read.split("\nstep ", 1)[0] + "\n\nbranches finished:\n- b")


@pytest.mark.parametrize("thinking", [True, False])
def test_a_reminded_reply_and_its_close_are_one_trained_leaf(monkeypatch, thinking):
    """core.REMIND_NOTE: the reply without a tool call goes back into the history as AReaL recorded it and the reminder
    is a runtime note after it (thinking: a tool message, so the reasoning stays), so the next request extends the
    previous one message for message (AReaL's default_prefix_matcher) and, with thinking, rendered: one leaf in which
    both completions, the text reply's and the closing call's, are trained (group.assign_threads: owned from the first
    prompt on), the reminder between them on the prompt side."""
    from octopus.rl.group import assign_threads
    from octopus.runtime.core import REMIND_NOTE
    from test_rl_plan import _episode, _leaf
    monkeypatch.setenv("OCTOPUS_CHAT_TEMPLATE_KWARGS", json.dumps({"enable_thinking": thinking}))
    start = {"branch_id": "main", "session_id": None, "task": "",
             "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Which constellation?"}]}
    text = {"role": "assistant", "content": "It is Monoceros.", **({"reasoning_content": "reasoning 1"} if thinking
                                                                     else {})}
    done = reply(2, "done", answer="Monoceros")["choices"][0]["message"]
    done = {"role": "assistant", **done} if thinking else {"role": "assistant", **done, "reasoning_content": None}
    sent = []

    def complete(messages, _sid):
        sent.append(copy.deepcopy(messages))
        return {"id": f"r{len(sent)}", "choices": [{"message": dict([text, done][len(sent) - 1])}]}
    report = asyncio.run(run_branch_continuation(start, complete, lambda *_: ("Final answer recorded", True),
                                                 max_steps=None, max_calls=5, closing={"done"},
                                                 refused=lambda *_: False))
    assert len(sent) == 2 and report["reminded"] == 1 and report["steps"][-1]["name"] == "done"
    assert sent[1] == sent[0] + [text, {"role": "tool" if thinking else "user", "content": REMIND_NOTE.format(
        calls="done")}]  # AReaL's record of the reply (model_dump(exclude_none)), then the reminder
    if not thinking:  # AReaL's concat leaf appends the reminder's tokens to the reply's: no re-render to compare
        return
    first, with_reply, second = render(sent[0]), render(sent[0] + [text], generation=False), render(sent[1])
    assert with_reply.startswith(first) and second.startswith(with_reply)
    assert REMIND_NOTE.format(calls="done") in second[len(with_reply):] and "reasoning 1" in second
    completion = len(with_reply) - len(first)  # characters for tokens: the leaf is the closing request's
    requests = [("r1", "main", len(first), completion), ("r2", "main", len(second), 7)]
    (row,) = assign_threads([_leaf("r2", list(range(len(second) + 7)))], _episode(requests))
    assert (row["own_start"], row["adv"], row["spans"]) == (len(first), 0.5, [])
