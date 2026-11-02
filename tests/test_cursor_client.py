"""OCTOPUS_CURSOR (docs/cursor-protocol.md), agent side: after its first request a thread sends only the messages after
its last reply, naming that reply's response id. A cursor-aware server rebuilds each request as the cursor request's
messages + its reply + the delta; for every request that equals what full mode sends."""

import asyncio
import json
import sys

import pytest

from test_rl_plan import B1, DONE, S0, S1, _call, _commit, _fork_call, _import_agent, _run_tree

SHELL = _call("shell", {"command": "true"})


def _episode(monkeypatch, tmp_path, cursor, control, serve=None, tool=False):
    """A tree (control: where the no-fork control starts, main's fork or branch-1's): main makes a call, an edit and
    three failing calls (OCTOPUS_REWIND rewinds to before the edit; tool: a rewind call to that step instead,
    OCTOPUS_REWIND_TOOL), forks (5, 6), joins and is done; branch-1 calls,
    forks (8, 9), joins and commits. serve: _run_tree's (default: the fake model). Returns each request body (and under
    "reply" the calls the fake model replied with) by response id, and tree.json."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _call("edit", {"path": "/testbed/f.py", "old": "a", "new": "b"}),
                    3: _call("rewind", {"to_step": 1, "note": "x"}) if tool else _call("shell", {"command": "false"}) * 3,
                    4: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {1: SHELL, 2: _fork_call(8, 9)}.get(turn, _commit(thread))
        if thread == "control":
            return SHELL if turn == 1 else DONE if control == "fork0" else _commit(B1)
        return _commit(thread)

    bodies = {}
    env = {"OCTOPUS_PLAN_CONTROL": "1", "OCTOPUS_REWIND" + "_TOOL" * tool: "1",
           "pick": lambda seed, pool: next(c for c in pool if c["path"] == control)}
    serve = serve or (lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps({**body, "reply": calls}))}))
    tree = _run_tree(monkeypatch, tmp_path, reply, serve=serve, OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR=cursor, **env)[2]
    return bodies, tree


@pytest.mark.parametrize("control, tool", [("fork0", False), ("fork0/branch-1/fork0", False), ("fork0", True), ("fork0/branch-1/fork0", True)])
def test_cursor_requests_rebuild_the_full_mode_requests(monkeypatch, tmp_path, control, tool):
    """tool: the rewind is the model's call (OCTOPUS_REWIND_TOOL): it continues from step 1's reply, exactly."""
    (tmp_path / "full").mkdir()
    (tmp_path / "cursor").mkdir()
    full, tree = _episode(monkeypatch, tmp_path / "full", "0", control, tool=tool)
    sent, cursor_tree = _episode(monkeypatch, tmp_path / "cursor", "1", control, tool=tool)
    assert sent.keys() == full.keys()
    assert cursor_tree["main"] == tree["main"] and "cursor" not in json.dumps(cursor_tree)  # tree.json as in full mode
    for rid, body in sent.items():
        whole = full[rid]["messages"]
        assert not any("cursor" in m for m in body["messages"])  # the marker never reaches the server
        assert {k: v for k, v in body.items() if k not in ("messages", "cursor")} == {
            k: v for k, v in full[rid].items() if k != "messages"}
        if "cursor" not in body:
            assert body["messages"] == whole
            continue
        parent = full[body["cursor"]]["messages"]  # the server's record of the cursor: its prompt and reply
        reply = whole[len(parent)]  # what full mode sends after that prompt: the server's reply as it sent it
        assert whole[:len(parent)] == parent and reply["role"] == "assistant" and reply["content"] == ""
        assert reply.get("tool_calls") == full[body["cursor"]]["reply"]
        assert whole[len(parent) + 1:] == body["messages"]
    cursors = {rid: body.get("cursor") for rid, body in sent.items()}
    # the only full sends: a thread's first request, main's control
    assert [rid for rid, c in cursors.items() if c is None] == ["main#1"] + ["control#1"] * (control == "fork0")
    assert sum(len(json.dumps(b)) for b in sent.values()) < sum(len(json.dumps(b)) for b in full.values())
    assert cursors["main#3"] == "main#2" and cursors["main#4"] == "main#1"  # the rewind continues from before the edit
    if tool:  # after step 1's result, the note: the tool's request is the rule's but for the note and the step tags
        assert sent["main#4"]["messages"][-1]["content"].startswith(  # a tool message, tagged
            "[context left: 32768 | step 1]\n[runtime] Rewound to step 1 at your request")
        assert sent["main#4"]["messages"][0]["content"].startswith("[context left: ")
        assert sent["main#4"]["messages"][0]["content"].split("\n")[0].endswith(" | step 1]")
    assert cursors["fork0/branch-0#1"] == cursors[f"{B1}#1"] == "main#4"  # a fork: children of the forking reply
    assert cursors[f"{S0}#1"] == cursors[f"{S1}#1"] == f"{B1}#2"
    assert cursors["main#5"] == "main#4" and cursors[f"{B1}#3"] == f"{B1}#2"  # after the joins
    if control != "fork0":  # branch-1's control: from main's fork, its task message and replies resent
        assert cursors["control#1"] == "main#4" and cursors["control#2"] == "control#1"


def test_a_reply_the_budget_cuts_gets_no_cursor(monkeypatch):
    """closing_only drops the calls of the reply that used up the budget: the history keeps another reply than the
    server's, so the transport gives it no cursor (with_cursor leaves it unmarked)."""
    monkeypatch.setenv("OCTOPUS_CURSOR", "1")
    mod = _import_agent(monkeypatch)
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent._budget = {"limit": 10, "used": 0}
    calls = _call("edit", {}) + _call("done", {})
    agent._complete = lambda messages, s=None, c=None: {
        "id": "r", "choices": [{"message": {"content": "", "tool_calls": list(calls)}}], "usage": {"completion_tokens": 6}}
    history = [{"role": "user", "content": "go"}]
    assert asyncio.run(agent._complete_async(history))["cursor"] == "r"  # 4 tokens left: kept as sent
    assert "cursor" not in asyncio.run(agent._complete_async(history))  # the budget is gone: done only


def test_cursor_request_bytes_stay_flat_on_a_long_thread(monkeypatch, tmp_path):
    """One thread of 48 replies, each answered by a ~1.5 kB tool result: full mode resends the history, cursor mode
    the last result."""
    from octopus.runtime.core import run_branch_continuation

    def sizes(cursor):
        monkeypatch.setenv("OCTOPUS_CURSOR", cursor)
        mod = _import_agent(monkeypatch)
        agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
        agent.logs_dir = tmp_path / cursor
        out = []

        def post_chat(url, body, headers, log):
            out.append(len(json.dumps(body)))
            return {"id": f"r{len(out):03d}", "choices": [{"message": {"content": "", "tool_calls": SHELL}}]}
        monkeypatch.setattr(sys.modules["octopus.runtime.tree"], "post_chat", post_chat)
        start = {"branch_id": "main", "session_id": None, "task": "", "messages": [
            {"role": "system", "content": "s" * 8000}, {"role": "user", "content": "u" * 2000}]}
        asyncio.run(run_branch_continuation(start, lambda m, s: agent._complete_async(m),
                                            lambda name, args, b: ("x" * 1500, False), 48, budget=None))
        return out

    full, cursor = sizes("0"), sizes("1")
    print("request bytes (turn: full, cursor):", [(i + 1, full[i], cursor[i]) for i in (0, 1, 7, 15, 31, 47)])
    assert full[0] == cursor[0] and full[-1] > 8 * cursor[-1] and max(cursor[1:]) == min(cursor[1:])
