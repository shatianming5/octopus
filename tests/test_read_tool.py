"""OCTOPUS_READ_TOOL (docs/cross-branch-read.md), agent side: read_thread(thread, from_step, to_step) mounts steps of another
thread of the episode with the reader's next request; the tool result names what was mounted, never its text."""

import json
import sys

import pytest

from test_rl_plan import DONE, _call, _commit, _fork_call, _import_agent, _run_tree

SHELL = _call("shell", {"command": "true"})
B0, B1 = "fork0/branch-0", "fork0/branch-1"


def _calls(*calls):
    return [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]


def _read(thread, a, b):
    return "read_thread", {"thread": thread, "from_step": a, "to_step": b}


def test_read_mounts_the_next_request_and_refuses_bad_reads(monkeypatch, tmp_path):
    """main: a call and a read before any fork (step 1), a fork (step 2: branch-0 calls twice and commits, steps 3-5;
    branch-1 tries main's steps 1 and 2 and commits), then two reads in one reply (step 3), then five invalid reads
    (step 4: one of steps it mounted already), done. Only the first read of step 3 mounts, on main's next request alone; every result is a short note
    (refusals name the reader and the other threads' readable steps), and every request offers read_thread with step
    tags on its tool results."""
    def reply(thread, turn):
        if thread == "main":
            return {1: _calls(("shell", {"command": "true"}), _read("branch-0", 1, 1)), 2: _fork_call(5, 6),
                    3: _calls(_read("branch-0", 3, 4), _read("branch-0", 3, 3)),
                    4: _calls(_read("branch-9", 1, 1), _read("main", 1, 1), _read("branch-0", 3, 4),
                              _read("branch-0", 1, 2), _read("branch-0", 4, 3))}.get(turn, DONE)
        if thread == B1:
            return _calls(_read("main", 1, 1), _read("main", 2, 2)) if turn == 1 else _commit(thread)
        return SHELL if turn < 3 else _commit(thread)

    bodies = {}

    def serve(rid, body, calls):  # prompt tokens grow by 1000 per turn of a thread
        bodies[rid] = json.loads(json.dumps(body))
        return {"id": rid, "usage": {"completion_tokens": 1, "prompt_tokens": 1000 * int(rid.split("#")[1])},
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}

    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, serve=serve, OCTOPUS_HTTP_WORKERS="1",
                                        OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1")
    assert not any("read_thread" in [t["function"]["name"] for t in b["tools"]] for b in bodies.values())
    assert {rid for rid, b in bodies.items() if "mount" in b} == {"main#4"}
    assert bodies["main#4"]["mount"] == {"thread": "branch-0", "from": f"{B0}#1", "to": f"{B0}#3"}
    assert bodies["main#4"]["cursor"] == "main#3"
    assert [r.get("mount") for r in rows if r["id"] == "main#4"] == [bodies["main#4"]["mount"]]
    tag, *read = [m["content"] for m in bodies["main#4"]["messages"]][0].split("\n", 1)
    assert tag.startswith("[main | context left: ") and tag.endswith(" | step 3]")
    # each tag names the thread whose history holds the result: a branch's inherited results keep main's, the fork result
    # (left untagged by main's forking reply) gets each branch's own name, not the sibling's that sent first
    for bid, child in agent._main_thread["started"][0]["children"].items():
        assert [m["content"].split(" | ")[0] for m in child["start"]["messages"] if m["role"] == "tool"] == [
            "[main"] * 2 + [f"[{bid}"] * 3 + ["commit recorded: " + bid]
    assert {m["content"].split(" | ")[0] for m in tree["main"] if m["role"] == "tool"} == {"[main", "exit 0"}  # done: unsent
    results = lambda rid: [m["content"].split("\n", 1)[1] for m in bodies[rid]["messages"] if m["role"] == "tool"]
    others = "Other threads: branch-0 (readable steps 5 to 5), branch-1 (readable steps 3 to 4)."  # ended: through its commit;
    # branch-0's steps 3-4 mounted already: never twice
    assert results("main#2")[1] == "No thread branch-0 in this episode. There is no other thread yet: fork first. " \
                                   "Nothing was read."
    assert results("main#4") == [
        "Mounted branch-0's steps 3 to 4 (2000 tokens; branch-0 has readable steps 3 to 5): from your next reply on "
        "you attend to them; their full text is not copied here.\nstep 3: shell(true) -> exit 0\nstep 4: shell(true) -> exit 0",
        "One read_thread per reply: the first one of this reply is mounted with your next reply. Nothing else was read."]
    assert results("main#5") == [
        f"No thread branch-9 in this episode. {others} Nothing was read.",
        f"You are main: read_thread reads another thread, never your own. {others} Nothing was read.",
        "branch-0's step 3 is in your own history already (branch-0 has readable steps 5 to 5). Nothing was read.",
        "branch-0 has readable steps 5 to 5: it has ended and gets no more. Nothing was read.",
        "read_thread needs thread (another thread: a branch id) and, if given, the "
        "integers from_step <= to_step (for files use view). Nothing was read."]
    waits = "main waits for your fork to close: none of its steps can be read. Other threads: branch-0 (no readable " \
            "step yet). Nothing was read."
    assert results(f"{B1}#2") == [waits, waits]  # main waits at the fork
    assert '"mount":' not in json.dumps(tree) and "cursor" not in json.dumps(tree)


def test_a_mount_sglang_chat_does_not_confirm_fails_the_request(monkeypatch, tmp_path):
    """SGLang's own chat endpoint (evaluation: its responses carry metadata) answers a mount request with
    metadata.octopus_mount; one without the chat mount ignores the mount silently, so the client fails that request
    (main ends "request failed") and logs it as not a 200: the episode is an infra failure, not scored. Confirmed, the
    episode runs on; the AReaL proxy (no metadata) is not checked."""
    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _calls(_read("branch-0", 2, 3))}.get(turn, DONE)
        return SHELL if turn < 3 else _commit(thread)

    def run(confirm):
        def serve(rid, body, calls):
            meta = {"weight_version": "default", **({"octopus_mount": 7} if confirm and "mount" in body else {})}
            return {"id": rid, "usage": {"completion_tokens": 1, "prompt_tokens": 1000 * int(rid.split("#")[1])},
                    "metadata": meta, "choices": [{"message": {"content": "", "tool_calls": calls}}]}
        (tmp_path / str(confirm)).mkdir()
        return _run_tree(monkeypatch, tmp_path / str(confirm), reply, serve=serve, OCTOPUS_HTTP_WORKERS="1",
                         OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1")

    _, rows, tree, _, _ = run(False)
    assert [r["id"] for r in rows if r.get("mount")] == ["main#3"] and tree["main_end"] == "request failed"
    # its row is not a 200: both dev_evals (workflow.trial_outcome, search_dev_eval.outcome) count an infra failure
    assert [r["http_code"] for r in rows if r.get("mount")] == ["unmounted"]
    _, rows, tree, _, _ = run(True)
    assert [r["id"] for r in rows if r.get("mount")] == ["main#3"] and tree["main_end"] == "done"
    assert all(r["http_code"] == 200 for r in rows)


def test_read_mounts_a_range_with_the_sources_read_and_refuses_a_rewritten_edge_a_fork_reply_and_the_control(monkeypatch):
    """branch-0's read reply a3 was rewritten, so its mount request continued from a2: the mount's placeholders sit
    after a2, inside steps 4 to 5, which mount (the reader also attends to what branch-0 read: its 50 tokens count in
    the size, as prompt_tokens leave them out; steps 6 to 6 start at a4, after them); a range with a rewritten edge, a
    read beside a fork and a read by the control are refused."""
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    reply = lambda rid: {"role": "assistant", "content": "", **({"cursor": rid} if rid else {})}  # noqa: E731
    tool = {"role": "tool", "content": "ok"}
    a_msgs = [{"role": "user", "content": "task"}, reply("a1"), tool, reply("a2"), tool, reply(None), tool, reply("a4"),
              tool, reply("a5")]  # steps 3 to 6; step 6 readable: a5 is back
    a = {"name": "fork0/branch-0", "started": [],
         "start": {"messages": a_msgs, "marks": (2, [{"end": 1}, {"end": 3}, {"end": 5}, {"end": 7}, {"end": 9}])}}
    b = {"name": "fork0/branch-1", "started": [], "start": {"messages": [], "marks": (2, [{"end": 0}])}}
    agent._main_thread = {"name": "main", "started": [{"children": {"branch-0": a, "branch-1": b}}], "start": {}}
    agent._usage, agent._mounted = {"a4": {"prompt_tokens": 300}, "a5": {"prompt_tokens": 420}}, {"a4": (50, set())}
    fork = {"choices": [{"message": {"tool_calls": _calls(("read_thread", {}), ("fork", {}))}}]}
    read = lambda x, y, payload=None: agent._read(b, {"thread": "branch-0", "from_step": x, "to_step": y},  # noqa: E731
                                                  b["start"]["messages"], payload or {})
    assert read(4, 5).startswith("Mounted branch-0's steps 4 to 5 (350 tokens; branch-0 has readable steps 3 to 6)")
    assert b.pop("mount") == {"thread": "branch-0", "from": "a2", "to": "a4"}
    assert read(5, 5).startswith("branch-0's steps 5 to 5 cannot be mounted: a reply at their edge was rewritten")
    assert read(6, 6, fork).startswith("read_thread cannot share a reply with fork or rewind") and "mount" not in b
    assert read(6, 6).startswith("Mounted branch-0's steps 6 to 6 (120 tokens; branch-0 has readable steps 3 to 6)")
    assert b.pop("mount") == {"thread": "branch-0", "from": "a4", "to": "a5"}
    child = {"name": "fork0/branch-0/fork0/branch-0.0", "started": [], "start": {"messages": a_msgs[:3]}}
    a["started"] = [{"children": {"branch-0.0": child}}]  # forked from branch-0's step 3: branch-0 waits for it
    assert agent._read(child, {"thread": "branch-0", "from_step": 3, "to_step": 3}, a_msgs[:3], {}) == (
        "branch-0 waits for your fork to close: none of its steps can be read. Other threads: branch-1 (no readable "
        "step yet). Nothing was read.")
    assert "Other threads: branch-1 (no readable step yet)." in agent._read(
        child, {"thread": "branch-0.0", "from_step": 1, "to_step": 1}, a_msgs[:3], {})
    agent._control_running = True
    assert read(6, 6) == "read_thread is not available to the control. Nothing was read." and "mount" not in b


def test_search_refuses_a_read_of_a_thread_that_waits_for_the_readers_fork(monkeypatch):
    """Search (not isolated): branch-0.0's forking branch-0 waits for its fork, and main for branch-0's, so a read of
    either is refused as never readable and the roster leaves them out."""
    from types import SimpleNamespace
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    agent.ws = SimpleNamespace(isolated=False, tool_names=frozenset(), read_by_name=False, read_when_readable=False)
    agent._usage, agent._mounted = {}, {}
    empty = lambda name: {"name": name, "started": [], "start": {"messages": [], "marks": (1, [{"end": 0}])}}  # noqa: E731
    child, b1 = empty("fork0/branch-0/fork0/branch-0.0"), empty("fork0/branch-1")
    b0 = {**empty("fork0/branch-0"), "started": [{"children": {"branch-0.0": child}}]}
    agent._main_thread = {**empty("main"), "started": [{"children": {"branch-0": b0, "branch-1": b1}}]}
    read = lambda name: agent._read(child, {"thread": name}, [], {})  # noqa: E731
    others = "Other threads: branch-1 (no readable step yet)."
    assert read("main") == f"main waits for your fork to close: none of its steps can be read. {others} Nothing was read."
    assert read("branch-0").startswith("branch-0 waits for your fork to close") and "mount" not in child


def test_a_read_of_an_ended_thread_does_not_say_wait(monkeypatch):
    """A closed branch never replies again, so its refusal says it has ended, not that a step is readable once its next
    reply is back: one text in search and coding."""
    from types import SimpleNamespace
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    for isolated, names in ((False, frozenset()), (True, frozenset({"view"}))):
        agent.ws = SimpleNamespace(isolated=isolated, tool_names=names, read_by_name=False, read_when_readable=False)
        agent._usage, agent._mounted = {}, {}
        b0 = {"name": "fork0/branch-0", "started": [], "ended": True,
              "start": {"messages": [{"role": "user", "content": "task"}], "marks": (1, [{"end": 1}])}}
        agent._main_thread = {"name": "main", "started": [{"children": {"branch-0": b0}}], "start": {}}
        assert agent._read(agent._main_thread, {"thread": "branch-0"}, [], {}) == (
            "branch-0 has no readable step: it has ended and gets no more. Nothing was read.")
        assert "Other threads: branch-0 (no readable step)." in agent._read(agent._main_thread, {"thread": "x"}, [], {})


def test_a_rewritten_first_reply_is_not_called_the_readers_history(monkeypatch):
    """Review of #137: main reads branch-0 (forked after main's step 3), whose first own reply b4 was rewritten (no
    cursor). Step 4 is skipped as unreadable, but main never had it: the refusal names the rewrite, not main's history."""
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    reply = lambda rid: {"role": "assistant", "content": "", **({"cursor": rid} if rid else {})}  # noqa: E731
    tool = {"role": "tool", "content": "ok"}
    m_msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}, reply("r1"), tool, reply("r2"),
              tool, reply("r3"), tool]
    branch = {"name": "fork0/branch-0", "started": [], "start": {
        "messages": m_msgs + [{"role": "user", "content": "task"}, reply(None), tool, reply("b5"), tool, reply("b6")],
        "marks": (3, [{"end": 9}, {"end": 11}, {"end": 13}])}}  # steps 4-5 done, b6 back
    main = {"name": "main", "started": [{"children": {"branch-0": branch}}], "start": {}}
    agent._main_thread, agent._usage, agent._mounted = main, {}, {}
    assert agent._read(main, {"thread": "branch-0", "from_step": 4, "to_step": 4}, m_msgs, {}) == (
        "branch-0's step 4 cannot be mounted: its reply was rewritten (branch-0 has readable steps 5 to 5). Nothing was "
        "read.")
    assert agent._read(main, {"thread": "branch-0", "from_step": 5, "to_step": 5}, m_msgs, {}).startswith("Mounted ")


def test_a_nested_controls_tags_name_its_branch(monkeypatch, tmp_path):
    """The no-fork control from branch-1's fork continues branch-1: its tool results are tagged [branch-1 | ...], as
    branch-1's own were (main's control keeps main's name)."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {1: SHELL, 2: _fork_call(8, 9)}.get(turn, _commit(thread))
        return (SHELL if turn == 1 else _commit(B1)) if thread == "control" else _commit(thread)

    bodies = {}
    _run_tree(monkeypatch, tmp_path, reply, serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
              pick=lambda seed, pool: next(c for c in pool if c["path"] == "fork0/branch-1/fork0"), OCTOPUS_HTTP_WORKERS="1",
              OCTOPUS_PLAN_CONTROL="1", OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1")
    tags = lambda rid: [m["content"].split(" | ")[0] for m in bodies[rid]["messages"] if m["role"] == "tool"]  # noqa: E731
    assert tags(f"{B1}#2") == ["[branch-1"] and tags("control#2") == ["[branch-1"]


def test_the_tag_parser_reads_a_named_tag():
    from octopus.tools.kv_read_probe import TAG
    tags = ("[token budget left: 9 | step 2]\n[branch-1 | token budget left: 7 | calls left: 3 | step 4]\n"  # old runs
            "[context left: 9 | step 2]\n[branch-1 | context left: 7 | calls left: 3 | step 4]")
    assert all(TAG.match(t) for t in tags.splitlines())


def test_a_branchs_first_read_gets_a_paired_control_without_the_mount(monkeypatch, tmp_path):
    """OCTOPUS_READ_CONTROL: branch-1 reads branch-0's step 3 (its second reply) and commits two replies later. After the
    episode its read control starts from that state: its first request is the reader's mount request (the same cursor,
    the same read result but its step listing) without the mount, in branch-1's workdir put back to its tree then; it commits, its requests
    are thread readctl/fork0/branch-1, and its patch is graded on branch-1's task (plan.tsv)."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 4: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn == 1 else _commit(B1)
        return SHELL if turn < 5 else _commit(thread)

    bodies = {}
    agent, rows, tree, _, commands = _run_tree(
        monkeypatch, tmp_path, reply, serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
        OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1")
    reader = bodies[f"{B1}#3"]
    assert reader["mount"]["thread"] == "branch-0" and reader["messages"][-1]["content"].split("\n", 1)[1].startswith(
        "Mounted branch-0's steps 3 to 3")
    read = reader["messages"][-1]["content"]  # the control's copy of the read result has no step listing
    assert read.endswith("\nstep 3: shell(true) -> exit 0") and read.split("\n")[0].startswith("[branch-1 | ")
    assert bodies["control#1"] == {k: v for k, v in reader.items() if k not in ("mount", "session_id")} | {
        "messages": reader["messages"][:-1] + [dict(reader["messages"][-1], content=read.split("\nstep ", 1)[0])]}
    assert [r["thread"] for r in rows if r["thread"].startswith("readctl/")] == [f"readctl/{B1}"] * 2
    [rec] = tree["read_controls"]
    assert {k: rec[k] for k in ("of", "read_rid", "end")} == {"of": B1, "read_rid": f"{B1}#2", "end": "commit"}
    assert "cd /tmp/branches/fork0-branch-1 && git reset -q --hard head0 && git clean -fdq" in [c for c, _ in commands]
    plan = (tmp_path / "container/tmp/octopus-plan/plan.tsv").read_text().splitlines()
    assert plan[-1].split("\t") == ["readctl-fork0-branch-1", "head0", "6", "/tmp/octopus-plan/readctl-fork0-branch-1.patch"]

    # graded: branch-1's patch solves task 6, its read control's does not; plan_outcome pairs them, the group step gives
    # the read reply r_reader − r_rc = 1 − (−1) (a failing commit), and nothing else moves
    import copy

    from octopus.envs.swe.reward import plan_group_step, plan_outcome
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("1\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({"5": 1, "6": 1}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": "fork0-branch-0", "task": "5", "passed": True}, {"key": "fork0-branch-1", "task": "6", "passed": True},
        {"key": "readctl-fork0-branch-1", "task": "6", "passed": False}]}))
    out = plan_outcome(tmp_path, rows)
    assert out["read_controls"] == [{"of": "fork0-branch-1", "read_rid": f"{B1}#2", "read_call": 0,
                                     "read_calls": ["read_thread"],
                                     "forked_after": False, "control_forked": False, "ended_with": "commit",
                                     "accepted": True,  # its commit's acceptance_verified, as a branch's
                                     "passed": False}]
    episode = plan_group_step([copy.deepcopy(out)])[0]
    alone = plan_group_step([{**copy.deepcopy(out), "read_controls": []}])[0]
    assert episode["read_adv"] == {f"{B1}#2": {"calls": ["read_thread"], "call": 0, "v": 2.0}} and alone["read_adv"] == {}
    assert [b["A_branch"] for b in episode["branches"]] == [b["A_branch"] for b in alone["branches"]]
    assert (episode["A_main"], episode["R"]) == (alone["A_main"], alone["R"])


def test_a_single_task_read_control_is_valued_by_branch_reward(monkeypatch, tmp_path):
    """Not a bundle (no tasks.json): branch-1 reads branch-0's step 3 and commits a passing patch; its read control
    aborts. read_values values the pair with the single-task leaf reward (branch_reward on each leaf's own grade): the
    reader's commit+pass +1, the control's abort of a task the group solved −1, so the read reply gets 2."""
    import copy

    from octopus.envs.swe.reward import plan_group_step, plan_outcome

    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 4: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn == 1 else _call("abort", {"branch_id": "branch-1", "reason": "stuck"})
        return SHELL if turn < 5 else _commit(thread)

    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1",
                                        OCTOPUS_BUNDLE="0", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1")
    [rec] = tree["read_controls"]
    assert {k: rec[k] for k in ("of", "read_rid", "end")} == {"of": B1, "read_rid": f"{B1}#2", "end": "abort"}
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("1\n")
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": "fork0-branch-0", "passed": False}, {"key": "fork0-branch-1", "passed": True},
        {"key": "readctl-fork0-branch-1", "passed": False}]}))
    out = plan_outcome(tmp_path, rows)
    assert out["tasks"] is None and out["read_controls"][0]["ended_with"] == "abort"
    episode = plan_group_step([copy.deepcopy(out)])[0]
    alone = plan_group_step([{**copy.deepcopy(out), "read_controls": []}])[0]
    assert episode["read_adv"] == {f"{B1}#2": {"calls": ["read_thread"], "call": 0, "v": 2.0}} and alone["read_adv"] == {}
    assert [b["A_branch"] for b in episode["branches"]] == [b["A_branch"] for b in alone["branches"]]


def test_a_self_stopped_reader_and_its_control_score_alike_and_the_mounting_call_is_named(monkeypatch, tmp_path):
    """Review of #148: branch-1's read reply holds a refused read, then the one that mounts (read_call 1); branch-1 then
    stops itself unclosed (a reply with no tool call). No deadline (no outside stop): its control runs
    to its own cap (OCTOPUS_BRANCH_MAX_TURNS 6 less branch-1's 2 replies) and stops unclosed too. Both score the matrix's
    −1, so the pair's A_read is 0."""
    import copy

    from octopus.envs.swe.reward import plan_group_step, plan_outcome

    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-9", 1, 1), _read("branch-0", 3, 3)), 4: None}.get(turn, SHELL)
        if thread == "control":
            return SHELL
        return SHELL if turn < 5 else _commit(thread)

    bodies = {}
    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1",
                                        serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
                                        OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1", OCTOPUS_BRANCH_MAX_TURNS="6")
    [rec] = tree["read_controls"]
    (*same, refused, read), (*csame, crefused, cread) = bodies[f"{B1}#3"]["messages"], bodies["control#1"]["messages"]
    assert (csame, crefused) == (same, refused) and read["content"].endswith("\nstep 3: shell(true) -> exit 0")
    assert cread == dict(read, content=read["content"].split("\nstep ", 1)[0])  # the mounting result: its listing cut
    assert {k: rec[k] for k in ("of", "read_rid", "read_call", "read_calls", "deadline", "cut", "forked_after",
                                "control_forked", "end")} == {"of": B1, "read_rid": f"{B1}#2", "read_call": 1,
                                                              "read_calls": ["read_thread"] * 2, "deadline": None,
                                            "cut": False, "forked_after": False, "control_forked": False, "end": "shell"}
    assert sum(r["thread"] == f"readctl/{B1}" for r in rows) == 4
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("0.5\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({"5": 1, "6": 0}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": "fork0-branch-0", "task": "5", "passed": True}, {"key": "fork0-branch-1", "task": "6", "passed": False},
        {"key": "readctl-fork0-branch-1", "task": "6", "passed": False}]}))
    episode = plan_group_step([copy.deepcopy(plan_outcome(tmp_path, rows))])[0]
    assert episode["read_adv"] == {f"{B1}#2": {"calls": ["read_thread"] * 2, "call": 1, "v": 0.0}}


def test_a_rule_rewind_drops_the_read_of_the_reply_it_removes(monkeypatch, tmp_path):
    """Review of #152: branch-1 writes and fails twice, then its read reply fails again, so OCTOPUS_REWIND removes that
    reply: its read is not sent (no mount anywhere), no read control starts, and branch-1 goes on and commits (the stale
    mount used to reach the read control's capture and kill the branch)."""
    false = ("shell", {"command": "false"})

    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {1: _calls(("write", {"path": "/testbed/x.py", "content": "x"}), false, false),
                    2: _calls(_read("branch-0", 3, 3), false), 5: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn == 1 else _commit(B1)
        return SHELL if turn < 5 else _commit(thread)

    bodies = {}
    agent, rows, tree, _, _ = _run_tree(
        monkeypatch, tmp_path, reply, serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
        OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1", OCTOPUS_REWIND="2")
    b1 = next(b for b in tree["forks"][0]["branches"] if b["branch_id"] == "branch-1")
    assert b1["ended_with"] == "commit" and len(b1["rewinds"]) == 1
    assert not any("mount" in b for b in bodies.values()) and not tree.get("read_controls")


def test_read_deadline_per_stop_type(monkeypatch):
    """_read_controls' deadline (review of #148): the shared pool's end or, for the reader alone, its own stop by an
    empty pool, from its mount request at t = 10. Main's end stops no branch."""
    _import_agent(monkeypatch)
    d = sys.modules["octopus.runtime.tree"].read_deadline
    assert d("", None, 10, None) == (None, None)  # closed (or stopped itself: "" or its own stop), no outside stop
    assert d("no tool call", 30, 10, None) == (None, None)  # self-stopped
    for stop in ("", "request failed: token budget used up"):
        assert d(stop, 36, 10, 35) == ([25, 0], "pool")  # the pool ran out first: closed or starved alike
    assert d("request failed: token budget used up", 30, 10, None) == ([20, 0], "pool")  # its own empty-pool stop


def test_a_pool_ended_episode_gives_every_read_control_the_pools_deadline_and_the_pool_left(monkeypatch, tmp_path):
    """Review of #148 (A, C): main never says done and the runtime's shared pool (2000 tokens; search's
    OCTOPUS_SEARCH_BUDGET, here on a bundle) runs out under main and branch-0. branch-1 read and committed before that:
    its control still faces the pool's end (not no deadline), and it draws on the pool left at the read."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, SHELL)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 4: _commit(B1)}.get(turn, SHELL)
        return SHELL

    bodies = {}

    def serve(rid, body, calls):  # main and branch-0 draw 20 tokens a reply, branch-1 and the control 1
        bodies[rid] = json.loads(json.dumps(body))
        heavy = rid.startswith("main") or rid.startswith(B0)
        return {"id": rid, "usage": {"completion_tokens": 20 if heavy else 1, "prompt_tokens": 100 * int(rid.split("#")[1])},
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}
    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, budget=2000, serve=serve, OCTOPUS_HTTP_WORKERS="1",
                                        OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1")
    [rec] = tree["read_controls"]
    b1 = next(b for b in tree["forks"][0]["branches"] if b["branch_id"] == "branch-1")
    assert tree["main_end"] == "budget" and b1["ended_with"] == "commit"
    assert rec["deadline_from"] == "pool" and rec["deadline"][1] == 0
    left = int(bodies[f"{B1}#3"]["messages"][-1]["content"].split("token budget left: ")[1].split(" ")[0])
    assert rec["budget_before"] == 2000 - left  # the pool used then


def test_a_branch_reply_that_empties_the_pool_still_sets_every_controls_deadline(monkeypatch, tmp_path):
    """Re-review of #148: main waits at its fork; branch-0's shell reply empties the shared pool, so its calls are
    dropped and it stops with a raise. The pool's end is still recorded: branch-1, which read and committed before, has
    a control with the pool's deadline (not none)."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 4: _commit(B1)}.get(turn, SHELL)
        return SHELL

    def serve(rid, body, calls):  # branch-0 draws 20 tokens a reply, every other thread 1
        return {"id": rid, "usage": {"completion_tokens": 20 if rid.startswith(B0) else 1,
                                     "prompt_tokens": 100 * int(rid.split("#")[1])},
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}
    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, budget=2000, serve=serve, OCTOPUS_HTTP_WORKERS="1",
                                        OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1")
    stops = {b["branch_id"]: (b["ended_with"], b.get("stop")) for b in tree["forks"][0]["branches"]}
    assert stops["branch-1"][0] == "commit" and "token budget used up" in str(stops["branch-0"][1])
    [rec] = tree["read_controls"]
    assert agent._pool_out_t is not None and rec["deadline_from"] == "pool" and rec["deadline"][1] == 0


def test_a_reader_main_waited_for_gets_no_main_end_deadline(monkeypatch, tmp_path):
    """Re-review of #148: main's fork waits for branch-1 and main is done right after: main's end followed the reader,
    and would have followed its control. The control gets no deadline, so a control slower than its reader runs to its
    own close."""
    import itertools
    import time as time_module
    clock = itertools.count()
    monkeypatch.setattr(time_module, "time", lambda: float(next(clock)))

    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 6: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn < 8 else _commit(B1)  # slower than its reader
        return SHELL

    agent, rows, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1",
                                        OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1")
    [rec] = tree["read_controls"]
    assert (rec["deadline"], rec["deadline_from"], rec["cut"], rec["end"]) == (None, None, False, "commit")


def test_a_branch_task_names_the_threads_it_can_read(monkeypatch):
    """Informed reads (1): a nested fork's branch is told its sibling with the first line of the sibling's task (80
    characters at most), as a read_thread id, then read_thread's definition; main and its forking branch wait for the
    fork and are left out. Without OCTOPUS_READ_TOOL nothing is added."""
    from octopus.runtime.core import branch_starts, read_card, READ_TOOL
    from octopus.runtime.core import RuntimePort
    from octopus.envs.search.dry_run import stub_tokenizer
    port = RuntimePort("branch-1")
    port.fork(["fix a\nsteps", "x" * 100])
    tasks = lambda: [s["messages"][-1]["content"] for s in branch_starts(port, [])]  # noqa: E731
    assert not any("read_thread" in t for t in tasks())
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    monkeypatch.setenv("OCTOPUS_TOKENIZER", stub_tokenizer())  # Qwen3.5's tool rendering
    first, second = [t.splitlines()[-5] for t in tasks()]
    assert all(t.endswith("\n" + read_card()) for t in tasks())  # the definition, where a read can succeed
    assert read_card() == (  # as the system prompt shows it: the servers' tool shape in the chat template's tags
        'read_thread is available from here on:\n<tools>\n{"type": "function", "function": {"description": "'
        + READ_TOOL["function"]["description"] + '", "name": "read_thread", "parameters": {"type": "object", '
        '"additionalProperties": false, "properties": {"thread": {"type": "string", "description": "The other thread: a '
        'branch id such as branch-1."}, "from_step": {"type": "integer", "description": '
        '"The first step to read (omitted: from its first step)."}, "to_step": {"type": "integer", "description": "The '
        'last step to read, from_step or later (omitted: to its latest readable step)."}}, "required": ["thread"]}, '
        '"strict": false}}\n</tools>')
    assert first == f"Other threads at this fork (read_thread ids): branch-1.1 (task: {'x' * 80})."
    assert second == "Other threads at this fork (read_thread ids): branch-1.0 (task: fix a)."


def _source(monkeypatch, steps, history=()):
    """branch-0 with `steps` done steps (reply r<k>, then a tool result tagged [branch-0 | step k]) and its next reply
    back; branch-1 reads it, holding branch-0's replies `history` (a reply m in its history mounted them)."""
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    msgs, marks = [{"role": "user", "content": "task"}], [{"end": 1}]
    for k, calls in enumerate(steps, 1):
        msgs.append({"role": "assistant", "content": "", "cursor": f"r{k}", "tool_calls": _calls(*[c[:2] for c in calls])})
        msgs += [{"role": "tool", "content": f"[branch-0 | step {k}]\n{out}"} for _, _, out in calls]
        marks.append({"end": len(msgs)})
    msgs.append({"role": "assistant", "content": "", "cursor": f"r{len(steps) + 1}"})
    a = {"name": "fork0/branch-0", "started": [], "start": {"messages": msgs, "marks": (0, marks)}}
    b = {"name": "fork0/branch-1", "started": [], "start": {"messages": [], "marks": (0, [{"end": 0}])}}
    agent._main_thread = {"name": "main", "started": [{"children": {"branch-0": a, "branch-1": b}}], "start": {}}
    agent._usage, agent._mounted = {}, {"m": (0, set(history))}
    own = [{"role": "assistant", "content": "", "cursor": "m"}]
    read = lambda **args: agent._read(b, {"thread": "branch-0", **args}, own, {})  # noqa: E731
    return read, b


def test_a_read_without_steps_mounts_the_latest_sixteen(monkeypatch):
    """Informed reads (2): read_thread(thread) mounts its latest 16 readable steps (5 to 20 of 20) and sets mount and
    read_call as a range does."""
    shell = [("shell", {"command": "true"}, "exit 0")]
    read, b = _source(monkeypatch, [shell] * 20)
    assert read().startswith("Mounted branch-0's steps 5 to 20 (0 tokens; branch-0 has readable steps 1 to 20)")
    assert b.pop("mount") == {"thread": "branch-0", "from": "r5", "to": "r21"} and b.pop("read_call") == 0


def test_a_range_mounts_its_readable_part(monkeypatch):
    """Readable part: with branch-0's steps 1 to 40 mounted by the reader already and 41 to 45 readable, a range is cut to
    its readable part (1 to 50, "0" to 43, from_step or to_step alone: the other end open), and refused only when none
    of it is readable: 1 to 40 the reader holds, 46 to 50 is not back yet, 0 to 0 is no step of branch-0's;
    from_step > to_step and a non-integer step are malformed (docs/cross-branch-read.md "Readable part of a range")."""
    shell = [("shell", {"command": "true"}, "exit 0")]
    read, b = _source(monkeypatch, [shell] * 45, history=[f"r{k}" for k in range(1, 41)])
    for args, (a, z) in [({"from_step": 1, "to_step": 50}, (41, 45)), ({"from_step": "0", "to_step": 43}, (41, 43)),
                         ({"from_step": 44}, (44, 45)), ({"to_step": 42}, (41, 42))]:
        assert read(**args).startswith(f"Mounted branch-0's steps {a} to {z} (0 tokens; branch-0 has readable steps "
                                       "41 to 45)")
        assert b.pop("mount") == {"thread": "branch-0", "from": f"r{a}", "to": f"r{z + 1}"}
    assert read(from_step=1, to_step=40) == read(to_step=40) == (
        "branch-0's step 1 is in your own history already (branch-0 has readable steps 41 to 45). Nothing was read.")
    assert read(from_step=46, to_step=50) == read(from_step=0, to_step=0) == (
        "branch-0 has readable steps 41 to 45 (a step is readable once branch-0's next reply is back). Nothing was read.")
    assert read(from_step=5, to_step=4).startswith("read_thread needs thread")
    assert read(from_step="x").startswith("read_thread needs thread") and "mount" not in b


def test_a_mount_lists_its_steps(monkeypatch):
    """Informed reads (3): each mounted step is one line: tool(path for view/edit/write, command for shell, else the
    first argument; its first line) -> the first line of its result after the tag, each part cut to 60 characters, a
    step's calls joined."""
    long = "/testbed/" + "p" * 80
    read, b = _source(monkeypatch, [
        [("view", {"path": long, "view_range": [1, 9]}, "1 import os\n2 import re")],
        [("shell", {"command": "pytest -q"}, "exit 1\nFAILED"), ("edit", {"old": "a", "path": "x.py"}, "edited x.py")],
        [("join", {"branch_ids": ["branch-1"]}, "joined")],
        [("shell", {"command": "cat > t.py <<'X'\nimport os\nX"}, "exit 0")]])
    assert read(from_step=1, to_step=4).split("\n")[1:] == [
        f"step 1: view({long[:60]}) -> 1 import os",
        "step 2: shell(pytest -q) -> exit 1; edit(x.py) -> edited x.py",
        "step 3: join(['branch-1']) -> joined",
        "step 4: shell(cat > t.py <<'X') -> exit 0"]  # a multi-line argument: its first line


def test_read_thread_needs_only_the_thread(monkeypatch):
    """Informed reads (4): from_step and to_step are optional, and the description says what omitting both does."""
    from octopus.runtime.core import READ_TOOL
    fn = READ_TOOL["function"]
    assert fn["parameters"]["required"] == ["thread"]
    assert "without from_step and to_step, its latest readable steps are mounted" in fn["description"]
    assert "a branch's steps continue the numbering of the thread it forked from" in fn["description"]


def test_main_gets_read_threads_definition_with_its_first_fork_only(monkeypatch, tmp_path):
    """read_thread is in no request's tool list (the shared prefix stays fixed): main's result of its first fork (the
    join note) ends with its definition; its second fork's does not repeat it."""
    from octopus.runtime.core import read_card

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _fork_call(8, 9)}.get(turn, DONE)
        return _commit(thread)

    bodies = {}
    _, _, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_READ_TOOL="1",
                                 serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}))
    assert not any("read_thread" in [t["function"]["name"] for t in b["tools"]] for b in bodies.values())
    notes = [m["content"] for m in tree["main"] if m["role"] == "user"][1:]  # after the task
    assert len(notes) == 2 and notes[0].endswith("\n" + read_card()) and "read_thread" not in notes[1]


def test_mains_later_fork_control_gets_no_read_thread_definition(monkeypatch, tmp_path):
    """Main's no-fork control from its second fork continues a history holding the first fork's result, but cannot
    read (linear tools): that result reaches it without read_thread's definition, the rest unchanged."""
    from octopus.runtime.core import read_card

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _fork_call(8, 9)}.get(turn, DONE)
        return DONE if thread == "control" else _commit(thread)

    bodies = {}
    _, _, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_PLAN_CONTROL="1", OCTOPUS_READ_TOOL="1",
                                 serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
                                 pick=lambda seed, pool: next(c for c in pool if c["path"] == "fork1"))
    main, ctrl, card = bodies["main#2"]["messages"], bodies["control#1"]["messages"], read_card()
    assert sum(card in str(m.get("content")) for m in main) == 1 and card not in json.dumps(ctrl)
    assert [m["content"].replace("\n" + card, "") for m in main[1:]] == [m["content"] for m in ctrl[1:]]


def test_a_siblings_edit_and_commit_of_a_file_a_branch_viewed_are_mounted_once_each(monkeypatch, tmp_path):
    """OCTOPUS_READ_EVENT (Gate 0): branch-0 views f, then branch-1 edits f (its 3rd reply) and commits (its 5th). Once
    branch-1's edit step is readable (its 4th reply back), branch-0's next request alone carries that step; after
    branch-1's commit, a later request carries branch-1's readable steps. No call, no text: the episode is the same with
    the flag off, with no mount and no read_event in tree.json. branch-1 viewed nothing of branch-0's: no event."""
    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn == 1 else DONE
        if thread == B0:
            return _calls(("view", {"path": "/testbed/f"})) if turn == 1 else SHELL if turn < 9 else _commit(thread)
        return {3: _calls(("edit", {"path": "f", "old": "a", "new": "b"})), 5: _commit(thread)}.get(turn, SHELL)

    def exec_out(command):  # the container's tool results: view and edit as container_tools reports them
        if command.startswith("python3 /tmp/harbor_tools.py "):
            name = json.loads(__import__("base64").b64decode(command.split()[2]))["name"]
            return json.dumps({"observation": {"view": "1:a", "edit": "replaced 1 occurrence in f"}.get(name, "exit 0")})
        return "f\n" if command.endswith("git diff --cached --name-only HEAD") else None  # a commit's own patch files

    def arm(flag):
        bodies = {}
        out = _run_tree(monkeypatch, tmp_path / flag, reply, exec_out=exec_out, OCTOPUS_HTTP_WORKERS="1",
                        serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
                        OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_EVENT=flag)
        return bodies, out[2]

    (tmp_path / "1").mkdir(), (tmp_path / "0").mkdir()
    on, tree = arm("1")
    off, tree_off = arm("0")
    edit, *commit = tree["read_event"]
    assert all(e["reader"] == B0 and e["source"] == B1 for e in tree["read_event"])
    assert edit["mounted"] and edit["files"] == ["f"] and edit["steps"] == [edit["step"]] * 2
    assert (edit["from"], edit["to"]) == (f"{B1}#3", f"{B1}#4")
    # the commit mounts the steps B0 does not hold, one run per request: before the edit step, then through the commit
    # reply (B1 has ended); none twice
    assert [(c["step"], c["mounted"], c["from"], c["to"]) for c in commit] == [
        ("commit", True, f"{B1}#1", f"{B1}#3"), ("commit", True, f"{B1}#4", f"{B1}#5")]
    mounts = {rid: b["mount"] for rid, b in on.items() if "mount" in b}
    assert mounts == {f"{B0}#{r['turn'] + 1}": {"thread": "branch-1", "from": r["from"], "to": r["to"], **e}
                      for r, e in zip(tree["read_event"], ({}, {}, {"ended": True}))}
    assert on.keys() == off.keys() and all(on[r]["messages"] == off[r]["messages"] for r in on)
    assert not any("mount" in b for b in off.values()) and "read_event" not in tree_off


def test_events_mount_every_step_of_an_ended_sibling_once(monkeypatch, tmp_path):
    """OCTOPUS_READ_EVENT: branch-0 viewed f; branch-1 (steps 2.., replies #1..) edits f and commits, (a) both in its
    first reply, (b) edit at reply 3, commit at reply 5, (c) edit and commit at reply 3 after two shell replies. The
    edit step's event mounts that step (an ended thread's last one through its reply, delta_render.read_range); the
    commit mounts the steps branch-0 does not hold yet, one run per request. Together: every step of branch-1 once,
    no range twice."""
    edit, commit = ("edit", {"path": "f", "old": "a", "new": "b"}), ("commit", {"branch_id": "branch-1",
                                                                       "summary": "x", "acceptance_passed": True})

    def exec_out(command):
        if command.startswith("python3 /tmp/harbor_tools.py "):
            name = json.loads(__import__("base64").b64decode(command.split()[2]))["name"]
            return json.dumps({"observation": {"view": "1:a", "edit": "replaced 1 occurrence in f"}.get(name, "exit 0")})
        return "f\n" if command.endswith("git diff --cached --name-only HEAD") else None

    cases = {"a": ({1: _calls(edit, commit)}, [((2, 2), "#1", "#1")]),
             "b": ({3: _calls(edit), 5: _calls(commit)}, [((4, 4), "#3", "#4"), ((2, 3), "#1", "#3"),
                                                          ((5, 6), "#4", "#5")]),
             "c": ({3: _calls(edit, commit)}, [((4, 4), "#3", "#3"), ((2, 3), "#1", "#3")])}
    for case, (turns, want) in cases.items():
        def reply(thread, turn):
            if thread == "main":
                return _fork_call(5, 6) if turn == 1 else DONE
            if thread == B0:
                return _calls(("view", {"path": "/testbed/f"})) if turn == 1 else SHELL if turn < 9 else _commit(thread)
            return turns.get(turn, SHELL)
        (tmp_path / case).mkdir()
        tree = _run_tree(monkeypatch, tmp_path / case, reply, exec_out=exec_out, OCTOPUS_HTTP_WORKERS="1",
                         OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_EVENT="1")[2]
        got = [e for e in tree["read_event"] if e["mounted"]]
        assert [(tuple(e["steps"]), e["from"], e["to"]) for e in got] == [(s, B1 + f, B1 + t) for s, f, t in want]
        steps = [k for e in got for k in range(e["steps"][0], e["steps"][1] + 1)]
        assert sorted(steps) == list(range(2, max(steps) + 1)), case  # every step once: union, no overlap
        assert all(e["why"] == "no mountable step" for e in tree["read_event"] if not e["mounted"])


def test_read_thread_mounts_only_steps_it_does_not_hold(monkeypatch):
    """read_thread with steps 3 and 6 held (mounted before): a range is cut before its first held step, a held start
    moves on to the next step not held, the latest-steps read takes the first run not held, and only a range wholly
    held says "in your own history already"; nothing is mounted twice."""
    shell = [("shell", {"command": "true"}, "exit 0")]
    read, b = _source(monkeypatch, [shell] * 8, history=["r3", "r6"])
    for args, (a, z) in [({"from_step": 1, "to_step": 5}, (1, 2)), ({"from_step": 3, "to_step": 8}, (4, 5)),
                         ({}, (1, 2)), ({"from_step": 7}, (7, 8))]:
        assert read(**args).startswith(f"Mounted branch-0's steps {a} to {z} ("), args
        assert b.pop("mount") == {"thread": "branch-0", "from": f"r{a}", "to": f"r{z + 1}"}
    for k in (3, 6):
        assert read(from_step=k, to_step=k) == (f"branch-0's step {k} is in your own history already (branch-0 has "
                                                "readable steps 1 to 8). Nothing was read.")


def test_a_nested_commit_event_names_its_own_patchs_files_only(monkeypatch, tmp_path):
    """OCTOPUS_READ_EVENT, depth 2: branch-1 forks branch-1.0 and branch-1.1; branch-1.0 views f (changed by branch-1
    before the fork, so in branch-1.1's graded patch text) and g; branch-1.1 commits a patch of its own with g alone
    (git diff --cached --name-only HEAD in its copy). The commit event for branch-1.0 names g, not f."""
    N0, N1 = f"{B1}/fork0/branch-1.0", f"{B1}/fork0/branch-1.1"

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn == 1 else DONE
        if thread == B1:
            return _fork_call(7, 8) if turn == 1 else _commit(thread)
        if thread == N0:
            return _calls(("view", {"path": "f"}), ("view", {"path": "g"})) if turn == 1 else \
                SHELL if turn < 9 else _commit(thread)
        return SHELL if turn < 4 else _commit(thread)

    def exec_out(command):
        if command.startswith("python3 /tmp/harbor_tools.py "):
            return json.dumps({"observation": "1:a"})
        return "g\n" if command.endswith("git diff --cached --name-only HEAD") else None

    _, _, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, exec_out=exec_out, OCTOPUS_HTTP_WORKERS="1",
                                 OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_EVENT="1")
    assert [(e["reader"], e["source"], e["step"], e["files"], e["mounted"]) for e in tree["read_event"]] == [
        (N0, N1, "commit", ["g"], True)]


def test_auto_fork_starts_a_branch_per_task_from_the_initial_request_and_main_from_the_join(monkeypatch, tmp_path):
    """OCTOPUS_AUTO_FORK (Gate 0): before any model call the environment forks one branch per bundle task; each
    branch's first request is the initial one (system, task) plus its task message, sent whole (no cursor: no reply
    precedes it), and main's first request, after every branch closed, is the initial one plus the join note (no
    forking reply in its history). Arm 3 (read tool + events) sends the same system prompt, task and tools; its
    branch task message adds the read roster and card; a sibling's edit of a file branch-0 viewed is mounted."""
    from octopus.envs.swe.prompt import _AUTO_FORK

    def reply(thread, turn):
        if thread == "main":
            return DONE
        if thread == B0:
            return _calls(("view", {"path": "/testbed/f"})) if turn == 1 else SHELL if turn < 9 else _commit(thread)
        return {3: _calls(("edit", {"path": "f", "old": "a", "new": "b"})), 5: _commit(thread)}.get(turn, SHELL)

    def exec_out(command):
        if command.startswith("python3 /tmp/harbor_tools.py "):
            name = json.loads(__import__("base64").b64decode(command.split()[2]))["name"]
            return json.dumps({"observation": {"view": "1:a", "edit": "replaced 1 occurrence in f"}.get(name, "exit 0")})
        if "/tasks/" in command:
            return json.dumps({str(k): f"Statement of task {k}.\n" for k in (1, 2, 3)})
        return "f\n" if command.endswith("git diff --cached --name-only HEAD") else None

    def arm(name, **env):
        bodies, order = {}, []
        (tmp_path / name).mkdir()
        out = _run_tree(monkeypatch, tmp_path / name, reply, exec_out=exec_out, OCTOPUS_HTTP_WORKERS="1",
                        serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}) or order.append(rid),
                        OCTOPUS_AUTO_FORK="1", OCTOPUS_CURSOR="1", **env)
        return bodies, order, out[2]

    off, order, tree_off = arm("off")
    on, _, tree = arm("on", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_EVENT="1")
    assert order[-1] == "main#1" and len({r.split("#")[0] for r in order}) == 4  # main, then 3 branches before it
    for bodies, t in ((off, tree_off), (on, tree)):
        main = bodies["main#1"]["messages"]
        assert [m["role"] for m in main] == ["system", "user", "user"] and main[2]["content"].startswith(
            "branches finished:") and _AUTO_FORK in main[0]["content"] and "cursor" not in bodies["main#1"]
        for k in (1, 2, 3):
            first = bodies[f"fork0/branch-{k - 1}#1"]
            assert "cursor" not in first and first["messages"][:2] == main[:2] and len(first["messages"]) == 3
            assert f"Your task: Fix task {k}.\nNo acceptance check was given." in first["messages"][2]["content"]
            assert f"Statement of task {k}." in first["messages"][2]["content"]
        assert t["forks"][0]["fork_index"] is None and [b["task_id"] for b in t["forks"][0]["branches"]] == ["1", "2", "3"]
    for r in off:
        if r.endswith("#1") and r != "main#1":
            a, b = off[r]["messages"], on[r]["messages"]
            assert a[:2] == b[:2] and b[2]["content"].startswith(a[2]["content"]) and off[r]["tools"] == on[r]["tools"]
    assert any("mount" in b for b in on.values()) and not any("mount" in b for b in off.values())
    assert tree["read_event"] and "read_event" not in tree_off


def test_a_rewind_to_the_read_step_mounts_the_read_again(monkeypatch, tmp_path):
    """main reads branch-0 (step 3, after its fork joined); the next request carries the mount and its reply rewinds to
    step 3. That reply goes, and with it what the mount gave: the read's result ("from your next reply on you attend to
    them") stays in main's history, so the next request carries the same mount again."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6), 3: _calls(_read("branch-0", 3, 4)),
                    4: _call("rewind", {"to_step": 3, "note": "n"})}.get(turn, DONE)
        return SHELL if turn < 3 else _commit(thread)

    bodies = {}
    _run_tree(monkeypatch, tmp_path, reply, serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
              OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_REWIND_TOOL="1")
    assert bodies["main#4"]["mount"]["thread"] == "branch-0"
    assert bodies["main#5"]["mount"] == bodies["main#4"]["mount"]


def test_main_gets_read_threads_definition_again_after_rewinding_past_it(monkeypatch, tmp_path):
    """main forks (step 1: its join note ends with read_thread's definition), rewinds to step 0, before that fork, and
    forks again: that fork's join note carries the definition, its one copy in main's history."""
    from octopus.runtime.core import read_card

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _call("rewind", {"to_step": 0, "note": "n"}), 3: _fork_call(8, 9)}.get(
                turn, DONE)
        return _commit(thread)

    _, _, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_READ_TOOL="1", OCTOPUS_REWIND_TOOL="1")
    assert sum(read_card() in str(m.get("content")) for m in tree["main"]) == 1


@pytest.mark.parametrize("to", [3, 4])
def test_a_reader_that_rewinds_to_or_past_its_read_gets_no_read_control(monkeypatch, tmp_path, to):
    """OCTOPUS_READ_CONTROL: branch-1 reads branch-0 (its step 4) and in the reply its mount request carried rewinds to
    the step before the read (3) or the read's own (4). Its history from the mount request on is no longer the one
    its read control would be paired with: no read control. Rewound to the read's own step, the next request carries
    the mount again."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 3: _call("rewind", {"to_step": to, "note": "n"}),
                    5: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn == 1 else _commit(B1)
        return SHELL if turn < 5 else _commit(thread)

    bodies = {}
    _, _, tree, _, _ = _run_tree(
        monkeypatch, tmp_path, reply, serve=lambda rid, body, calls: bodies.update({rid: json.loads(json.dumps(body))}),
        OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1", OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1",
        OCTOPUS_REWIND_TOOL="1")
    assert bodies[f"{B1}#3"]["mount"]["thread"] == "branch-0" and not tree.get("read_controls")
    assert bodies[f"{B1}#4"].get("mount") == (bodies[f"{B1}#3"]["mount"] if to == 4 else None)


def test_a_fork_after_the_read_that_a_rewind_removed_still_counts(monkeypatch, tmp_path):
    """OCTOPUS_READ_CONTROL: branch-1 reads (step 4), shells (5), forks (6) and rewinds to step 5, which removes its
    fork from its history: its pair still records that it forked after its read (forked_after)."""
    def reply(thread, turn):
        if thread == "main":
            return {1: SHELL, 2: _fork_call(5, 6)}.get(turn, DONE)
        if thread.count("/") > 1:  # branch-1's own branches
            return _commit(thread)
        if thread == B1:
            return {2: _calls(_read("branch-0", 3, 3)), 4: _fork_call(7, 8),
                    5: _call("rewind", {"to_step": 5, "note": "n"}), 6: _commit(B1)}.get(turn, SHELL)
        if thread == "control":
            return SHELL if turn == 1 else _commit(B1)
        return SHELL if turn < 5 else _commit(thread)

    _, _, tree, _, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_HTTP_WORKERS="1", OCTOPUS_CURSOR="1",
                                 OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1", OCTOPUS_REWIND_TOOL="1")
    [rec] = tree["read_controls"]
    assert rec["of"] == B1 and rec["forked_after"] is True


def test_a_rewind_drops_the_events_of_the_steps_it_removes(monkeypatch):
    """OCTOPUS_READ_EVENT: a sibling's queued events of the rewound thread's steps past the one it rewinds to are
    dropped (that step number names another reply from then on); its earlier steps' events and its commit's stay."""
    agent = _import_agent(monkeypatch).GrokHarborAgent.__new__(_import_agent(monkeypatch).GrokHarborAgent)
    agent._event = []
    src = {"name": B1, "started": [], "start": {"marks": (2, [{"end": 5}, {"end": 8}, {"end": 11}])}}
    events = [{"source": src, "step": 3, "files": ["f"]}, {"source": src, "step": 4, "files": ["f"]},
              {"source": src, "step": "commit", "files": ["f"]}]
    sibling = {"name": B0, "events": list(events)}
    src["parent"] = {"started": [{"children": {"branch-0": sibling, "branch-1": src}}]}
    assert agent._rewound(src, 8) == ""  # to step 3: its history then ends at 8
    assert sibling["events"] == [events[0], events[2]]
