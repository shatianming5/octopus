"""Octopus runtime tools are host-side, and the default port does not touch KV."""

import asyncio
import json

import pytest

from octopus.envs.swe.prompt import TOOLS, offered_tools
from octopus.runtime.core import (
    apply_runtime, branch_starts, decode_tool_call, run_branch_continuation, RUNTIME_NAMES, RuntimePort,
)
from test_rl_plan import _import_agent


def _harbor(monkeypatch):
    """The SWE environment's module (octopus/envs/swe/agent.py, Harbor stubbed) and its workspace, no container."""
    mod = _import_agent(monkeypatch)
    return mod, mod.HarborWorkspace()


async def _branches(frozen, port, complete, dispatch, max_steps, workdirs=None):
    """Every branch cloned from the port's last fork, run concurrently until it closes; reports in branch order."""
    starts = branch_starts(port, frozen, workdirs)
    for start in starts:  # the runtime names a branch's session after its thread; here after the branch
        start["session_id"] = start["branch_id"]
    return list(await asyncio.gather(*(run_branch_continuation(start, complete, dispatch, max_steps) for start in starts)))


def test_runtime_catalog_is_fork_commit_abort():
    assert RUNTIME_NAMES == ("fork", "commit", "abort")
    coding = {item["function"]["name"] for item in TOOLS}
    assert set(RUNTIME_NAMES).isdisjoint(coding)


def test_apply_runtime_records_without_claiming_kv():
    port = RuntimePort()
    seen, stop = apply_runtime("fork", {"tasks": ["read logs", "read code"]}, port)
    assert stop is False and seen == "fork started: branches branch-0, branch-1"
    seen, _ = apply_runtime("commit", {"branch_id": "branch-0", "summary": "bug is in auth"}, port)
    assert seen == "commit recorded: branch-0"
    seen, _ = apply_runtime("abort", {"branch_id": "branch-1"}, port)
    assert seen == "abort recorded: branch-1"
    assert [name for name, _ in port.calls] == ["fork", "commit", "abort"]


def test_offered_tools_are_working_then_runtime_tools():
    names = [item["function"]["name"] for item in offered_tools()]
    assert names[:8] == [item["function"]["name"] for item in TOOLS]
    assert names[8:] == ["fork", "commit", "abort"]


def test_apply_runtime_rejects_empty_args():
    seen, stop = apply_runtime("fork", {"tasks": []}, RuntimePort())
    assert "at least two tasks" in seen and stop is False
    port = RuntimePort()
    seen, _ = apply_runtime("fork", {"tasks": ["only one"]}, port)
    assert "at least two tasks" in seen
    assert port.last_fork is None
    seen, _ = apply_runtime("commit", {"branch_id": "", "summary": "x"}, RuntimePort())
    assert "needs branch_id" in seen


def test_cloned_branches_runs_two_branch_loops():
    import asyncio

    port = RuntimePort()
    trunk = [{"role": "user", "content": "schedule Job __str__ and __repr__"}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)
    seen_complete = []

    def complete(messages, session_id=None):
        seen_complete.append({"session_id": session_id, "prefix": messages[0], "task": messages[-1]})
        branch = session_id  # branch_starts names each session after its branch (the runtime: after its thread)
        n = sum(1 for item in seen_complete if item["session_id"] == session_id)
        if n == 1:
            return {"choices": [{"message": {
                "tool_calls": [{"id": "v", "function": {
                    "name": "view", "arguments": json.dumps({"path": "."}),
                }}],
            }}]}
        return {"choices": [{"message": {
            "tool_calls": [{"id": "c", "function": {
                "name": "commit",
                "arguments": json.dumps({"branch_id": branch, "summary": "ok"}),
            }}],
        }}]}

    async def dispatch(name, args, branch_id=None):
        return f"{name} on {args}", name in {"commit", "abort", "done"}

    reports = asyncio.run(_branches(trunk, port, complete, dispatch, max_steps=4))
    assert [item["branch_id"] for item in reports] == ["branch-0", "branch-1"]
    assert reports[0]["session_id"] != reports[1]["session_id"]
    from octopus.runtime.core import branch_task_message

    for report, task in zip(reports, [branch_task_message("branch-0", "fix __str__", 2), branch_task_message("branch-1", "fix __repr__", 2)]):
        assert report["messages"][0] == trunk[0]
        assert report["messages"][1]["content"] == task
        assert [step["name"] for step in report["steps"]] == ["view", "commit"]
        assert all(step["session_id"] == report["session_id"] for step in report["steps"])
    assert {item["session_id"] for item in seen_complete} == {
        reports[0]["session_id"], reports[1]["session_id"],
    }


def test_decode_tool_call_reads_openai_arguments():
    name, args, call_id = decode_tool_call({
        "id": "c1",
        "function": {"name": "view", "arguments": json.dumps({"path": "furl/furl.py"})},
    })
    assert (name, args, call_id) == ("view", {"path": "furl/furl.py"}, "c1")


def test_cloned_branches_runs_every_tool_call_and_keeps_arguments():
    import asyncio

    port = RuntimePort()
    trunk = [{"role": "user", "content": "Job __str__ and __repr__"}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)
    seen = {}

    def complete(messages, session_id=None, session_params=None):
        n = seen.get(session_id, 0) + 1
        seen[session_id] = n
        if n == 1:
            return {"choices": [{"message": {
                "tool_calls": [
                    {"id": "v1", "function": {
                        "name": "view",
                        "arguments": json.dumps({"path": "schedule/job.py"}),
                    }},
                    {"id": "v2", "function": {
                        "name": "view",
                        "arguments": json.dumps({"path": "schedule/__init__.py"}),
                    }},
                ],
            }}]}
        return {"choices": [{"message": {
            "tool_calls": [{"id": "c", "function": {
                "name": "commit",
                "arguments": json.dumps({
                    "branch_id": "branch-0", "summary": "both files",
                }),
            }}],
        }}]}

    async def dispatch(name, args, branch_id=None):
        return f"{name}:{args}", name in {"commit", "abort", "done"}

    reports = asyncio.run(_branches(trunk, port, complete, dispatch, max_steps=4))
    steps = reports[0]["steps"]
    assert [item["name"] for item in steps] == ["view", "view", "commit"]
    assert steps[0]["arguments"] == {"path": "schedule/job.py"}
    assert steps[1]["arguments"] == {"path": "schedule/__init__.py"}
    assert steps[2]["arguments"] == {"branch_id": "branch-0", "summary": "both files"}
    assert [item["tool_call_id"] for item in steps] == ["v1", "v2", "c"]


def test_cloned_branches_runs_branches_concurrently():
    import asyncio

    port = RuntimePort()
    trunk = [{"role": "user", "content": "Job __str__ and __repr__"}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)

    async def main():
        both_in_flight = asyncio.Barrier(2)

        async def complete(messages, session_id=None, session_params=None):
            # Deadlocks (then times out) unless both branches are waiting on the model at once.
            await asyncio.wait_for(both_in_flight.wait(), timeout=2)
            return {"choices": [{"message": {"tool_calls": [{"id": "c", "function": {
                "name": "commit",
                "arguments": json.dumps({"branch_id": session_id, "summary": "ok"}),
            }}]}}]}

        async def dispatch(name, args, branch_id=None):
            return f"{name} ok", name == "commit"

        return await _branches(trunk, port, complete, dispatch, max_steps=2)

    reports = asyncio.run(main())
    assert [item["branch_id"] for item in reports] == ["branch-0", "branch-1"]
    assert all(item["steps"][-1]["name"] == "commit" for item in reports)


def test_cloned_branches_survives_one_branch_request_failure():
    import asyncio

    port = RuntimePort()
    trunk = [{"role": "user", "content": "Job __str__ and __repr__"}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)
    first = port.last_fork["branches"][0]["branch_id"]

    async def complete(messages, session_id=None, session_params=None):
        if session_id == first:
            raise RuntimeError("TimeoutError: timed out")
        return {"choices": [{"message": {"content": "no tools"}}]}

    async def dispatch(name, args, branch_id=None):
        return "ok", False

    reports = asyncio.run(_branches(trunk, port, complete, dispatch, max_steps=2))
    assert "request failed" in reports[0]["steps"][0]["text"]
    assert reports[1]["steps"][0]["text"] == "no tools"


def test_fork_record_keeps_prefix_fork_and_each_branch_turns():
    import asyncio

    from octopus.runtime.core import fork_record

    port = RuntimePort()
    trunk = [{"role": "system", "content": "s"}, {"role": "user", "content": "Job __str__ and __repr__"}]
    main = trunk + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "f", "function": {"name": "fork", "arguments": "{}"}}]}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)

    def complete(messages, session_id=None, session_params=None):
        return {"choices": [{"message": {"tool_calls": [{"id": "c", "function": {
            "name": "commit", "arguments": json.dumps({"branch_id": session_id, "summary": "ok"})}}]}}]}

    async def dispatch(name, args, branch_id=None):
        return "committed", True

    # every branch starts from the forking thread's whole history at the fork, its forking reply included
    reports = asyncio.run(_branches(main, port, complete, dispatch, max_steps=2))
    assert all(r["messages"][:len(main)] == main for r in reports)
    record = fork_record(main, reports)
    assert record["prefix_len"] == len(main)
    assert record["fork_index"] == 2 and record["join_index"] == 3
    first = record["branches"][0]
    assert first["messages"][0]["role"] == "user" and "Your task: fix __str__" in first["messages"][0]["content"]
    assert first["messages"][1]["tool_calls"][0]["function"]["name"] == "commit"
    assert first["messages"][2]["role"] == "tool" and first["ended_with"] == "commit"


def test_branch_is_told_its_id_and_how_to_close():
    from octopus.runtime.core import branch_task_message

    text = branch_task_message("branch-1", "reproduce the bug", 2)
    assert "You are branch-1" in text and "Your task: reproduce the bug" in text
    assert 'commit with branch_id="branch-1"' in text and 'abort with branch_id="branch-1"' in text


def test_join_note_carries_each_branch_outcome():
    from octopus.envs.swe.prompt import join_note

    reports = [
        {"branch_id": "branch-0", "task": "read source", "steps": [
            {"name": "view", "arguments": {}, "observed": "..."},
            {"name": "commit", "arguments": {"branch_id": "branch-0", "summary": "bug is in foo.py:12"}}]},
        {"branch_id": "branch-1", "task": "reproduce", "steps": [
            {"name": "abort", "arguments": {"branch_id": "branch-1"}}]},
        {"branch_id": "branch-2", "task": "tests", "steps": [
            {"name": "view", "arguments": {}, "observed": "tests/test_foo.py"}]},
        {"branch_id": "branch-3", "task": "search", "steps": [
            {"name": "done", "arguments": {"answer": "Ada Lovelace"}}]},
        {"branch_id": "branch-4", "task": "verify", "steps": [
            {"name": "search", "arguments": {}, "observed": "5 docs"},
            {"name": None, "text": "It is Ada Lovelace (doc 12).", "arguments": {}}]},
    ]
    note = join_note(reports)
    assert note.startswith("branches finished:")
    assert "- branch-0 (read source): committed: bug is in foo.py:12" in note
    assert "- branch-1 (reproduce): abort" in note
    assert "- branch-3 (search): done: Ada Lovelace" in note
    # an unclosed branch hands nothing: neither its last result nor the reply it ended on without a tool call
    assert "- branch-2 (tests): stopped after 1 steps without commit" in note.splitlines()
    assert "- branch-4 (verify): stopped after 2 steps without commit" in note.splitlines()
    assert "test_foo" not in note and "doc 12" not in note
    from octopus.runtime import core
    assert core.join_note(reports[4:], close="done") == (
        "branches finished:\n- branch-4 (verify): stopped after 2 steps without done")


def test_branches_get_their_workdir_and_dispatch_knows_the_branch():
    import asyncio

    port = RuntimePort()
    trunk = [{"role": "user", "content": "Job __str__ and __repr__"}]
    apply_runtime("fork", {"tasks": ["fix __str__", "fix __repr__"]}, port)
    workdirs = {"branch-0": "/tmp/branches/fork0-branch-0", "branch-1": "/tmp/branches/fork0-branch-1"}
    seen = []

    async def complete(messages, session_id=None, session_params=None):
        return {"choices": [{"message": {"tool_calls": [{"id": "c", "function": {
            "name": "commit", "arguments": json.dumps({"branch_id": session_id, "summary": "ok"})}}]}}]}

    async def dispatch(name, args, branch_id=None):
        seen.append(branch_id)
        return "ok", True

    reports = asyncio.run(_branches(trunk, port, complete, dispatch, max_steps=2, workdirs=workdirs))
    assert sorted(seen) == ["branch-0", "branch-1"]
    assert "private copy of the repo at /tmp/branches/fork0-branch-0" in reports[0]["messages"][1]["content"]


def test_join_note_points_at_each_branch_patch():
    from octopus.envs.swe.prompt import join_note

    reports = [
        {"branch_id": "branch-0", "task": "fix", "steps": [{"name": "commit", "arguments": {"summary": "fixed foo"}}]},
        {"branch_id": "branch-1", "task": "read", "steps": [{"name": "commit", "arguments": {"summary": "nothing"}}]},
    ]
    note = join_note(reports, {
        "branch-0": ("/tmp/branches/fork0-branch-0.patch", "1 file changed, 2 insertions(+)"),
        "branch-1": ("/tmp/branches/fork0-branch-1.patch", ""),
    })
    assert "changes saved to /tmp/branches/fork0-branch-0.patch (1 file changed, 2 insertions(+))" in note
    assert "git apply --3way /tmp/branches/fork0-branch-0.patch" in note
    assert "branch-1 (read): committed: nothing\n  no file changes" in note


def test_agent_and_udocker_keep_event_loop_free():
    from pathlib import Path

    agent = Path("octopus/runtime/tree.py").read_text()
    assert "await self._complete_async(history, calls_left=" in agent
    assert "_HTTP_POOL, self._complete," in agent
    assert "asyncio.to_thread(self.ws.runtime," in agent
    assert "async with self._dispatch_lock:" in agent
    env = Path("octopus/envs/swe/udocker.py").read_text()
    assert "await _off_loop(self._start_sync)" in env
    assert "asyncio.to_thread" not in env
    assert "_off_loop(self._exec_sync," in env
    for name in ("_upload_file", "_upload_dir", "_download_file", "_download_dir"):
        assert f"_off_loop(self.{name}," in env


def test_udocker_start_clones_golden_and_cleans_up_cancelled_starts():
    from pathlib import Path

    env = Path("octopus/envs/swe/udocker.py").read_text()
    # One extraction per image, trials copy it; creates/copies are bounded.
    assert "def _golden(self, image: str) -> str:" in env
    assert "cloned = self._clone(golden)" in env
    assert "UDOCKER_GOLDEN_DIR" in env and "UDOCKER_CREATE_SLOTS" in env
    assert "UDOCKER_GOLDEN_STORE" not in env and "/mnt/HDD3" not in env
    assert "environment_dir" not in env  # never write into the dataset's environment dir
    assert "--nosysdirs" not in env  # tests need host /dev/shm and /dev/pts
    assert "with _CREATE_SLOTS:" in env
    assert "cid = lines[0] if lines else \"\"" in env
    # stop() during a start kills the tracked copy and the start thread removes what it made.
    assert "self._stopping.set()" in env and "_kill_group(proc)" in env
    assert "def _check_cancel(self) -> None:" in env
    assert "if self._cid and (delete or not self._started):" in env
    # Log download after a failed start does not call udocker inspect with an empty id.
    assert env.count("if not self._cid:\n            return") == 2
    # Commands die with Harbor; they signal only their own processes (an agent's pkill once killed the model servers).
    assert "pass_fds=(fd,)" in env and "read -u 9 _; kill -9 $c;" in env
    assert '(root / "etc/ld.so.preload").write_text(_NOKILL' in env
    assert '"-q", "run", sys.argv[2], "/bin/bash", "-lc", "__CMD__"' in env  # _capture_proot


def test_agent_run_runs_branch_continuations():
    from pathlib import Path

    text = Path("octopus/runtime/tree.py").read_text()
    assert "run_branch_continuation" in text
    assert 'item.get("arguments")' in text
    assert '{"branch_id": report["branch_id"]}' not in text
    assert "starts = branch_starts(port, messages," in text  # a branch starts from the history at its fork
    assert "self._write_tree(messages, forks)" in text
    # branches work in private copies of /testbed and hand back patches
    assert "await self.ws.open_branches(root, workdirs, info)" in text
    assert 'root=thread["root"]' in text
    assert 'note = self.ws.join_note(reports, patches, merged, not thread["depth"])' in text  # + OCTOPUS_AUTO_MERGE results
    assert '"patch_text": text' in text  # each branch patch is logged for per-branch verification
    assert 'body["chat_template_kwargs"] = json.loads(template_kwargs)' in text


def test_refused_commit_does_not_close_the_branch():
    import asyncio
    from octopus.runtime.core import closing_name, run_branch_continuation

    replies = iter([
        {"name": "commit", "arguments": {"branch_id": "branch-0"}},                    # refused: no summary
        {"name": "commit", "arguments": {"branch_id": "branch-0", "summary": "fixed"}},  # accepted
    ])

    def complete(messages, session_id=None, session_params=None):
        call = next(replies)
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}}]}

    def dispatch(name, args, branch_id):
        from octopus.runtime.core import apply_runtime
        return apply_runtime(name, args)

    start = {"branch_id": "branch-0", "session_id": "s", "task": "t", "messages": [{"role": "user", "content": "go"}]}
    report = asyncio.run(run_branch_continuation(start, complete, dispatch, 5))
    assert [s.get("rejected", False) for s in report["steps"]] == [True, False]
    assert closing_name(report["steps"]) == "commit"
    assert closing_name(report["steps"][:1]) is None  # a refused commit alone is not a close


def test_calls_after_a_closing_call_in_its_reply_do_not_run():
    """One reply calls commit then shell (225__0c642ff fork0/branch-0): the branch stops at its commit, the shell never
    runs, the branch ended with commit, and its history keeps only the calls that ran."""
    import asyncio
    from octopus.runtime.core import apply_runtime, closing_name, run_branch_continuation

    def call(name, args):
        return {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def complete(messages, session_id=None):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            call("commit", {"branch_id": "branch-0", "summary": "fixed"}), call("shell", {"command": "ls"})]}}]}

    ran = []

    def dispatch(name, args, branch_id):
        ran.append(name)
        return apply_runtime(name, args) if name == "commit" else ("exit 0\n", False)

    start = {"branch_id": "branch-0", "session_id": "s", "task": "t", "messages": [{"role": "user", "content": "go"}]}
    report = asyncio.run(run_branch_continuation(start, complete, dispatch, 5))
    assert ran == ["commit"] and closing_name(report["steps"]) == "commit"
    assert [c["id"] for c in start["messages"][1]["tool_calls"]] == ["commit"] and len(start["messages"]) == 3


def test_only_an_accepted_fork_starts_branches():
    from pathlib import Path

    text = Path("octopus/runtime/tree.py").read_text()
    assert "before = port.last_fork" in text
    assert "port.last_fork is not before" in text


def _replying(calls, prompts):
    """A fake model: one tool call per turn, with the given prompt_tokens usage."""
    replies = iter(zip(calls, prompts))

    def complete(messages, session_id=None, session_params=None):
        (name, args), prompt = next(replies)
        return {"usage": {"prompt_tokens": prompt}, "choices": [{"message": {"content": "", "tool_calls": [
            {"id": f"c{prompt}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}
    return complete


def test_branch_stops_once_it_grows_past_its_token_budget():
    import asyncio
    from octopus.runtime.core import closing_name, run_branch_continuation

    calls = [("shell", {"command": "ls"})] * 3 + [("commit", {"branch_id": "branch-0", "summary": "s"})]
    start = {"branch_id": "branch-0", "session_id": "s", "task": "t", "messages": [{"role": "user", "content": "go"}]}
    report = asyncio.run(run_branch_continuation(
        start, _replying(calls, [1000, 1400, 1800, 2200]), lambda n, a, b: ("exit 0\n", False), 10, budget=1000))
    assert [s["name"] for s in report["steps"]] == ["shell", "shell", "shell", "context"]  # 2200 - 1000 > 1000: commit dropped
    assert closing_name(report["steps"]) == "context"


def test_a_thread_resends_each_reply_as_areal_recorded_it():
    """bundle-e1c pyopenssl b02-3:1-0, fork0/branch-5: a reply came back with reasoning_content (SGLang's
    qwen3 parser on a stray </think>); the next request re-sent it without, AReaL's prefix match (messages + its record
    of the reply) fell back to the request before, and the thread split into two exported sequences."""
    import asyncio
    import copy
    from octopus.runtime.core import run_branch_continuation

    replies = iter([
        ("shell", {"command": "ls -la /testbed"}, None),
        ("view", {"path": "crypto.py"}, "让我开始逐个回答任务：\n\n"),
        ("shell", {"command": "sed -n '1372,1405p' crypto.py"}, None),
        ("commit", {"branch_id": "branch-5", "summary": "s"}, None),
    ])
    sent = []  # per request: (messages, AReaL's record of the reply = ChatCompletionMessage.model_dump(exclude_none))

    def complete(messages, session_id=None, session_params=None):
        name, args, reasoning = next(replies)
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{len(sent)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        if reasoning:
            message["reasoning_content"] = reasoning
        sent.append((copy.deepcopy(messages), copy.deepcopy(message)))
        return {"choices": [{"message": message}]}

    start = {"branch_id": "branch-5", "session_id": "fork0/branch-5", "task": "t",
             "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "You are branch-5"}]}
    asyncio.run(run_branch_continuation(start, complete, lambda n, a, b: ("exit 0\n", False), 10))
    assert len(sent) == 4
    for (before, reply), (after, _) in zip(sent, sent[1:]):
        assert after[:len(before) + 1] == before + [reply]  # AReaL's default_prefix_matcher


def test_commit_runs_the_acceptance_check_and_the_join_sees_the_real_result(monkeypatch):
    import asyncio
    from octopus.envs.swe.prompt import ACCEPTANCE_PREFIX, join_note
    from octopus.runtime.core import run_branch_continuation

    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    ran = []

    def dispatch(name, args, branch_id):
        ran.append((name, args.get("command")))
        return ("exit 1\nAssertionError", False) if name == "shell" else ("committed", False)

    start = {"branch_id": "branch-0", "session_id": "s", "task": f"fix x\n{ACCEPTANCE_PREFIX}python check.py",
             "messages": [{"role": "user", "content": "go"}]}
    commit = ("commit", {"branch_id": "branch-0", "summary": "fixed x", "acceptance_passed": True})
    # the first commit the check fails is refused with its output; a second one closes the branch as it is
    report = asyncio.run(run_branch_continuation(start, _replying([commit, commit], [100, 110]), dispatch, 5,
                                                 ws=_harbor(monkeypatch)[1]))
    assert ran == [("shell", "python check.py"), ("shell", "python check.py"), ("commit", None)]
    gated, closed = report["steps"]
    assert gated["rejected"] and gated["gated"] and gated["observed"].startswith(
        "commit not accepted yet: the acceptance check `python check.py` does not exit 0 on your tree:\nexit 1\nAssertionError")
    assert closed["acceptance_verified"] is False and closed["after_gate"] and not closed.get("rejected")
    assert "acceptance FAILED, checked by the runtime; committed again after the runtime refused it" in join_note([report])


def test_a_branch_with_an_unreliable_check_commits_its_own_verified_both_ways(monkeypatch, tmp_path):
    """A weak task's branch passes its own check to commit: the runtime runs it on the branch's tree (it must exit 0)
    and on a scratch copy of its fork point, the HEAD commit of its private copy (it must fail there)."""
    import asyncio, subprocess
    from octopus.envs.swe.prompt import join_note, weak_acceptance
    from octopus.runtime.core import run_branch_continuation
    mod, ws = _harbor(monkeypatch)
    fork_point_command, weak_why = mod.fork_point_command, mod.weak_why

    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    repo = tmp_path / "copy"
    repo.mkdir()
    subprocess.run("git init -q && echo bug > f && git add -A && git -c user.name=t -c user.email=t@t commit -qm fork-point "
                   "&& echo fixed > f", shell=True, cwd=repo, check=True)

    def dispatch(name, args, branch_id):  # the branch's shell: /testbed is its private copy
        if name != "shell":
            return "committed", False
        r = subprocess.run(["bash", "-c", args["command"].replace("/testbed", str(repo))], capture_output=True, text=True)
        return f"exit {r.returncode}\n{r.stdout}{r.stderr}", False

    task = weak_acceptance("Task 3: fix f\nAcceptance (exits 0 when done): python -c 'import nope'",
                           weak_why("own", "ModuleNotFoundError: No module named 'nope'"))
    assert "(unreliable: it fails because of an error in the check itself (ModuleNotFoundError: No module named 'nope')" in task
    start = {"branch_id": "branch-0", "session_id": "fork0/branch-0", "task": task, "messages": [{"role": "user", "content": "go"}]}
    weak_check = ("commit", {"branch_id": "branch-0", "summary": "s", "acceptance_passed": True, "acceptance": "true"})
    good = ("commit", {"branch_id": "branch-0", "summary": "s", "acceptance_passed": True, "acceptance": "grep -q fixed /testbed/f"})
    report = asyncio.run(run_branch_continuation(start, _replying([weak_check, good], [100, 110]), dispatch, 5, ws=ws))
    first, last = report["steps"]
    # `true` passes on the fork point too: refused; the grep fails there and passes on the branch's tree
    assert first["gated"] and "also exits 0 on the code you forked from" in first["observed"]
    assert last["acceptance_verified"] is True and last["acceptance_own"] == "grep -q fixed /testbed/f"
    assert (repo / "f").read_text() == "fixed\n"  # the fork-point run used a scratch copy
    assert "checked by the runtime with the branch's own check, which fails on the fork point" in join_note([report])
    assert "cd /tmp/octopus-fork-point/fork0_branch-0 && git reset -q --hard" in fork_point_command("x", "fork0/branch-0")
    # a control thread (GrokHarborAgent._maybe_control) runs with session_id None: its scratch copy is named by branch_id
    control = dict(start, session_id=None, branch_id="control")
    report = asyncio.run(run_branch_continuation(control, _replying([good], [100]), dispatch, 5, ws=ws))
    assert report["steps"][0]["acceptance_verified"] is True


def test_a_fix_branch_committing_without_any_check_is_asked_for_its_own_once(monkeypatch):
    """A weak task's branch that passes no acceptance to commit has nothing the runtime can run: the first commit is
    refused and asks for its own check; a second one closes the branch as it is, unverified."""
    import asyncio
    from octopus.envs.swe.prompt import join_note, weak_acceptance
    from octopus.runtime.core import run_branch_continuation
    mod, ws = _harbor(monkeypatch)

    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    ran = []

    def dispatch(name, args, branch_id):
        ran.append(name)
        return "committed", False

    task = weak_acceptance("Task 2: fix f\nAcceptance (exits 0 when done): python -c 'import nope'", mod.weak_why("0"))
    start = {"branch_id": "branch-1", "session_id": "fork0/branch-1", "task": task, "messages": [{"role": "user", "content": "go"}]}
    commit = ("commit", {"branch_id": "branch-1", "summary": "s", "acceptance_passed": True})
    report = asyncio.run(run_branch_continuation(start, _replying([commit, commit], [100, 110]), dispatch, 5, ws=ws))
    first, last = report["steps"]
    assert ran == ["commit"]  # nothing to run: no shell
    assert first["gated"] and first["observed"].startswith("commit not accepted yet: your task has no reliable acceptance check")
    assert "your fix was not verified" in first["observed"]
    assert last["after_gate"] and "acceptance_verified" not in last
    assert "acceptance passed, as reported; committed again after the runtime refused it" in join_note([report])


def test_own_failure_flags_checks_that_fail_because_of_themselves(monkeypatch):
    own_failure = _harbor(monkeypatch)[0].own_failure
    own = 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\nTypeError: f() takes 3 arguments'
    lib = ('Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n  File "/testbed/m.py", line 9, '
           'in f\nTypeError: bad operand')
    assert own_failure("python -c 'f(1)'", "1", own) == "TypeError: f() takes 3 arguments"
    assert own_failure("python -c 'f(1)'", "1", lib) is None  # raised in the product code: the bug may be why
    assert own_failure("python -c x", "1", own.replace("TypeError", "AssertionError")) is None
    assert own_failure("python -c x", "1", "  File \"/testbed/a.py\", line 2\nModuleNotFoundError: No module named 'w'")
    assert own_failure("python -m pytest tests/t.py", "4", "ERROR: file or directory not found") == \
        "pytest found no such test file or directory (exit 4)"
    assert own_failure("pytest -k x", "5", "") and own_failure("python t.py", "5", "") is None
    assert own_failure("x", "0", own) is None and own_failure("x", "127", own) is None


def test_branch_fix_abort_needs_a_reason_and_the_join_sees_it(monkeypatch):
    from octopus.envs.swe.prompt import join_note
    from octopus.runtime.core import rejected

    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    observed, _ = _harbor(monkeypatch)[1].runtime("abort", {"branch_id": "branch-0"}, RuntimePort())
    assert rejected("abort", observed)
    report = {"branch_id": "branch-0", "task": "t", "steps": [
        {"name": "abort", "arguments": {"branch_id": "branch-0", "reason": "hypothesis wrong: tests still fail"}}]}
    assert "aborted: hypothesis wrong: tests still fail" in join_note([report])


# fork tasks strings the tool-call parser left (not strict JSON) in bundle-e11/e12 tree.json files, verbatim or trimmed
STRING_TASKS_ESCAPE = '[{"task": "Fix validator_for() to warn about unknown $schema", "acceptance": "cd /testbed && python -c \\"from jsonschema import validators; import warnings; warnings.simplefilter(\'error\', DeprecationWarning); schema = {\'\\$schema\': \'unknownSchema\'}; v = validators.validator_for(schema); print(\'OK if warning raised\')\\"", "task_id": 4, "will_pass": false}, {"task": "Make deprecated attributes accessible in jsonschema.validators", "acceptance": "cd /testbed && python -c \\"from jsonschema import validators; import warnings; warnings.filterwarnings(\'ignore\', category=DeprecationWarning); v = validators.validators; print(\'OK\' if v else \'FAIL\')\\"", "task_id": 3, "will_pass": true}]'
STRING_TASKS_NEWLINE = '[{"task_id": 6, "task": "Fix bug 6: Color code validation now accepts any length of hexadecimal digits", "acceptance": "python - <<\'EOF\'\\nfrom grafanalib import validators\\n\\nclass _Attr(object):\\n    name = \\"color\\"\\n\\n# Wrong-length hex must be rejected  \nfor color in (\\"#1111111\\", \\"#11111\\"):  # 8 characters, 6 characters  \n    try:\\n        validators.is_color_code(None, _Attr(), color)\\n        raise AssertionError(f\\"{color!r} should raise ValueError (invalid color code length)\\")\\n    except ValueError:\\n        pass  # expected\\n\\n# #RRGGBB (7 chars total: # + 6 hex) should be valid\\nvalidators.is_color_code(None, _Attr(), \\"#37872D\\")\\n# #RGB (4 chars total: # + 3 hex) should be valid\\nvalidators.is_color_code(None, _Attr(), \\"#378\\")\\nprint(\\"All color validations passed\\")\\nprint(\\"PASSED\\")\\nEOF\\n", "will_pass": true}]'
STRING_TASKS_BROKEN = '[{"acceptance": "python3 -c \\"from dotenv import get_key; import tempfile,os; p=tempfile.mktemp(suffix=\'.env\'); open(p,\'w\').write(\'foo\\\\n\'); r=get_key(p,\'foo\'); os.unlink(p); exit(0 if r is None else 1)\\"", "task": "Task 1: get_key returns empty string instead of None for unquoted keys"}}]'


def _bundle_env(monkeypatch):
    for key, value in {"OCTOPUS_STRATEGY": "branch_fix", "OCTOPUS_BUNDLE": "1", "OCTOPUS_PLAN_EPISODE": "1"}.items():
        monkeypatch.setenv(key, value)


def test_fork_string_tasks_are_read_as_written_or_refused_with_the_shape(monkeypatch):
    """e11/e12: 146 of main's fork calls had tasks as a string (the parser's fallback when the model's array is not
    strict JSON), all refused as "needs at least two tasks". A backslash before a character JSON does not escape and a
    raw newline inside a string are read as written; anything after the array is ignored. A string that still does not
    parse is refused saying where, with the expected shape."""
    from octopus.envs.swe.prompt import fork_checks, stated_task_ids
    from octopus.runtime.core import fork_task_list
    _bundle_env(monkeypatch)
    apply_runtime = _harbor(monkeypatch)[1].runtime
    port = RuntimePort()
    seen, _ = apply_runtime("fork", {"tasks": STRING_TASKS_ESCAPE}, port)
    assert "branch-1" in seen and len(port.last_fork["branches"]) == 2
    checks = fork_checks({"tasks": STRING_TASKS_ESCAPE})
    assert [k for k, _ in checks] == ["4", "3"] and "{'\\$schema': 'unknownSchema'}" in checks[0][1]  # bash: $ kept
    call = {"function": {"name": "fork", "arguments": json.dumps({"tasks": STRING_TASKS_ESCAPE})}}
    assert stated_task_ids({"tool_calls": [call]}) == ["4", "3"]
    (task,) = fork_task_list(STRING_TASKS_NEWLINE + "}")
    assert "rejected  \nfor color in" in task["acceptance"] and task["task_id"] == 6
    seen, stop = apply_runtime("fork", {"tasks": STRING_TASKS_BROKEN}, port := RuntimePort())
    assert seen.startswith("fork refused: tasks is not a JSON array (Expecting ',' delimiter at char 288: ...")
    assert seen.endswith('<-- here). Send [{"task": "...", "acceptance": "...", "task_id": 1}, ...]; '
                         'inside a string escape every " as \\" and every \\ as \\\\.') and port.last_fork is None
    assert fork_checks({"tasks": STRING_TASKS_BROKEN}) is None and stop is False
    # where in the string as written: a kept backslash (\$) is shown once, not doubled
    broken = '[{"task": "a", "acceptance": "echo \\$x "y"", "task_id": 1}]'
    seen = apply_runtime("fork", {"tasks": broken}, RuntimePort())[0]
    assert f"at char {broken.index('y')}: ...{broken[:broken.index('y')]}<-- here)" in seen


def test_one_branch_is_told_main_covers_the_rest():
    """A fork's only branch is not "one of 1 parallel branches"; with siblings the text is as before."""
    from octopus.runtime.core import branch_task_message
    one = branch_task_message("branch-0", "Task 5", 1)
    assert one.startswith("You are branch-0, the one branch cloned from the context above.\nYour task: Task 5\n"
                          "Work only on this task; the thread that forked you covers the rest. When it is done, "
                          "call commit")
    assert branch_task_message("branch-0", "Task 5", 2).startswith(
        "You are branch-0, one of 2 parallel branches cloned from the context above.\nYour task: Task 5\n"
        "Work only on this task; the other branches cover the rest. When it is done, call commit with ")


def test_a_fork_needs_two_tasks(monkeypatch):
    """A fork of one task is refused, main's or a branch's; the schema and prompt say at least two."""
    from octopus.envs.swe.prompt import delivery, runtime_tools
    _bundle_env(monkeypatch)
    apply_runtime = _harbor(monkeypatch)[1].runtime
    tool = next(t for t in runtime_tools() if t["function"]["name"] == "fork")["function"]
    assert tool["parameters"]["properties"]["tasks"]["minItems"] == 2 and "Need at least two tasks." in tool["description"]
    assert "至少两条且彼此不同" in delivery()
    for port in (RuntimePort(), RuntimePort("branch-0")):
        assert apply_runtime("fork", {"tasks": STRING_TASKS_NEWLINE}, port)[0] == (
            "fork needs at least two tasks, each with task and acceptance") and port.last_fork is None
    monkeypatch.delenv("OCTOPUS_BUNDLE")
    one = [{"task": "x", "acceptance": "true"}]
    assert "at least two" in apply_runtime("fork", {"tasks": one}, RuntimePort())[0]


def test_rewind_tool_restores_the_tree_and_cuts_the_history_to_the_named_step(monkeypatch):
    """OCTOPUS_REWIND_TOOL: a thread starting after 3 inherited steps (a branch) views (step 4), edits (5), fails (6),
    names a step it does not have (7: answered, nothing changes), then rewinds to 4: its tree is the one before the edit,
    its history ends with step 4's result and the note, and its next step is 5 again. The tree is snapshot only before
    a call that may change it, once per step."""
    import asyncio
    from octopus.runtime.core import run_branch_continuation
    monkeypatch.setenv("OCTOPUS_REWIND_TOOL", "1")
    calls = iter([("view", {"path": "a.py"}), ("edit", {"path": "a.py", "old": "a", "new": "b"}),
                  ("shell", {"command": "false"}), ("rewind", {"to_step": 9, "note": "?"}),
                  ("rewind", {"to_step": 4, "note": "wrong file"}), None, None])  # text, reminded, text: it ends
    tree, snaps, restored, seen = {"a.py": "a"}, [], [], []

    def complete(messages, session_id=None):
        seen.append((start["step"], [dict(m) for m in messages]))
        call = next(calls)
        return {"choices": [{"message": {"content": "" if call else "stop", "tool_calls": call and [
            {"id": f"c{len(seen)}", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}}]}

    def dispatch(name, args, branch_id):
        if name == "edit":
            tree["a.py"] = args["new"]
        return "exit 1\nfailed" if name == "shell" else f"{name} ok", False

    async def snapshot():
        snaps.append(json.dumps(tree))
        return snaps[-1]

    async def restore(snap):
        restored.append(snap)
        tree.clear()
        tree.update(json.loads(snap))
    start = {"branch_id": "branch-0", "session_id": "s", "task": "t", "step": 3,
             "messages": [{"role": "user", "content": "go"}]}
    report = asyncio.run(run_branch_continuation(start, complete, dispatch, 10, snapshot=snapshot, restore=restore,
                                                 ws=_harbor(monkeypatch)[1]))
    assert [step for step, _ in seen] == [3, 4, 5, 6, 7, 4, 5]  # the reminded reply is a step
    seen.pop()
    assert snaps == ['{"a.py": "a"}', '{"a.py": "b"}'] and restored == ['{"a.py": "a"}'] and tree == {"a.py": "a"}
    invalid = report["steps"][3]["observed"]
    assert invalid == ("rewind needs to_step, one of this thread's steps: an integer from 3 (where it started) to 6. "
                       "Nothing was rewound.")
    assert seen[4][1][-1] == {"role": "tool", "tool_call_id": "c4", "content": invalid}  # answered, nothing cut
    history = seen[-1][1]
    assert history[:-1] == seen[1][1] and history[-2] == {"role": "tool", "tool_call_id": "c1", "content": "view ok"}
    note = ("[runtime] Rewound to step 4 at your request: the steps after it were removed and the repository is back as "
            "it was after step 4. They edited a.py; 1 commands failed, last with: failed.\nYour note: wrong file")
    assert history[-1] == {"role": "tool", "content": note}  # tagged like a tool result
    (rewind,) = report["rewinds"]  # with the messages it removed: steps 5 to 7 and the rewinding reply
    assert {k: v for k, v in rewind.items() if k != "messages"} == {
        "step": 5, "dropped_messages": 8, "to_step": 4, "failures": 1, "edited": ["a.py"], "note": note}
    assert rewind["messages"][:6] == seen[-2][1][len(history) - 1:]
    assert [m["role"] for m in rewind["messages"]] == ["assistant", "tool"] * 4


@pytest.mark.parametrize("script, steps", [
    ([("edit", {}), ("rewind", {"to_step": 0, "note": "n"}), ("edit", {}), ("shell", {}), None, None],
     [0, 1, 0, 1, 0, 1]),
    ([("view", {}), ("edit", {}), ("rewind", {"to_step": 1, "note": "n"}), ("edit", {}), ("shell", {}), ("view", {}),
      None, None], [0, 1, 2, 1, 2, 1, 2, 3])])
def test_rule_and_tool_rewinds_together_keep_both_notes_and_the_step_numbers(monkeypatch, script, steps):
    """OCTOPUS_REWIND=2 with OCTOPUS_REWIND_TOOL=1: the model rewinds to the step at whose end the rule's checkpoint
    sits (its start, or after a view), edits again and fails three times: the rule rewinds to before that edit with
    the model's note kept, and the steps go on from there (a rewind to the start once left no step: IndexError)."""
    import asyncio
    from octopus.runtime.core import run_branch_continuation
    monkeypatch.setenv("OCTOPUS_REWIND", "2")
    monkeypatch.setenv("OCTOPUS_REWIND_TOOL", "1")
    calls, seen = iter(script), []

    def complete(messages, session_id=None):
        seen.append((start["step"], [m["content"] for m in messages if m["content"].startswith("[runtime] Rewound")]))
        call = next(calls)
        n = 3 if call and call[0] == "shell" else 1  # three failing shells in one reply
        return {"choices": [{"message": {"content": "" if call else "stop", "tool_calls": call and [
            {"id": f"c{len(seen)}-{i}", "type": "function", "function": {"name": call[0], "arguments": json.dumps(
                {"path": "a.py", "old": "a", "new": "b"} if call[0] == "edit" else call[1])}} for i in range(n)]}}]}

    async def snapshot():
        return "head0"

    async def restore(snap):
        pass
    start = {"branch_id": "main", "session_id": None, "task": "", "messages": [{"role": "user", "content": "go"}]}
    asyncio.run(run_branch_continuation(start, complete, lambda name, args, b: (
        "exit 1\nfailed" if name == "shell" else "ok", False), 20, snapshot=snapshot, restore=restore,
        ws=_harbor(monkeypatch)[1]))
    assert [step for step, _ in seen] == steps
    notes = seen[-1][1]
    assert len(notes) == 2 and notes[0].startswith("[runtime] Rewound to step") and notes[1].startswith(
        "[runtime] Rewound: the attempt")


TEXT = "It is **Monoceros**."  # search v8, qid 2884: main's answer as a reply without done


def _texting(script, prompts=None, **payload):
    """A fake model: a script item is a (tool, args) call or None, a reply without a tool call (TEXT, with reasoning);
    prompts: each request's prompt_tokens (default 100); payload: extra keys of each response (budget_out)."""
    replies, prompts, n = iter(script), iter(prompts or [100] * len(script)), [0]

    def complete(messages, session_id=None):
        call, n[0] = next(replies), n[0] + 1
        return {"id": f"r{n[0]}", "usage": {"prompt_tokens": next(prompts), "completion_tokens": 1}, **payload,
                "choices": [{"message": {"content": "" if call else TEXT, "reasoning_content": "so: Monoceros",
                                         "tool_calls": call and [{"id": f"c{n[0]}", "type": "function", "function": {
                                             "name": call[0], "arguments": json.dumps(call[1])}}]}}]}
    complete.requests = n
    return complete


def _thread(complete, closing=frozenset({"done"}), **kw):
    """main's loop (CursorRuntime.run: closing done, capped in calls) unless kw says otherwise."""
    from octopus.runtime.core import run_branch_continuation
    start = {"branch_id": "main", "session_id": None, "task": "", "messages": [{"role": "user", "content": "q"}]}
    kw = {"max_steps": None, "max_calls": 10, **kw}
    return asyncio.run(run_branch_continuation(start, complete, lambda name, args, _b: ("ok", False), closing=closing,
                                               refused=lambda *_: False, **kw))


@pytest.mark.parametrize("env", ["search main", "search branch", "swe branch"])
def test_a_reply_without_a_tool_call_is_reminded_once_then_closes(env, monkeypatch):
    """Since #306 only a closing call hands a thread's result: its first reply without a tool call is answered with
    core.REMIND_NOTE naming the thread's own closing tools (the environment's: main's done, search's branch done, a
    SWE branch's abort/commit/done), and it goes on; here it closes with done. The reply stays in its history as sent
    (AReaL's record of it: the next prompt extends it) and the reminder is a runtime note after it."""
    HarborWorkspace = _harbor(monkeypatch)[0].HarborWorkspace  # Harbor stubbed: the agents import it (tree.py)
    from octopus.envs.search.agent import SearchWorkspace
    from octopus.runtime.core import REMIND_NOTE, closing_name
    closing = {"search main": {"done"}, "search branch": SearchWorkspace.branch_closing,
               "swe branch": HarborWorkspace.branch_closing}[env]
    complete = _texting([("search", {}), None, ("done", {"answer": "Monoceros"})])
    report = _thread(complete, closing, **({} if env == "search main" else {"max_steps": 8, "max_calls": None}))
    assert closing_name(report["steps"]) == "done" and report["reminded"] == 1 and complete.requests[0] == 3
    names = {"search main": "done", "search branch": "done", "swe branch": "abort or commit or done"}[env]
    assert report["messages"][3:5] == [
        {"role": "assistant", "content": TEXT, "reasoning_content": "so: Monoceros"},
        {"role": "user", "content": f"You replied without a tool call. A thread's result counts only through its "
                                    f"closing call: close with {names} now, or continue with a tool call."}]
    assert REMIND_NOTE.format(calls=names) == report["messages"][4]["content"]


def test_a_second_reply_without_a_tool_call_ends_the_thread():
    """Reminded once per thread: the next reply without a tool call ends it as before #306's reminder (no answer)."""
    complete = _texting([None, ("search", {}), None])
    report = _thread(complete)
    assert complete.requests[0] == 3 and report["reminded"] == 1
    assert [s["name"] for s in report["steps"]] == ["search", None] and report["steps"][-1]["text"] == TEXT
    assert [m["role"] for m in report["messages"]] == ["user", "assistant", "user", "assistant", "tool"]


@pytest.mark.parametrize("cap", ["turns", "reserve", "budget", "window", "budget_out"])
def test_no_reminder_without_room_for_another_turn(monkeypatch, cap):
    """The reminder costs a turn: a thread without one left (its turns), inside its answer reserve, at its growth
    budget or the serving window's room, or whose reply used up the shared token budget, ends on that reply as before,
    unreminded. (Calls: a thread makes a request only with a call left, and a reply without a tool call spends none.)"""
    from octopus.envs.search.dry_run import stub_tokenizer
    from octopus.runtime.settings import max_reply
    script, prompts, kw, payload = [("search", {}), None], [100, 100], {}, {}
    if cap == "turns":
        kw = {"max_steps": 2, "max_calls": None}
    elif cap == "reserve":  # grown 601 of 1000: 399 left, under the 500 kept for closing
        monkeypatch.setenv("OCTOPUS_ANSWER_RESERVE", "500")
        monkeypatch.setenv("OCTOPUS_TOKENIZER", stub_tokenizer())
        prompts, kw = [100, 700], {"budget": 1000}
    elif cap == "budget":  # grown 501 of 500 (the prompt itself 500 past the start: not yet a "context" stop)
        prompts, kw = [100, 600], {"budget": 500}
    elif cap == "window":  # the first reply: the window leaves no room past the first prompt
        monkeypatch.setenv("OCTOPUS_CONTEXT_WINDOW", str(100 + max_reply()))
        script, prompts, kw = [None], [100], {"budget": 10_000}
    else:
        payload = {"budget_out": True}
    complete = _texting(script, prompts, **payload)
    report = _thread(complete, **kw)
    assert report["reminded"] == 0 and report["steps"][-1] == {"name": None, "text": TEXT, "session_id": None,
                                                                "arguments": {}}
    assert complete.requests[0] == len(script)


@pytest.mark.parametrize("to_step", [1, 2])
def test_a_rewind_keeps_or_drops_the_reminder_with_its_step(monkeypatch, to_step):
    """OCTOPUS_REWIND_TOOL: the reminded reply and its reminder are a step of their own (step 2 here). A rewind to a
    step before it drops both with the rest (kept in the rewind's record), one to it keeps both; either way the thread
    was reminded (start["reminded"] is never undone, as rewinds_used), so its next reply without a tool call ends it."""
    from octopus.runtime.core import REMIND_NOTE
    monkeypatch.setenv("OCTOPUS_REWIND_TOOL", "1")

    async def snapshot():
        return "snap"

    async def restore(_snap):
        pass
    complete = _texting([("view", {"path": "a.py"}), None, ("rewind", {"to_step": to_step, "note": "n"}), None])
    report = _thread(complete, frozenset({"done"}), snapshot=snapshot, restore=restore, ws=_harbor(monkeypatch)[1])
    assert complete.requests[0] == 4 and report["reminded"] == 1 and report["steps"][-1]["text"] == TEXT
    reminder = {"role": "user", "content": REMIND_NOTE.format(calls="done")}
    (rewind,) = report["rewinds"]
    assert rewind["to_step"] == to_step
    assert (reminder in report["messages"]) == (to_step == 2) and (reminder in rewind["messages"]) == (to_step == 1)
    assert report["messages"][-1]["content"].startswith("[runtime] Rewound to step %d" % to_step)
