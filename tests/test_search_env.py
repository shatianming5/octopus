"""Search environment (octopus/envs/search): tools and logs of one episode, reward (EM, judge), group advantages."""

import ast
import asyncio
import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from octopus.rl.group import assign_threads, plan_group_stats
from octopus.envs.search.dry_run import _Json, serve
from octopus.envs.search.reward import exact_match, grade, normalize, search_group_step, search_outcome
from test_rl_plan import _Ids

HARBOR_STUBS = ("harbor", "harbor.agents", "harbor.agents.base", "harbor.agents.capabilities", "harbor.environments",
                "harbor.environments.base", "harbor.models", "harbor.models.agent", "harbor.models.agent.context",
                "harbor.models.trajectories", "harbor.models.trajectories.agent",
                "harbor.models.trajectories.observation", "harbor.models.trajectories.observation_result",
                "harbor.models.trajectories.step", "harbor.models.trajectories.tool_call",
                "harbor.models.trajectories.trajectory")


def _search_modules(monkeypatch):
    """octopus.envs.search.agent and dry_run on stand-ins for harbor (not installed here); the env they set is undone."""
    class Any:
        def __init__(self, *a, **k):
            pass

        def to_json_dict(self):
            return {}

    class BaseAgent(Any):
        session_id = None

        def __init__(self, logs_dir=None, model_name=None, **k):
            self.logs_dir, self.model_name = logs_dir, model_name
    for name in HARBOR_STUBS:
        module = types.ModuleType(name)
        for attr in ("AgentCapabilities", "BaseEnvironment", "AgentContext", "Agent", "Observation",
                     "ObservationResult", "Step", "ToolCall", "Trajectory"):
            setattr(module, attr, Any)
        module.BaseAgent = BaseAgent
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("octopus.runtime.tree", "octopus.envs.search.agent", "octopus.eval.search_dev_eval"):
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]  # imported fresh on the stand-ins, removed again after the test
    from octopus.envs.search import agent
    for key in ("OCTOPUS_SEARCH_URL", "OCTOPUS_CHAT_URL", "OCTOPUS_MODEL", "OCTOPUS_HTTP_ATTEMPTS", "OCTOPUS_PLAN_CONTROL",
                *agent.DEFAULTS):
        monkeypatch.setenv(key, agent.DEFAULTS.get(key, ""))
    monkeypatch.setenv("OCTOPUS_READ_TOOL", "1")  # rl_run.search.env's (defaults.DEFAULTS has none)
    for key in ("OCTOPUS_BUNDLE", "OCTOPUS_PLAN_EPISODE", "OCTOPUS_STRATEGY", "OCTOPUS_JUDGE_URL",
                "OCTOPUS_FORK_DEPTH", "OCTOPUS_MAX_THREADS"):
        monkeypatch.delenv(key, raising=False)
    from octopus.envs.search import dry_run
    monkeypatch.setenv("OCTOPUS_TOKENIZER", dry_run.stub_tokenizer())
    return agent, dry_run


def test_search_episode_tools_threads_and_logs(monkeypatch, tmp_path):
    """One dry-run episode on the search DEFAULTS and read_thread on: main forks two branches that search, open and
    close with done (answer and evidence), fork returns their findings and main answers; the main-first control answers without fork. No episode budget: each tool result
    shows the thread's own growth left and calls left; no join offered. Its record is search_outcome's, from the
    runtime's logs and verifier/answers.json (the verifier dir's only file)."""
    agent, dry_run = _search_modules(monkeypatch)
    summary = dry_run.main(tmp_path)
    trial = tmp_path / summary["trial"].rpartition("/")[2]
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    assert all(r["http_code"] == 200 for r in rows)
    assert summary["threads"] == {"main": 2, "fork0/branch-0": 3, "fork0/branch-1": 3, "control": 2}
    tools = {r["thread"]: set(r["tools"]) for r in rows}
    assert {"search", "open", "done", "fork"} == tools["main"] == tools["fork0/branch-0"]  # no commit / abort
    assert tools["control"] == {"search", "open", "done"}

    tree = json.loads((trial / "agent/tree.json").read_text())
    assert tree["schema"] == "octopus-tree-v1" and tree["main_end"] == "done" and tree["budget"] is None
    system = tree["main"][0]["content"]
    assert "context left" in system and "budget" not in system and agent._SYNC in system
    # fork's own result: the port's, ended with what happens to the findings; fork takes tasks only
    assert tree["main"][3]["content"].endswith("]\nfork started: branches branch-0, branch-1; their findings follow "
                                               "when they finish.")
    fork_tool = next(t for t in tree["tools"] if t["function"]["name"] == "fork")["function"]["parameters"]
    assert set(fork_tool["properties"]) == {"tasks"} and fork_tool["properties"]["tasks"]["minItems"] == 1
    assert "at least two" not in system and "Fork when it helps; whether, when and how many is your choice." in system
    fork = tree["forks"][0]
    assert fork["response_id"] == "main#1" and fork["join_response_id"] == "main#1"  # sync: joined in its own reply
    branch = fork["branches"][1]
    assert branch["ended_with"] == "done" and branch["task"] == "Find the birth year of Mira Okonkwo"
    assert branch["summary"] == "1971"  # what its done hands main (core.handed)
    # the branch limits in effect, and a branch's calls left are its turns, in the one shared system prompt
    assert ("fork waits for every branch to close and returns their findings. Limits: at most 10 branches in the "
            "session; a branch cannot fork; a branch has at most 50 turns. Every tool result's tag shows") in system
    assert "and your calls left (a branch: its turns left); when" in system
    # the first fork: main waits for it and its branches start together, so no thread has a readable step: no
    # read_thread roster or card (read_when_readable); done closes the branch
    task = next(m["content"] for m in branch["messages"] if "You are fork0/branch-1," in str(m.get("content")))
    assert "Other threads at this fork" not in task and "read_thread" not in task
    assert ("Work only on this task; the other branches cover the rest. When it is done, call done with your answer "
            "and its evidence (docids, short quotes); if you found nothing, say so in your answer.") in task
    assert "commit" not in task and "abort" not in task
    results = [m["content"] for m in branch["messages"] if m["role"] == "tool"]  # the fork's result, then its own
    assert "] docid=d2 | Mira Okonkwo |" in results[1] and "[characters 0-71 of 71; end of document]" in results[2]
    # thread_budget() (32768) less its growth through its 2nd reply (prompt_tokens: the stub's
    # message size / 4; 20 completion)
    grown = [r["prompt_tokens"] for r in rows if r["thread"] == "fork0/branch-1"]
    assert results[2].startswith(f"[branch-1 | context left: {32768 - (grown[1] - grown[0]) - 20} | calls left: 48 | step 3]")
    joined = tree["main"][4]["content"]
    assert ("- fork0/branch-1 (Find the birth year of Mira Okonkwo): done: 1971\n  evidence: d2: Mira Okonkwo (born 1971)"
            in joined)
    assert "changes" not in joined and "/testbed" not in joined  # no patch lines without a workspace
    assert joined.endswith("\n" + agent.SearchWorkspace().read_card) and "view" not in joined  # the search card
    assert tree["control"]["kind"] == "main-first" and tree["control"]["end"] == "done"

    outcome = search_outcome(trial, rows)
    assert outcome["y"] is True and outcome["forked"] and outcome["C"] == 100.0  # 20 + max(60, 60) + 20
    assert outcome["control"]["y"] and outcome["control"]["C"] == 40.0
    # no reader: no branch graded on its own; the verifier dir holds the graded answers only
    assert [(b["key"], b["passed"]) for b in outcome["branches"]] == [("fork0-branch-0", None), ("fork0-branch-1", None)]
    assert [p.name for p in (trial / "verifier").iterdir()] == ["answers.json"]
    assert summary["answers"]["main"] == {"answer": "1971", "correct": True, "how": "em"}


def _scripted(dry_run, scripts: dict, nofork: str = "control"):
    """A policy serving scripts {thread: [(tool, args, prompt_tokens), ...]}: a branch is its session id, a request
    without fork in its tools is `nofork`'s, any other main's. requests: (thread, its tool results) per request."""
    import collections
    import threading
    turns, lock, requests = collections.Counter(), threading.Lock(), []

    class Policy(_Json):
        def do_POST(self):
            body = self.body()
            fork = "fork" in {t["function"]["name"] for t in body.get("tools") or []}
            thread = body.get("session_id") or ("main" if fork else nofork)
            with lock:
                turns[thread] += 1
                turn = turns[thread]
                requests.append((thread, [m["content"] for m in body["messages"] if m["role"] == "tool"]))
            name, args, prompt = scripts[thread][turn - 1]  # name: a tool, or a reply's [(tool, args), ...]
            calls = [c for n, a in (name if isinstance(name, list) else [(name, args)]) for c in dry_run._call(n, **a)]
            self.reply(200, {"id": f"{thread}#{turn}", "object": "chat.completion",
                             "usage": {"prompt_tokens": prompt, "completion_tokens": 20},
                             "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                 "role": "assistant", "content": "", "tool_calls": calls}}]})
    return Policy, requests


def test_search_thread_growth_caps(monkeypatch, tmp_path):
    """OCTOPUS_THREAD_BUDGET=1000: main ends at the reply that grew past 1000 tokens since its first request (dropped,
    main_end "context"); a branch has the same cap (thread_budget: branch-0 grows 4000 and is cut, branch-1 grows 500
    and closes); the main-first control gets what main had
    left at the fork (1000 - 300), not a fresh 1000; each tool result shows the thread's growth left through its last reply (prompt and completion), before the results. A linear run is
    capped the same; unset, the one default 32768. OCTOPUS_ANSWER_RESERVE=0: no reserve, every call runs up to the
    cap."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_PLAN_CONTROL="1",
                           OCTOPUS_THREAD_BUDGET="1000", OCTOPUS_ANSWER_RESERVE="0").items():
        monkeypatch.setenv(key, value)
    branch = [("search", {"query": "Mira Okonkwo"}, 5000), ("open", {"docid": "d2"}, 9000)]  # grows 4000 > 1000
    small = [("search", {"query": "Mira Okonkwo"}, 5000), ("open", {"docid": "d2"}, 5500)]  # grows 500
    scripts = {"main": [("search", {"query": "Zorblax"}, 100),
                        ("fork", {"tasks": ["Find the founder", "Find the birth year"]}, 400),  # grown 300
                        ("done", {"answer": "1971"}, 1200)],  # grown 1100 > 1000: dropped
               "fork0/branch-0": branch + [("done", {"answer": "1971", "evidence": "d2"}, 9100)],
               "fork0/branch-1": small + [("done", {"answer": "1971", "evidence": "d2"}, 5600)],
               "control": [("search", {"query": "Mira"}, 400), ("done", {"answer": "1971"}, 1200)]}  # grown 800 > 700
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path / "a"))
    tree = json.loads((trial / "agent/tree.json").read_text())
    assert tree["main_end"] == "context" and tree["control"]["end"] == "context"
    assert [b["ended_with"] for b in tree["forks"][0]["branches"]] == ["context", "done"]
    main = [results for thread, results in requests if thread == "main"]
    assert main[1][-1].startswith("[main | context left: 980 | calls left: 99 | step 1]")  # its first reply: 20 completion tokens
    assert main[2][-1].startswith("[main | context left: 680 | calls left: 98 | step 2]")
    control = [results for thread, results in requests if thread == "control"]
    assert control[1][-1].startswith("[main | context left: 680 | calls left: 98 | step 2]")
    assert not json.loads((trial / "verifier/answers.json").read_text())["control"]["answer"]

    monkeypatch.setenv("OCTOPUS_LINEAR", "1")  # a single agent: the same cap
    monkeypatch.delenv("OCTOPUS_PLAN_CONTROL")
    policy, requests = _scripted(dry_run, {"main": [scripts["main"][0], scripts["main"][2]]}, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path / "b"))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "context"

    monkeypatch.delenv("OCTOPUS_THREAD_BUDGET")  # unset or 0: the one default, 32768 (settings.thread_budget)
    policy, requests = _scripted(dry_run, {"main": [scripts["main"][0], scripts["main"][2]]}, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path / "c"))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "done"
    assert requests[1][1][-1].startswith("[main | context left: 32748 | calls left: 99 | step 1]")


def test_search_answer_reserve(monkeypatch, tmp_path):
    """OCTOPUS_THREAD_BUDGET=1000, OCTOPUS_ANSWER_RESERVE=300: a reply that leaves a thread less than 300 of its growth
    has its other calls refused with RESERVE_NOTE (its tag still the truthful growth left), and its next done closes it:
    a one-task fork's branch hands main its answer and evidence, main answers."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="1000",
                           OCTOPUS_ANSWER_RESERVE="300").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("fork", {"tasks": ["Find the birth year of Mira Okonkwo"]}, 100),
                        ("search", {"query": "Zorblax"}, 800),  # grown 720: 280 left, refused
                        ("done", {"answer": "1971"}, 900)],
               "fork0/branch-0": [("search", {"query": "Mira Okonkwo"}, 5000),
                                  ("open", {"docid": "d2"}, 5750),  # grown 770: 230 left, refused
                                  ("done", {"answer": "1971", "evidence": "d2: born 1971"}, 5800)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    assert tree["main_end"] == "done" and tree["forks"][0]["branches"][0]["ended_with"] == "done"
    note = "Refused: your context is nearly used up ({} tokens left). Close now: call done with what you have so far. " \
           "Until you do, every other call is refused."
    said = {thread: results[-1] for thread, results in requests if results}  # each thread's last request: its last result
    assert said["main"] == "[main | context left: 280 | calls left: 98 | step 2]\n" + note.format(280)
    assert said["fork0/branch-0"] == "[branch-0 | context left: 230 | calls left: 48 | step 3]\n" + note.format(230)
    assert "- fork0/branch-0 (Find the birth year of Mira Okonkwo): done: 1971\n  evidence: d2: born 1971" in tree["main"][4][
        "content"]
    assert json.loads((trial / "verifier/answers.json").read_text())["main"]["answer"] == "1971"


@pytest.mark.parametrize("branch", [False, True])
def test_search_result_past_the_reserve_is_clipped(monkeypatch, tmp_path, branch):
    """OCTOPUS_THREAD_BUDGET=400, OCTOPUS_ANSWER_RESERVE=300: a search result larger than the 80 tokens left above the
    reserve is clipped there (CLIPPED, then CLOSE_NOTE), the thread is in its reserve, and its next reply closes it
    with done: main's (a linear run) or a one-task fork's branch's."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.runtime.core import CLIPPED
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="400",
                           OCTOPUS_ANSWER_RESERVE="300", **({} if branch else {"OCTOPUS_LINEAR": "1"})).items():
        monkeypatch.setenv(key, value)
    work = [("search", {"query": "Zorblax Mira Okonkwo garden robot"}, 1000), ("done", {"answer": "1971"}, 1100)]
    scripts = ({"main": [("fork", {"tasks": ["Find the birth year"]}, 100), ("done", {"answer": "1971"}, 200)],
                "fork0/branch-0": work} if branch else {"main": [(n, a, p - 900) for n, a, p in work]})
    policy, requests = _scripted(dry_run, scripts, nofork="control" if branch else "main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    thread = "fork0/branch-0" if branch else "main"
    result = [r for th, r in requests if th == thread][1][-1]
    assert CLIPPED in result and result.endswith("\n\nYour context is nearly used up. Close now: call done with what "
                                                 "you have so far; any other call is refused.")
    assert len(result.split("\n", 1)[1]) < 3 * 80  # the room above the reserve, in characters
    assert tree["main_end"] == "done"
    assert not branch or tree["forks"][0]["branches"][0]["ended_with"] == "done"


def test_search_answer_reserve_cannot_loop(monkeypatch, tmp_path):
    """A main that keeps calling search inside its reserve is refused each time and ends at the cap (main_end
    "context"), not in a loop."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="1000",
                           OCTOPUS_ANSWER_RESERVE="300", OCTOPUS_LINEAR="1").items():
        monkeypatch.setenv(key, value)
    search = ("search", {"query": "Zorblax"})
    policy, requests = _scripted(dry_run, {"main": [(*search, p) for p in (100, 800, 900, 1200, 1300)]}, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "context" and len(requests) == 4
    assert all("\nRefused: your context is nearly used up" in r[-1] for _, r in requests[2:])


def test_search_branch_commit_is_no_close(monkeypatch, tmp_path):
    """A search branch's commit (no search tool) is answered as an unknown tool and is a rejected step, never the
    branch's close: the branch that searched, committed and hit its turn cap ended unclosed, hands main no summary,
    and the join note says it stopped without done."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_BRANCH_MAX_TURNS="3").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("fork", {"tasks": ["Find the birth year"]}, 100), ("done", {"answer": "1971"}, 300)],
               "fork0/branch-0": [("search", {"query": "Mira Okonkwo"}, 1000),
                                  ("commit", {"branch_id": "branch-0", "summary": "WRONG 1968"}, 1100),
                                  ("search", {"query": "Okonkwo born"}, 1200)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    branch = tree["forks"][0]["branches"][0]
    assert branch["ended_with"] == "search" and branch["summary"] is None
    joined = tree["main"][4]["content"]
    assert "committed" not in joined and "WRONG" not in joined and "stopped after 3 steps without done" in joined


def test_search_read_thread_offered_where_a_read_can_succeed(monkeypatch, tmp_path):
    """read_when_readable: the first fork's branches (siblings that start together, main waiting) get no read_thread
    roster or card, and a read_thread call there is refused; a later fork's branches do, since the first fork's ended
    branches are readable. read_by_name: the roster, the join note and the refusals name threads by their names
    (fork1/branch-1), and a branch is told its own ("You are fork1/branch-0"); a bare id reads the one thread that has it
    (branch-2: fork0's), and is refused as ambiguous, naming them, where several do (branch-0). The mount does not
    count toward the reader's growth: its tag's context left is 32768 less its own
    prompt growth and completion only (prompt_tokens leave the mount out)."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_CURSOR="1").items():
        monkeypatch.setenv(key, value)
    first = [("search", {"query": "Mira Okonkwo"}, 1000), ("open", {"docid": "d2"}, 3000), ("done", {"answer": "1971"}, 3100)]
    scripts = {"main": [("fork", {"tasks": ["Find the founder", "Find the birth year", "Find the company"]}, 100),
                        ("fork", {"tasks": ["Check the founder", "Check the year"]}, 300),
                        ("done", {"answer": "1971"}, 500)],
               **{f"fork0/branch-{i}": first for i in range(3)},
               "fork0/branch-0": [("read_thread", {"thread": "branch-1"}, 900)] + first,  # offered none: refused
               "fork1/branch-0": [("read_thread", {"thread": "branch-2"}, 5000),  # fork0's: fork1 has no branch-2
                                  ("search", {"query": "Mira Okonkwo born"}, 5100),  # its request carries the mount
                                  ("done", {"answer": "1971"}, 5200)],
               "fork1/branch-1": [("read_thread", {"thread": "branch-0"}, 5000),  # fork0's and fork1's: ambiguous
                                  ("done", {"answer": "1971"}, 5100)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    tasks = [[next(m["content"] for m in b["messages"] if f"You are fork{k}/{b['branch_id']}," in str(m.get("content")))
              for b in f["branches"]] for k, f in enumerate(tree["forks"])]  # told its thread name
    assert not any("Other threads at this fork" in t or "read_thread" in t for t in tasks[0])
    assert all("Other threads at this fork (read_thread names): " in t and "read_thread is available" in t for t in tasks[1])
    assert ("(read_thread names): fork1/branch-1 (task: Check the year); fork0/branch-0 (task: Find the founder); "
            "fork0/branch-1 (task: Find the birth year); fork0/branch-2 (task: Find the company).") in tasks[1][0]
    assert "- fork0/branch-2 (Find the company): done: 1971" in tree["main"][4]["content"]  # the join note's names
    assert [r for th, r in requests if th == "fork0/branch-0"][1][-1].endswith(
        "]\nread_thread is not available in this branch. Nothing was read.")
    ambiguous = [r for th, r in requests if th == "fork1/branch-1"][1][-1].split("\n", 1)[1]
    assert ambiguous == ("branch-0 is ambiguous: fork0/branch-0, fork1/branch-0 have it; name one of them. Nothing "
                         "was read.")
    reader = [results for thread, results in requests if thread == "fork1/branch-0"]  # OCTOPUS_CURSOR: each its delta
    read = reader[1][-1].split("\n", 1)[1]  # the read's result, sent with the mount
    assert read.startswith("Mounted fork0/branch-2's steps") and int(read.split("(", 1)[1].split(" tokens")[0]) > 0
    assert reader[2][-1].startswith(f"[branch-0 | context left: {32768 - (5100 - 5000) - 20} |")


def test_search_fork_takes_one_task_swe_two(monkeypatch):
    """The fewest tasks a fork takes is the environment's (WorkspaceProvider.fork_min): search 1, the runtime's
    default (SWE's) 2."""
    agent, _ = _search_modules(monkeypatch)
    from octopus.runtime.core import RuntimePort
    from octopus.runtime.tree import WorkspaceProvider
    assert agent.SearchWorkspace().runtime("fork", {"tasks": ["one"]}, RuntimePort()) == (
        "fork started: branches branch-0", False)
    assert agent.SearchWorkspace().runtime("fork", {"tasks": []}, RuntimePort())[0] == (
        "fork needs tasks, each a non-empty string")
    assert WorkspaceProvider().runtime("fork", {"tasks": ["one"]}, RuntimePort())[0] == "fork needs at least two tasks"


def test_search_rewind_tool_cuts_the_history_and_its_steps_still_count(monkeypatch, tmp_path):
    """OCTOPUS_REWIND_TOOL=1 in search (no workspace), a linear run: rewind is offered with search's description
    (history only, no repository) and read_thread's card names it. A rewind cuts main's history back to the named
    step; its note names no workspace and is a tool message tagged with what is left now (calls stay spent); the
    episode goes on to its answer. The removed steps still count: their search results in bcp's recall, and the
    episode record's rewinds (calls, steps dropped, note characters)."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.eval import bcp
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_REWIND_TOOL="1",
                           OCTOPUS_LINEAR="1").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("search", {"query": "Zorblax"}, 100), ("search", {"query": "garden"}, 200),
                        ("rewind", {"to_step": 1, "note": "wrong lead"}, 300), ("done", {"answer": "1971"}, 400)]}
    policy, requests = _scripted(dry_run, scripts, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    rewind = next(t for t in tree["tools"] if t["function"]["name"] == "rewind")["function"]["description"]
    assert "history" in rewind and "repository" not in rewind and "not refunded" not in rewind
    assert "not in a reply that calls fork or rewind." in agent.SearchWorkspace().read_card
    note = ("[runtime] Rewound to step 1 at your request: the steps after it were removed from your history.\n"
            "Your note: wrong lead")
    result, told = requests[3][1]  # step 1, the note: 3 calls spent, context back to step 1's
    assert tree["main_end"] == "done" and "docid=d1" in result and "docid=d3" not in result
    assert told == "[main | context left: 32668 | calls left: 97 | step 1]\n" + note
    assert bcp.trial_run(trial)["retrieved_docids"] == ["d1", "d2", "d3"]  # d3: the rewound search's
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    assert search_outcome(trial, rows)["rewinds"] == {"calls": 1, "steps": 2, "note_chars": len(note)}


def test_search_reply_that_rewinds_is_not_clipped(monkeypatch, tmp_path):
    """OCTOPUS_THREAD_BUDGET=2000, OCTOPUS_ANSWER_RESERVE=300: a reply that rewinds with little room left above the
    reserve has its results kept, not clipped (they go with the rewind): the context comes back and the thread is not
    stuck in its reserve, so its next search runs. The rewind note's tag shows the context left back at step 0: all
    2000 (the growth of the step it went back to, not of the rewinding reply)."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="2000",
                           OCTOPUS_ANSWER_RESERVE="300", OCTOPUS_LINEAR="1", OCTOPUS_REWIND_TOOL="1").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("search", {"query": "Zorblax"}, 100), ("rewind", {"to_step": 0, "note": "back"}, 1750),
                        ("search", {"query": "Okonkwo"}, 200), ("done", {"answer": "1971"}, 300)]}
    policy, requests = _scripted(dry_run, scripts, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "done"
    assert "docid=d2" in requests[3][1][-1] and "Refused" not in requests[3][1][-1]
    (told,) = requests[2][1]
    assert told.startswith("[main | context left: 2000 | calls left: 98 | step 0]\n[runtime] Rewound to step 0"), told


@pytest.mark.parametrize("branch", [False, True])
def test_search_rewind_runs_in_the_reserve(monkeypatch, tmp_path, branch):
    """OCTOPUS_THREAD_BUDGET=400, OCTOPUS_ANSWER_RESERVE=300, OCTOPUS_REWIND_TOOL=1: a thread whose search result was
    clipped (in its reserve; the note offers rewind) may rewind to where it started: the rewind runs, the context comes
    back (all 400), the thread leaves its reserve and its next search runs (clipped again). Main of a linear run, or a
    one-task fork's branch."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.runtime.core import CLIPPED
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="400",
                           OCTOPUS_ANSWER_RESERVE="300", OCTOPUS_REWIND_TOOL="1",
                           **({} if branch else {"OCTOPUS_LINEAR": "1"})).items():
        monkeypatch.setenv(key, value)
    big = ("search", {"query": "Zorblax Mira Okonkwo garden robot"})
    work = [(*big, 1000), ("rewind", {"to_step": int(branch), "note": "back"}, 1090), (*big, 1020),
            ("done", {"answer": "1971"}, 1110)]
    scripts = ({"main": [("fork", {"tasks": ["Find the birth year"]}, 100), ("done", {"answer": "1971"}, 200)],
                "fork0/branch-0": work} if branch else {"main": [(n, a, p - 900) for n, a, p in work]})
    policy, requests = _scripted(dry_run, scripts, nofork="control" if branch else "main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    said = [r[-1] for th, r in requests[1:] if th == ("fork0/branch-0" if branch else "main")][-3:]
    assert CLIPPED in said[0] and said[0].endswith(
        "Close now: call done with what you have so far, or rewind to an earlier step to free context; any other call "
        "is refused.")
    assert "| context left: 400 |" in said[1] and "[runtime] Rewound to step" in said[1], said[1]
    assert CLIPPED in said[2] and "Refused" not in said[2]  # out of the reserve: the search ran
    assert tree["main_end"] == "done"
    assert not branch or tree["forks"][0]["branches"][0]["ended_with"] == "done"


def test_search_rewind_cap_per_thread(monkeypatch, tmp_path):
    """OCTOPUS_REWIND_MAX=2: the rewind tool's description names the cap; main's third rewind is refused (REWIND_SPENT,
    nothing rewound, the thread goes on) while an invalid one counts for nothing; a branch has its own two; the
    main-first control goes on from main's two (its rewind refused)."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.runtime.core import REWIND_SPENT
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_REWIND_TOOL="1",
                           OCTOPUS_REWIND_MAX="2", OCTOPUS_PLAN_CONTROL="1").items():
        monkeypatch.setenv(key, value)
    back = ("rewind", {"to_step": 1, "note": "back"})
    scripts = {"main": [("search", {"query": "Zorblax"}, 100), (*back, 200),
                        ("rewind", {"to_step": 99, "note": "no such step"}, 210), (*back, 220), (*back, 230),
                        ("fork", {"tasks": ["Find the birth year"]}, 240), ("done", {"answer": "1971"}, 300)],
               "fork0/branch-0": [("search", {"query": "Mira Okonkwo"}, 1000),
                                  ("rewind", {"to_step": 3, "note": "back"}, 1100),
                                  ("rewind", {"to_step": 3, "note": "back"}, 1110), ("done", {"answer": "1971"}, 1200)],
               "control": [("rewind", {"to_step": 2, "note": "back"}, 250), ("done", {"answer": "1971"}, 260)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    rewind = next(t for t in tree["tools"] if t["function"]["name"] == "rewind")["function"]["description"]
    assert "at most 2 times per thread" in rewind
    said = [r[-1] for th, r in requests[1:] if th == "main"]
    assert "rewind needs to_step" in said[2]
    assert said[4].endswith("| step 2]\n" + REWIND_SPENT.format(2)), said[4]  # step 2: the refused reply's own
    assert tree["main_end"] == "done" and [r["to_step"] for r in tree["rewinds"]] == [1, 1]
    branch = tree["forks"][0]["branches"][0]
    assert branch["ended_with"] == "done" and [r["to_step"] for r in branch["rewinds"]] == [3, 3]
    assert [r[-1] for th, r in requests if th == "control"][1].endswith(REWIND_SPENT.format(2))


def test_search_reserve_note_offers_rewind_only_while_rewinds_are_left(monkeypatch, tmp_path):
    """OCTOPUS_REWIND_MAX=1, a linear run in its reserve (OCTOPUS_THREAD_BUDGET=400, OCTOPUS_ANSWER_RESERVE=300): the
    clipped result's note offers rewind while the thread has one left; after its rewind the notes (clipped and
    refused) no longer do, and a second rewind is refused."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.runtime.core import CLIPPED, REWIND_SPENT
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_THREAD_BUDGET="400",
                           OCTOPUS_ANSWER_RESERVE="300", OCTOPUS_REWIND_TOOL="1", OCTOPUS_REWIND_MAX="1",
                           OCTOPUS_LINEAR="1").items():
        monkeypatch.setenv(key, value)
    big = ("search", {"query": "Zorblax Mira Okonkwo garden robot"})
    scripts = {"main": [(*big, 100), ("rewind", {"to_step": 0, "note": "back"}, 190), (*big, 120),
                        ("search", {"query": "Okonkwo"}, 130), ("rewind", {"to_step": 0, "note": "back"}, 140),
                        ("done", {"answer": "1971"}, 150)]}
    policy, requests = _scripted(dry_run, scripts, nofork="main")
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    said = [r[-1] for _, r in requests[1:]]
    assert CLIPPED in said[0] and said[0].endswith("so far, or rewind to an earlier step to free context; any other "
                                                   "call is refused.")
    assert "[runtime] Rewound to step 0" in said[1]
    assert CLIPPED in said[2] and said[2].endswith("Close now: call done with what you have so far; any other call is "
                                                   "refused.")
    assert said[3].endswith("Close now: call done with what you have so far. Until you do, every other call is "
                            "refused.")
    assert said[4].endswith(REWIND_SPENT.format(1))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "done"


def test_search_branch_searches_count_after_a_rewind(monkeypatch, tmp_path):
    """A branch that searched, rewound to where it started and answered made a search call of its own: search_outcome
    counts it (rl.branch_credit=gate's searched), though its history no longer holds it."""
    from octopus.rl.group import search_branch_ok
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_REWIND_TOOL="1").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("fork", {"tasks": ["Find the birth year"]}, 100), ("done", {"answer": "1971"}, 300)],
               "fork0/branch-0": [("search", {"query": "Mira Okonkwo"}, 1000),
                                  ("rewind", {"to_step": 1, "note": "noted 1971"}, 1100),
                                  ("done", {"answer": "1971"}, 1200)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    branch, = search_outcome(trial, rows)["branches"]
    assert branch["searched"] == 1 and search_branch_ok(branch)


def test_search_reply_that_forks_cannot_also_answer(monkeypatch, tmp_path):
    """A main reply that calls fork and done: the fork runs, done is refused (main's reply that forked would be in no
    leaf of main), and main answers in its next reply, with the branch's finding."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [([("fork", {"tasks": ["Find the birth year"]}), ("done", {"answer": "1968"})], None, 100),
                        ("done", {"answer": "1971"}, 300)],
               "fork0/branch-0": [("done", {"answer": "1971"}, 1000)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    assert tree["main_end"] == "done" and len(tree["forks"]) == 1
    assert "done refused: a reply whose fork started cannot also call done" in tree["main"][4]["content"]
    assert json.loads((trial / "verifier/answers.json").read_text())["main"]["answer"] == "1971"
    fork = next(t for t in tree["tools"] if t["function"]["name"] == "fork")["function"]["description"]
    assert "A reply whose fork starts cannot also call done:" in fork


def test_search_refused_fork_leaves_done_alone(monkeypatch, tmp_path):
    """A branch (it cannot fork: OCTOPUS_FORK_DEPTH 1) replies fork and done: its fork is refused, nothing extends that
    reply, so its done closes it in that reply."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("fork", {"tasks": ["Find the birth year"]}, 100), ("done", {"answer": "1971"}, 300)],
               "fork0/branch-0": [([("fork", {"tasks": ["deeper"]}), ("done", {"answer": "1971"})], None, 1000),
                                  ("done", {"answer": "1971"}, 1100)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    branch = json.loads((trial / "agent/tree.json").read_text())["forks"][0]["branches"][0]
    assert branch["ended_with"] == "done" and [t for t, _ in requests].count("fork0/branch-0") == 1


def test_search_branch_cap_refuses_a_fork_past_it(monkeypatch, tmp_path):
    """OCTOPUS_MAX_BRANCHES=3: main's first fork of two starts; its second of two would make four and is refused with
    the number left, nothing started: one, which a search fork of one task may still take."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_MAX_BRANCHES="3").items():
        monkeypatch.setenv(key, value)
    fork = ("fork", {"tasks": ["Find the founder", "Find the birth year"]}, 100)
    close = [("done", {"answer": "x"}, 200)]
    policy, requests = _scripted(dry_run, {"main": [fork, fork, ("done", {"answer": "1971"}, 300)],
                                           "fork0/branch-0": close, "fork0/branch-1": close})
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    assert len(tree["forks"]) == 1 and tree["main_end"] == "done"
    refusal = [results for thread, results in requests if thread == "main"][2][-1]
    assert refusal.endswith("]\nfork refused: 2 more branches would pass the episode's limit of 3 branches; 1 left: "
                            "fork fewer branches or finish this task yourself")


def test_search_system_prompt_states_the_branch_limits_in_effect(monkeypatch):
    """The fork run's system prompt states the knob values in effect: no branch cap leaves it out, a fork depth D > 1
    says how deep forks nest, no depth limit (0) leaves it out; the linear prompt (control, linear run) names no
    branch."""
    agent, _ = _search_modules(monkeypatch)
    for key, value in dict(OCTOPUS_MAX_BRANCHES="0", OCTOPUS_FORK_DEPTH="2", OCTOPUS_BRANCH_MAX_TURNS="7").items():
        monkeypatch.setenv(key, value)
    ws = agent.SearchWorkspace()
    assert ("Limits: forks nest at most 2 levels below the main thread; a branch has at most 7 turns. Every tool "
            "result") in ws.system(False)
    assert "branch" not in ws.system(True)
    monkeypatch.setenv("OCTOPUS_FORK_DEPTH", "0")
    assert "Limits: a branch has at most 7 turns. Every tool result" in ws.system(False)


def test_search_has_no_commit_or_abort(monkeypatch, tmp_path):
    """Search protocol v6: commit and abort are no search tools; main's call of either is answered as any unknown tool
    (146__9a614ef committed and was told "commit recorded"), and main answers with done."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    monkeypatch.setenv("OCTOPUS_MODEL", "stub")
    monkeypatch.setenv("OCTOPUS_HTTP_ATTEMPTS", "1")
    policy, requests = _scripted(dry_run, {"main": [("commit", {"branch_id": "1", "summary": "1971"}, 100),
                                                    ("abort", {"branch_id": "A"}, 200), ("done", {"answer": "1971"}, 300)]})
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    said = [results[-1] for _, results in requests[1:]]
    refusal = "]\n{} is not a tool. Available tools: done, fork, open, search."
    assert said[0].endswith(refusal.format("commit")) and said[1].endswith(refusal.format("abort"))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "done"


@pytest.mark.parametrize("reader, control, v", [
    ("born 1971", "born 1968", 2.0),  # read helped
    ("born 1971", "1971, from d2", 0.0),  # both correct
    ("born 1968", "born 1971", -2.0)])  # read hurt
def test_search_read_control_is_graded_and_valued(monkeypatch, tmp_path, reader, control, v):
    """OCTOPUS_READ_CONTROL in search: main forks twice; the second fork's branch reads the first fork's ended branch by
    its name (fork0/branch-0) and closes with done; after the main-first control, its read control (no mount) closes
    with done. A read control's done is a branch's (the control's graded answer stays its
    own). The read control and its reader are graded on their own (answers.json under their leaf keys, the record's
    passed): each one's done answer judged against the question's answer, the judge called for those two leaves only;
    every other branch is ungraded (None). read_adv's v = 2·passed(reader) − 2·passed(control): the read's value on
    the answer's scale."""
    agent, dry_run = _search_modules(monkeypatch)
    import collections
    import threading
    turns, lock, mounted, judged = collections.Counter(), threading.Lock(), [], []
    scripts = {
        "main": [("fork", {"tasks": ["Find who founded Zorblax Inc."]}),
                 ("fork", {"tasks": ["Find the birth year of Mira Okonkwo"]}), ("done", {"answer": "1971"})],
        "control": [("search", {"query": "Mira Okonkwo born"}), ("done", {"answer": "1971"})],
        "fork0/branch-0": [("search", {"query": "Zorblax founder"}), ("open", {"docid": "d1"}),
                           ("done", {"answer": "Mira Okonkwo", "evidence": "d1"})],
        "fork1/branch-0": [("search", {"query": "Mira Okonkwo"}), ("read_thread", {"thread": "fork0/branch-0"}),
                           ("done", {"answer": reader, "evidence": "d2"})],
        "readctl": [("done", {"answer": control})]}

    class Judge(_Json):
        def do_POST(self):
            prompt = self.body()["messages"][0]["content"]
            response = prompt.split("[response]: ")[1].split("\n\n[correct_answer]")[0]
            judged.append(response)
            self.reply(200, {"choices": [{"message": {"content": f"correct: {'yes' if '1971' in response else 'no'}"}}]})

    class Policy(_Json):
        def do_POST(self):
            body = self.body()
            fork = "fork" in {t["function"]["name"] for t in body.get("tools") or []}
            thread = body.get("session_id") or ("control" if not fork else "main" if turns["main"] < 3 else "readctl")
            with lock:
                turns[thread] += 1
                turn = turns[thread]
            if "mount" in body:
                mounted.append(f"{thread}#{turn}")
            script = scripts[thread]
            calls = dry_run._call(script[turn - 1][0], **script[turn - 1][1]) if turn <= len(script) else None
            self.reply(200, {"id": f"{thread}#{turn}", "object": "chat.completion",
                             "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                             "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                                          "message": {"role": "assistant", "content": "" if calls else "no tool call",
                                                      **({"tool_calls": calls} if calls else {})}}]})
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(Policy) + "/v1/chat/completions")
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_PLAN_CONTROL="1", OCTOPUS_CURSOR="1",
                           OCTOPUS_READ_TOOL="1", OCTOPUS_READ_CONTROL="1", OCTOPUS_JUDGE_URL=serve(Judge) + "/v1").items():
        monkeypatch.setenv(key, value)
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    assert mounted == ["fork1/branch-0#3"] and turns["readctl"] == 1
    assert json.loads((trial / "verifier/answers.json").read_text())["control"]["answer"] == "1971"
    # main's and the control's "1971" are exact matches; branch-0 is no reader
    assert judged == [reader, control]
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    out = search_outcome(trial, rows)
    assert {b["key"]: b["passed"] for b in out["branches"]} == {"fork0-branch-0": None,
                                                                 "fork1-branch-0": "1971" in reader}
    assert [{k: c[k] for k in ("of", "read_rid", "ended_with", "passed")} for c in out["read_controls"]] == [
        {"of": "fork1-branch-0", "read_rid": "fork1/branch-0#2", "ended_with": "done", "passed": "1971" in control}]
    [episode] = search_group_step([out])
    assert episode["read_adv"] == {"fork1/branch-0#2": {"calls": ["read_thread"], "call": 0, "v": v}}
    assert all(b["A_branch"] == episode["A_main"] for b in episode["branches"])


@pytest.mark.parametrize("then", ["text", "done"])
@pytest.mark.parametrize("linear", [False, True])
def test_search_text_reply_without_done_is_reminded_once(monkeypatch, tmp_path, linear, then):
    """main (forked, or a linear run's) and the control reply text without done, the gold answer (search v8: 7.4% of
    forked mains after the join): each is reminded once to close with done (core.REMIND_NOTE). then=text: a second text
    reply ends it with no answer, incorrect (y 0) in the training outcome and in the dev_eval row alike; then=done:
    its done answer is graded (y 1)."""
    from octopus.runtime.core import REMIND_NOTE
    agent, dry_run = _search_modules(monkeypatch)
    after = [None] + ([dry_run._call("done", answer="1971")] if then == "done" else [])  # None: a text reply
    scripts = {"main": [dry_run._call("search", query="Mira Okonkwo born")] + after if linear else
               [dry_run._call("fork", tasks=["Find who founded Zorblax Inc.", "Find the birth year of Mira Okonkwo"])]
               + after,
               "control": [dry_run._call("search", query="Mira Okonkwo born")] + after}

    class Policy(dry_run.StubPolicy):
        def do_POST(self):
            body = self.body()
            fork = "fork" in {t["function"]["name"] for t in body.get("tools") or []}
            thread = body.get("session_id") or ("main" if fork or linear else "control")
            with self.lock:
                self.turns[thread] += 1
                turn = self.turns[thread]
            calls = (scripts[thread][turn - 1] if turn <= len(scripts[thread]) else None) if thread in scripts \
                else dry_run.stub_reply(thread, turn, body["messages"])
            text = "" if calls else "1971"
            self.reply(200, {"id": f"{thread}#{turn}", "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                             "choices": [{"message": {"role": "assistant", "content": text,
                                                      **({"tool_calls": calls} if calls else {})}}]})
    dry_run.StubPolicy.turns.clear()
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(Policy) + "/v1/chat/completions")
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_PLAN_CONTROL="1",
                           **({"OCTOPUS_LINEAR": "1"} if linear else {})).items():
        monkeypatch.setenv(key, value)
    trial = asyncio.run(agent.run_trial({**dry_run.QUESTION, "answer": "1971"}, tmp_path))
    assert json.loads((trial / "result.json").read_text())["exception_info"] is None
    tree = json.loads((trial / "agent/tree.json").read_text())
    y = then == "done"
    assert tree["main_end"] == ("done" if y else "no tool call") and bool(tree["forks"]) != linear
    assert tree["reminded"] == 1 and (linear or tree["control"]["reminded"] == 1)
    assert {"role": "user", "content": REMIND_NOTE.format(calls="done")} in tree["main"]
    graded = ({"answer": "1971", "correct": True, "how": "em"} if y else
              {"answer": None, "correct": False, "how": "none"})
    assert json.loads((trial / "verifier/answers.json").read_text()) == {
        "main": graded, **({} if linear else {"control": graded})}
    out = search_outcome(trial, [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()])
    assert out["y"] == y and (out["control"] is None if linear else out["control"]["y"] == y)
    assert out["reminded"] == {"main": 1, "branches": 0}
    pytest.importorskip("areal")  # search_dev_eval imports octopus.rl.config
    from octopus.eval import search_dev_eval as dev_eval
    row = dev_eval.outcome(out)
    assert row["y"] == y and (row["control"] is None if linear else row["control"]["y"] == y)
    assert row["reminded"] == {"main": 1, "branches": 0}


def test_search_tool_errors(monkeypatch, tmp_path):
    """An unknown docid is the model's mistake (a tool result); a retrieval service that is down marks the episode
    failed, so AReaL rejects it."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    a = agent.SearchAgent(logs_dir=tmp_path)
    observed, stop = asyncio.run(a._dispatch_locked("open", {"docid": "nope"}))
    assert observed.startswith("open failed: HTTP 404") and not stop and a.ws.retrieval_error is None
    observed = asyncio.run(a._dispatch_locked("open", {"docid": "d1", "offset": "x"}))[0]
    assert observed == "open: offset is an integer"
    assert asyncio.run(a._dispatch_locked("done", {"answer": " 1971 "})) == ("Final answer recorded: 1971", True)
    assert a.ws.answers == {"main": "1971"}
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", "http://127.0.0.1:9")  # nothing listens
    assert asyncio.run(a._dispatch_locked("search", {"query": "x"}))[0] == "retrieval service error"
    assert a.ws.retrieval_error.startswith("search: URLError")


def test_search_tool_is_a_fixed_top_10(monkeypatch, tmp_path):
    """search takes only a query: no k in the tool schema; the agent asks the service for CLM's fixed top 10, whatever
    k the model passes."""
    agent, dry_run = _search_modules(monkeypatch)
    bodies = []

    class Recording(dry_run.FakeRetrieval):
        def body(self):
            bodies.append(super().body())
            return bodies[-1]
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(Recording))
    a = agent.SearchAgent(logs_dir=tmp_path)
    (search,) = [t["function"] for t in a._tools() if t["function"]["name"] == "search"]
    assert search["parameters"]["properties"] == {"query": {"type": "string"}} and "top 10" in search["description"]
    asyncio.run(a._dispatch_locked("search", {"query": "Zorblax", "k": 20}))
    assert bodies == [{"query": "Zorblax", "k": 10}]


def test_judge_without_a_verdict_grades_incorrect(monkeypatch, tmp_path):
    """A judge that gives no verdict, asked again for it on the same conversation, is no episode error: the answer is
    graded incorrect ("judge_unparsed", counted in the episode record), so AReaL keeps the episode's group."""
    agent, dry_run = _search_modules(monkeypatch)
    asked = []

    class Mute(_Json):
        def do_POST(self):
            asked.append(self.body()["messages"])
            self.reply(200, {"choices": [{"message": {"content": "reasoning: long\nconfidence: 100"}}]})
    monkeypatch.setenv("OCTOPUS_JUDGE_URL", serve(Mute) + "/v1")
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(dry_run.StubPolicy) + "/v1/chat/completions")
    monkeypatch.setenv("OCTOPUS_MODEL", "stub")
    monkeypatch.setenv("OCTOPUS_HTTP_ATTEMPTS", "1")
    dry_run.StubPolicy.turns.clear()
    trial = asyncio.run(agent.run_trial({**dry_run.QUESTION, "answer": "1972"}, tmp_path))  # 1971: no exact match
    assert json.loads((trial / "result.json").read_text())["exception_info"] is None
    main = json.loads((trial / "verifier/answers.json").read_text())["main"]
    assert main == {"answer": "1971", "correct": False, "how": "judge_unparsed"}
    from octopus.envs.search.reward import JUDGE_FOLLOW_UP
    assert len(asked) == 2 and asked[1][1:] == [{"role": "assistant", "content": "reasoning: long\nconfidence: 100"},
                                                {"role": "user", "content": JUDGE_FOLLOW_UP}]
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    assert search_outcome(trial, rows)["judge_unparsed"] == 1


def _eval_episodes(monkeypatch, agent, *overrides):
    """search dev_eval's proxy and episode on stand-ins: proxy.serve yields the first server's /v1 (the stub policy),
    and an episode is SearchEpisodeAgent.run_episode's, its `python -m octopus.envs.search.agent` run in this process
    with the command's environment. The run config: the repo's, as a search run sets it (rl_run.search.env)."""
    import contextlib
    import shlex
    pytest.importorskip("areal")
    # grpo.yaml's ${oc.env:PYTHONPATH} (the launcher's): unset under a bare pytest
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parents[1])))
    from octopus.rl import config, proxy
    from octopus.envs.search import rl

    @contextlib.asynccontextmanager
    async def serve(config, servers, lora, port=0):
        yield f"http://{servers[0]}/v1"

    async def episode(agent, data, url, task_id):
        return await agent.run_episode(data, url, "key", task_id)

    async def run_group(command, **_):
        words = shlex.split(command)
        for word, before in zip(words[1:], words):
            if before == "export":
                monkeypatch.setenv(*word.split("=", 1))
        i = words.index("octopus.envs.search.agent")
        await agent.run_trial(json.loads(Path(words[i + 1]).read_text()), Path(words[i + 2]))
        return 0
    monkeypatch.setattr(proxy, "serve", serve)
    monkeypatch.setattr(proxy, "episode", episode)
    monkeypatch.setattr(rl, "_run_group", run_group)
    return config.load(config.CONFIG, ["workflow=octopus.envs.search.rl.SearchEpisodeAgent", "tokenizer_path=unused",
                                       *overrides])


def test_search_dev_eval_runner_and_bcp_converter(monkeypatch, tmp_path):
    """The eval runner on the dry-run question (fake retrieval, stub policy): one row per (question, sample) with the
    fork and main-first control fields (rl.plan_control=true), a rerun resumes; browsecomp-plus grades with the
    official judge and adds recall, with no question or answer text in its rows; eval_bcp.trial_run turns a trial into
    eval_bcp's input. Branches get the search wording of their task."""
    agent, dry_run = _search_modules(monkeypatch)
    config = _eval_episodes(monkeypatch, agent, "rl.plan_control=true")
    from octopus.eval import search_dev_eval as dev_eval, bcp as eval_bcp
    for key in ("OCTOPUS_JUDGE_URL", "OCTOPUS_JUDGE_MODEL", "OCTOPUS_JUDGE_TEMPERATURE"):
        monkeypatch.setenv(key, "")  # the runner sets them; undone after the test
    monkeypatch.setenv("OCTOPUS_JUDGE_CACHE", str(tmp_path / "judge-cache.jsonl"))  # never used by the runner
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    policy = serve(dry_run.StubPolicy).removeprefix("http://")
    judged = []

    class Judge(_Json):
        def do_POST(self):
            body = self.body()
            assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0.0
            prompt = body["messages"][0]["content"]
            judged.append(prompt)
            yes = "[response]: 1971" in prompt and "[correct_answer]: 1971" in prompt
            self.reply(200, {"choices": [{"message": {"content": f"correct: {'yes' if yes else 'no'}\nconfidence: 80"}}]})

    def run(name, k, out, qs, *extra):
        dry_run.StubPolicy.turns.clear()
        args = dev_eval.parse(["--set", name, "--dataset-dir", "-", "--servers", policy, "--lora", "rq1-v10", "--out",
                               str(out), "--judge-url", serve(Judge) + "/v1", "--k", str(k), *extra])
        asyncio.run(dev_eval.main_async(args, config, qs))
        return [json.loads(x) for x in out.read_text().splitlines()]

    (tmp_path / "val.jsonl").write_text(json.dumps({"qid": "dry-1", "sample": 0, "error": "URLError"}) + "\n")
    rows = run("validation", 1, tmp_path / "val.jsonl", [dry_run.QUESTION], "--jobs-dir", str(tmp_path / "jobs"))
    assert len(rows) == 2 and judged == []  # the infra failure is run again; exact match: no judge call
    assert dev_eval.summary(rows).endswith("infra failures (not scored): 0")  # the retry supersedes it
    assert rows[1]["judge_unparsed"] == 0 and "judge_unparsed" not in dev_eval.summary(rows)
    assert " | judge_unparsed 2\n" in dev_eval.summary([{**rows[1], "judge_unparsed": 2}])
    row = rows[1]
    assert "recall" not in row
    assert (row["task_dir"], row["qid"], row["sample"], row["lora"], row["y"], row["forked"]) == \
        ("search-dry-1", "dry-1", 0, "rq1-v10", True, True)
    control = row["control"]
    assert control == {"kind": "main-first", "y": True, "C": 40.0, "C_wall": control["C_wall"],
                       "wall_s": control["wall_s"], "calls": 2, "prompt_tokens": control["prompt_tokens"],
                       "completion_tokens": 40}
    # the row's totals are the episode tree's (main 2 + two branches 3 each), the control's apart
    assert (row["C"], row["calls"], row["completion_tokens"], row["branches"], row["ended_with"]) == \
        (100.0, 8, 160, 2, "done")
    assert [r["sample"] for r in run("validation", 2, tmp_path / "val.jsonl", [dry_run.QUESTION], "--jobs-dir",
                                     str(tmp_path / "jobs"))] == [0, 0, 1]  # resumed

    tree = json.loads((Path(row["trial"]) / "agent/tree.json").read_text())
    task = next(m["content"] for m in tree["forks"][0]["branches"][0]["messages"] if "Your task:" in str(m["content"]))
    assert agent.SearchWorkspace.branch_contract in task and "files" not in task
    assert "view" not in task and "rewind" not in task
    # the first fork's branches: no thread is readable yet, so no read_thread card (read_when_readable)
    assert agent.SearchWorkspace().read_card not in task

    bcp = {**dry_run.QUESTION, "answer": "nineteen seventy-one", "evidence_docids": ["d2", "d3"], "gold_docids": ["d2"]}
    (row,) = run("browsecomp-plus", 1, tmp_path / "bcp.jsonl", [bcp], "--keep-trials")
    jobs = Path(row["trial"]).parents[1]  # the decrypted question only on tmpfs, in a private dir
    assert jobs.parent == Path("/dev/shm") and jobs.stat().st_mode & 0o777 == 0o700
    assert not (tmp_path / "judge-cache.jsonl").exists() and sum(p.is_dir() for p in (tmp_path / "jobs").iterdir()) == 2
    assert row["y"] is False and len(judged) == 2  # main and control, no exact match: the official prompt
    assert all(p.startswith("Judge whether the following [response]") for p in judged)
    assert (row["recall"], row["open_recall"], row["open_gold_recall"]) == (0.5, 0.5, 1.0)
    text = (tmp_path / "bcp.jsonl").read_text()
    assert dry_run.QUESTION["question"] not in text and '"answer"' not in text
    assert eval_bcp.trial_run(row["trial"]) == {"query_id": "dry-1", "response": "1971",
                                                "retrieved_docids": ["d1", "d2"], "opened_docids": ["d1", "d2"]}
    shutil.rmtree(jobs)
    (row,) = run("browsecomp-plus", 1, tmp_path / "bcp2.jsonl", [bcp])
    assert row["y"] is False and not Path(row["trial"]).parent.exists()  # removed when the run ends
    base = ["--set", "browsecomp-plus", "--dataset-dir", "-", "--servers", policy, "--lora", "rq1-v10", "--out", "o"]
    for extra in (["--judge-url", "u", "--jobs-dir", str(tmp_path / "jobs")],  # no persistent trial dirs
                  []):  # no judge and no kept trials: nothing left to grade
        with pytest.raises(SystemExit):
            dev_eval.parse(base + extra)


def test_search_dev_eval_knobs(monkeypatch, tmp_path):
    """Every request carries the training episode's settings (SearchEpisodeAgent's reply cap rl.MAX_NEW_TOKENS and
    chat template) and the run's (here OCTOPUS_SEARCH_BUDGET and OCTOPUS_LINEAR=1, a single agent: no fork in the
    system prompt or the tool list); --qids runs exactly these, in order; no --judge-url: exact match only."""
    agent, dry_run = _search_modules(monkeypatch)
    config = _eval_episodes(monkeypatch, agent)
    from octopus.eval import search_dev_eval as dev_eval
    from octopus.envs.search.rl import MAX_NEW_TOKENS
    monkeypatch.setenv("OCTOPUS_LINEAR", "1")
    monkeypatch.setenv("OCTOPUS_SEARCH_BUDGET", "500")
    monkeypatch.setenv("OCTOPUS_JUDGE_URL", "http://127.0.0.1:9/v1")  # dropped: no --judge-url
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    bodies = []

    class Policy(dry_run.StubPolicy):
        def body(self):
            bodies.append(super().body())
            return bodies[-1]
    dry_run.StubPolicy.turns.clear()
    (tmp_path / "qids.txt").write_text("dry-1\n")
    other = {**dry_run.QUESTION, "qid": "dry-2"}
    args = dev_eval.parse(["--set", "validation", "--dataset-dir", "-", "--servers", serve(Policy).removeprefix("http://"),
                           "--lora", "rq1-v0", "--out", str(tmp_path / "o.jsonl"), "--k", "1", "--jobs-dir",
                           str(tmp_path / "jobs"), "--qids", str(tmp_path / "qids.txt")])
    asyncio.run(dev_eval.main_async(args, config, [other, dry_run.QUESTION]))
    (row,) = [json.loads(x) for x in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert (row["qid"], row["y"], row["forked"]) == ("dry-1", True, False) and len(bodies) == 2
    for body in bodies:
        assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == MAX_NEW_TOKENS
        assert body["model"] == "policy"
        assert not {"fork", "join", "commit"} & {t["function"]["name"] for t in body["tools"]}
    system = bodies[0]["messages"][0]["content"]
    assert "budget of 500" in system and agent._LINEAR in system and agent._FORK not in system
    assert "branches" not in system  # a single agent has none
    monkeypatch.delenv("OCTOPUS_LINEAR")  # the training control's prompt: a linear run's, no branches either
    assert agent.SearchWorkspace.system(None, True) == system and "shared by" in agent.limits_note(True)


def test_search_dev_eval_refuses_a_code_workflow(monkeypatch, tmp_path):
    """A config whose workflow runs no search episodes (the repo's default, HarborEpisodeAgent) is refused before any
    episode, not run as plan episodes on search rows."""
    pytest.importorskip("areal")
    # grpo.yaml's ${oc.env:PYTHONPATH} (the launcher's): unset under a bare pytest
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parents[1])))
    from octopus.rl import config
    from octopus.eval import search_dev_eval as dev_eval
    args = dev_eval.parse(["--set", "validation", "--dataset-dir", "-", "--servers", "h:1", "--out", "o"])
    with pytest.raises(SystemExit, match="runs no search episodes"):
        asyncio.run(dev_eval.main_async(args, config.load(config.CONFIG), []))


def test_rq1_yaml_has_every_rl_key():
    """Hydra's struct config: an rl.KEY override (rl_run.search.env: rl.search_questions, rl.search_split,
    rl.filter_zero_variance) fails unless grpo.yaml lists KEY."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "octopus/rl/config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLPoolConfig")
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert fields <= set(yaml.safe_load((root / "octopus/rl/grpo.yaml").read_text())["rl"])


def test_search_episode_python(monkeypatch, tmp_path):
    """octopus/envs/search/rl.py runs the episode with $OCTOPUS_SEARCH_PYTHON when set (harbor a uv tool shim: no python
    next to it), else the python next to the harbor CLI."""
    pytest.importorskip("areal")  # search/rl.py imports octopus.rl.config
    workflow = types.ModuleType("octopus.rl.workflow")

    async def run_group(command, **kw):
        return await (await asyncio.create_subprocess_exec("bash", "-c", command, **kw)).wait()
    workflow.HARBOR, workflow.CODE, workflow.EpisodeAgent, workflow.GROUP_STEPS = tmp_path, tmp_path, object, {}
    workflow._run_group, workflow.trial_outcome = run_group, lambda job_dir, build: job_dir
    monkeypatch.setitem(sys.modules, "octopus.rl.workflow", workflow)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=None))
    monkeypatch.delitem(sys.modules, "octopus.envs.search.rl", raising=False)
    monkeypatch.delattr("octopus.envs.search.rl", raising=False)  # imported fresh on the stand-in, not the package's
    from octopus.envs.search import rl
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/env.sh").write_text("")
    python = tmp_path / "venv-python"
    python.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/ran"\n')
    python.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no harbor CLI on PATH
    monkeypatch.setenv("OCTOPUS_SEARCH_PYTHON", str(python))
    episode = object.__new__(rl.SearchEpisodeAgent)
    episode.jobs_dir, episode.model = tmp_path / "jobs", "policy"
    episode.model_env = lambda base_url, api_key: {}  # EpisodeAgent's (stubbed here)
    episode.jobs_dir.mkdir()
    q = {"question": "?", "answer": "a", "plan_control": False}
    job_dir = asyncio.run(episode.run_episode({"qid": "q1", **q}, "http://x/v1", "k"))
    assert (tmp_path / "ran").read_text().split()[:2] == ["-m", "octopus.envs.search.agent"] and job_dir.is_dir()
    (tmp_path / "bin").mkdir()  # unset: the python next to the harbor CLI, as before
    python.rename(tmp_path / "bin/python")
    (tmp_path / "bin/harbor").write_text("")
    (tmp_path / "bin/harbor").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/usr/bin:/bin")
    monkeypatch.delenv("OCTOPUS_SEARCH_PYTHON")
    asyncio.run(episode.run_episode({"qid": "q2", **q}, "http://x/v1", "k"))
    assert "search-q2-" in (tmp_path / "bin/ran").read_text()


def test_reward_exact_match_and_judge(monkeypatch, tmp_path):
    assert normalize("The  Beatles!") == "beatles"
    assert exact_match("1971.", "1971") and exact_match("the Beatles", "Beatles") and not exact_match("1970", "1971")
    for response, gold in (("35", "3.5"), ("4", "-4"), ("130", "1:30"), ("12 billion", "$1.2 billion")):
        assert not exact_match(response, gold)  # different numbers, the same once punctuation is dropped
    assert not exact_match("", "") and grade("q", None, "1971") == (False, "none")
    # a \boxed{} answer (29% of OpenResearcher questions ask for one) matches its gold
    assert exact_match(r"\boxed{1971}", "1971") and exact_match(r"$\boxed{3.5}$", "3.5")
    assert normalize(r"\boxed{\text{SS Normandie}}") == "ss normandie" and not exact_match(r"\boxed{1970}", "1971")
    monkeypatch.delenv("OCTOPUS_JUDGE_URL", raising=False)
    assert grade("q", "nineteen seventy-one", "1971") == (False, "none")  # no judge: exact match only

    calls, served = [], ["/models/Qwen3-32B"]  # the model behind the served name judge-m

    class Judge(_Json):
        def do_GET(self):  # as SGLang 0.5.19 serves them: /v1/models' root is the served name, /model_info has the path
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": "judge-m", "root": "judge-m"}]})
            else:
                self.reply(*((200, {"model_path": served[0], "served_model_name": "judge-m"})
                             if self.path == "/model_info" else (404, {"detail": "Not Found"})))

        def do_POST(self):
            body = self.body()
            calls.append(body)
            prompt = body["messages"][0]["content"]
            yes = "nineteen seventy-one" in prompt or "MCMLXXI" in prompt
            verdict = "" if "MCMLXXI" in prompt and len(body["messages"]) == 1 else f"correct: {'yes' if yes else 'no'}\n"
            self.reply(200, {"id": "j", "choices": [{"message": {"role": "assistant", "content":
                             f"extracted_final_answer: x\nreasoning: r\n{verdict}confidence: 90"}}]})
    monkeypatch.setenv("OCTOPUS_JUDGE_URL", serve(Judge) + "/v1")
    monkeypatch.setenv("OCTOPUS_JUDGE_MODEL", "judge-m")
    monkeypatch.setenv("OCTOPUS_JUDGE_CACHE", str(tmp_path / "judge.jsonl"))
    monkeypatch.delenv("OCTOPUS_JUDGE_TEMPERATURE", raising=False)
    assert grade("When?", "nineteen seventy-one", "1971") == (True, "judge")
    assert grade("When?", "1970", "1971") == (False, "judge")
    assert grade("When?", "nineteen seventy-one", "1971") == (True, "judge")  # from the cache
    assert grade("When?", "1971", "1971") == (True, "em")  # no judge call
    assert len(calls) == 2 and calls[0]["model"] == "judge-m" and calls[0]["temperature"] == 0  # training: greedy
    from octopus.envs.search import reward
    served[0] = "/models/Qwen3.5-27B"  # another judge under the same served name: its verdicts are its own
    reward.judge_identity.cache_clear()  # a new episode process
    assert grade("When?", "nineteen seventy-one", "1971") == (True, "judge") and len(calls) == 3
    assert [json.loads(x)["key"][1] for x in (tmp_path / "judge.jsonl").read_text().splitlines()] == [
        "/models/Qwen3-32B", "/models/Qwen3-32B", "/models/Qwen3.5-27B"]
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "[correct_answer]: 1971" in calls[0]["messages"][0]["content"]
    # a first reply without a verdict: the follow-up on the same conversation gives it, and it is cached
    assert grade("When?", "MCMLXXI", "1971") == (True, "judge") and len(calls) == 5
    assert [m["role"] for m in calls[-1]["messages"]] == ["user", "assistant", "user"]
    assert grade("When?", "MCMLXXI", "1971") == (True, "judge") and len(calls) == 5


YS = [1, 1, 1, 0, 0, 0, 0, 0]


def _group() -> list[dict]:
    """8 search episodes built from group.py's contract alone: y = YS, the even ones forked (two branches and a control
    each, only episode 0's control correct); episode 0's branch-1 read (correct) and its read control (wrong) are the
    only leaves graded on their own."""
    return [{
        "y": bool(y), "C": 100.0, "forked": i % 2 == 0,
        "control": {"y": i == 0, "C": 40.0, "kind": "main-first"} if i % 2 == 0 else None,
        "branches": [{"key": f"fork0-branch-{b}", "passed": True if i == b == 0 else None} for b in range(2)]
        if i % 2 == 0 else [],
        "read_controls": [{"of": "fork0-branch-0", "read_rid": "b1", "read_call": 0, "read_calls": ["read_thread"],
                           "forked_after": False, "control_forked": False, "passed": False}] if i == 0 else [],
        "requests": []}
        for i, y in enumerate(YS)]


def test_search_group_advantages_group_of_8():
    """G = 8: R = 2y-1, leave-one-out A over the 8 mains (no std; the forking episodes' controls are graded, never
    trained); every thread of an episode (main, branches) takes its A_main."""
    episodes = search_group_step(_group())
    R = [2 * y - 1 for y in YS]
    assert [e["A_main"] for e in episodes] == pytest.approx([r - (sum(R) - r) / 7 for r in R])
    for e in episodes:
        assert all(b["A_branch"] == e["A_main"] for b in e["branches"])
    assert episodes[0]["read_adv"] == {"b1": {"calls": ["read_thread"], "call": 0, "v": 2.0}}  # (2·1−1) − (2·0−1)
    stats = plan_group_stats(episodes)
    assert stats["fork_vs_control"] == {"main-first": [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}
    assert (stats["read_adv"], stats["read_adv_n"]) == (2.0, 1)

    episode = episodes[0]
    episode["requests"] = [("m1", "main", 10, 5), ("b1", "fork0/branch-0", 20, 5), ("b2", "fork0/branch-1", 20, 5),
                           ("m2", "main", 30, 5), ("c1", "control", 10, 5)]
    leaves = [SimpleNamespace(interaction_id=rid, _cache={"input_ids": _Ids(range(40))})
              for rid in ("m2", "b1", "b2", "c1")]
    rows = assign_threads(leaves, episode)
    assert [r["adv"] for r in rows] == [episode["A_main"]] * 3 + [0.0]  # the control: graded, never trained
    assert [r["own_start"] for r in rows] == [10, 20, 20, 40] and all(r["spans"] == [] for r in rows)


def test_search_dev_eval_signals_keep_nohup():
    """dev_eval.main turns SIGTERM / SIGHUP into Ctrl-C (the tmpfs trial dirs are removed), but a SIGHUP ignored by
    nohup stays ignored: an ssh disconnect does not end the run."""
    import signal
    import subprocess
    pytest.importorskip("areal")
    code = ("import signal, sys, types\n"
            "from octopus.eval import search_dev_eval as dev_eval\n"
            "dev_eval.parse = lambda: types.SimpleNamespace(set=None, dataset_dir=None)\n"
            "dev_eval.questions, dev_eval.main_async, dev_eval.asyncio.run, dev_eval.eval_config = (lambda *a: None,) * 4\n"
            "dev_eval.main()\n"
            "print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN,"
            " signal.getsignal(signal.SIGTERM) is signal.default_int_handler)")
    root = Path(__file__).resolve().parents[1]
    run = lambda **kw: subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,  # noqa: E731
                                      check=True, **kw).stdout.split()
    assert run(preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)) == ["True", "True"]  # nohup
    assert run() == ["False", "True"]


def test_top_p_is_an_evaluations_alone():
    """Training's episode requests carry only gconfig's temperature (the trainer's logprobs assume top_p 1.0); an
    evaluation's (search_dev_eval: rl.EVAL_SAMPLING) carry its temperature and top_p."""
    from octopus.rl.run_config import episode_env, request_kwargs
    config = SimpleNamespace(gconfig=SimpleNamespace(temperature=1.0, max_new_tokens=8192),
                             rollout=SimpleNamespace(agent=SimpleNamespace(chat_template_kwargs={})))
    assert "top_p" not in request_kwargs(config)  # training's agent kwargs (workflow_kwargs) carry none
    assert json.loads(episode_env("policy", 1.0, 4096, {})["OCTOPUS_SAMPLING_JSON"]) == {"temperature": 1.0}
    assert json.loads(episode_env("policy", 0.7, 4096, {}, 0.95)["OCTOPUS_SAMPLING_JSON"]) == {"temperature": 0.7,
                                                                                               "top_p": 0.95}


def test_search_agent_caps_replies_and_takes_eval_sampling(tmp_path):
    """SearchEpisodeAgent caps every reply at rl.MAX_NEW_TOKENS whatever gconfig.max_new_tokens; training's episodes
    sample at gconfig's temperature alone, search_dev_eval's (its kwargs plus rl.EVAL_SAMPLING) at 0.7 / 0.95."""
    pytest.importorskip("areal")
    from octopus.envs.search.rl import EVAL_SAMPLING, MAX_NEW_TOKENS, SearchEpisodeAgent
    kw = dict(jobs_dir=str(tmp_path), temperature=1.0, max_new_tokens=8192, chat_template_kwargs={})
    train = SearchEpisodeAgent(**kw).model_env("http://h/v1", "k")
    assert train["OCTOPUS_MAX_TOKENS"] == str(MAX_NEW_TOKENS) == "4096"
    assert json.loads(train["OCTOPUS_SAMPLING_JSON"]) == {"temperature": 1.0}
    evaluation = SearchEpisodeAgent(**{**kw, **EVAL_SAMPLING}).model_env("http://h/v1", "k")
    assert json.loads(evaluation["OCTOPUS_SAMPLING_JSON"]) == {"temperature": 0.7, "top_p": 0.95}


TABLES = {"numeric": "| year | value |\n" + "".join(f"| {1900 + i} | {i * 37 % 1000}.{i % 10} |\n" for i in range(400)),
          "cjk": "米拉·奥孔科沃于一九七一年出生，是佐布拉克斯公司的创始人。" * 150}


@pytest.mark.parametrize("kind", sorted(TABLES))
def test_clipped_result_leaves_the_closing_turn_room(monkeypatch, tmp_path, kind):
    """clip_results counts the policy's tokens (here the stub's, one a UTF-8 byte): a numeric table or CJK text past
    the room above the reserve is cut to fit it, and the linear main, whose prompt_tokens are its messages' tokens,
    still answers with done inside OCTOPUS_THREAD_BUDGET (a character estimate would let them through)."""
    agent, dry_run = _search_modules(monkeypatch)
    from octopus.runtime.core import MESSAGE_TOKENS, clip_results, policy_tokenizer
    tok = policy_tokenizer(dry_run.stub_tokenizer())
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)  # noqa: E731
    (cut,) = clip_results([TABLES[kind]], 1000, "close now")
    assert count(cut) + MESSAGE_TOKENS <= 1000 < count(TABLES[kind])

    class Retrieval(dry_run.FakeRetrieval):
        def do_POST(self):
            self.body()
            self.reply(200, {"results": [{"docid": "t1", "title": "Table", "url": "", "snippet": TABLES[kind]}]})

    class Policy(_Json):  # prompt_tokens: the tokens of every message, as a server would count them
        turns = []

        def do_POST(self):
            body = self.body()
            prompt = sum(count(str(m.get("content") or "")) + 10 for m in body["messages"])
            self.turns.append(prompt)
            name, args = [("search", {"query": "table"}), ("done", {"answer": "1971"})][len(self.turns) - 1]
            self.reply(200, {"id": f"m{len(self.turns)}", "usage": {"prompt_tokens": prompt, "completion_tokens": 20},
                             "choices": [{"message": {"role": "assistant", "content": "",
                                                      "tool_calls": dry_run._call(name, **args)}}]})
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(Retrieval))
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(Policy) + "/v1/chat/completions")
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_LINEAR="1",
                           OCTOPUS_THREAD_BUDGET="3000", OCTOPUS_ANSWER_RESERVE="500").items():
        monkeypatch.setenv(key, value)
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    assert json.loads((trial / "agent/tree.json").read_text())["main_end"] == "done"
    assert Policy.turns[1] - Policy.turns[0] < 3000 - 400  # the clipped result left the reserve nearly whole


def test_answer_reserve_needs_the_policy_tokenizer(monkeypatch):
    from octopus.runtime.settings import answer_reserve
    monkeypatch.setenv("OCTOPUS_ANSWER_RESERVE", "2048")
    monkeypatch.delenv("OCTOPUS_TOKENIZER", raising=False)
    with pytest.raises(RuntimeError, match="OCTOPUS_TOKENIZER"):
        answer_reserve()
    monkeypatch.setenv("OCTOPUS_ANSWER_RESERVE", "0")
    assert answer_reserve() == 0


def _slow(policy, thread: str, seconds: float):
    """The scripted policy answering each request of `thread` after `seconds` (the other threads run meanwhile)."""
    import time

    class Slow(policy):
        def body(self):
            body = super().body()
            if body.get("session_id") == thread:
                time.sleep(seconds)
            return body
    return Slow


NESTED = "fork1/branch-0/fork0/branch-0.0"


def _nested_episode(monkeypatch, tmp_path):
    """OCTOPUS_FORK_DEPTH=2: main forks t1, then t2 and t3; fork1/branch-0 forks again and answers from its branch
    NESTED (depth 2), which reads fork0's ended branch and searches. (agent, trial, tree, requests)."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_CURSOR="1",
                           OCTOPUS_FORK_DEPTH="2").items():
        monkeypatch.setenv(key, value)
    s = lambda q, p: ("search", {"query": q}, p)  # noqa: E731
    scripts = {"main": [("fork", {"tasks": ["t1"]}, 100), ("fork", {"tasks": ["t2", "t3"]}, 200),
                        ("done", {"answer": "1971"}, 900)],
               "fork0/branch-0": [s("Zorblax", 150), ("done", {"answer": "x"}, 160)],
               "fork1/branch-0": [("fork", {"tasks": ["deep"]}, 300), ("done", {"answer": "Mira Okonkwo"}, 800)],
               "fork1/branch-1": [s("Mira born", 300), s("Mira 1971", 320), ("done", {"answer": "1971"}, 400)],
               NESTED: [("read_thread", {"thread": "fork0/branch-0"}, 400), s("Zorblax Inc founder", 450),
                        ("done", {"answer": "Mira Okonkwo"}, 500)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    return agent, trial, json.loads((trial / "agent/tree.json").read_text()), requests


def test_search_nested_branches_read_any_thread_they_may(monkeypatch, tmp_path):
    """A depth-2 branch is offered read_thread at its fork, with a roster of every thread the runtime lets it read: an
    earlier fork's ended branch and its uncle, not main or its forking branch (they wait for its fork); its read of
    the earlier fork's branch mounts."""
    agent, _, tree, requests = _nested_episode(monkeypatch, tmp_path)
    nested = tree["forks"][1]["branches"][0]["forks"][0]["branches"][0]
    task = next(m["content"] for m in nested["messages"] if f"You are {NESTED}," in str(m.get("content")))
    assert ("Other threads at this fork (read_thread names): fork0/branch-0 (task: t1); fork1/branch-1 (task: t3).\n"
            + agent.SearchWorkspace().read_card) in task
    read = [r for th, r in requests if th == NESTED][1][-1].split("\n", 1)[1]
    assert read.startswith("Mounted fork0/branch-0's steps")


def test_search_gate_counts_a_branchs_subtree_searches(monkeypatch, tmp_path):
    """rl.branch_credit gate: fork1/branch-0 searched nothing itself but answered from its branch's search: its
    subtree searched, so it keeps A_main like every other ok branch."""
    _, trial, _, _ = _nested_episode(monkeypatch, tmp_path)
    rows = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    rec = search_outcome(trial, rows)
    searched = {b["key"]: b["searched"] for b in rec["branches"]}
    assert searched == {"fork0-branch-0": 1, "fork1-branch-0": 1, "fork1-branch-0-fork0-branch-0.0": 1,
                        "fork1-branch-1": 2}
    search_group_step([rec, dict(rec, y=0, branches=[dict(b) for b in rec["branches"]])], branch_credit="gate")
    assert rec["A_main"] > 0 and all(b["A_branch"] == rec["A_main"] for b in rec["branches"])


def test_search_branch_is_offered_read_thread_once_a_thread_is_readable(monkeypatch, tmp_path):
    """The first fork's branches start together, so neither is offered read_thread at its fork; once fork0/branch-0
    has ended (readable), the slow fork0/branch-1 is offered it before its next request (a note with the roster and the
    card), and its read mounts."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_CURSOR="1").items():
        monkeypatch.setenv(key, value)
    s = lambda q, p: ("search", {"query": q}, p)  # noqa: E731
    scripts = {"main": [("fork", {"tasks": ["A", "B"]}, 100), ("done", {"answer": "1971"}, 900)],
               "fork0/branch-0": [s("Zorblax", 200), ("done", {"answer": "x"}, 250)],
               "fork0/branch-1": [s("Mira", 200), s("Mira born", 250),
                                  ("read_thread", {"thread": "fork0/branch-0"}, 300), ("done", {"answer": "1971"}, 400)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(_slow(policy, "fork0/branch-1", 0.3)) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    b1 = json.loads((trial / "agent/tree.json").read_text())["forks"][0]["branches"][1]
    task = next(m["content"] for m in b1["messages"] if "You are fork0/branch-1," in str(m.get("content")))
    assert "read_thread" not in task
    offer = [m["content"] for m in b1["messages"] if "Other threads now" in str(m.get("content"))]
    assert offer == ["Other threads now (read_thread names): fork0/branch-0 (task: A).\n"
                     + agent.SearchWorkspace().read_card]
    read = [r for th, r in requests if th == "fork0/branch-1"][3][-1].split("\n", 1)[1]
    assert read.startswith("Mounted fork0/branch-0's steps")


def test_search_deep_thread_ends_by_the_serving_window(monkeypatch, tmp_path):
    """OCTOPUS_FORK_DEPTH=0: no depth limit, the serving window (OCTOPUS_CONTEXT_WINDOW 20000, replies of at most 1000
    tokens) decides. Main forks; its branch starts at 9000 tokens, so its growth cap is 20000 - 9000 - 1000 = 10000
    (not 32768): its tags say so, its search results past that cap less the answer reserve are clipped and it closes
    with done; no request passes the window (prompt + 1000 <= 20000). Its own fork would start branches with 9980
    tokens of room, under the answer reserve (2048) + 8192: refused, no branch made."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_FORK_DEPTH="0",
                           OCTOPUS_CONTEXT_WINDOW="20000", OCTOPUS_MAX_TOKENS="1000").items():
        monkeypatch.setenv(key, value)
    scripts = {"main": [("fork", {"tasks": ["dig"]}, 100), ("done", {"answer": "1971"}, 500)],
               "fork0/branch-0": [("fork", {"tasks": ["deeper"]}, 9000), ("search", {"query": "Mira"}, 16900),
                                  ("done", {"answer": "1971"}, 17100)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    trial = asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    tree = json.loads((trial / "agent/tree.json").read_text())
    said = [r[-1] for th, r in requests if th == "fork0/branch-0"]
    assert said[1].startswith("[branch-0 | context left: 9980 |")
    assert said[1].endswith("]\nForking is not available here: not enough context window is left at this depth; "
                            "finish this task yourself.")  # tree.FORK_WINDOW_REFUSAL
    assert "[clipped: context nearly used up]" in said[2] and "Your context is nearly used up. Close now" in said[2]
    branch = tree["forks"][0]["branches"][0]
    assert branch["ended_with"] == "done" and not branch["forks"] and tree["main_end"] == "done"
    assert {th for th, _ in requests} == {"main", "fork0/branch-0"}


def test_search_window_counts_mounted_tokens(monkeypatch, tmp_path):
    """The serving window counts a thread's mounts, which prompt_tokens leave out (OCTOPUS_CONTEXT_WINDOW 20000, replies
    of at most 1000 tokens): fork1/branch-0 (first prompt 4000) mounts N tokens of fork0's ended branch, so its growth
    cap drops from 15000 to 15000 - N and its fork, which would leave it room without the mount, is refused; its
    sibling (first prompt 13000) is refused the same read, whose mount would not fit."""
    agent, dry_run = _search_modules(monkeypatch)
    monkeypatch.setenv("OCTOPUS_SEARCH_URL", serve(dry_run.FakeRetrieval))
    for key, value in dict(OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_FORK_DEPTH="0", OCTOPUS_CURSOR="1",
                           OCTOPUS_CONTEXT_WINDOW="20000", OCTOPUS_MAX_TOKENS="1000").items():
        monkeypatch.setenv(key, value)
    read = ("read_thread", {"thread": "fork0/branch-0"})
    scripts = {"main": [("fork", {"tasks": ["t1"]}, 100), ("fork", {"tasks": ["t2", "t3"]}, 200),
                        ("done", {"answer": "1971"}, 900)],
               "fork0/branch-0": [("search", {"query": "Mira"}, 150), ("search", {"query": "Mira born"}, 5150),
                                  ("done", {"answer": "1971"}, 5200)],
               "fork1/branch-0": [(*read, 4000), ("fork", {"tasks": ["deeper"]}, 4200),
                                  ("done", {"answer": "x"}, 4300)],
               "fork1/branch-1": [(*read, 13000), ("done", {"answer": "x"}, 13100)]}
    policy, requests = _scripted(dry_run, scripts)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", serve(policy) + "/v1/chat/completions")
    asyncio.run(agent.run_trial(dry_run.QUESTION, tmp_path))
    said = [r[-1] for th, r in requests if th == "fork1/branch-0"]
    mounted = said[1].split("\n", 1)[1]
    assert mounted.startswith("Mounted fork0/branch-0's steps")
    n = int(mounted.split("(", 1)[1].split(" tokens")[0])
    assert n > 5000 and said[1].startswith(f"[branch-0 | context left: {15000 - n - 20} |")
    assert said[2].endswith("]\nForking is not available here: not enough context window is left at this depth; "
                            "finish this task yourself.")
    refused = [r[-1] for th, r in requests if th == "fork1/branch-1"][1].split("\n", 1)[1]
    assert refused == (f"fork0/branch-0's steps 2 to 4 hold {n} tokens, more than your context window has left: "
                       "read fewer steps. Nothing was read.")
