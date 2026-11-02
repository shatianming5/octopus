"""Plan-episode RL: reward/cost/advantages, statement tokens, verifier, tool schema."""

import asyncio
import base64
import collections
import json
import os
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from octopus.rl.group import assign_threads, control_cost, critical_path, group_step, leave_one_out, plan_group_stats
from octopus.envs.swe.reward import plan_group_step, plan_outcome

QWEN35 = [os.environ.get("QWEN35_TOKENIZER", ""), "/mnt/disk1/user/models/Qwen3.5-4B",
          "/home/user/octo-env/models/Qwen3.5-4B"]


def test_baseline_is_the_group_mains_and_ties_give_exactly_zero():
    """R = 2y−1, A_main leave-one-out over the G mains only: an episode's no-fork control (an evaluation's) is not in any
    baseline, so the mains' advantages sum to 0 (with the controls in it they did not); every branch takes
    its episode's A_main. Equal rewards, a 1/6 solved fraction too, give exactly 0 (the zero-signal filter)."""
    ys = [1, 0, 0, 0, 0, 0, 0, 1]
    episodes = [{"y": y, "C": 10, "forked": True, "branches": [{"key": "fork0-branch-0", "passed": None}],
                 "control": {"y": i % 2 == 0, "C": 5, "kind": "main-first"}} for i, y in enumerate(ys)]
    plan_group_step(episodes)
    adv = [e["A_main"] for e in episodes]
    assert adv == pytest.approx([12 / 7] + [-4 / 7] * 6 + [12 / 7]) and sum(adv) == pytest.approx(0, abs=1e-12)
    assert all("A_main" not in e["control"] and "R" not in e["control"] for e in episodes)
    assert [b["A_branch"] for e in episodes for b in e["branches"]] == adv
    assert leave_one_out([2 / 6 - 1] * 8) == [0.0] * 8 and leave_one_out([0.5]) == [0.0]


def test_critical_path_main_plus_max_branch_per_fork():
    reqs = [
        {"thread": "main", "completion_tokens": 10},
        {"thread": "main", "completion_tokens": 4},
        {"thread": "fork0/a", "completion_tokens": 3},
        {"thread": "fork0/b", "completion_tokens": 9},
        {"thread": "fork1/a", "completion_tokens": 2},
        {"thread": "control", "completion_tokens": 100},
    ]
    assert critical_path(reqs) == 14 + 9 + 2


def test_control_cost_counts_main_tokens_before_the_first_fork():
    reqs = [
        {"id": "m0", "thread": "main", "completion_tokens": 10},
        {"id": "m1", "thread": "main", "completion_tokens": 4},  # the accepted fork call: not in the control's history
        {"id": "b0", "thread": "fork0/branch-0", "completion_tokens": 3},
        {"id": "m2", "thread": "main", "completion_tokens": 7},
        {"id": "c0", "thread": "control", "completion_tokens": 20},
    ]
    assert control_cost(reqs, "m1") == 10 + 20


# (format, tokenizer in the local HF cache): each core.TOOL_FORMATS format on its model family's chat template
FORMAT_TOKENIZERS = [
    ("qwen3_coder", "Qwen/Qwen3-Coder-30B-A3B-Instruct"), ("qwen25", "Qwen/Qwen3-0.6B"),
    ("qwen25", "Qwen/Qwen2.5-7B-Instruct"), ("llama3", "unsloth/Llama-3.2-1B-Instruct"),
    ("llama3", "unsloth/Meta-Llama-3.1-8B-Instruct"), ("mistral", "unsloth/mistral-7b-instruct-v0.3"),
    ("mistral", "unsloth/Mistral-Small-3.2-24B-Instruct-2506"), ("deepseekv3", "deepseek-ai/DeepSeek-V3"),
    ("deepseekv31", "deepseek-ai/DeepSeek-V3.1"), ("glm45", "zai-org/GLM-4.5"), ("glm47", "zai-org/GLM-4.7"),
    ("kimi_k2", "moonshotai/Kimi-K2-Instruct"), ("gpt-oss", "openai/gpt-oss-20b"),
]


def _cached_tokenizer(repo: str):
    transformers = pytest.importorskip("transformers")
    try:
        if repo.startswith("moonshotai/"):  # tiktoken, its own tokenizer class
            return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True, trust_remote_code=True)
        return transformers.PreTrainedTokenizerFast.from_pretrained(repo, local_files_only=True)  # (DeepSeek's config fails)
    except (OSError, ValueError, ImportError):
        pytest.skip(f"no {repo} tokenizer here")


@pytest.mark.parametrize("fmt, repo", FORMAT_TOKENIZERS)
def test_tool_format_spans_and_positions_on_its_chat_template(monkeypatch, fmt, repo):
    """With rollout.agent.tool_call_parser = fmt, on a conversation its model family's chat template renders: the calls
    are found."""
    from octopus.rl.group import call_spans
    monkeypatch.setenv("OCTOPUS_TOOL_FORMAT", fmt)
    tok = _cached_tokenizer(repo)
    tasks = [{"task": "fix a", "will_pass": True}, {"task": "fix b", "will_pass": False}]
    calls = [("fork", {"tasks": tasks}), ("commit", {"branch_id": "b", "summary": "s"}),
             ("abort", {"branch_id": "b", "reason": "r"}), ("done", {"summary": "ok"})]
    deepseek = fmt.startswith("deepseek")  # its templates take the arguments as a JSON string and no content
    messages = [{"role": "user", "content": "bug"}]
    for i, (name, args) in enumerate(calls):  # one call per reply: the Llama 3 templates take no more
        call_id = f"functions.{name}:{i}" if fmt == "kimi_k2" else f"call0000{i}"  # Kimi renders it; Mistral: 9 chars
        messages += [{"role": "assistant", "content": None if deepseek else "", "tool_calls": [
                         {"id": call_id, "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args) if deepseek else args}}]},
                     {"role": "tool", "tool_call_id": call_id, "content": "ok"}]
    text = tok.apply_chat_template(messages[:-1], tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    token_text = lambda i: tok.decode([i])
    found = [tok.decode(ids[a:b]) for a, b in call_spans(ids, token_text, [name for name, _ in calls])]
    assert len(found) == 4 and all(name in s and all(k in s for k in args) for s, (name, args) in zip(found, calls)), found


# per JSON-call format: one call as it writes it (@ = the object), and a tokenizer of its family
JSON_CALLS = [("qwen25", "Qwen/Qwen3-0.6B", "<tool_call>\n@\n</tool_call>"),
              ("llama3", "unsloth/Llama-3.2-1B-Instruct", "<|python_tag|>@<|eom_id|>"),
              ("mistral", "unsloth/mistral-7b-instruct-v0.3", "[TOOL_CALLS] [@]</s>")]


@pytest.mark.parametrize("fmt, repo, wrap", JSON_CALLS)
def test_json_tool_calls_in_any_spelling(monkeypatch, fmt, repo, wrap):
    """JSON calls: compact, arguments before the name, whitespace before the object or inside it all still give the
    call; a call whose name is not a string is none."""
    from octopus.rl.group import call_spans
    monkeypatch.setenv("OCTOPUS_TOOL_FORMAT", fmt)
    fork = '{"name": "fork", "arguments": {"tasks": [{"task": "a", "will_pass": true}, {"task": "b }{", "will_pass": false}]}}'
    read, done = '{"arguments": {"thread": "x"}, "name": "read_thread"}', '{"name": "done", "arguments": {"summary": "ok"}}'
    bad_name = '{"name": ["done"], "arguments": {}}'
    text = "".join(wrap.replace("@", c) for c in (fork, read, "{bad}", bad_name, done))
    ids = [ord(c) for c in text]
    span = lambda a, b: text[a:b]
    assert [span(*s) for s in call_spans(ids, chr, ["fork", "read_thread", "done"])] == [fork, read, done]

    tok = _cached_tokenizer(repo)
    token_text = lambda i: tok.decode([i])
    variants = ['{"name":"commit","arguments":{"branch_id":"b"}}', '{"arguments": {"branch_id": "b"}, "name": "commit"}',
                '  {"name": "commit", "arguments": {}}', '{ "name": "commit", "arguments": {} }',
                '{"arguments":{},"name":"commit"}']
    for v in variants:
        for name in ("commit", "abort"):
            call = v.replace("commit", name)
            ids = tok.encode("ok " + wrap.replace("@", call), add_special_tokens=False)
            spans = call_spans(ids, token_text, [name])  # (an end token may carry a bracket too)
            assert [tok.decode(ids[a:b]).strip(" \n[]") for a, b in spans] == [call.strip()], call
    ids = tok.encode(wrap.replace("@", bad_name.replace("done", "commit")), add_special_tokens=False)
    assert call_spans(ids, token_text, ["commit"]) == []


def test_unknown_tool_format_fails():
    from octopus.runtime.core import tool_format
    with pytest.raises(ValueError, match="tool-call format 'qwen4'"):
        tool_format("qwen4")


class _Ids(list):  # the bits of a torch tensor assign_threads uses
    def reshape(self, *shape):
        return self

    def tolist(self):
        return list(self)


def _leaf(rid, ids):
    """Shaped like AReaL's deserialize_interactions output: no parent, no model_response."""
    return SimpleNamespace(interaction_id=rid, reward=0.0, rollout_index=0, metadata={},
                           _cache={"input_ids": _Ids(ids), "loss_mask": _Ids([0] * len(ids))})


def _episode(requests, **kw):
    episode = {"R": 1.0, "A_main": 0.5, "requests": requests, "control": {"y": False},
               "branches": [{"key": "fork0-branch-0", "A_branch": 0.25}, {"key": "fork0-branch-1", "A_branch": -0.25}]}
    episode.update(kw)
    return episode


# main: m0 (a rejected fork call), m1 (the accepted fork), m2 (done); branch-0 starts from the main thread before the
# fork call: b0, b0b. (id, thread, prompt_tokens, completion_tokens)
M0, M1, M2 = [4, 1, 2, 3, 8], [1, 2, 3, 8, 1, 2, 3, 9], [5, 6, 18]
MAIN = [0, 0] + M0 + [0] + M1 + [0] + M2
BRANCH = [0, 0] + M0 + [0] + [0, 0] + [7, 7] + [0] + [7]
REQUESTS = [("m0", "main", 2, 5), ("m1", "main", 8, 8), ("b0", "fork0/branch-0", 10, 2),
            ("b0b", "fork0/branch-0", 13, 1), ("m2", "main", 17, 3)]


def test_assign_threads_masks_inherited_prefix():
    """B1 on deserialized leaves: a branch trains only from its first request's prompt length on."""
    branch, mainrow = assign_threads([_leaf("b0b", BRANCH), _leaf("m2", MAIN)], _episode(REQUESTS))
    assert (branch["own_start"], branch["adv"], branch["reward"], branch["spans"]) == (10, 0.25, 1.0, [])
    assert mainrow["own_start"] == 2 and mainrow["adv"] == 0.5


def test_assign_threads_rejects_unknown_threads():
    """B3: a completion without a requests.jsonl row, or an ungraded branch thread, rejects the group; a control leaf
    is graded, never trained."""
    with pytest.raises(RuntimeError, match="no requests.jsonl row"):
        assign_threads([_leaf("x", MAIN)], _episode(REQUESTS))
    with pytest.raises(RuntimeError, match="no graded branch"):
        assign_threads([_leaf("b9", [1])], _episode([("b9", "fork1/branch-0", 0, 1)]))
    row = assign_threads([_leaf("c1", [3, 3, 1, 4])], _episode([("c0", "control", 2, 1), ("c1", "control", 3, 1)]))[0]
    assert (row["reward"], row["adv"], row["own_start"]) == (1.0, 0.0, 4)


# a rewound thread (OCTOPUS_REWIND): x0, x1 (the reply the rewind goes back to), x2 (a fork) and x3 abandoned; x4, x5
# (done) continue from x1. Two concat leaves sharing x0 and x1. (rid, prompt, completion)
REWOUND = [("0", 2, 2), ("1", 5, 2), ("2", 8, 4), ("3", 13, 2), ("4", 9, 2), ("5", 12, 3)]
ABANDONED = [0, 0, 4, 4, 0, 4, 4, 0, 1, 2, 3, 8, 0, 4, 4]
CONTINUED = [0, 0, 4, 4, 0, 4, 4, 0, 0, 4, 4, 0, 5, 6, 18]
CHAINS = {"3": "0123", "5": "0145"}  # each leaf's requests: AReaL puts loss mask 1 on every one's completion


def test_assign_threads_trains_a_rewound_thread_once_with_its_advantage():
    """A rewound main / branch thread is several leaves: each gets the thread's reward and advantage; each
    leaf trains only the requests after the previous leaf's last one (own_start), so every completion token is trained
    exactly once. A thread that did not rewind keeps its one leaf as before."""
    threads = {"m": "main", "b": "fork0/branch-0"}
    requests = [(p + rid, t, prompt, n) for p, t in threads.items() for rid, prompt, n in REWOUND]
    requests.insert(3, ("x0", "fork0/branch-1", 3, 4))  # interleaved in time with the rewound threads
    episode = _episode(requests)
    leaves = [_leaf(p + rid, ids) for p in threads for rid, ids in (("3", ABANDONED), ("5", CONTINUED))]
    rows = assign_threads(leaves + [_leaf("x0", [0] * 7)], episode)
    for p, values in {"m": (1.0, 0.5), "b": (1.0, 0.25)}.items():
        first, second = (r for leaf, r in zip(leaves, rows) if leaf.interaction_id.startswith(p))
        assert (first["reward"], first["adv"]) == (second["reward"], second["adv"]) == values
        assert (first["own_start"], second["own_start"]) == (2, 9)
        trained = collections.Counter()  # (request, position) -> leaves training it
        for leaf, row in ((leaf, r) for leaf, r in zip(leaves, rows) if leaf.interaction_id.startswith(p)):
            for rid in CHAINS[leaf.interaction_id[1:]]:
                _, prompt, n = REWOUND[int(rid)]
                trained.update((rid, i) for i in range(prompt, prompt + n) if i >= row["own_start"])
        assert set(trained.values()) == {1} and len(trained) == sum(n for _, _, n in REWOUND)
    assert (rows[-1]["own_start"], rows[-1]["adv"]) == (3, -0.25)  # the thread that did not rewind
    # the rewound threads' leaves leave a lone leaf's row unchanged
    assert rows[-1] == assign_threads([_leaf("x0", [0] * 7)], _episode([requests[3]]))[0]


@pytest.mark.parametrize("same_turn, tool", [(False, False), (True, False), (False, True)])
def test_rewind_past_a_running_fork_records_it(monkeypatch, tmp_path, same_turn, tool):
    """slice 7: a rewind dropping the forking reply of a fork (one made in an earlier turn, or in the very turn that
    rewinds) crashed fork_record ("max() iterable argument is empty").
    The fork is recorded from the history it started from, and a control from it starts there. tool: the rewind is
    the model's (OCTOPUS_REWIND_TOOL), back to its edit's step, after the fork's step (a rewind call in the forking
    reply is refused: test_a_reply_that_forks_cannot_also_close_or_rewind)."""
    def reply(thread, turn):
        if thread == "main" and tool:
            script = {1: _call("edit", {"path": "a.py", "old_str": "a", "new_str": "b"}), 2: _fork_call(5, 6),
                      3: _call("rewind", {"to_step": 1, "note": "wrong approach"})}
            return script.get(turn) or DONE
        if thread == "main":
            fail = _call("shell", {"command": "false"})
            script = {1: _call("edit", {"path": "a.py", "old_str": "a", "new_str": "b"}), 2: _fork_call(5, 6)}
            if same_turn:
                script = {1: script[1], 2: fail, 3: fail, 4: _fork_call(5, 6) + [dict(fail[0], id="f")]}
            return script.get(turn) or (fail if turn < 8 else _call("done", {"summary": "ok"}))
        if thread == "control":
            return _call("done", {"summary": "ok"})
        return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})
    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_PLAN_CONTROL="1",
                                              **{f"OCTOPUS_REWIND{'_TOOL' * tool}": "1"})
    assert len(tree["rewinds"]) == 1
    fork, = tree["forks"]
    assert [b["task_id"] for b in fork["branches"]] == ["5", "6"]
    assert fork["response_id"] == ("main#4" if same_turn and not tool else "main#2")
    assert not any(c.get("function", {}).get("name") == "fork" for m in tree["main"] for c in m.get("tool_calls") or [])
    edit = [m for m in history["control"] if m.get("role") == "assistant"]  # the forking thread's history then
    assert edit and edit[0]["tool_calls"][0]["function"]["name"] == "edit"


def _trial(tmp_path, branches_json):
    (tmp_path / "verifier").mkdir(parents=True)
    (tmp_path / "agent").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("1\n")
    if branches_json is not None:
        (tmp_path / "verifier/branches.json").write_text(json.dumps(branches_json))
    fork = {"response_id": "m1", "branches": [{"branch_id": "branch-0", "ended_with": "commit"},
                                              {"branch_id": "branch-1", "ended_with": "abort"}]}
    (tmp_path / "agent/tree.json").write_text(json.dumps({"forks": [fork]}))
    return tmp_path


def test_plan_outcome_labels_and_rejects_missing_or_short_branches_json(tmp_path):
    """b2: no branches.json (verifier crash) or a branch count mismatch rejects the episode."""
    reqs = [{"id": "m1", "thread": "main", "prompt_tokens": 9, "completion_tokens": 5},
            {"id": "b", "thread": "fork0/branch-0", "prompt_tokens": 12, "completion_tokens": 3},
            {"id": "m2", "prompt_tokens": 20, "completion_tokens": 1}]
    with pytest.raises(RuntimeError, match="plan verifier"):
        plan_outcome(_trial(tmp_path / "a", None), reqs)
    short = {"branches": [{"key": "fork0-branch-0", "passed": True}]}
    with pytest.raises(RuntimeError, match="plan verifier"):
        plan_outcome(_trial(tmp_path / "b", short), reqs)
    full = {"branches": [{"key": "fork0-branch-0", "passed": True}, {"key": "fork0-branch-1", "passed": False}]}
    out = plan_outcome(_trial(tmp_path / "c", full), reqs)
    assert out["requests"] == [("m1", "main", 9, 5), ("b", "fork0/branch-0", 12, 3), ("m2", "main", 20, 1)]
    assert [b["passed"] for b in out["branches"]] == [True, False] and out["control"] is None


def test_plan_verifier_shell_grades_final_then_branches_then_restores():
    from octopus.envs.swe.taskset import plan_verifier_shell, verifier_shell
    shell = plan_verifier_shell()
    assert "/tmp/octopus-plan/plan.tsv" in shell
    assert "/logs/verifier/reward.txt" in shell
    assert "/logs/verifier/branches.json" in shell
    assert "/tmp/octopus-plan/control.snap" in shell and "/tmp/branches" not in shell
    assert "git stash apply" in shell
    assert "/tests/check.py $(cat /tests/hidden_files.txt)" in shell
    assert shell.index("reward.txt") < shell.index("plan.tsv")
    assert "cp -a /tests/hidden_full/. /testbed/" in shell
    default = verifier_shell()
    assert "plan.tsv" not in default
    assert "/logs/verifier/reward.txt" in default


def test_plan_verifier_applies_a_branch_patch_over_a_main_edit(tmp_path):
    """b1: the main thread edited a file before the fork and the branch patch edits it again: restore must leave
    index == tree so git apply --3way accepts the patch."""
    import re
    from octopus.envs.swe.taskset import plan_verifier_shell
    shell = plan_verifier_shell()
    funcs = "\n".join(re.search(rf"^{name}\(\) {{.*?^}}", shell, re.S | re.M).group(0) for name in ("take_snap", "restore"))
    repo = tmp_path / "repo"
    script = f"""set -e
git init -q {repo} && cd {repo} && git config user.email t@t && git config user.name t
printf 'a\\nb\\nc\\n' > f.py && git add f.py && git commit -qm base
{funcs}
printf 'A\\nb\\nc\\n' > f.py        # main thread edit before the fork
point=$(take_snap)
printf 'A\\nb\\nC\\n' > f.py && git diff > {tmp_path}/branch.patch  # the branch's patch on the fork-point tree
git add -A && git commit -qm final  # the episode moves on
restore "$point"
git apply --3way --whitespace=nowarn {tmp_path}/branch.patch
cat f.py
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["A", "b", "C"]


def _import_agent(monkeypatch):
    class Any:
        def __init__(self, *a, **k):
            pass

    class BaseAgent(Any):
        model_name = "policy"  # Harbor's -m (an agent made with __new__ skips Harbor's __init__)
    for name in ("harbor", "harbor.agents", "harbor.agents.base", "harbor.agents.capabilities", "harbor.environments",
                 "harbor.environments.base", "harbor.models", "harbor.models.agent", "harbor.models.agent.context",
                 "harbor.models.trajectories", "harbor.models.trajectories.agent",
                 "harbor.models.trajectories.observation", "harbor.models.trajectories.observation_result",
                 "harbor.models.trajectories.step", "harbor.models.trajectories.tool_call",
                 "harbor.models.trajectories.trajectory"):
        module = types.ModuleType(name)
        for attr in ("BaseAgent", "AgentCapabilities", "BaseEnvironment", "AgentContext", "Agent", "Observation",
                     "ObservationResult", "Step", "ToolCall", "Trajectory"):
            setattr(module, attr, Any)
        module.BaseAgent = BaseAgent
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "octopus.envs.swe.agent", raising=False)
    monkeypatch.delitem(sys.modules, "octopus.runtime.tree", raising=False)
    monkeypatch.setenv("OCTOPUS_CHAT_URL", "http://proxy/v1/chat/completions")  # RL: model_env's (tests stub post_chat)
    import octopus.envs.swe.agent as agent_module
    return agent_module


def test_plan_verifier_inputs_survive_the_policy_deleting_them(monkeypatch, tmp_path):
    """The policy ran rm -rf /tmp/branches (and could fill /tmp/octopus-plan): the verifier inputs are written from
    the agent's fork records after the episode's last tool call."""
    mod = _import_agent(monkeypatch)
    root = tmp_path / "container"

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            assert command.startswith("rm -rf /")
            subprocess.run(["rm", "-rf", str(root / command[len("rm -rf /"):])], check=True)

        async def upload_dir(self, source_dir, target_dir):
            import shutil
            shutil.copytree(source_dir, root / target_dir.lstrip("/"), dirs_exist_ok=True)

    (root / "tmp/octopus-plan").mkdir(parents=True)
    (root / "tmp/octopus-plan/plan.tsv").write_text("junk the policy wrote\n")  # /tmp/branches itself is gone
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.logs_dir, agent._control_snap = tmp_path / "logs", "head0 stash9"
    forks = [{"fork_point": "head0 stash1", "branches": [{"branch_id": "branch-0", "patch_text": "diff --git a/f b/f\n"},
                                                         {"branch_id": "branch-1", "patch_text": ""}]}]
    agent.ws = mod.HarborWorkspace(Env())
    asyncio.run(agent._write_plan_inputs(forks))
    plan = root / "tmp/octopus-plan"
    assert (plan / "plan.tsv").read_text() == ("fork0-branch-0\thead0 stash1\t/tmp/octopus-plan/fork0-branch-0.patch\n"
                                               "fork0-branch-1\thead0 stash1\t\n")
    assert (plan / "fork0-branch-0.patch").read_text() == "diff --git a/f b/f\n"
    assert (plan / "control.snap").read_text() == "head0 stash9"


def test_control_rejects_fork_and_sends_history_verbatim(monkeypatch):
    """b5: the control's dispatch refuses fork (and the runtime records no fork)."""
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    monkeypatch.setenv("OCTOPUS_PLAN_EPISODE", "1")
    from octopus.runtime.core import RuntimePort
    from octopus.envs.swe.prompt import DELIVERY_LINEAR, delivery
    mod = _import_agent(monkeypatch)
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent._runtime, agent._control_linear = RuntimePort(), True
    tasks = [{"task": t, "acceptance": "true"} for t in "ab"]
    observed, stop = asyncio.run(agent._dispatch("fork", {"tasks": tasks}))
    assert "fork is not a tool" in observed and not stop and agent._runtime.last_fork is None
    assert delivery(linear=True).startswith(DELIVERY_LINEAR)


def test_fork_takes_no_will_pass_in_plan_episodes(monkeypatch):
    """As in search, a fork task states nothing about its branch: no will_pass in the schema or the system message,
    with or without OCTOPUS_PLAN_EPISODE, and a fork without one is accepted."""
    monkeypatch.setenv("OCTOPUS_PLAN_EPISODE", "1")
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    from octopus.envs.swe.prompt import offered_tools
    from octopus.runtime.core import RuntimePort
    from octopus.envs.swe.prompt import delivery
    apply_runtime = _import_agent(monkeypatch).HarborWorkspace().runtime

    def fork_props():
        fork = next(t for t in offered_tools() if t["function"]["name"] == "fork")
        return fork["function"]["parameters"]["properties"]["tasks"]["items"]

    def done_props():
        done = next(t for t in offered_tools() if t["function"]["name"] == "done")
        return done["function"]["parameters"]["properties"]

    assert "will_pass" not in fork_props().get("properties", {}) and "will_pass" not in fork_props()["required"]
    assert "patch_right" not in done_props()
    assert "will_pass" not in delivery()
    tasks = [{"task": "a", "acceptance": "true"}, {"task": "b", "acceptance": "true"}]
    msg, stop = apply_runtime("fork", {"tasks": tasks}, RuntimePort())
    assert stop is False and "branch-0" in msg


@pytest.mark.parametrize("merge", ["1", "0"])
def test_swe_branch_is_told_done_does_not_merge_under_auto_merge(monkeypatch, merge):
    """search's branch_close (#193) for SWE: under OCTOPUS_AUTO_MERGE=1 a branch's task message says done does not merge
    its patch and commit does; without auto merge (main sees every branch's patch) it says nothing."""
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    monkeypatch.setenv("OCTOPUS_AUTO_MERGE", merge)
    from octopus.runtime.core import branch_task_message
    close = _import_agent(monkeypatch).HarborWorkspace().branch_close
    text = branch_task_message("branch-0", "Fix f", 2, close=close)
    sentence = ("done only closes this branch: the join does not merge its patch into the forking thread's tree; "
                "commit is what merges it.")
    assert (close, sentence in text) == ((sentence, True) if merge == "1" else ("", False))


def test_allocate_prefix_groups_fits_plan_main_branch_control():
    np = pytest.importorskip("numpy")
    from octopus.rl.prefix_tree import allocate_prefix_groups, build_tree
    rng = np.random.default_rng(2)
    s0 = rng.integers(0, 1000, 20_000)
    seqs = []
    for _ in range(8):
        main = np.concatenate([s0, rng.integers(0, 1000, 4_000)])
        seqs.append(main)
        seqs.append(np.concatenate([main[:22_000], rng.integers(0, 1000, 3_000)]))  # branch after a short join
        seqs.append(np.concatenate([s0, rng.integers(0, 1000, 4_000)]))  # control from s0
    assert all(len(s) <= 90_112 for s in seqs)
    mbs = allocate_prefix_groups(seqs, cap=90_112)
    assert sorted(i for m in mbs for i in m) == list(range(len(seqs)))
    for m in mbs:
        tree = build_tree(np.concatenate([seqs[i] for i in m]),
                          [0, *np.cumsum([len(seqs[i]) for i in m]).tolist()])
        assert len(tree.tokens) <= 90_112


# --- task bundles (OCTOPUS_BUNDLE) ---

def _bundle_branch(ended_with, passed, landed, task="1"):
    return {"task": task, "ended_with": ended_with, "passed": passed, "landed": landed, "accepted": True}


def test_pure_outcome_gives_every_branch_its_episode_a_main():
    """Every branch (committed, aborted, unclosed) takes its episode's A_main."""
    cut = _bundle_branch(None, False, False)
    episodes = [{"y": 0.5, "C": 10, "forked": True, "control": None, "tasks": {"1": 1, "2": 0},
                 "branches": [_bundle_branch("commit", True, True), _bundle_branch("abort", False, False), cut]},
                {"y": 1.0, "C": 30, "forked": True, "control": None, "tasks": {"1": 1, "2": 1},
                 "branches": [_bundle_branch("commit", False, False, task="2")]},
                {"y": 0.0, "C": 20, "forked": False, "control": None, "tasks": {"1": 0, "2": 0}, "branches": []}]
    eps = plan_group_step(episodes)
    assert [e["A_main"] for e in eps] == pytest.approx([0.0, 1.5, -1.5])  # 2y-1, leave-one-out
    assert all(b["A_branch"] == e["A_main"] for e in eps for b in e["branches"])


def test_bundle_group_stats():
    """The group's stats line: bundle branches naming no task, committed passing branches that landed, |A|, C_wall."""
    branches = [
        _bundle_branch("commit", True, True),     # fix that landed
        _bundle_branch("commit", True, False),    # its patch solved the task, the final tree does not
        _bundle_branch("commit", False, True),   # its patch failed, main solved the task
        _bundle_branch("commit", False, False),   # failed
        _bundle_branch("abort", True, False),    # threw away a fix
        _bundle_branch("abort", False, False),    # unsolved
        _bundle_branch(None, False, False),       # unclosed
        _bundle_branch("commit", False, None, task=None),  # no task: own patch, any task
    ]
    episodes = [{"y": 0.5, "C": 10, "C_wall": 600.4, "forked": True, "control": None, "tasks": {"1": 1, "2": 0},
                 "branches": branches},
                {"y": 1.0, "C": 30, "C_wall": 299.6, "forked": False, "control": None, "tasks": {"1": 1, "2": 1},
                 "branches": []}]
    eps = plan_group_step(episodes)
    assert [e["R"] for e in eps] == [0.0, 1.0]  # 2y-1
    stats = plan_group_stats(eps)
    assert stats["no_task_id"] == round(1 / 8, 4) and stats["committed_passing_landed"] == 0.5
    assert stats["abs_A_main"] == 1.0
    assert stats["C_wall"] == {"forked": [600], "unforked": [300], "control": []}  # recorded, R ignores it


def test_plan_outcome_bundle_reads_tasks_budget_and_branch_tasks(tmp_path):
    trial = _trial(tmp_path, {"branches": [
        {"key": "fork0-branch-0", "task": "2", "tasks": {"2": 1}, "passed": True},
        {"key": "fork0-branch-1", "task": None, "tasks": {"1": 0, "2": 0}, "passed": False}]})
    (trial / "verifier/reward.txt").write_text("0.5\n")
    (trial / "verifier/tasks.json").write_text(json.dumps({"1": 0, "2": 1}))
    tree = json.loads((trial / "agent/tree.json").read_text())
    (trial / "agent/tree.json").write_text(json.dumps({**tree, "budget": {"limit": 100000, "used": 4321}}))
    out = plan_outcome(trial, [{"id": "m1", "thread": "main", "prompt_tokens": 9, "completion_tokens": 5}])
    assert out["y"] == 0.5
    assert out["tasks"] == {"1": 0, "2": 1} and out["budget"] == {"limit": 100000, "used": 4321}
    assert [(b["task"], b["passed"], b["landed"]) for b in out["branches"]] == [("2", True, True), (None, False, None)]


def test_bundle_verifier_grades_branches_alone_on_their_task_then_the_final_tree(tmp_path):
    """Bundle of two tasks (task k is solved when f.py holds fix<k>). The final tree fixed task 1; branch a fixed task
    1 for task 1, b fixed task 2 naming none, c named task 2 but fixed task 1, d named a task the bundle lacks; the
    no-fork control's tree fixed task 2 (graded on every task, before the final tree). check.py reads stdin, as python-pinyin's CLI tests do: it must not consume the remaining plan.tsv rows."""
    from octopus.envs.swe.taskset import bundle_verifier_shell
    t = tmp_path
    for d in ("tests/hidden_full", "logs", "plan"):
        (t / d).mkdir(parents=True)
    (t / "tests/tasks.json").write_text(json.dumps({"1": {"f2p": [], "p2p": []}, "2": {"f2p": [], "p2p": []}}))
    (t / "tests/check.py").write_text(f"""import json, sys
from pathlib import Path
sys.stdin.read()
tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())
text = Path("{t}/testbed/f.py").read_text()
solved = {{k: int(f"fix{{k}}" in text) for k in tasks}}
Path("{t}/logs/verifier/tasks.json").write_text(json.dumps(solved))
Path("{t}/logs/verifier/reward.txt").write_text(f"{{sum(solved.values()) / len(solved)}}\\n")
""")
    repo = t / "testbed"
    patch = {}
    for name, text in (("a", "fix1\nbug2\n"), ("b", "bug1\nfix2\n"), ("c", "fix1\nbug2\n")):
        patch[name] = t / "plan" / f"{name}.patch"
    script = f"""set -e
git init -q {repo} && cd {repo} && git config user.email t@t && git config user.name t
printf 'bug1\\nbug2\\n' > f.py && git add f.py && git commit -qm initial
for x in a:fix1:bug2 b:bug1:fix2 c:fix1:bug2; do IFS=: read n l1 l2 <<< "$x"
  printf '%s\\n%s\\n' $l1 $l2 > f.py && git diff > {t}/plan/$n.patch; done
printf 'bug1\\nfix2\\n' > f.py && git add -A && echo "$(git rev-parse HEAD) $(git stash create)" > {t}/plan/control.snap
printf 'fix1\\nbug2\\n' > f.py  # the final tree
git rev-parse HEAD
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    head = out.stdout.split()[-1]
    rows = [("fork0-a", "1", "a"), ("fork0-b", "-", "b"), ("fork0-c", "2", "c"), ("fork1-d", "7", "-")]
    (t / "plan/plan.tsv").write_text("".join(
        f"{k}\t{head}\t{task}\t{t / 'plan' / (p + '.patch') if p != '-' else '-'}\n" for k, task, p in rows))
    # check validity: each check on the initial tree and with its task's reference fix (/tests/fixes)
    (t / "tests/fixes").mkdir()
    for k, text in (("1", "fix1\nbug2\n"), ("2", "bug1\nfix2\n")):
        subprocess.run(["bash", "-c", f"cd {repo} && git show HEAD:f.py > /tmp/_orig_{os.getpid()} && cp f.py /tmp/_cur_{os.getpid()}"
                        f" && cp /tmp/_orig_{os.getpid()} f.py && printf %s {shlex.quote(text)} > f.py"
                        f" && git diff HEAD > {t}/tests/fixes/fix_{k}.patch && cp /tmp/_cur_{os.getpid()} f.py"], check=True)
    b64 = lambda c: base64.b64encode(c.replace("/testbed", str(repo)).encode()).decode()
    (t / "plan/checks.tsv").write_text("".join(f"{k}\t{kind}\t{head}\t{task}\t{b64(c)}\n" for k, kind, task, c in (
        ("fork0-a", "fork", "1", "grep -q fix1 /testbed/f.py"), ("fork0-c", "fork", "2", "grep -q fix1 /testbed/f.py"),
        ("fork0-b", "own", "2", "true"))))
    shell = bundle_verifier_shell()
    for a, b in (("/opt/miniconda3/envs/testbed/bin/python", sys.executable), ("/testbed", str(repo)),
                 ("/tests/", f"{t}/tests/"), ("/logs/", f"{t}/logs/"), ("/tmp/octopus-plan", f"{t}/plan")):
        shell = shell.replace(a, b)
    out = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads((t / "logs/verifier/tasks.json").read_text()) == {"1": 1, "2": 0}
    assert (t / "logs/verifier/reward.txt").read_text().strip() == "0.5"
    graded = json.loads((t / "logs/verifier/branches.json").read_text())["branches"]
    assert [(b["key"], b["task"], b["tasks"], b["passed"]) for b in graded] == [
        ("fork0-a", "1", {"1": 1}, True), ("fork0-b", None, {"1": 0, "2": 1}, True),
        ("fork0-c", "2", {"2": 0}, False), ("fork1-d", None, {}, False)]
    assert json.loads((t / "logs/verifier/branches.json").read_text())["control"] == {"tasks": {"1": 0, "2": 1}}
    assert (repo / "f.py").read_text() == "fix1\nbug2\n"  # graded as it stands
    assert json.loads((t / "logs/verifier/checks.json").read_text()) == {  # valid: fails on the bug, passes on the fix
        "fork0-a:fork": {"task": "1", "fix_applied": True, "bug": "1", "fix": "0"},
        "fork0-c:fork": {"task": "2", "fix_applied": True, "bug": "1", "fix": "1"},
        "fork0-b:own": {"task": "2", "fix_applied": True, "bug": "0", "fix": "0"}}


def test_shared_pool_prefixes_tool_results_and_ends_with_closing_only(monkeypatch):
    mod = _import_agent(monkeypatch)
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent._budget = {"limit": 10, "used": 0}
    calls = [{"id": i, "type": "function", "function": {"name": name, "arguments": "{}"}}
             for i, name in enumerate(("edit", "done"))]
    agent._complete = lambda messages, s=None, p=None: {
        "choices": [{"message": {"content": "", "tool_calls": list(calls)}}], "usage": {"completion_tokens": 6}}
    history = [{"role": "user", "content": "go"}, {"role": "tool", "content": "a"}]
    first = asyncio.run(agent._complete_async(history))
    assert history[1]["content"] == "[token budget left: 10]\na" and len(first["choices"][0]["message"]["tool_calls"]) == 2
    history.append({"role": "tool", "content": "b"})
    last = asyncio.run(agent._complete_async(history))
    assert history[1]["content"] == "[token budget left: 10]\na" and history[2]["content"] == "[token budget left: 4]\nb"
    assert agent._budget["used"] == 12
    assert [c["function"]["name"] for c in last["choices"][0]["message"]["tool_calls"]] == ["done"]
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(agent._complete_async(history))


def test_bundle_fork_task_id_schema_record_and_plan_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    monkeypatch.delenv("OCTOPUS_AUTO_MERGE", raising=False)
    from octopus.envs.swe.prompt import (
        _MERGE_AUTO, _MERGE_AUTO_BUNDLE, _MERGE_MANUAL, TASK_ID_NEEDS, delivery, fork_tasks, offered_tools,
    )
    from octopus.envs.swe.container_tools import _view
    from octopus.runtime.core import RuntimePort, fork_record
    apply_runtime = _import_agent(monkeypatch).HarborWorkspace().runtime

    def items():
        fork = next(t for t in offered_tools() if t["function"]["name"] == "fork")
        return fork["function"]["parameters"]["properties"]["tasks"]["items"]
    assert "task_id" not in items()["properties"] and "/tasks" not in delivery()
    assert _MERGE_MANUAL in delivery() and _MERGE_AUTO not in delivery()
    monkeypatch.setenv("OCTOPUS_BUNDLE", "1")
    # required in bundle mode (was optional: 20-41% of branches named no task, diluting per-task credit)
    assert items()["properties"]["task_id"]["type"] == "integer" and "task_id" in items()["required"]
    assert "task_id" in delivery() and "/tasks/INDEX.md" in delivery()
    assert _MERGE_AUTO_BUNDLE in delivery() and _MERGE_MANUAL not in delivery()  # bundle default: the runtime merges
    assert "冲突标记" not in delivery() and "主线在所有任务都修好后调用 done" in delivery()
    for bad in ({}, {"task_id": True}, {"task_id": "two"}):
        forks = [{"task": "x", "acceptance": "true", "task_id": 1}, {"task": "y", "acceptance": "true", **bad}]
        assert apply_runtime("fork", {"tasks": forks}, RuntimePort()) == (TASK_ID_NEEDS, False)
    forks = [{"task": "x", "acceptance": "true", "task_id": 1}, {"task": "y", "acceptance": "true", "task_id": "2"}]
    assert "branch-0" in apply_runtime("fork", {"tasks": forks}, RuntimePort())[0]
    with pytest.raises(ValueError, match="escapes"):
        _view(tmp_path, {"path": "/tasks/../etc/passwd"})
    tasks = [{"task": "x", "acceptance": "true", "task_id": 2}, {"task": "y", "acceptance": "true", "task_id": "3"},
             {"task": "z", "acceptance": "true"}, {"task": "w", "acceptance": "true", "task_id": True}]
    main = [{"role": "assistant", "tool_calls": [{"function": {"name": "fork", "arguments": json.dumps({"tasks": tasks})}}]}]
    reports = [{"branch_id": f"branch-{i}", "task": t["task"], "messages": [], "steps": []} for i, t in enumerate(tasks)]
    record = fork_record(main, reports, fork_tasks)
    assert [b["task_id"] for b in record["branches"]] == ["2", "3", None, None]

    mod = _import_agent(monkeypatch)
    root = tmp_path / "container"

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            pass

        async def upload_dir(self, source_dir, target_dir):
            import shutil
            shutil.copytree(source_dir, root / target_dir.lstrip("/"), dirs_exist_ok=True)

    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.logs_dir = tmp_path / "logs"
    forks = [{"fork_point": "head1 stash1", "branches": [{"branch_id": "branch-0", "patch_text": "diff\n", "task_id": "2"},
                                                         {"branch_id": "branch-1", "patch_text": ""}]}]
    agent.ws = mod.HarborWorkspace(Env())
    agent.ws.initial = "head0"
    asyncio.run(agent._write_plan_inputs(forks))
    assert (root / "tmp/octopus-plan/plan.tsv").read_text() == (
        "fork0-branch-0\thead0\t2\t/tmp/octopus-plan/fork0-branch-0.patch\nfork0-branch-1\thead0\t-\t-\n")


def test_shared_pool_cutoff_ends_concurrent_branches_cleanly(monkeypatch):
    """Two branches share the budget: whichever call finds nothing left ends its branch as a "request failed" step
    (run_branch_continuation catches it), so the fork returns and no exception reaches the episode."""
    mod = _import_agent(monkeypatch)
    from octopus.runtime.core import run_branch_continuation
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent._budget = {"limit": 10, "used": 0}
    agent._complete = lambda messages, s=None, p=None: {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "shell", "arguments": '{"command": "true"}'}}]}}],
        "usage": {"completion_tokens": 4}}

    async def dispatch(name, args, branch_id):  # branch-1 works longer: branch-0's next reply uses up the budget first
        await asyncio.sleep(0.3 if branch_id == "branch-1" else 0)
        return "exit 0\n", False

    async def both():
        starts = [{"branch_id": b, "session_id": b, "task": "t", "messages": [{"role": "user", "content": "go"}]}
                  for b in ("branch-0", "branch-1")]
        return await asyncio.gather(*(run_branch_continuation(s, agent._complete_async, dispatch, max_steps=20)
                                      for s in starts))
    reports = asyncio.run(both())
    assert agent._budget["used"] >= agent._budget["limit"]
    for report in reports:
        assert report["steps"][-1]["name"] is None  # cut off: unclosed, its patch is still saved and graded
    assert reports[0]["steps"][-1]["text"] == "" and "token budget used up" in reports[1]["steps"][-1]["text"]


def test_critical_path_follows_nested_forks():
    reqs = [{"thread": "main", "completion_tokens": 14}, {"thread": "fork0/a", "completion_tokens": 3},
            {"thread": "fork0/b", "completion_tokens": 9}, {"thread": "fork0/a/fork0/x", "completion_tokens": 8},
            {"thread": "fork0/a/fork0/y", "completion_tokens": 1}, {"thread": "fork0/a/fork1/z", "completion_tokens": 2}]
    assert critical_path(reqs) == 14 + (3 + 8 + 2)  # fork0/a's own + its forks' longest, past fork0/b


def _call(name, args):
    return [{"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def _fork_call(*task_ids):
    return _call("fork", {"tasks": [{"task": f"Task {k}", "acceptance": "true", "task_id": k}
                                    for k in task_ids]})


FORK_DEPTH_REFUSAL = "Forking is not available at this depth; finish this task yourself."  # octopus/runtime/tree.py


def _run_tree(monkeypatch, tmp_path, reply, budget=None, red=None, pick=None, drive=None, serve=None, fail=None,
              exec_out=None, **env):
    """GrokHarborAgent.run on a bundle plan episode with a fake model and container. reply(thread, turn) -> tool calls
    or None, where thread is the request's session id (an unbound branch's session is its thread name; "control" for the
    no-fork control). red(command) -> the fork red check's output (default: none, so no fork is refused). pick(seed,
    pool) -> the control's fork call (default: random.Random(seed).choice). drive(run) -> the coroutine run in place of
    agent.run's (e.g. under a timeout). serve(rid, body, calls) -> the response to the request body as sent (e.g. from a
    proxy), or None for the fake model's (response id rid, reply(thread, turn)'s calls). A shell call of `false`
    fails, as does a container command fail(command) gives stderr for; exec_out(command) -> a container command's
    stdout, or None for the fake's. Returns the
    agent, requests.jsonl rows, tree.json, each thread's messages, and the container commands (command, cwd)."""
    import threading

    from octopus.envs.search.dry_run import stub_tokenizer
    for key in ("OCTOPUS_PLAN_CONTROL", "OCTOPUS_MAIN_MAX_CALLS", "OCTOPUS_FORK_DEPTH",
                "OCTOPUS_MAX_THREADS", "OCTOPUS_REWIND"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"OCTOPUS_STRATEGY": "branch_fix", "OCTOPUS_BUNDLE": "1",
                       "OCTOPUS_PLAN_EPISODE": "1", "OCTOPUS_TOKENIZER": stub_tokenizer(),
                       **env}.items():
        monkeypatch.setenv(key, value)
    mod = _import_agent(monkeypatch)
    turns, history, commands, lock, online = collections.Counter(), {}, [], threading.Lock(), []

    def post_chat(url, body, headers, log):
        control = getattr(agent, "_control_running", False)  # the no-fork control, from main's fork or a branch's
        thread = body.get("session_id") or ("control" if control else "main")
        with lock:
            turns[thread] += 1
            turn = turns[thread]
        history[thread] = body["messages"]  # the thread's own list, sent verbatim in plan episodes
        rid, calls = f"{thread}#{turn}", reply(thread, turn)
        served = serve and serve(rid, body, calls)
        if served:
            log({"id": served["id"], "http_code": 200, **served["usage"]})
            return served
        log({"id": rid, "http_code": 200, "prompt_tokens": 10 * turn, "completion_tokens": 1})
        return {"id": rid, "usage": {"completion_tokens": 1}, "choices": [{
            "message": {"content": "", "tool_calls": calls}}]}

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            commands.append((command, cwd))
            if env != {"OCTOPUS_NO_EGRESS": "1"}:  # udocker_nokill.c: no network for any agent command
                online.append(command)
            err = fail and fail(command)
            if err:
                return SimpleNamespace(stdout="", stderr=err, return_code=1)
            if command.startswith("python3 /tmp/harbor_tools.py ") and json.loads(base64.b64decode(
                    command.split()[2]))["args"].get("command") == "false":
                return SimpleNamespace(stdout="exit 1\nfailed", stderr="", return_code=1)
            out = (exec_out and exec_out(command)) or (
                   "head0\n" if "git stash create" in command else (red(command) if red else "") + "fp\n"
                   if command.endswith("git rev-parse HEAD")
                   else "1 file changed, 1 insertion(+)\n" if "--shortstat" in command
                   else "diff --git a/f b/f\n" if "head -c" in command
                   else json.dumps({str(k): f"Statement of task {k}.\n" for k in range(1, 17)}) if "/tasks/" in command
                   else "")
            return SimpleNamespace(stdout=out, stderr="", return_code=0)

        async def upload_dir(self, source_dir, target_dir):
            import shutil
            shutil.copytree(source_dir, tmp_path / "container" / target_dir.lstrip("/"), dirs_exist_ok=True)

    monkeypatch.setattr(sys.modules["octopus.runtime.tree"], "post_chat", post_chat)
    if pick:
        monkeypatch.setattr(sys.modules["octopus.runtime.tree"], "random", SimpleNamespace(Random=lambda seed: SimpleNamespace(
            choice=lambda pool: pick(seed, pool))))
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.logs_dir = tmp_path / "agent"
    agent.logs_dir.mkdir()
    agent._write_trajectory = lambda steps: None
    if budget:  # the runtime's shared pool (SWE has none; search's OCTOPUS_SEARCH_BUDGET), here on a bundle
        monkeypatch.setattr(mod.HarborWorkspace, "budget", lambda self, instruction: {"limit": budget, "used": 0})
    run = agent.run("Fix the bugs.", Env(), SimpleNamespace())
    asyncio.run(drive(run) if drive else run)
    assert not online  # tool calls and red checks included
    rows = [json.loads(x) for x in (tmp_path / "agent/requests.jsonl").read_text().splitlines()]
    return agent, rows, json.loads((tmp_path / "agent/tree.json").read_text()), history, commands


def test_bundle_fork_red_check_starts_every_task_and_warns_weak_ones(monkeypatch, tmp_path):
    """bundle-e2b audit (docs/task-bundles.md): many acceptance checks exit 0 before any fix and cannot detect their
    bug, and some task texts are wrong. A bundle fork runs every check in its branch's copy of the forking thread's tree
    first. Every task still starts (bundle-e4: a dropped task was solved 16% of the time by main, a started one 37% by
    its branch), but a task whose check passes or is broken has its check shown as unreliable, so the runtime does not
    verify its commit with it, and its branch is told to write a check that fails first; the forking thread's note
    names those tasks. A task id without a /tasks/<k>.md refuses the fork. Branches see their task's own statement.
    Every check is in tree.json red_checks."""
    from octopus.envs.swe.prompt import TASK_ID_NEEDS
    own_failure = _import_agent(monkeypatch).own_failure
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug").write_text("")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    checks = ["test -f bug", "touch made && test ! -f /testbed/bug", "Check that bug is fixed", "echo (",
              "python3 -c 'print(1'", "python3 -c 'print(1' || true", "python3 -c 'import bug'",
              "python3 -c 'compile(\"x(\", \"<string>\", \"exec\")'"]
    workdirs, results = _red_in_copies(monkeypatch, tmp_path, repo, checks)
    # /testbed is the copy: its bug is there and its edits stay there; prose and code that does not parse are broken,
    # a traceback through python -c is not, nor is product code compiled at run time failing (bottle's templates)
    assert [code for code, _ in results] == ["0", "1", "127", "syntax", "syntax", "0", "1", "1"]
    assert sorted(p.name for p in repo.iterdir()) == [".git", "bug"]
    assert not (Path(workdirs[1]) / "made").exists()  # each copy is back at its fork point
    # the output's tail comes back too: an import of a module that does not exist fails because of the check itself
    own = [own_failure(c, code, tail) for c, (code, tail) in zip(checks, results)]
    assert own == [None] * 6 + ["ModuleNotFoundError: No module named 'bug'", None]

    def task(k, check):
        return {"task": f"Task {k}: placeholder", "acceptance": check, "task_id": k}

    calls = {1: [task(5, "false"), task(99, "false")],  # task 99 does not exist
             2: [task(5, "nope"), task(6, "true")],  # task 5's check is broken, task 6's passes: both start, warned
             3: [task(5, "false"), task(6, "true"), task(7, "nope"), task(8, "false")]}  # 6 and 7 warned

    def reply(thread, turn):
        return _call("fork", {"tasks": calls[turn]}) if thread == "main" and turn in calls else None

    def red(command):  # the fake container: exit 0 for `true`, 127 for `nope`, else 1
        return "".join(f"{i} {dict(true=0, nope=127).get(c, 1)}\n"
                       for i, c in enumerate(re.findall(r"bash -lc (\S+) >", command)))
    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply, red=red)
    results = [m["content"].split("\n", 1)[1] for m in history["main"] if m["role"] == "tool"]  # budget line first
    assert results[0] == TASK_ID_NEEDS
    notes = [m["content"] for m in history["main"] if "Started with an unreliable acceptance check" in m["content"]]
    assert len(notes) == 2 and notes[1].endswith(
        "\nStarted with an unreliable acceptance check: the acceptance check of task 6 already exits 0 on the current "
        "tree, before any fix, so it cannot detect the bug; the acceptance check of task 7 is not a working command "
        "(command not found, or its code does not parse), so it fails whatever the code does. Those branches were told "
        "to write a check that fails on the current code first. A task that is already fixed needs no branch.")
    assert [[b["task_id"] for b in f["branches"]] for f in tree["forks"]] == [["5", "6"], ["5", "6", "7", "8"]]
    assert [f["weak"] for f in tree["forks"]] == [[0, 1], [1, 2]] and not any("dropped" in f for f in tree["forks"])
    assert tree["red_checks"] == [{"thread": "main", "task_ids": t, "exits": e, "weak": w, "own": {}} for t, e, w in (
        (["5", "6"], ["127", "0"], [0, 1]), (["5", "6", "7", "8"], ["1", "0", "127", "1"], [1, 2]))]
    # in the branch copies, in the command that makes them: /testbed is copied once per branch, not twice
    assert any("cp -a /testbed /tmp/branches/fork1-branch-0 " in c and
               "cd /tmp/branches/fork1-branch-1 && timeout 60 bash -lc true > /tmp/branches/fork1-branch-1.red" in c
               for c, _ in commands)
    starts = {t: next(m["content"] for m in h if m["role"] == "user" and m["content"].startswith("You are branch-"))
              for t, h in history.items() if t.startswith("fork1/")}
    assert "Your task: Task 8: placeholder\nAcceptance (exits 0 when done): false\n" \
           "The statement of task 8 (/tasks/8.md):\nStatement of task 8.\n" in starts["fork1/branch-3"]
    weak6, weak7 = starts["fork1/branch-1"], starts["fork1/branch-2"]
    assert "Acceptance (exits 0 when done)" not in weak6 + weak7
    assert "Acceptance check given at the fork (unreliable: it already exits 0 on the unfixed code, so it cannot " \
           "tell a fix from no fix): true\nFirst write your own check that fails on the current code" in weak6
    assert "(unreliable: it is not a working command, so it cannot tell a fix from no fix): nope" in weak7
    from octopus.envs.swe.prompt import acceptance_of, weak_acceptance
    assert acceptance_of(weak_acceptance("Task 6\nAcceptance (exits 0 when done): true", True)) is None


def test_bundle_fork_with_a_weak_check_is_a_control_candidate(monkeypatch, tmp_path):
    """A fork with a task whose check the red check found unreliable is an accepted fork with every task started: the
    no-fork control can start from it, and every branch is graded on its own task."""
    seen = {}

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(1, 2, 3) if turn == 1 else _call("done", {"summary": "ok"})
        if thread == "control":
            return _call("done", {"summary": "ok"})
        return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})

    def pick(seed, pool):
        seen["paths"] = [c["path"] for c in pool]
        return pool[0]
    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply, red=lambda c: "0 0\n", pick=pick,
                                                     OCTOPUS_PLAN_CONTROL="1")  # task 1's check passes already
    assert seen["paths"] == ["fork0"] and tree["control"]["path"] == "fork0"
    assert [b["task_id"] for b in tree["forks"][0]["branches"]] == ["1", "2", "3"] and tree["forks"][0]["weak"] == [0]
    plan = (tmp_path / "container/tmp/octopus-plan/plan.tsv").read_text().splitlines()
    assert [(r.split("\t")[0], r.split("\t")[2]) for r in plan] == [
        ("fork0-branch-0", "1"), ("fork0-branch-1", "2"), ("fork0-branch-2", "3")]
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("0.5\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({"1": 1, "2": 1, "3": 0}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": "fork0-branch-0", "task": "1", "tasks": {"1": 1}, "passed": True},
        {"key": "fork0-branch-1", "task": "2", "tasks": {"2": 1}, "passed": True},
        {"key": "fork0-branch-2", "task": "3", "tasks": {"3": 0}, "passed": False}],
        "control": {"tasks": {"1": 1, "2": 0, "3": 0}}}))
    assert [b["passed"] for b in plan_outcome(tmp_path, rows)["branches"]] == [True, True, False]


B1, S0, S1 = "fork0/branch-1", "fork0/branch-1/fork0/branch-1.0", "fork0/branch-1/fork0/branch-1.1"


def test_a_fork_whose_branch_copies_fail_is_refused_not_the_episode(monkeypatch, tmp_path):
    """e11: three episodes died with "branch copies of ... failed". tldextract: main's first fork could not copy
    /testbed (cp: a dir main had made 0555, Permission denied); graphene, webargs: a branch forked after its own workdir
    was gone (graphene's main ran `rm -rf /tmp/branches`). The forking thread is told and works on: main forks again
    (the same fork0 ids) and its branch-1, whose copy is gone, commits without sub-branches."""
    denied = "cp: cannot create directory '/tmp/branches/fork0-branch-0/page2': Permission denied"
    gone = "cp: cannot stat '/tmp/branches/fork0-branch-1': No such file or directory"
    copies = collections.Counter()

    def fail(command):
        if " cp -a /testbed /tmp/branches/fork0-branch-0 " in command:
            copies["main"] += 1
            return denied if copies["main"] == 1 else None
        return gone if " cp -a /tmp/branches/fork0-branch-1 " in command else None

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn <= 2 else DONE
        return _fork_call(8, 9) if thread == B1 and turn == 1 else _commit(thread)
    agent, _, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, fail=fail)
    note = "The fork failed, no branch started (branch copies of {} failed: {}). Fork again or do these tasks yourself."
    assert note.format("/testbed", denied) in [m["content"] for m in history["main"]]
    assert note.format("/tmp/branches/fork0-branch-1", gone) in [m["content"] for m in history[B1]]
    assert [b["branch_id"] for f in tree["forks"] for b in f["branches"]] == ["branch-0", "branch-1"]
    assert tree["forks"][0]["branches"][1]["forks"] == [] and agent._live == 1


def test_a_branch_forking_again_runs_real_sub_branches_graded_and_credited(monkeypatch, tmp_path):
    """bundle543-b django-money b01.r1-3:2 (PR #38): branch-1 of a 3-way fork forks again while branch-2 still runs.
    Its sub-branches now run in copies of its tree from its conversation, branch-1 gets their join note, and every
    branch at either depth is named by its path the same way in requests.jsonl, tree.json, plan.tsv and branches.json,
    graded, and credited by its own task."""
    import threading
    nested = threading.Event()

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6, 7) if turn == 1 else None
        if thread == B1:
            return _fork_call(8, 9) if turn == 1 else _call(
                "commit", {"branch_id": "branch-1", "summary": "merged", "acceptance_passed": True}) if turn == 2 else None
        if thread == S0:
            nested.set()
            return _call("shell", {"command": "true"}) if turn == 1 else _call(
                "commit", {"branch_id": "branch-1.0", "summary": "fixed 8", "acceptance_passed": True})
        if thread == S1:  # depth 2 = OCTOPUS_FORK_DEPTH: refused, never a fake success
            return _fork_call(1, 2) if turn == 1 else _call("abort", {"branch_id": "branch-1.1", "reason": "stuck"})
        if thread == "fork0/branch-2":
            assert nested.wait(10)  # still running while branch-1 forks
            return _call("shell", {"command": "true"}) if turn == 1 else _call(
                "commit", {"branch_id": "branch-2", "summary": "fixed 7", "acceptance_passed": True})
        return None  # fork0/branch-0 stops without a call

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply)
    threads = {"main", "fork0/branch-0", B1, "fork0/branch-2", S0, S1}
    assert {r["thread"] for r in rows} == threads and agent._live == 1
    # S0 starts from branch-1's conversation up to its fork call, in a copy of branch-1's tree
    parent, sub = history[B1], history[S0]
    at = [i for i, m in enumerate(parent) if m["role"] == "assistant"][1] + 1  # after branch-1's fork call
    assert sub[:at] == parent[:at] and sub[at + 1]["content"].startswith("You are branch-1.0, one of 2")
    assert any("cp -a /tmp/branches/fork0-branch-1 /tmp/branches/fork0-branch-1-fork0-branch-1.0" in c for c, _ in commands)
    assert "/tmp/branches/fork0-branch-1-fork0-branch-1.0" in {cwd for c, cwd in commands if "harbor_tools.py" in c}
    assert any(m["role"] == "user" and "fork0-branch-1-fork0-branch-1.0.patch" in m["content"] for m in parent)
    # at the depth limit fork is not offered (task message) and a call is refused
    assert history[S1][at + 1]["content"].endswith(FORK_DEPTH_REFUSAL)
    assert any(m["role"] == "tool" and m["content"].endswith(FORK_DEPTH_REFUSAL) for m in history[S1])
    # graded on its tree since branch-1's fork point (the main fork's), which includes branch-1's changes
    assert ("cd /tmp/branches/fork0-branch-1-fork0-branch-1.0 && git diff --cached --binary fp | head -c 2000000"
            in [c for c, _ in commands])

    sub_fork = tree["forks"][0]["branches"][1]["forks"][0]
    assert [b["branch_id"] for b in sub_fork["branches"]] == ["branch-1.0", "branch-1.1"]
    assert sub_fork["response_id"] == f"{B1}#1"
    assert [b["ended_with"] for b in sub_fork["branches"]] == ["commit", "abort"]
    plan_rows = [r.split("\t") for r in (tmp_path / "container/tmp/octopus-plan/plan.tsv").read_text().splitlines()]
    assert [r[0] for r in plan_rows] == ["fork0-branch-0", "fork0-branch-1", "fork0-branch-1-fork0-branch-1.0",
                                         "fork0-branch-1-fork0-branch-1.1", "fork0-branch-2"]
    assert plan_rows[2][1:] == ["head0", "8", "/tmp/octopus-plan/fork0-branch-1-fork0-branch-1.0.patch"]

    solved = {"7", "8"}  # the verifier echoes each plan.tsv key (test_bundle_verifier_grades_...)
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("0.0\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({k: 0 for k in "56789"}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": key, "task": task, "tasks": {task: int(task in solved)}, "passed": task in solved}
        for key, _, task, _ in plan_rows]}))
    outcome = plan_outcome(tmp_path, rows)
    # main forked and then replied without a tool call: it never called done (ended_with was "done" for any fork)
    assert tree["main_end"] == outcome["ended_with"] == "no tool call"
    # OCTOPUS_AUTO_MERGE (bundle default): branch-1.0's commit is applied to its parent branch-1's copy at that join
    assert any(c.startswith("cd /tmp/branches/fork0-branch-1 && t=$(git write-tree) && if err=$(git apply --3way")
               and "/tmp/branches/fork0-branch-1-fork0-branch-1.0.patch" in c for c, _ in commands)
    assert [b["passed"] for b in outcome["branches"]] == [False, False, True, False, True]  # by their own tasks
    episode = plan_group_step([outcome])[0]
    last = {r["thread"]: r["id"] for r in rows}
    out = dict(zip(last, assign_threads([_leaf(rid, [0] * 40) for rid in last.values()], episode)))
    assert out[S0]["own_start"] == 10 and all(r["adv"] == episode["A_main"] for r in out.values())


@pytest.mark.parametrize("env, refusal", [({"OCTOPUS_FORK_DEPTH": "1"}, FORK_DEPTH_REFUSAL),
                                          ({"OCTOPUS_MAX_THREADS": "4"}, "fork refused: 4 threads are running and 2")])
def test_a_branch_fork_is_refused_at_the_depth_limit_and_the_thread_cap(monkeypatch, tmp_path, env, refusal):
    """Main + 3 running branches = 4 threads: a branch's 2-way fork passes OCTOPUS_MAX_THREADS=4, and with
    OCTOPUS_FORK_DEPTH=1 a branch may not fork at all (and is told so where it starts)."""
    import threading
    answered = threading.Event()  # branch-0 has its fork result: its siblings run until then

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6, 7) if turn == 1 else None
        if thread == "fork0/branch-0":
            if turn == 2:
                answered.set()
            return _fork_call(8, 9) if turn == 1 else None
        assert answered.wait(10)
        return None

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, **env)
    assert {r["thread"] for r in rows} == {"main", "fork0/branch-0", "fork0/branch-1", "fork0/branch-2"}
    assert any(m["role"] == "tool" and refusal in m["content"] for m in history["fork0/branch-0"])
    assert not any(b.get("forks") for b in tree["forks"][0]["branches"]) and agent._live == 1
    task = history["fork0/branch-0"][-3]["content"]  # its task message, before its fork call and the result
    assert task.endswith(FORK_DEPTH_REFUSAL) == ("OCTOPUS_FORK_DEPTH" in env)


def test_budget_cutoff_closes_threads_at_every_depth(monkeypatch, tmp_path):
    """The shared budget runs out while a branch waits on its sub-branches: every thread ends at its next request,
    the waiting branch gets its join note, and the episode returns with the nested fork recorded."""
    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn == 1 else None
        if thread == "fork0/branch-0" and turn == 1:
            return _fork_call(7, 8)
        return _call("shell", {"command": "true"})

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, budget=12)
    assert agent._budget["used"] >= 12 and agent._live == 1 and tree["main_end"] == "budget"
    sub = tree["forks"][0]["branches"][0]["forks"][0]["branches"]
    assert len(sub) == 2 and all(b["ended_with"] in (None, "shell") for b in sub)
    assert history["fork0/branch-0"][-1]["content"].startswith("branches finished")
    assert {"fork0/branch-0/fork0/branch-0.0", "fork0/branch-0/fork0/branch-0.1"} <= {r["thread"] for r in rows}


def test_branch_ids_are_unique_across_depths_and_depth_1_requests_are_unchanged(monkeypatch, tmp_path):
    """A depth-1 branch is told branch-<i> as before; its sub-branches are told <its id>.<n>, numbered on across its
    forks, so none shares an id with a depth-1 branch or a sibling's child (pilot: sub-branch "branch-5" read the
    depth-1 branch-5's tree)."""
    from octopus.runtime.core import RuntimePort, tree_branches

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(*range(1, 8)) if turn == 1 else None
        if thread in ("fork0/branch-5", "fork0/branch-6") and turn in (1, 2):
            return _fork_call(8, 9)  # two forks in a row: the second numbers on
        return None

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply)
    ids = {name: b["branch_id"] for name, _, b in tree_branches(tree["forks"])}
    deep = [i for name, i in ids.items() if name.count("/") == 3]
    assert [i for name, i in ids.items() if name.count("/") == 1] == [f"branch-{i}" for i in range(7)]
    assert sorted(deep) == ["branch-5.0", "branch-5.1", "branch-5.2", "branch-5.3",
                            "branch-6.0", "branch-6.1", "branch-6.2", "branch-6.3"]
    assert len(set(ids.values())) == len(ids)
    task = next(m for m in history["fork0/branch-5/fork1/branch-5.3"] if "You are branch-5.3" in str(m.get("content")))
    assert "/tmp/branches/fork0-branch-5-fork1-branch-5.3:" in task["content"]
    # depth 1 as before: the main fork's result and each branch's task message and session
    main_fork = next(m for m in history["main"] if m.get("role") == "tool")
    assert main_fork["content"].endswith("branches branch-0, branch-1, branch-2, branch-3, branch-4, branch-5, branch-6")
    for i in range(7):
        start = history[f"fork0/branch-{i}"][tree["forks"][0]["prefix_len"]]
        assert start["content"].startswith(f"You are branch-{i}, one of 7 parallel branches")
        assert "private copy of the repo at /tmp/branches/fork0-branch-%d:" % i in start["content"]
    port = RuntimePort("branch-2.1")
    port.fork(["a", "b"]), port.fork(["c"])
    assert [b["branch_id"] for b in port.last_fork["branches"]] == ["branch-2.1.2"]


def test_auto_merge_applies_committed_patches_in_commit_order_one_conflicting(monkeypatch, tmp_path):
    """OCTOPUS_AUTO_MERGE on a real repo: branch-1 committed first and merges; branch-0 edits the same line, conflicts
    and is taken back whole (no markers in the tree); aborted branch-2's patch is not applied. The join note says so."""
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    mod = _import_agent(monkeypatch)
    from octopus.envs.swe.prompt import join_note
    repo = tmp_path / "testbed"
    script = f"""set -e
git init -q {repo} && cd {repo} && git config user.email t@t && git config user.name t
printf 'a\\nb\\nc\\nd\\ne\\n' > f.py && git add f.py && git commit -qm base
for x in 0:X:b:e 1:A:b:E 2:a:B:e; do IFS=: read n l1 l2 l5 <<< "$x"
  printf '%s\\n%s\\nc\\nd\\n%s\\n' $l1 $l2 $l5 > f.py && git diff --binary > {tmp_path}/b$n.patch; done
git checkout -q f.py
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            ran = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=ran.stdout, stderr=ran.stderr, return_code=ran.returncode)

    ended = {"branch-0": "commit", "branch-1": "commit", "branch-2": "abort"}
    branches = [{"branch_id": b, "ended_with": e, "patch": f"{tmp_path}/b{b[-1]}.patch", "diffstat": "1 file changed"}
                for b, e in ended.items()]
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.ws = mod.HarborWorkspace(Env())
    merged = asyncio.run(agent._merge_committed(str(repo), branches, ["branch-2", "branch-1", "branch-0"]))
    assert merged == {"branch-1": "merged", "branch-0": "conflict: f.py"}
    assert (repo / "f.py").read_text() == "A\nb\nc\nd\nE\n"  # branch-1's alone
    assert not subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=repo, capture_output=True,
                              text=True).stdout

    reports = [{"branch_id": b, "task": "t", "steps": [{"name": e, "arguments": {"summary": "s", "reason": "r"}}]}
               for b, e in ended.items()]
    note = join_note(reports, {b["branch_id"]: (b["patch"], b["diffstat"]) for b in branches}, merged)
    lines = note.splitlines()
    assert lines[2].endswith("(1 file changed); not applied: it conflicts in f.py; your files are unchanged")
    assert lines[4].endswith("(1 file changed); merged")
    assert lines[6].endswith(f"git apply --3way {tmp_path}/b2.patch")  # aborted: not applied, left to the model
    assert "do not apply those again" in note and "was not applied" in note


def test_auto_merge_keeps_a_siblings_scratch_files_and_merges_the_rest(monkeypatch, tmp_path):
    """A patch whose only conflicts are scratch files it creates (a root file, a test_ module) that root already has
    merges without them; a conflicting new file elsewhere (pkg/util.py) or a tracked one still takes it back whole."""
    mod = _import_agent(monkeypatch)
    repo = tmp_path / "testbed"
    script = f"""set -e
git init -q {repo} && cd {repo} && git config user.email t@t && git config user.name t
seq 12 > f.py && git add f.py && git commit -qm base
for x in 1:1:A:one 0:12:L:two 2:6:F:two 3:1:X:three; do IFS=: read n line text w <<< "$x"
  sed -i "${{line}}s/.*/$text/" f.py && mkdir -p tests && echo $w > test_bug.py && echo $w > tests/test_x.py
  [ $n = 0 ] || [ $n = 3 ] || {{ mkdir -p pkg && echo $w > pkg/util.py; }}
  git add -A && git diff --cached --binary HEAD > {tmp_path}/b$n.patch && git reset -q --hard && git clean -fdq; done
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            ran = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=ran.stdout, stderr=ran.stderr, return_code=ran.returncode)

    ws = mod.HarborWorkspace(Env())
    merged = asyncio.run(ws.apply(str(repo), [(f"b{n}", f"{tmp_path}/b{n}.patch") for n in (1, 0, 2, 3)]))
    assert merged == {"b1": "merged",
                      "b0": "merged: except its new files test_bug.py, tests/test_x.py (a sibling's of the same names kept)",
                      "b2": "conflict: pkg/util.py, test_bug.py, tests/test_x.py",
                      "b3": "conflict: f.py, test_bug.py, tests/test_x.py"}
    assert (repo / "f.py").read_text().split() == ["A", *map(str, range(2, 12)), "L"]  # b1's, b0's; not b2's, b3's
    assert (repo / "test_bug.py").read_text() == (repo / "pkg/util.py").read_text() == "one\n"
    assert not subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=repo, capture_output=True,
                              text=True).stdout


def test_main_ending_and_a_done_episode(monkeypatch, tmp_path):
    """tree.json main_end: how the main thread really ended (plan_outcome ended_with)."""
    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn == 1 else _call("done", {"summary": "ok"})
        return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply)
    assert tree["main_end"] == "done"
    merges = [c for c, _ in commands if c.startswith("cd /testbed && t=$(git write-tree) && if err=$(git apply --3way")]
    assert sorted(m.split()[11] for m in merges) == ["/tmp/branches/fork0-branch-0.patch", "/tmp/branches/fork0-branch-1.patch"]
    end = sys.modules["octopus.runtime.tree"].main_ending
    assert end([{"name": "done", "rejected": True}, {"name": None, "text": "bye"}], False) == "no tool call"
    assert end([{"name": "shell"}], False) == "step cap"  # OCTOPUS_MAIN_MAX_CALLS calls
    assert end([{"name": "shell"}], True) == "budget"
    assert end([{"name": "context"}], False) == "context"  # it grew past thread_budget
    assert end([{"name": None, "text": "stopped: no successful fork within 5 turns"}], False) == "step cap"
    assert end([{"name": None, "text": "request failed: timeout"}], False) == "request failed"


def test_bundle_group_with_controls_rewards_advantages_and_log():
    """A bundle group with controls: A_main is leave-one-out over the mains alone (y = solved fraction), as without
    them, and the group log pairs each forked y with its control's y."""

    def group(control):
        return [{"y": 0.5, "C": 10, "forked": True, "tasks": {"1": 1, "2": 0},
                 "control": {"y": 0.25, "C": 8, "kind": "main-first"} if control else None,
                 "branches": [_bundle_branch("commit", True, True), _bundle_branch("abort", False, False)]},
                {"y": 0.75, "C": 20, "forked": True, "tasks": {"1": 1, "2": 1},
                 "control": {"y": 1.0, "C": 9, "kind": "main-later"} if control else None,
                 "branches": [_bundle_branch("commit", False, False)]},
                {"y": 0.0, "C": 30, "forked": False, "control": None, "tasks": {"1": 0, "2": 0}, "branches": []}]
    with_control, without = plan_group_step(group(True)), plan_group_step(group(False))
    assert [e["R"] for e in with_control] == [0.0, 0.5, -1.0]
    assert [e["A_main"] for e in with_control] == [e["A_main"] for e in without] == pytest.approx([0.25, 1.0, -1.25])
    assert all(b["A_branch"] == e["A_main"] for e in with_control for b in e["branches"])
    stats = plan_group_stats(with_control)
    assert stats["fork_vs_control"] == {"main-first": [[0.5, 0.25]], "main-later": [[0.75, 1.0]]}
    assert stats["fork_minus_control"] == {"main-first": 0.25, "main-later": -0.25}
    assert plan_group_stats(without)["fork_vs_control"] == {} == plan_group_stats(without)["fork_minus_control"]


def test_plan_outcome_bundle_control_y_is_its_solved_fraction(tmp_path):
    """The bundle verifier's control {tasks} -> control y = solved fraction, C = main's tokens before the fork + the
    control's. A control the verifier did not grade rejects the episode."""
    branches = [{"key": "fork0-branch-0", "task": "2", "tasks": {"2": 1}, "passed": True},
                {"key": "fork0-branch-1", "task": "1", "tasks": {"1": 0}, "passed": False}]
    trial = _trial(tmp_path, {"branches": branches, "control": {"tasks": {"1": 1, "2": 0, "3": 0, "4": 0}}})
    (trial / "verifier/reward.txt").write_text("0.5\n")
    (trial / "verifier/tasks.json").write_text(json.dumps({"1": 0, "2": 1, "3": 1, "4": 0}))
    tree = json.loads((trial / "agent/tree.json").read_text())
    (trial / "agent/tree.json").write_text(json.dumps({**tree, "control": {"kind": "main-first", "response_id": "m1"}}))
    reqs = [{"id": "m0", "thread": "main", "completion_tokens": 7}, {"id": "m1", "thread": "main", "completion_tokens": 5},
            {"id": "b", "thread": "fork0/branch-0", "completion_tokens": 3}, {"id": "m2", "thread": "main", "completion_tokens": 1},
            {"id": "c0", "thread": "control", "completion_tokens": 4}, {"id": "c1", "thread": "control", "completion_tokens": 2}]
    out = plan_outcome(trial, reqs)
    assert out["control"] == {"y": 0.25, "C": 7 + 4 + 2, "kind": "main-first", "C_wall": 0.0} and out["y"] == 0.5
    del json.loads((trial / "verifier/branches.json").read_text())["control"]
    (trial / "verifier/branches.json").write_text(json.dumps({"branches": branches}))
    with pytest.raises(RuntimeError, match="control=False"):
        plan_outcome(trial, reqs)


def test_main_control_and_branches_share_one_context_cap_and_label(monkeypatch, tmp_path):
    """SWE as search: with OCTOPUS_THREAD_BUDGET unset every thread (main, each branch, the main-first control) may add
    32768 tokens to its context, its tool results show what is left, and its reply past it is dropped."""
    monkeypatch.delenv("OCTOPUS_THREAD_BUDGET", raising=False)

    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) if turn == 1 else DONE
        return _call("shell", {"command": "true"}) if turn == 1 else DONE if thread == "control" else _commit(thread)

    def serve(rid, body, calls):  # every thread's second request grew 32769 tokens past its first
        prompt = 1000 + 32769 * (int(rid.split("#")[1]) - 1)
        return {"id": rid, "usage": {"prompt_tokens": prompt, "completion_tokens": 1},
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply, serve=serve,
                                                     OCTOPUS_PLAN_CONTROL="1")
    from octopus.runtime.core import limits_note
    assert tree["main_end"] == "context" and tree["control"]["end"] == "context"
    assert [b["ended_with"] for b in tree["forks"][0]["branches"]] == ["context", "context"]
    # the system prompt states the tag as search's does (main's calls uncapped here)
    assert limits_note(True, calls=False) in history["main"][0]["content"]
    assert limits_note(False, calls=False) in history["control"][0]["content"]
    for thread in ("main", "fork0/branch-0", "fork0/branch-1", "control"):  # after its first reply: 1 token of growth
        tag = [m["content"] for m in history[thread] if m["role"] == "tool"][-1].split("\n")[0]
        assert re.fullmatch(r"\[context left: 32767( \| calls left: \d+)?\]", tag), tag
    # no shared token pool in SWE: no thread is told one, none is stopped by one
    assert not any("token budget" in str(m.get("content")) for msgs in history.values() for m in msgs)
    assert tree["budget"] is None and not any("budget used up" in str(b["stop"]) for b in tree["forks"][0]["branches"])


def test_bundle_control_starts_where_main_forked_with_its_budget_and_calls_left(monkeypatch, tmp_path):
    """OCTOPUS_PLAN_CONTROL in a bundle: main makes 2 calls (2 tokens), forks, and finishes. The control restores the
    fork point, replays main's history up to the forking reply under the linear bundle system message and tools (no
    fork), and continues with main's budget and step cap left there: 4 − 2 calls, tokens counted from 2. main's
    tree.json budget stays its own; control.snap is a verifier input; the control leaf is graded, never trained."""
    from octopus.envs.swe.prompt import delivery

    def reply(thread, turn):
        if thread == "main":
            return _call("shell", {"command": "true"}) if turn < 3 else _fork_call(5, 6) if turn == 3 else _call(
                "done", {"summary": "ok"})
        if thread == "control":
            return _call("shell", {"command": "true"})  # never done: ends at the step cap
        return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply, budget=100,
                                                     OCTOPUS_PLAN_CONTROL="1", OCTOPUS_MAIN_MAX_CALLS="4")
    control = [r for r in rows if r["thread"] == "control"]
    assert len(control) == 2 and all("fork" not in r["tools"] for r in control)
    assert tree["control"] == {"kind": "main-first", "path": "fork0", "depth": 0, "response_id": "main#3",
                               "calls_before": 2, "budget_before": 2, "end": "step cap", "budget_used": 4,
                               "reminded": 0, "wall_s": tree["control"]["wall_s"]}
    assert tree["budget"] == {"limit": 100, "used": 6} and tree["main_end"] == "done"  # main 4 + 2 branches
    sent = history["control"]
    assert sent[0]["content"] == delivery(linear=True) and "/tasks/INDEX.md" in sent[0]["content"]
    assert not any(w in sent[0]["content"] for w in ("task_id", "commit"))
    assert sent[1:6] == history["main"][1:6]  # main's history before the fork call, budget tags included
    # tagged by main at its forking request (2 of its 4 calls made), then by the control with the calls it has left,
    # 100 − 2 − its first reply tokens and the context main had left at the fork (32768 less its growth 30 − 10)
    assert sent[5]["content"].startswith("[token budget left: 98 | context left: 32768 | calls left: 2]\n")
    assert sent[7]["content"].startswith("[token budget left: 97 | context left: 32748 | calls left: 1]\n")
    restores = [c for c, _ in commands if "git reset -q --hard" in c and "cp -a" not in c]  # not the red checks'
    assert len(restores) == 2  # to the fork point, then back to the final tree
    assert (tmp_path / "container/tmp/octopus-plan/control.snap").read_text() == "head0"

    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("1.0\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({"5": 1, "6": 1}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"control": {"tasks": {"5": 1, "6": 0}}, "branches": [
        {"key": f"fork0-branch-{i}", "task": k, "tasks": {k: 1}, "passed": True} for i, k in enumerate("56")]}))
    episode = plan_group_step([plan_outcome(tmp_path, rows)])[0]
    assert episode["control"]["y"] == 0.5 and episode["control"]["C"] == 2 + 2
    last = {r["thread"]: r["id"] for r in rows}
    out = dict(zip(last, assign_threads([_leaf(rid, [0] * 40) for rid in last.values()], episode)))
    assert (out["control"]["reward"], out["control"]["adv"], out["control"]["own_start"]) == (1.0, 0.0, 40)


def _three_fork_tree(monkeypatch, tmp_path, control_reply, path, **env):
    """A bundle episode with every kind of fork call: main's first (refused: task 99 names no task, no fork point), main's
    fork0 (5, 6), branch-1's nested fork (8, 9) at its second turn, main's fork1 (3, 4), then done. The control's fork
    call is the one at `path`; returns _run_tree's output plus (seed, candidate paths) the choice saw."""
    red_calls, seen = [], {}

    def red(command):
        red_calls.append(command)
        return ""

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(1, 99), 2: _fork_call(5, 6), 3: _fork_call(3, 4)}.get(
                turn, _call("done", {"summary": "ok"}))
        if thread == "control":
            return control_reply(turn)
        if thread == B1 and turn < 3:
            return _call("shell", {"command": "true"}) if turn == 1 else _fork_call(8, 9)
        return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})

    def pick(seed, pool):
        seen.update(seed=seed, paths=[c["path"] for c in pool])
        return next(c for c in pool if c["path"] == path)
    out = _run_tree(monkeypatch, tmp_path, reply, red=red, pick=pick, OCTOPUS_PLAN_CONTROL="1", **env)
    assert len(red_calls) == 3  # main's 2 accepted fork calls and branch-1's; refused calls run no check
    return (*out, seen)


def _assistant_at(messages, n):
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"][n]


def test_control_point_is_any_accepted_fork_call_and_a_later_main_fork_restarts_there(monkeypatch, tmp_path):
    """The control's fork call is drawn (seeded by the trial's name) from every accepted fork call of the tree, nested
    ones included and a refused one not; main's later fork restarts from that fork's tree with main's history up
    to it (the first fork's join included) and the calls main had left."""
    agent, rows, tree, history, commands, seen = _three_fork_tree(
        monkeypatch, tmp_path, lambda turn: _call("done", {"summary": "ok"}), "fork1")
    assert seen == {"seed": tmp_path.name, "paths": ["fork0", "fork0/branch-1/fork0", "fork1"]}
    assert tree["control"] == {"kind": "main-later", "path": "fork1", "depth": 0, "response_id": "main#3",
                               "calls_before": 2, "budget_before": None, "end": "done",
                               "budget_used": None, "reminded": 0, "wall_s": tree["control"]["wall_s"]}
    at = _assistant_at(history["main"], 2)  # main's forking reply
    sent = history["control"]
    assert sent[1:at] == history["main"][1:at] and len(sent) == at + 2
    assert any(m["role"] == "user" and m["content"].startswith("branches finished") for m in sent)
    assert len([c for c, _ in commands if c.startswith("cd /testbed && git reset -q --hard")]) == 2  # there and back
    assert (tmp_path / "container/tmp/octopus-plan/control.snap").exists()


def test_nested_control_restarts_the_branch_from_its_fork_without_fork_and_is_not_trained(monkeypatch, tmp_path):
    """A nested fork chosen: branch-1's workdir goes back to its tree at the fork, its history up to the forking reply
    replays with fork refused as at the depth limit (statement block kept), under its turns left (cap 3 − 1); the patch is graded like the branch's, on its task. Training: the control is graded (the group log's
    nested pair), never trained, and moves nothing else."""
    import copy
    replies = {1: _fork_call(1, 2), 2: _call("shell", {"command": "true"}),
               3: _call("commit", {"branch_id": "branch-1", "summary": "x", "acceptance_passed": True})}
    agent, rows, tree, history, commands, seen = _three_fork_tree(
        monkeypatch, tmp_path, replies.get, "fork0/branch-1/fork0", OCTOPUS_BRANCH_MAX_TURNS="3")
    assert tree["control"] == {"kind": "nested", "path": "fork0/branch-1/fork0", "depth": 1, "response_id": f"{B1}#2",
                               "calls_before": 1, "budget_before": None, "end": "shell",
                               "budget_used": None, "reminded": 0, "wall_s": tree["control"]["wall_s"]}
    assert tree["forks"][0]["branches"][1]["forks"][0]["workdir_point"] == "head0"
    assert "cd /tmp/branches/fork0-branch-1 && git reset -q --hard head0 && git clean -fdq" in [c for c, _ in commands]
    assert not [c for c, _ in commands if c.startswith("cd /testbed && git reset")]  # main's tree is not touched
    sent, own = history["control"], history[B1]
    task = next(i for i, m in enumerate(own) if m["role"] == "user" and m["content"].startswith("You are branch-1,"))
    at = task + 1 + _assistant_at(own[task + 1:], 1)  # branch-1's forking reply, its second turn
    assert sent[:task] == own[:task] and sent[task + 1:at] == own[task + 1:at]
    assert sent[task]["content"] == own[task]["content"] + "\n" + FORK_DEPTH_REFUSAL
    assert "Statement of task 6." in sent[task]["content"]
    assert sent[at + 1]["content"].endswith(FORK_DEPTH_REFUSAL)  # its fork call is refused
    assert sum(r["thread"] == "control" for r in rows) == 2  # OCTOPUS_BRANCH_MAX_TURNS 3 − branch-1's 1 turn before
    assert ("cd /tmp/branches/fork0-branch-1 && git diff --cached --binary fp | head -c 2000000"
            in [c for c, _ in commands])  # its changes since the branch's own fork point
    plan = tmp_path / "container/tmp/octopus-plan"
    assert plan.joinpath("plan.tsv").read_text().splitlines()[-1].split("\t") == [
        "control", "head0", "6", "/tmp/octopus-plan/control.patch"]
    assert not plan.joinpath("control.snap").exists()

    tasks = {"5": "0", "6": "1", "8": "1", "9": "0", "3": "1", "4": "0"}
    keys = {"fork0-branch-0": "5", "fork0-branch-1": "6", "fork0-branch-1-fork0-branch-1.0": "8",
            "fork0-branch-1-fork0-branch-1.1": "9", "fork1-branch-0": "3", "fork1-branch-1": "4"}
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("0.5\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps({k: int(v) for k, v in tasks.items()}))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": key, "task": k, "tasks": {k: int(tasks[k])}, "passed": tasks[k] == "1"} for key, k in keys.items()]
        + [{"key": "control", "task": "6", "tasks": {"6": 0}, "passed": False}]}))
    out = plan_outcome(tmp_path, rows)
    assert out["control"] is None and out["y"] == 0.5
    assert out["branch_control"] == {"of": "fork0-branch-1", "passed": False}
    episode = plan_group_step([copy.deepcopy(out)])[0]
    alone = plan_group_step([{**copy.deepcopy(out), "branch_control": None}])[0]
    assert {k: v for k, v in episode.items() if k != "branch_control"} == {k: v for k, v in alone.items()
                                                                            if k != "branch_control"}
    last = {r["thread"]: r["id"] for r in rows}
    out_rows = dict(zip(last, assign_threads([_leaf(rid, [0] * 40) for rid in last.values()], episode)))
    assert (out_rows["control"]["adv"], out_rows["control"]["own_start"], out_rows["control"]["reward"]) == (
        0.0, 40, episode["R"])
    assert plan_group_stats([episode])["fork_vs_control"] == {"nested": [[1.0, 0.0]]}


DONE = _call("done", {"summary": "ok"})


def _commit(thread):
    return _call("commit", {"branch_id": thread.split("/")[-1], "summary": "fixed", "acceptance_passed": True})


def test_bundle_fork_waits_for_its_branches_and_returns_their_join(monkeypatch, tmp_path):
    """main's fork returns only once both branches have closed: main's next request comes after every branch request,
    and the fork's result is followed by the join note (both patches merged into /testbed). No join tool is offered.
    Every tool result carries the calls left (main: OCTOPUS_MAIN_MAX_CALLS; a branch: OCTOPUS_BRANCH_MAX_TURNS turns)."""
    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _call("view", {"path": "."})}.get(turn, DONE)
        return _call("shell", {"command": "true"}) if turn == 1 else _commit(thread)

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_MAIN_MAX_CALLS="10")
    sent = [r["thread"] for r in rows if r.get("id")]
    assert sent.index("main", 1) > max(i for i, t in enumerate(sent) if t.startswith("fork0/"))
    assert not any("join" in r["tools"] for r in rows)
    main = history["main"]
    at = _assistant_at(main, 0)  # the forking reply: its result, then the join note
    assert main[at + 1]["role"] == "tool" and main[at + 1]["content"].endswith("fork started: branches branch-0, branch-1")
    note = main[at + 2]["content"]
    assert main[at + 2]["role"] == "user" and note.startswith("branches finished") and note.count("changes saved to") == 2
    assert "is fixed: do not re-check or redo it" in note and "re-run the acceptance" not in note
    results = [m["content"] for m in main if m["role"] == "tool"]
    assert [r.split("]\n", 1)[0].split(" | ")[1] for r in results[:2]] == ["calls left: 9", "calls left: 8"]
    own = [m["content"] for m in history["fork0/branch-0"] if m["role"] == "tool"][-2]  # its shell result
    assert own.split("\n")[0].endswith(" | calls left: 99]")  # OCTOPUS_BRANCH_MAX_TURNS 100 − its 1 turn
    fork = tree["forks"][0]
    assert (fork["response_id"], fork["join_response_id"], tree["main_end"]) == ("main#1", "main#1", "done")
    assert fork["join_index"] == at + 2  # where the join note goes
    for b in fork["branches"]:
        assert fork["fork_time"] <= b["start_time"] <= b["end_time"] <= fork["join_time"]
        assert b["ended_with"] == "commit" and b["merge"] and b["join_response_id"] == "main#1"
    assert len([c for c, _ in commands if "git apply --3way" in c]) == 2 and agent._live == 1


@pytest.mark.parametrize("name", ["commit", "abort"])
def test_main_commit_or_abort_in_a_bundle_is_answered_as_a_no_op_not_gated_or_recorded(monkeypatch, tmp_path, name):
    """e10 s40: main made 3.7 commit calls per forked episode, gated as if it were a branch ("call commit again") and
    then answered "commit recorded". main's commit or abort closes nothing (its fork has joined its branches): one reply
    says so and done ends main."""
    args = {"branch_id": "branch-0", "summary": "s", "acceptance_passed": True, "reason": "r"}

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: _call(name, args)}.get(turn, DONE)
        return _commit(thread)

    agent, rows, tree, history, commands = _run_tree(monkeypatch, tmp_path, reply)
    results = [m["content"] for m in history["main"] if m["role"] == "tool"]
    assert results[1].endswith("\n" + sys.modules["octopus.runtime.tree"].MAIN_CLOSE.format(name))
    assert [n for n, _ in agent._runtime.calls] == ["fork"]
    commit = next(t for t in tree["tools"] if t["function"]["name"] == "commit")
    assert "the main thread never commits" in commit["function"]["description"]


def test_join_merges_in_commit_order_and_takes_a_conflicting_patch_back_whole(monkeypatch, tmp_path):
    """On a real repo: before its fork main edited g.py line 5 and k.py and added h.py (unstaged). The join stages
    them, then merges in commit order: branch-1 (g.py line 1) beside main's g.py edit, branch-0 (f.py line 1); branch-2
    edits f.py line 1 too, conflicts with branch-0's merged patch and is taken back whole: nothing else is lost."""
    mod = _import_agent(monkeypatch)
    repo = tmp_path / "testbed"
    script = f"""set -e
git init -q {repo} && cd {repo} && git config user.email t@t && git config user.name t
for f in f g k; do printf 'a\\nb\\nc\\nd\\ne\\n' > $f.py; done; git add . && git commit -qm base
sed -i 1s/a/A/ f.py && git diff --binary > {tmp_path}/b0.patch && git checkout -q f.py
sed -i 1s/a/X/ g.py && git diff --binary > {tmp_path}/b1.patch && git checkout -q g.py
sed -i 1s/a/Z/ f.py && git diff --binary > {tmp_path}/b2.patch && git checkout -q f.py
sed -i 5s/e/Y/ g.py && sed -i 3s/c/K/ k.py && echo new > h.py
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            ran = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=ran.stdout, stderr=ran.stderr, return_code=ran.returncode)

    branches = [{"branch_id": f"branch-{i}", "ended_with": "commit", "patch": f"{tmp_path}/b{i}.patch",
                 "diffstat": "1 file changed"} for i in (0, 1, 2)]
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.ws = mod.HarborWorkspace(Env())
    merged = asyncio.run(agent._merge_committed(str(repo), branches, ["branch-1", "branch-0", "branch-2"]))
    assert merged == {"branch-1": "merged", "branch-0": "merged", "branch-2": "conflict: f.py"}
    assert (repo / "g.py").read_text() == "X\nb\nc\nd\nY\n" and (repo / "k.py").read_text() == "a\nb\nK\nd\ne\n"
    assert (repo / "h.py").read_text() == "new\n"
    assert (repo / "f.py").read_text() == "A\nb\nc\nd\ne\n"


def test_critical_path_counts_overlapping_main_and_branch_work_once():
    """fork0 runs from main's reply m1 to its join at m3 while main works (m2): main resumes after the longer of its own
    work and the longest branch. Without the span the fork waits (as before), in tokens and in calls; a nested fork
    that waited records the same id twice."""
    from octopus.rl.group import fork_spans
    reqs = [{"id": "m0", "thread": "main", "completion_tokens": 5}, {"id": "m1", "thread": "main", "completion_tokens": 1},
            {"id": "a0", "thread": "fork0/branch-0", "completion_tokens": 4},
            {"id": "a1", "thread": "fork0/branch-0", "completion_tokens": 3},
            {"id": "a2", "thread": "fork0/branch-0", "completion_tokens": 3},
            {"id": "b0", "thread": "fork0/branch-1", "completion_tokens": 4},
            {"id": "m2", "thread": "main", "completion_tokens": 3}, {"id": "m3", "thread": "main", "completion_tokens": 1},
            {"id": "m4", "thread": "main", "completion_tokens": 2}, {"id": "c0", "thread": "control", "completion_tokens": 50}]
    tree = [{"response_id": "m1", "join_response_id": "m3", "branches": [
        {"branch_id": "branch-0", "forks": [{"response_id": "a1", "join_response_id": "a1", "branches": []}]},
        {"branch_id": "branch-1"}]}]
    spans = fork_spans(tree)
    assert spans == {"fork0/branch-0": ("m1", "m3"), "fork0/branch-1": ("m1", "m3")}  # branch-0's fork has no branch
    assert critical_path(reqs) == 12 + 10  # waits: main's 12 + branch-0's 10
    assert critical_path(reqs, spans) == max(5 + 1 + 3 + 1, 5 + 1 + 10) + 2
    assert critical_path(reqs, unit="calls") == 5 + 3
    assert critical_path(reqs, spans, "calls") == max(4, 2 + 3) + 1
    assert control_cost(reqs, "m1", spans, "calls") == 1 + 1  # m0, then the control's one reply


def test_critical_path_takes_the_longest_branch_of_a_fork():
    """A fork's branches start together: the thread resumes after its longest branch, whether the fork blocks (forked
    and collected in one reply, as every fork now) or ran in the background (old tree.json records)."""
    from octopus.rl.group import fork_spans
    # blocking: m0 (10) forks a (30) and b (40 + 20), collected in m0; then m1 (5)
    sync = [{"id": "m0", "thread": "main", "completion_tokens": 10}, {"id": "a0", "thread": "fork0/a", "completion_tokens": 30},
            {"id": "b0", "thread": "fork0/b", "completion_tokens": 40}, {"id": "b1", "thread": "fork0/b", "completion_tokens": 20},
            {"id": "m1", "thread": "main", "completion_tokens": 5}]
    spans = fork_spans([{"response_id": "m0", "join_response_id": "m0", "branches": [{"branch_id": "a"}, {"branch_id": "b"}]}])
    assert critical_path(sync, spans) == 10 + 60 + 5
    assert critical_path(sync, spans, "calls") == 1 + 2 + 1
    # background: m0 forks a (30) and b (60), main works m1 (25), m2 (5) collects both (its model call overlaps them)
    bg = [{"id": "m0", "thread": "main", "completion_tokens": 10}, {"id": "a0", "thread": "fork0/a", "completion_tokens": 30},
          {"id": "b0", "thread": "fork0/b", "completion_tokens": 60}, {"id": "m1", "thread": "main", "completion_tokens": 25},
          {"id": "m2", "thread": "main", "completion_tokens": 5}, {"id": "m3", "thread": "main", "completion_tokens": 2}]
    spans = fork_spans([{"response_id": "m0", "join_response_id": "m2", "branches": [{"branch_id": "a"}, {"branch_id": "b"}]}])
    assert critical_path(bg, spans) == max(10 + 25 + 5, 10 + 60) + 2
    # nested: main's background fork of x and y (m0 -> m2); x's reply x0 (5) blocks on its fork of c (8) and d (3)
    nested = [{"id": "m0", "thread": "main", "completion_tokens": 10}, {"id": "x0", "thread": "fork0/x", "completion_tokens": 5},
              {"id": "c0", "thread": "fork0/x/fork0/c", "completion_tokens": 8},
              {"id": "d0", "thread": "fork0/x/fork0/d", "completion_tokens": 3},
              {"id": "x1", "thread": "fork0/x", "completion_tokens": 2}, {"id": "y0", "thread": "fork0/y", "completion_tokens": 12},
              {"id": "m1", "thread": "main", "completion_tokens": 4}, {"id": "m2", "thread": "main", "completion_tokens": 1}]
    forks = [{"response_id": "m0", "join_response_id": "m2", "branches": [
        {"branch_id": "x", "forks": [{"response_id": "x0", "join_response_id": "x0",
                                      "branches": [{"branch_id": "c"}, {"branch_id": "d"}]}]},
        {"branch_id": "y"}]}]
    assert critical_path(nested, fork_spans(forks)) == max(10 + 4 + 1, 10 + (5 + 8 + 2), 10 + 12)


def test_critical_path_in_wall_seconds():
    """unit "wall": a request costs the seconds to its thread's next request (the last one: to the thread's end in
    `ends`, tools after its final reply included; else its model time); fork0 runs from main's m1 to its join m3, and
    main resumes after the later of its own time and the branch's."""
    from octopus.rl.group import fork_spans
    reqs = [{"id": "m0", "thread": "main", "t": 0, "elapsed_s": 1}, {"id": "m1", "thread": "main", "t": 10, "elapsed_s": 1},
            {"id": "a0", "thread": "fork0/branch-0", "t": 11, "elapsed_s": 1},
            {"id": "a1", "thread": "fork0/branch-0", "t": 30, "elapsed_s": 1},
            {"id": "m2", "thread": "main", "t": 12, "elapsed_s": 1}, {"id": "m3", "thread": "main", "t": 20, "elapsed_s": 2}]
    spans = fork_spans([{"response_id": "m1", "join_response_id": "m3", "branches": [{"branch_id": "branch-0"}]}])
    assert critical_path(reqs, spans, "wall") == max(10 + 2 + 8 + 2, 10 + 2 + 19 + 1)  # main to m2, then the branch
    assert critical_path(reqs, unit="wall") == 10 + 2 + 8 + 2 + 20  # no span: the fork waits
    # the branch ended at 40 (its commit's acceptance check ran 9 s after its last reply's model call)
    assert critical_path(reqs, spans, "wall", {"fork0/branch-0": 40}) == 10 + 2 + 19 + 10
    # a branch's own fork blocks (forked and collected in its reply b0): b0's cost (2 -> 51) already holds c's 4-50 s.
    # main forks b in the background at m0, waits in join (m1) until b ends at 55, done at 57-58
    nested = [{"id": "m0", "thread": "main", "t": 0, "elapsed_s": 1}, {"id": "m1", "thread": "main", "t": 1, "elapsed_s": 1},
              {"id": "b0", "thread": "fork0/b", "t": 2, "elapsed_s": 1}, {"id": "c0", "thread": "fork0/b/fork0/c", "t": 4},
              {"id": "c1", "thread": "fork0/b/fork0/c", "t": 40, "elapsed_s": 2}, {"id": "b1", "thread": "fork0/b", "t": 51},
              {"id": "m2", "thread": "main", "t": 57, "elapsed_s": 1}]
    forks = [{"response_id": "m0", "join_response_id": "m1", "branches": [
        {"branch_id": "b", "forks": [{"response_id": "b0", "join_response_id": "b0", "branches": [{"branch_id": "c"}]}]}]}]
    assert critical_path(nested, fork_spans(forks), "wall", {"fork0/b": 55, "fork0/b/fork0/c": 50}) == 58
    # main forks and calls done in one reply (model 0-1 s), collected at its end: that reply costs only its model
    # time, so its longest branch (b: 2-40 s; c: 2-30 s, alongside) still counts from the forking reply's end
    last = [{"id": "m0", "thread": "main", "t": 0, "elapsed_s": 1}, {"id": "b0", "thread": "fork0/b", "t": 2, "elapsed_s": 1},
            {"id": "c0", "thread": "fork0/c", "t": 2, "elapsed_s": 1}]
    spans = fork_spans([{"response_id": "m0", "join_response_id": "m0", "branches": [{"branch_id": "b"}, {"branch_id": "c"}]}])
    assert critical_path(last, spans, "wall", {"fork0/b": 40, "fork0/c": 30}) == 1 + 38


def test_control_cost_in_wall_seconds_runs_main_to_the_forking_reply():
    """A main control's wall C before the fork it replaces matches the forking tree's: main's last request before
    the forking reply costs up to that reply's start (its tools after the model call count), not its model time."""
    from octopus.rl.group import fork_spans
    control = [{"id": "c0", "thread": "control", "t": 100, "elapsed_s": 3}]
    # main-first: r0's model 0-2 s, then tools until the forking reply r1 at 20
    first = [{"id": "r0", "thread": "main", "t": 0, "elapsed_s": 2}, {"id": "r1", "thread": "main", "t": 20}]
    assert control_cost(first + control, "r1", unit="wall") == 20 + 3
    # main-later: fork0's branch b (model 1-2 s, tools to its end at 15) joined at m1 (2 s); main's tools until the
    # forking reply m2 at 16
    later = [{"id": "m0", "thread": "main", "t": 0, "elapsed_s": 1}, {"id": "b0", "thread": "fork0/b", "t": 1, "elapsed_s": 1},
             {"id": "m1", "thread": "main", "t": 2, "elapsed_s": 1}, {"id": "m2", "thread": "main", "t": 16}]
    spans = fork_spans([{"response_id": "m0", "join_response_id": "m1", "branches": [{"branch_id": "b"}]}])
    assert control_cost(later + control, "m2", spans, "wall", {"fork0/b": 15}) == 16 + 3
    assert control_cost(later + control, "m2", spans, "calls", {"fork0/b": 15}) == 2 + 1  # other units unchanged

def test_bundle_call_cap_is_stated_once(monkeypatch):
    """The main call cap is stated once in the bundle prompt, and once in the linear control's."""
    from octopus.envs.swe.prompt import delivery
    for key, value in {"OCTOPUS_STRATEGY": "branch_fix", "OCTOPUS_BUNDLE": "1",
                       "OCTOPUS_MAIN_MAX_CALLS": "120"}.items():
        monkeypatch.setenv(key, value)
    assert delivery().count("120 次工具") == 1 and "主线最多调用 120 次工具（分支各有自己的上限）" in delivery()
    assert "你最多调用 120 次工具" in delivery(linear=True)


def test_dev_eval_summary_reports_fork_minus_control():
    """dev_eval (rl.plan_control): plan rows carry their no-fork control; the summary reports y − control y per kind."""
    from octopus.eval.dev_eval import summary
    rows = [{"kind": "plan", "passed": False, "y": 0.5, "n": 2, "control": {"kind": "main-first", "y": 0.25}},
            {"kind": "plan", "passed": False, "y": 0.0, "n": 2, "control": {"kind": "main-first", "y": 0.5}},
            {"kind": "plan", "passed": True, "y": 1.0, "n": 2, "control": None}]
    rows = [{"task_dir": f"t{i}", "sample": 0, **r} for i, r in enumerate(rows)]
    assert "fork - control (main-first): n=2 -0.1250±0.3750" in summary(rows)


def test_dev_eval_totals_are_the_tree_and_a_rerun_retries_errors(monkeypatch, tmp_path):
    """dev_eval (rl.plan_control): a row's calls and tokens are the episode tree's, the control's own (and its wall_s)
    are under control, and the row's wall_s leaves the control's out; a rerun runs an episode whose row has an error
    again, and the summary takes the retry, not the error. The proxy and the episode are stand-ins."""
    import contextlib
    import types
    pytest.importorskip("areal")
    # grpo.yaml's ${oc.env:PYTHONPATH} (the launcher's): unset under a bare pytest
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parents[1])))
    from octopus.rl import config, proxy
    from octopus.eval import dev_eval
    trial, task = tmp_path / "trial", tmp_path / "tasks/t1"
    (trial / "agent").mkdir(parents=True)
    task.mkdir(parents=True)
    (trial / "agent/requests.jsonl").write_text("".join(
        json.dumps({"thread": t, "prompt_tokens": p, "completion_tokens": c}) + "\n"
        for t, p, c in (("main", 10, 1), ("fork0/branch-0", 20, 2), ("control", 40, 4))))
    (trial / "agent/tree.json").write_text(json.dumps({"forks": [{"branches": [{"branch_id": "branch-0"}]}],
                                                     "control": {"wall_s": 5.0}}))

    @contextlib.asynccontextmanager
    async def serve(config, servers, lora, port=0):
        yield "http://proxy"

    async def episode(agent, data, url, task_id):
        return {"trial": str(trial), "passed": True, "y": 1.0, "tasks": None, "branch_control": None,
                "control": {"kind": "main-first", "y": 0.0}}
    monkeypatch.setattr(proxy, "serve", serve)
    monkeypatch.setattr(proxy, "episode", episode)
    out = tmp_path / "o.jsonl"
    out.write_text(json.dumps({"task_dir": str(task), "kind": "plan", "sample": 0, "error": "URLError"}) + "\n")
    asyncio.run(dev_eval.main_async(types.SimpleNamespace(
        tasks_dir=str(tmp_path / "tasks"), offset=0, limit=None, servers="127.0.0.1:1", jobs_dir=str(tmp_path),
        parallel=1, lora="rq1-v0", out=str(out), k=1), config.load(config.CONFIG)))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2 and dev_eval.summary(rows).endswith("infra failures (not scored): 0")
    row = rows[1]
    assert (row["calls"], row["prompt_tokens"], row["completion_tokens"], row["branches"]) == (2, 30, 3, 1)
    assert row["control"] == {"kind": "main-first", "y": 0.0, "calls": 1, "prompt_tokens": 40, "completion_tokens": 4,
                              "cached_tokens": 0, "wall_s": 5.0}
    assert row["wall_s"] == -5  # the stub episode takes no time: only the control's 5 s come off


def test_apply_plan_adds_span_values_on_their_tokens():
    pytest.importorskip("areal")
    import torch
    from octopus.rl.workflow import _apply_plan
    result = {"input_ids": torch.zeros(1, 8, dtype=torch.long), "loss_mask": torch.ones(1, 8)}
    leaf = SimpleNamespace(reward=1.0, octopus_plan=1.0, octopus_adv=0.5, octopus_own_start=2,
                           octopus_spans=[(3, 5, 0.25), (6, 7, -0.5)])
    _apply_plan(result, leaf)
    assert result["octopus_adv"].view(-1).tolist() == [0.5, 0.5, 0.5, 0.75, 0.75, 0.5, 0.0, 0.5]
    assert result["loss_mask"].view(-1).tolist()[:2] == [0, 0]


def test_join_post_join_idle_and_post_join_work():
    from octopus.rl.group import join_post
    reqs = [{"id": "m0", "t": 0, "elapsed_s": 1}, {"id": "m1", "t": 2, "elapsed_s": 1}, {"id": "b0", "thread": "fork0/b", "t": 3},
            {"id": "m2", "t": 4, "elapsed_s": 1}, {"id": "m3", "t": 10, "elapsed_s": 2}, {"id": "m4", "t": 13, "elapsed_s": 1}]
    fork = {"response_id": "m1", "join_response_id": "m2", "join_time": 9}
    assert join_post(reqs, [fork]) == (5.0, 4.0)  # m2 waits 5 s in join; m3 (3 s), m4 (done, 1 s) after it
    late = {"response_id": "m3", "join_response_id": "m4", "join_time": 20}  # collected at main's last reply
    assert join_post(reqs, [fork, late]) == (11.0, 0.0)


# main's abort
def _verify(tmp_path, tasks, keys):
    """The plan verifier's files: tasks.json ({task: solved}) and branches.json (every branch key, unsolved)."""
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text(f"{sum(tasks.values()) / len(tasks)}\n")
    (tmp_path / "verifier/tasks.json").write_text(json.dumps(tasks))
    (tmp_path / "verifier/branches.json").write_text(json.dumps({"branches": [
        {"key": key, "task": k, "tasks": {k: 0}, "passed": False} for key, k in keys.items()]}))


def test_reply_that_uses_up_the_token_budget_without_closing_is_a_budget_stop(monkeypatch, tmp_path):
    """A shared pool of 8 (the runtime's, search's OCTOPUS_SEARCH_BUDGET); main draws 4 before it forks. Both branches
    only run shell: the reply that empties the pool has its shell call dropped (closing_only) and nothing left, so its
    branch stops "token budget used up", as the shared calls' last reply does, not on its text."""
    def reply(thread, turn):
        if thread == "main":
            return _call("shell", {"command": "true"}) if turn < 4 else _fork_call(5, 6) if turn == 4 else DONE
        return _call("shell", {"command": "true"})

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, budget=8)
    assert tree["main_end"] == "budget"
    assert [b["stop"] for b in tree["forks"][0]["branches"]] == ["request failed: token budget used up"] * 2


def test_text_only_reply_that_uses_up_the_token_budget_is_a_self_stop(monkeypatch, tmp_path):
    """The reply that empties the pool had no tool call: its branch stopped itself on its text, not the budget, and is
    not reminded (no request is left). Main draws 4 of 8 and forks; each branch runs shell once and then replies text
    only, so the reply that empties the pool is always one of those."""
    def reply(thread, turn):
        if thread == "main":
            return _call("shell", {"command": "true"}) if turn < 4 else _fork_call(5, 6) if turn == 4 else DONE
        return _call("shell", {"command": "true"}) if turn == 1 else None

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, budget=8)
    # the first text reply (7) is reminded, its next request already sent when the other's (8) empties the pool: that
    # one is not reminded (core: payload["budget_out"]); the reminded branch's second text reply (9) ends it
    assert agent._budget["used"] == 9 and tree["main_end"] == "budget"
    assert [b["stop"] for b in tree["forks"][0]["branches"]] == ["", ""]
    assert sorted(b["reminded"] for b in tree["forks"][0]["branches"]) == [0, 1]


def test_bundle_prompt_says_weak_tasks_start():
    """The bundle prompt states what the red check does: a task whose acceptance already exits 0, cannot run or fails
    because of itself still starts, its branch told to write its own failing check first (weak_acceptance); a fork is
    refused for a missing or unknown task_id (TASK_ID_NEEDS), not for the check, and the prompt never says "only" there
    (a fork of one task is refused too: "至少两条")."""
    import octopus.envs.swe.prompt as tools
    assert "不启动" not in tools._BUNDLE and "一条都不剩时这次 fork 被拒" not in tools._BUNDLE
    assert "照样启动" in tools._BUNDLE and "要先写一条在当前代码上失败的检查" in tools._BUNDLE
    assert "这项检查不会让 fork 被拒；task_id 缺失或不对应任何任务时 fork 会被拒" in tools._BUNDLE
    assert "只有" not in tools._BUNDLE and "至少两条" in tools.delivery()


@pytest.mark.parametrize("first, recorded", [(5, ["5", "7"]), (99, [])])
def test_one_fork_per_reply_main(monkeypatch, tmp_path, first, recorded):
    """[fork(tasks 5, 7), fork(tasks 6, 8)] in one reply: only the reply's first fork call may fork, as fork_record
    reads it; the second is refused, so no lost fork holds _live and the running branches are recorded under their own
    tasks. A refused first call (task 99) leaves no fork in that reply."""
    two = [dict(_fork_call(first, 7)[0], id="f1"), dict(_fork_call(6, 8)[0], id="f2")]

    def reply(thread, turn):
        if thread == "main":
            return {1: two}.get(turn, DONE)
        return _commit(thread)

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply)
    assert [b["task_id"] for f in tree["forks"] for b in f["branches"]] == recorded
    assert "fork refused: one fork per reply" in [m["content"] for m in history["main"] if m["role"] == "tool"][1]
    assert agent._live == 1


def test_one_fork_per_reply_branch(monkeypatch, tmp_path):
    """A branch's two blocking forks in one reply: the first runs and is recorded with its tasks, the second refused."""
    two = [dict(_fork_call(5, 6)[0], id="f1"), dict(_fork_call(7, 8)[0], id="f2")]

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6)}.get(turn, DONE)
        if thread == "fork0/branch-0":
            return two if turn == 1 else _commit(thread)
        return _commit(thread)

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply)
    (kids,) = tree["forks"][0]["branches"][0]["forks"]
    assert [b["task_id"] for b in kids["branches"]] == ["5", "6"]
    own = history["fork0/branch-0"]
    assert any(m["role"] == "tool" and "fork refused: one fork per reply" in m["content"] for m in own)
    assert agent._live == 1


def test_bundle_prompt_leaves_closing_branches_to_them_and_counts_no_tasks(monkeypatch):
    """main closes with done: the bundle prompt names done as main's own closing call (branches close themselves) and
    states no task count (1-task bundles exist)."""
    from octopus.envs.swe.prompt import delivery
    monkeypatch.setenv("OCTOPUS_BUNDLE", "1")
    text = delivery()
    assert "主线在所有任务都修好后调用 done" in text and "预算" not in text and "join" not in text
    assert "这个仓库里有若干个任务（见 /tasks/INDEX.md）" in text and "相互独立" not in text and "多个" not in text


def test_join_post_counts_every_reply_that_collected_a_branch_of_a_fork():
    """A fork's branches joined over two replies (named joins): both replies' waits are join idle
    J; the reply between them is inside the fork's window (neither J nor P); post-join work starts after the reply
    collecting its last branch."""
    from octopus.rl.group import join_post
    reqs = [{"id": "m0", "t": 0, "elapsed_s": 1}, {"id": "m1", "t": 2, "elapsed_s": 1}, {"id": "m2", "t": 4, "elapsed_s": 1},
            {"id": "m3", "t": 10, "elapsed_s": 2}, {"id": "m4", "t": 13, "elapsed_s": 1}, {"id": "m5", "t": 20, "elapsed_s": 1}]
    fork = {"response_id": "m1", "join_response_id": "m4", "join_time": 19, "branches": [
        {"join_response_id": "m2", "join_time": 9},
        {"join_response_id": "m4", "join_time": 19, "ended_with": "commit", "merge": "merged", "end_time": 18}]}
    assert join_post(reqs, [fork]) == (5.0 + 6.0, 1.0)  # m2 waits 5 s, m4 6 s; m5 (done, 1 s) after


def test_join_post_tail_after_a_join_any_is_post_work_when_done_sweeps_the_rest():
    """JOIN_ANY + GRACE=0: main joins branch-0 (m2), works (m3-m5) and calls done (m6), which collects branch-1.
    P is main's serial work after the parallel work it collected had finished: branch-1 stopped without a commit
    (done's grace) contributed nothing, so m3-m5 and done's model call are post-join work; had branch-1 merged a patch
    it committed at 58 s, only done (starting after that) is. Aborting branch-1 and joining it at m6 before done (m7)
    counts the same tail. A fork whose first branches the last reply collects keeps that reply out of P."""
    from octopus.rl.group import join_post
    reqs = [{"id": f"m{i}", "t": t, "elapsed_s": e} for i, (t, e) in enumerate(
        [(0, 1), (2, 1), (4, 1), (10, 2), (20, 3), (40, 5), (60, 1)])]
    b0 = {"join_response_id": "m2", "join_time": 9, "ended_with": "commit", "merge": "merged", "end_time": 8}
    b1 = {"join_response_id": "m6", "join_time": 65, "ended_with": None, "merge": None, "end_time": 64,
          "stop": "stopped: join grace"}
    fork = {"response_id": "m1", "join_response_id": "m6", "join_time": 65, "branches": [b0, b1]}
    tail = 10.0 + 20.0 + 20.0
    assert join_post(reqs, [fork]) == (5.0 + 4.0, tail + 1.0)
    merged = dict(b1, ended_with="commit", merge="merged", end_time=58, stop=None)
    assert join_post(reqs, [dict(fork, branches=[b0, merged])]) == (5.0 + 4.0, 1.0)
    # main aborts branch-1 and joins it explicitly at m6 (61 s), then done at m7: the same tail, plus its two replies
    explicit = reqs + [{"id": "m7", "t": 70, "elapsed_s": 1}]
    aborted = dict(b1, join_time=61, end_time=60.5, stop="stopped: main abort")
    J, P = join_post(explicit, [dict(fork, join_time=61, branches=[b0, aborted])])
    assert P == tail + 1.0 + 1.0
    # the last reply also collects a fork forked at m5 (its first and only branches): done's model call is no P
    late = {"response_id": "m5", "join_response_id": "m6", "join_time": 65, "branches": [
        {"join_response_id": "m6", "join_time": 65, "ended_with": "commit", "merge": "merged", "end_time": 62}]}
    assert join_post(reqs, [fork, late]) == (5.0 + 4.0, 10.0 + 20.0)


def test_cancelled_main_cancels_its_running_branches(monkeypatch, tmp_path):
    """review of #101: Harbor's timeout cancels main inside a join; every branch it started (each its own task, at any
    depth) is cancelled with it instead of running on in the event loop all trials share."""
    import time

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6)}.get(turn, DONE)
        if thread == B1 and turn == 1:
            return _fork_call(8, 9)
        time.sleep(0.05)
        return _call("shell", {"command": "true"})  # never closes
    left = []

    async def drive(run):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(run, 1)
        await asyncio.sleep(0.3)
        left.extend(t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done())

    _run_tree(monkeypatch, tmp_path, reply, drive=drive)
    assert left == []


def test_critical_path_follows_each_branch_to_the_reply_that_joined_it():
    """review of #101: fork0's branches joined at different replies (named joins): main's work after
    joining branch-0 depends on it, so it follows branch-0 instead of overlapping it (the fork's span to its last
    join gave 1010). In seconds main's join reply already ran to branch-0's end: the same either way."""
    from octopus.rl.group import fork_spans
    forks = [{"response_id": "m0", "join_response_id": "m3", "branches": [
        {"branch_id": "branch-0", "join_response_id": "m1"}, {"branch_id": "branch-1", "join_response_id": "m3"}]}]
    spans = fork_spans(forks)
    assert spans == {"fork0/branch-0": ("m0", "m1"), "fork0/branch-1": ("m0", "m3")}
    reqs = [{"id": "m0", "completion_tokens": 10, "t": 0, "elapsed_s": 1},
            {"id": "a0", "thread": "fork0/branch-0", "completion_tokens": 1000, "t": 2},
            {"id": "c0", "thread": "fork0/branch-1", "completion_tokens": 10, "t": 3},
            {"id": "m1", "completion_tokens": 10, "t": 1, "elapsed_s": 1},
            {"id": "m2", "completion_tokens": 500, "t": 50, "elapsed_s": 5},
            {"id": "m3", "completion_tokens": 20, "t": 60, "elapsed_s": 1}]
    assert critical_path(reqs, spans) == 10 + 1000 + 500 + 20
    assert critical_path(reqs, spans, "calls") == 4
    ends = {"fork0/branch-0": 50, "fork0/branch-1": 20}
    assert critical_path(reqs, spans, "wall", ends) == 61 == critical_path(
        reqs, {b: ("m0", "m3") for b in spans}, "wall", ends)


def test_bundle_join_note_tells_main_done_and_a_branch_commit(monkeypatch):
    """review of #101: a branch that forked gets the join note too; it closes with commit/abort (done would leave its
    patch unmerged), so only main is told to call done."""
    from octopus.envs.swe.prompt import join_note
    monkeypatch.setenv("OCTOPUS_BUNDLE", "1")
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    report = {"branch_id": "branch-0", "task": "t", "steps": [{"name": "commit", "arguments": {"summary": "s"},
                                                                "acceptance_verified": True}]}
    merged = {"branch-0": "merged"}
    main, branch = join_note([report], {"branch-0": ("p", "1 file")}, merged), join_note(
        [report], {"branch-0": ("p", "1 file")}, merged, main=False)
    assert "to /testbed" in main and "Call done once every task is fixed." in main and "commit your task" not in main
    assert "to your working copy" in branch and "Then commit your task, or abort it" in branch and "done" not in branch
    assert "do not re-check or redo it" in main and "acceptance FAILED" in main


def test_branch_copies_share_one_committed_fork_point(monkeypatch, tmp_path):
    """One command opens every branch's copy of the forking thread's tree: the fork point (its edits, new files) is
    committed once and copied, the source untouched; a failed copy raises."""
    mod = _import_agent(monkeypatch)
    monkeypatch.setattr(mod, "BRANCH_ROOT", str(tmp_path / "branches"))
    src = tmp_path / "testbed"
    src.mkdir()
    git = lambda d, *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, check=True).stdout
    (src / "f").write_text("a\n")
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    base = git(src, "rev-parse", "HEAD")
    (src / "f").write_text("edited\n")
    (src / "new").write_text("n\n")

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=r.stdout, stderr=r.stderr, return_code=r.returncode)

    workdirs = {f"branch-{i}": str(tmp_path / "branches" / f"fork0-branch-{i}") for i in range(3)}
    heads, _ = asyncio.run(mod.HarborWorkspace(Env())._open_branch_copies(workdirs, source=str(src)))
    assert git(src, "rev-parse", "HEAD") == base and git(src, "status", "--porcelain")  # source untouched
    point = heads[workdirs["branch-0"]]
    assert len(point) == 40 and set(heads.values()) == {point}
    for root in workdirs.values():  # its index fresh (before git status refreshes it): git apply --index works
        subprocess.run(["git", "-C", root, "diff-files", "--quiet"], check=True)
        assert git(root, "rev-parse", "HEAD").strip() == point and git(root, "rev-parse", "HEAD~1") == base
        assert not git(root, "status", "--porcelain") and (Path(root) / "f").read_text() == "edited\n"
        assert (Path(root) / "new").exists()
    (Path(workdirs["branch-0"]) / "f").write_text("fixed\n")  # a sibling's patch to f applies to copy 2's index
    patch = git(workdirs["branch-0"], "diff")
    subprocess.run(["git", "-C", workdirs["branch-1"], "apply", "--index"], input=patch, text=True, check=True)
    with pytest.raises(RuntimeError, match="branch copies"):
        asyncio.run(mod.HarborWorkspace(Env())._open_branch_copies(workdirs, source=str(tmp_path / "missing")))


def _red_in_copies(monkeypatch, tmp_path, src, checks):
    """_open_branch_copies of src (a git repo) with the red checks, in bash; returns (workdirs, red_results)."""
    mod = _import_agent(monkeypatch)
    monkeypatch.setattr(mod, "BRANCH_ROOT", str(tmp_path / "branches"))

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=r.stdout, stderr=r.stderr, return_code=r.returncode)
    workdirs = [str(tmp_path / "branches" / f"fork0-branch-{i}") for i in range(len(checks))]
    heads, results = asyncio.run(mod.HarborWorkspace(Env())._open_branch_copies(
        dict(enumerate(workdirs)), str(src), checks))
    assert len(set(heads.values())) == 1
    return workdirs, results


def test_red_checks_in_branch_copies_leave_them_at_the_fork_point(tmp_path, monkeypatch):
    """The red checks run in the branch copies; whatever a check does to its copy, the copy is put back to its fork
    point with a fresh index: its edits, deletions, new files and commits are undone, while files that were there at
    the fork (edited, new, ignored) all stay. Only the check's own ignored files (caches) are left."""
    src = tmp_path / "testbed"
    src.mkdir()
    git = lambda d, *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, check=True).stdout
    (src / "f").write_text("a\n")
    (src / ".gitignore").write_text("*.so\n__pycache__/\n")
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (src / "f").write_text("edited\n")  # the forking thread's edit, a new file and an ignored build product
    (src / "new").write_text("n\n")
    (src / "ext.so").write_text("built\n")
    checks = ["cd /testbed && echo x > f && rm new && echo y > made && mkdir -p d/__pycache__ && echo c > "
              "d/__pycache__/m.pyc && git add -A && git -c user.name=c -c user.email=c@c commit -qm c; false",
              "echo z >> /testbed/f; mkdir -p /testbed/newdir && touch /testbed/newdir/x; exit 0"]
    workdirs, results = _red_in_copies(monkeypatch, tmp_path, src, checks)
    assert [code for code, _ in results] == ["1", "0"]
    point = git(workdirs[0], "rev-parse", "HEAD")
    for root in workdirs:
        subprocess.run(["git", "-C", root, "diff-files", "--quiet"], check=True)  # index fresh before git status
        assert git(root, "rev-parse", "HEAD") == point and not git(root, "status", "--porcelain")
        assert (Path(root) / "f").read_text() == "edited\n" and (Path(root) / "new").read_text() == "n\n"
        assert (Path(root) / "ext.so").read_text() == "built\n"
        assert not (Path(root) / "made").exists() and not (Path(root) / "newdir").exists()
        assert not Path(root + ".red").exists()
    assert git(workdirs[0], "ls-files", "-o").split() == ["d/__pycache__/m.pyc", "ext.so"]  # the check's cache stays
    assert (src / "f").read_text() == "edited\n" and not (src / "made").exists()  # the source is never touched


def test_a_starved_branch_is_not_merged(monkeypatch, tmp_path):
    """main forks tasks 5 (check `true`), 6 (check `false`) and 7 (check `nope`, which the red check finds broken, so
    its task text has no runnable check: weak_acceptance); all three branches only run shell until the shared token
    budget stops them with changes. None closed: no check runs and nothing merges."""
    tasks = [{"task": f"Task {k}", "acceptance": c, "task_id": k}
             for k, c in ((5, "true"), (6, "false"), (7, "nope"))]

    def reply(thread, turn):
        return _call("fork", {"tasks": tasks}) if thread == "main" else _call("shell", {"command": "ls"})

    def red(command):  # 127 for `nope` (broken: weak), else 1 (a check that detects its bug)
        return "".join(f"{i} {127 if c == 'nope' else 1}\n"
                       for i, c in enumerate(re.findall(r"bash -lc (\S+) >", command)))

    agent, rows, tree, history, commands = _run_tree(
        monkeypatch, tmp_path, reply, budget=8, red=red,
        exec_out=lambda c: "merged\n" if "git apply --3way" in c else None)
    branches = tree["forks"][0]["branches"]
    assert all("token budget used up" in b["stop"] for b in branches) and tree["main_end"] == "budget"
    assert "Acceptance (exits 0 when done)" not in branches[2]["task"]
    assert all("starved_check" not in b and b["merge"] is None and "merged_starved" not in b for b in branches)
    assert "was stopped by the shared budget" not in json.dumps(tree["main"], ensure_ascii=False)
    assert not any("git apply --3way" in c for c, _ in commands)


def test_rewind_tool_in_main_a_branch_and_the_control(monkeypatch, tmp_path):
    """OCTOPUS_REWIND_TOOL: main shells (step 1), edits (2) and rewinds to 1; forks (2 again); its branch-0 starts
    after main's forking reply (2), edits (3) and rewinds to where it started (2); the control from main's fork starts
    before that reply (1), edits (2) and rewinds to 1.
    Each restores its own tree to the snapshot taken before its edit (the fake numbers a tree's snapshot by the edits
    it had), and each tool result's budget tag names its step."""
    import base64
    edits = collections.Counter()
    back = _call("rewind", {"to_step": 1, "note": "wrong approach"})
    edit = _call("edit", {"path": "/testbed/f.py", "old": "a", "new": "b"})

    def reply(thread, turn):
        if thread == "main":
            return {1: _call("shell", {"command": "true"}), 2: edit, 3: back, 4: _fork_call(5, 6)}.get(turn, DONE)
        if thread == "fork0/branch-0":  # its history ends with main's forking reply, step 2
            return {1: edit, 2: _call("rewind", {"to_step": 2, "note": "wrong approach"})}.get(turn, _commit(thread))
        if thread == "control":  # its history ends before main's forking reply: after step 1
            return {1: edit, 2: back}.get(turn, DONE)
        return _commit(thread)

    def exec_out(command):
        if command.startswith("python3 /tmp/harbor_tools.py "):
            if json.loads(base64.b64decode(command.split()[2]))["name"] == "edit":
                edits[command.split()[3]] += 1
        elif "git stash create" in command:
            root = command.split()[1]
            return f"{root}:{edits[root]}\n"
        elif "git reset -q --hard /" in command:  # a restore: the tree has the snapshot's edits again
            root, n = command.split("--hard ")[1].split()[0].split(":")
            edits[root] = int(n)
        return None
    agent, rows, tree, history, commands = _run_tree(
        monkeypatch, tmp_path, reply, exec_out=exec_out, OCTOPUS_REWIND_TOOL="1", OCTOPUS_PLAN_CONTROL="1",
        pick=lambda seed, pool: pool[0])
    restores = [c.split("--hard ")[1].split()[0] for c, _ in commands if "git reset -q --hard /" in c]  # not red checks
    # main's rewind, branch-0's, the control's start at main's fork point, its rewind, /testbed back to main's end
    assert restores == ["/testbed:0", "/tmp/branches/fork0-branch-0:0", "/testbed:0", "/testbed:0", "/testbed:0"]
    note = ("[runtime] Rewound to step 1 at your request: the steps after it were removed and the repository is back "
            "as it was after step 1. They edited /testbed/f.py; no command failed.\nYour note: wrong approach")
    # each tool result sent (the last one, of done or commit, never is: the list is the thread's own), a rewind note
    # too: a tool message tagged with the step it went back to (main's note is in the history its branch and the
    # control start from)
    for thread, to, steps in (("main", 1, [1, 1, 2]), ("fork0/branch-0", 2, [1, 1, 2, 2]), ("control", 1, [1, 1, 1])):
        told = [m["content"].split("\n", 1) for m in history[thread] if m["role"] == "tool"][:len(steps)]
        assert note.replace("step 1", f"step {to}") == told[-1 if thread != "main" else 1][1]
        tag = r"\[context left: \d+( \| calls left: \d+)? \| step (\d+)\]"
        assert [int(re.fullmatch(tag, t).group(2)) for t, _ in told] == steps, told
    task, kept, commit, _ = history["fork0/branch-0"][-4:]  # rewound to where it started: its task, the note, commit
    assert task["content"].startswith("You are branch-0") and commit["role"] == "assistant"
    assert kept["content"].endswith("]\n" + note.replace("step 1", "step 2"))
    assert [r["to_step"] for r in tree["rewinds"]] == [1]
    assert [r["to_step"] for r in tree["forks"][0]["branches"][0]["rewinds"]] == [2]
    assert {r["thread"] for r in rows if r["tools"][-1] == "rewind"} == {r["thread"] for r in rows}  # every thread's


def test_rewind_tool_note_names_undone_merges(monkeypatch, tmp_path):
    """A rewind past the join that merged a fork's patches says those merges were undone with the repository, with
    each patch's path."""
    def reply(thread, turn):
        if thread == "main":
            script = {1: _call("shell", {"command": "true"}), 2: _fork_call(5, 6),
                      3: _call("rewind", {"to_step": 1, "note": "x"})}
            return script.get(turn, DONE)
        return _commit(thread)
    agent, rows, tree, history, _ = _run_tree(
        monkeypatch, tmp_path, reply, exec_out=lambda c: "merged\n" if "git apply --3way" in c else None,
        OCTOPUS_REWIND_TOOL="1")
    note = next(m["content"] for m in history["main"] if "Rewound to step 1" in str(m["content"]))
    assert note.split("no command failed. ")[1] == (
        "Merges undone with the repository: branch-0 (/tmp/branches/fork0-branch-0.patch), branch-1 "
        "(/tmp/branches/fork0-branch-1.patch); these branches are joined already, apply a patch again with git apply "
        "--3way to keep it.\nYour note: x")
    # tree.json says so too: rl.branch_credit=gate gives an undone branch no merge credit
    from octopus.rl.group import branch_ok
    branches = tree["forks"][0]["branches"]
    assert [b["merge"] for b in branches] == ["undone: merged"] * 2
    assert not any(branch_ok(dict(b, passed=True)) for b in branches)


@pytest.mark.parametrize("think", ["", '{"enable_thinking": true}'])
def test_rewind_to_the_forking_step_keeps_its_merges(monkeypatch, tmp_path, think):
    """A rewind to the step that forked keeps that step's join (its merges are in the tree it restores), also with
    thinking on, where the join note joins the step's last tool result: no merge is called undone."""
    def reply(thread, turn):
        if thread == "main":
            return {1: _call("shell", {"command": "true"}), 2: _fork_call(5, 6),
                    3: _call("rewind", {"to_step": 2, "note": "x"})}.get(turn, DONE)
        return _commit(thread)
    agent, rows, tree, history, _ = _run_tree(
        monkeypatch, tmp_path, reply, exec_out=lambda c: "merged\n" if "git apply --3way" in c else None,
        OCTOPUS_REWIND_TOOL="1", OCTOPUS_CHAT_TEMPLATE_KWARGS=think)
    note = next(m["content"] for m in history["main"] if "Rewound to step 2" in str(m["content"]))
    assert "Merges undone" not in note and "no command failed.\nYour note: x" in note
    assert [b["merge"] for b in tree["forks"][0]["branches"]] == ["merged"] * 2


def test_main_control_has_the_calls_main_had_left_after_a_rewind(monkeypatch, tmp_path):
    """Main's calls before its forking reply include those a rewind removed from its history: shell, shell, rewind
    (to step 1), then fork, under OCTOPUS_MAIN_MAX_CALLS=7. Its main-first control starts with 7 - 3 calls left."""
    shell = _call("shell", {"command": "true"})

    def reply(thread, turn):
        if thread == "main":
            return {1: shell, 2: shell, 3: _call("rewind", {"to_step": 1, "note": "n"}), 4: _fork_call(5, 6)}.get(
                turn, DONE)
        if thread == "control":
            return shell if turn < 3 else DONE
        return _commit(thread)
    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_HTTP_WORKERS="1",
                                              OCTOPUS_REWIND_TOOL="1", OCTOPUS_PLAN_CONTROL="1",
                                              OCTOPUS_MAIN_MAX_CALLS="7")
    assert tree["control"]["calls_before"] == 3
    # main's step 1 result and its rewind note (tagged by main: 4 left), then the control's two shell results
    tags = [m["content"].split("\n", 1)[0] for m in history["control"] if m["role"] == "tool"][:-1]
    assert [re.search(r"calls left: (\d+)", t)[1] for t in tags] == ["6", "4", "3", "2"], tags


@pytest.mark.parametrize("close", ["rewind", "done", "rewind first"])
def test_a_reply_that_forks_cannot_also_close_or_rewind(monkeypatch, tmp_path, close):
    """A reply whose fork starts and that also calls rewind (after the fork or before it) or a closing call: the fork
    runs, the other call is refused (a rejected step, nothing rewound or closed), and the thread goes on to its next
    reply. So the forking reply, which its branches extend, is also in its thread's continuation: every main request
    is in a chain of main requests that ends in a leaf (a reply that forked and closed main was in no main leaf, so
    main was never trained)."""
    back = _call("rewind", {"to_step": 1, "note": "wrong approach"})
    bodies = []

    def serve(rid, body, calls):
        bodies.append((rid, [{k: m.get(k) for k in ("role", "content", "tool_calls")} for m in body["messages"]]))

    def reply(thread, turn):
        if thread == "main":
            other = {"rewind": back, "done": DONE}.get(close)
            script = {1: _call("edit", {"path": "a.py", "old_str": "a", "new_str": "b"}),
                      2: _fork_call(5, 6) + other if other else back + _fork_call(5, 6)}
            return script.get(turn) or DONE
        return _commit(thread)
    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, serve=serve, OCTOPUS_REWIND_TOOL="1")
    name = close.split()[0]
    refused = [m["content"] for m in history["main"] if m["role"] == "tool" and f"{name} refused" in m["content"]]
    assert len(refused) == 1 and f"a reply whose fork started cannot also call {name}" in refused[0]
    assert tree["main_end"] == "done" and tree["rewinds"] == []
    assert [r["thread"] for r in rows].count("main") == 3

    def extends(i, j):  # request j's messages start with request i's and then its reply
        a, b = bodies[i][1], bodies[j][1]
        return len(b) > len(a) and b[:len(a)] == a and b[len(a)]["role"] == "assistant"
    mains = [i for i, (rid, _) in enumerate(bodies) if rid.startswith("main")]
    assert all(extends(i, j) for i, j in zip(mains, mains[1:]))


@pytest.mark.parametrize("strategy", ["", "branch_fix"])
def test_swe_fork_description_names_rewind_only_when_offered(monkeypatch, strategy):
    """SWE's fork descriptions state FORK_ALONE: done, commit or abort, and rewind only under OCTOPUS_REWIND_TOOL."""
    from octopus.envs.swe.prompt import runtime_tools
    monkeypatch.setenv("OCTOPUS_STRATEGY", strategy)

    def fork():
        return next(t for t in runtime_tools() if t["function"]["name"] == "fork")["function"]["description"]
    monkeypatch.delenv("OCTOPUS_REWIND_TOOL", raising=False)
    assert fork().endswith("cannot also call done, commit or abort: call them in a later reply.")
    monkeypatch.setenv("OCTOPUS_REWIND_TOOL", "1")
    assert fork().endswith("cannot also call done, commit, abort or rewind: call them in a later reply.")


def test_a_call_refused_beside_a_fork_spends_no_call(monkeypatch, tmp_path):
    """OCTOPUS_MAIN_MAX_CALLS=2: main's first reply forks and calls done; done is refused (FORK_ALONE) and is not
    counted, so main has a call left and closes with done in its next reply."""
    def reply(thread, turn):
        if thread == "main":
            return _fork_call(5, 6) + DONE if turn == 1 else DONE
        return _commit(thread)
    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_MAIN_MAX_CALLS="2")
    assert tree["main_end"] == "done" and [r["thread"] for r in rows].count("main") == 2


def test_a_threads_requests_after_its_last_leaf_are_trained_in_its_first_branch():
    """main's fork m1 is its last request (its call cap): no main leaf. Its first branch's leaf owns m0 and m1 too
    (own_start at m0's prompt), at main's advantage (a span each); the other branch's leaf is unchanged. A request in
    no leaf that no branch extends rejects the group."""
    requests = [("m0", "main", 2, 5), ("m1", "main", 8, 8), ("b0", "fork0/branch-0", 18, 2),
                ("b1", "fork0/branch-1", 18, 3)]
    b0, b1 = assign_threads([_leaf("b0", [0] * 20), _leaf("b1", [0] * 21)], _episode(requests))
    assert (b0["own_start"], b0["adv"], b0["spans"]) == (2, 0.25, [(2, 7, 0.25), (8, 16, 0.25)])
    assert (b1["own_start"], b1["adv"], b1["spans"]) == (18, -0.25, [])
    with pytest.raises(RuntimeError, match="m1 of main is in no leaf and no branch extends it"):
        assign_threads([_leaf("m0", [0] * 7)], _episode(requests[:2]))


@pytest.mark.parametrize("who", ["main", "branch"])
def test_a_fork_that_ends_its_thread_is_trained_once(monkeypatch, tmp_path, who):
    """The fork is its thread's last request: main shells and forks at OCTOPUS_MAIN_MAX_CALLS=2 (main_end "step
    cap"), or branch-0 shells and forks at OCTOPUS_BRANCH_MAX_TURNS=2. The leaves are the requests no other request
    extends (AReaL's prefix tree on the bodies as sent); every completion token of every request is trained by exactly
    one leaf, at its own thread's advantage."""
    from octopus.runtime.core import branch_key
    shell = _call("shell", {"command": "true"})

    def reply(thread, turn):
        if thread == "main":
            return {1: shell, 2: _fork_call(5, 6)}.get(turn) or DONE if who == "main" else (
                _fork_call(5, 6) if turn == 1 else DONE)
        if thread == "fork0/branch-0" and who == "branch":
            return {1: shell, 2: _fork_call(7, 8)}[turn]
        return _commit(thread)
    bodies = {}

    def serve(rid, body, calls):  # prompt tokens grow with the messages: a request extending another has more
        msgs = [{k: m.get(k) for k in ("role", "content", "tool_calls")} for m in body["messages"]]
        bodies[rid] = msgs
        return {"id": rid, "usage": {"prompt_tokens": len(json.dumps(msgs)), "completion_tokens": 1},
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}
    env = {"OCTOPUS_MAIN_MAX_CALLS": "2"} if who == "main" else {"OCTOPUS_BRANCH_MAX_TURNS": "2"}
    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, serve=serve, OCTOPUS_HTTP_WORKERS="1",
                                              **env)
    assert who == "branch" or tree["main_end"] == "step cap"
    requests = [(r["id"], r["thread"], r["prompt_tokens"], r["completion_tokens"]) for r in rows]
    size = {rid: p + c for rid, _, p, c in requests}

    def extends(a, b):  # b's messages are a's and then a's reply
        return len(bodies[b]) > len(bodies[a]) and bodies[b][:len(bodies[a])] == bodies[a] and \
            bodies[b][len(bodies[a])]["role"] == "assistant"
    parent = {b: max((a for a, *_ in requests if a != b and extends(a, b)), key=size.get, default=None)
              for b, *_ in requests}
    leaves = [rid for rid, *_ in requests if rid not in parent.values()]
    adv = {t: 0.5 if t == "main" else 0.25 if t == "fork0/branch-0" else -0.25 for _, t, _, _ in requests}
    episode = _episode(requests, branches=[{"key": branch_key(t), "A_branch": a} for t, a in adv.items() if t != "main"])
    rows_out = assign_threads([_leaf(rid, [0] * size[rid]) for rid in leaves], episode)
    trained, at = collections.Counter(), {rid: (t, p, c) for rid, t, p, c in requests}
    for leaf, row in zip(leaves, rows_out):
        rid = leaf
        while rid:
            thread, p, c = at[rid]
            for i in range(max(p, row["own_start"]), p + c):
                trained[rid, i] += 1
                assert row["adv"] + sum(v for s, e, v in row["spans"] if s <= i < e) == adv[thread], (leaf, rid)
            rid = parent[rid]
    assert set(trained.values()) == {1} and len(trained) == sum(c for *_, c in requests)


def test_read_credit_replaces_the_advantage_on_the_read_call_and_read_controls_are_not_trained():
    """OCTOPUS_READ_CONTROL: the reader's first read_thread call in its read reply gets r_reader − r_rc in place of its
    A_branch (the rest of the reply and every other token keep it); a read control's leaf is graded but masked whole."""
    text = '<function=shell>x</function> <function=read_thread><parameter=thread>main</parameter></function>' \
           '<function=read_thread>y</function>'
    ids = [ord(c) for c in text]
    prompt = 4
    ep = _episode([("r1", "fork0/branch-1", prompt, len(ids)), ("c1", "readctl/fork0/branch-1", prompt + len(ids), 5)],
                  read_adv={"r1": {"calls": ["shell", "read_thread", "read_thread"], "call": 2, "v": 2.0}})
    # (the second read_thread call mounted, the first was refused)
    reader, control = assign_threads([_leaf("r1", [0] * prompt + ids), _leaf("c1", [0] * (prompt + len(ids) + 5))],
                                     ep, chr)
    a, b = text.rindex("<function=read_thread>"), len(text)
    assert reader["adv"] == -0.25 and reader["spans"] == [(prompt + a, prompt + b, 2.0 + 0.25)]
    assert (control["adv"], control["own_start"], control["spans"]) == (0.0, prompt + len(ids) + 5, [])


@pytest.mark.parametrize("text, calls", [
    # cut off inside its last call (8192 tokens), parsed by SGLang's parser (AReaL's fallback when it finds no complete
    # or only ambiguous blocks), which still took commit: the format matches no call there
    ('<tool_call>\n<function=read_thread>\n<parameter=thread>\nmain\n</parameter>\n</function>\n</tool_call>\n'
     '<tool_call>\n<function=commit>\n<parameter=summary>\nhalf a summ', ["read_thread", "commit"]),
    # a call written in prose before the real one: the format matches both, the server parsed one
    ('I could use <function=read_thread>x</function> here.\n<tool_call>\n<function=read_thread>\n'
     '<parameter=thread>\nmain\n</parameter>\n</function>\n</tool_call>', ["read_thread"]),
])
def test_read_credit_only_on_a_call_the_server_parsed(text, calls):
    """The parsed calls (the read reply's read_calls) define what the reply called; the format only locates them. A
    reply whose matches are not exactly those calls gets no read span: never one on prose or on the wrong call."""
    ids = [ord(c) for c in text]
    ep = _episode([("r1", "fork0/branch-1", 4, len(ids))], read_adv={"r1": {"calls": calls, "call": 0, "v": 2.0}})
    [reader] = assign_threads([_leaf("r1", [0] * 4 + ids)], ep, chr)
    assert reader["spans"] == []
    if "commit" in calls:  # AReaL's own parser takes complete blocks only: read_thread alone, which keeps its span
        ep["read_adv"]["r1"]["calls"] = ["read_thread"]
        [reader] = assign_threads([_leaf("r1", [0] * 4 + ids)], ep, chr)
        assert [(s, e) for s, e, _ in reader["spans"]] == [
            (4 + text.index("<function="), 4 + text.index("</function>") + len("</function>"))]
    # the same reply complete and without the prose: the span is the parsed call's
    clean = text[text.index("<tool_call>"):].split("<tool_call>\n<function=commit>")[0]
    ids = [ord(c) for c in clean]
    ep = _episode([("r1", "fork0/branch-1", 4, len(ids))],
                  read_adv={"r1": {"calls": ["read_thread"], "call": 0, "v": 2.0}})
    [reader] = assign_threads([_leaf("r1", [0] * 4 + ids)], ep, chr)
    a = clean.index("<function=read_thread>")
    assert [(s, e) for s, e, _ in reader["spans"]] == [(4 + a, 4 + clean.index("</function>") + len("</function>"))]


def test_read_values_score_both_sides_alike_and_leave_out_fork_pairs():
    """Review of #148: the same rules on both sides (bundle): unclosed −1, a commit +1 when its patch solved its task
    else −1. A pair where the reader forked after its read or the control called fork carries no value."""
    from octopus.rl.group import read_values
    from octopus.envs.swe.reward import code_read_rewards
    reader = {"key": "fork0-branch-1", "ended_with": "shell"}
    c = {"of": "fork0-branch-1", "read_rid": "r", "read_call": 0, "read_calls": ["read_thread"], "ended_with": "shell",
         "passed": False}
    episode = {"tasks": {"1": 0}, "branches": [reader], "read_controls": [c]}
    hook = lambda e, c, b: code_read_rewards(e, c, b, False)  # noqa: E731
    v = lambda: read_values(episode, hook).get("r", {}).get("v")  # noqa: E731
    assert v() == 0.0  # both unclosed
    c.update(ended_with="commit", passed=True)
    assert v() == -2.0  # the control solved it
    c.update(passed=False)
    assert v() == 0.0  # a failing commit scores as unclosed
    c.update(ended_with="shell")
    c["forked_after"] = True
    assert v() is None
    c.update(forked_after=False, control_forked=True)
    assert v() is None


def _read_pair(reader: dict, control: dict) -> dict:
    """A single-task episode (no tasks) as plan_outcome gives it: one branch and its read control."""
    return {"y": False, "C": 10.0, "forked": True, "tasks": None, "control": None, "branch_control": None,
            "branches": [{"key": "fork0-branch-0", "branch_id": "branch-0", "passed": False, "accepted": None,
                          "attempted": True, "task": None, **reader}],
            "read_controls": [{"of": "fork0-branch-0", "read_rid": "r", "read_call": 0, "read_calls": ["read_thread"],
                               "forked_after": False, "control_forked": False, "passed": False, "accepted": None,
                               **control}]}


def test_single_task_read_values_score_both_sides_alike():
    """Review of #163: an unclosed reader and control both score −1. A dishonest failing commit (acceptance_verified
    False) scores −1.5 on either side."""
    cut = {"ended_with": None}
    assert plan_group_step([_read_pair(cut, {"ended_with": "shell"})])[0]["read_adv"]["r"]["v"] == 0.0
    dishonest = {"ended_with": "commit", "accepted": False}
    assert plan_group_step([_read_pair(dishonest, dishonest)])[0]["read_adv"]["r"]["v"] == 0.0
    assert plan_group_step([_read_pair(dishonest, {"ended_with": "commit"})])[0]["read_adv"]["r"]["v"] == -0.5


def test_an_environment_supplies_its_read_rewards():
    """Another environment's per-leaf scalar grade (an F1, a proof score) through group_step's read_rewards hook
    (episode, c, b), with no change to the group step: the read reply gets r_reader − r_rc on that scale."""
    episode = _read_pair({"ended_with": "commit", "passed": 0.75}, {"ended_with": "commit", "passed": 0.25})
    got = group_step([episode], read_rewards=lambda e, c, b: (b["passed"], c["passed"]))
    assert got[0]["read_adv"] == {"r": {"calls": ["read_thread"], "call": 0, "v": 0.5}}


def test_conflict_by_labels_a_taken_back_patchs_files_sibling_or_own(monkeypatch, tmp_path):
    """OCTOPUS_CONFLICT_BY on a real repo: branches S and B fork from /testbed; S changes f.py's line 2 and commits
    first (merged); B changes f.py's line 2 too and g.py's line 2, which the forking thread itself changed after the
    fork. B is taken back with both files; f.py conflicts with S's patch alone (sibling), g.py with the thread's own
    edit (own)."""
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    monkeypatch.setenv("OCTOPUS_CONFLICT_BY", "1")
    mod = _import_agent(monkeypatch)
    root, br = tmp_path / "testbed", tmp_path / "branches"
    script = f"""set -e
git init -q {root} && cd {root} && git config user.email t@t && git config user.name t
for f in f g; do printf 'a\\nb\\nc\\nd\\ne\\nf\\ng\\n' > $f.py; done && git add -A && git commit -qm base
mkdir {br} && for w in S B; do cp -a {root} {br}/$w && git -C {br}/$w commit -qm 'fork point' --allow-empty; done
sed -i 2s/b/S/ {br}/S/f.py && sed -i 2s/b/B/ {br}/B/f.py && sed -i 2s/b/B/ {br}/B/g.py && sed -i 2s/b/own/ g.py
for w in S B; do (cd {br}/$w && git add -A && git diff --cached --binary HEAD > {br}/$w.patch); done
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            ran = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=cwd)
            return SimpleNamespace(stdout=ran.stdout, stderr=ran.stderr, return_code=ran.returncode)

    branches = [{"branch_id": b, "ended_with": "commit", "patch": f"{br}/{b}.patch", "diffstat": "1 file changed"}
                for b in ("S", "B")]
    agent, by = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent), {}
    agent.ws = mod.HarborWorkspace(Env())
    merged = asyncio.run(agent._merge_committed(str(root), branches, ["S", "B"], since=0, by=by))
    assert merged == {"S": "merged", "B": "conflict: f.py, g.py"}
    assert by == {"B": {"f.py": "sibling", "g.py": "own"}}


def test_swe_threads_are_reminded_once_with_their_own_closing_tools(monkeypatch, tmp_path):
    """core.REMIND_NOTE on a SWE plan episode: branch-0's first reply without a tool call is reminded with a branch's
    closing tools (abort, commit or done) and it commits; main's after the join is reminded with done and it closes;
    the no-fork control (plan_control) replies without a tool call twice: reminded once (with done), then it ends.
    tree.json counts each thread's reminders, and so does the episode record."""
    from octopus.rl.group import episode_record
    from octopus.runtime.core import REMIND_NOTE

    def reply(thread, turn):
        if thread == "main":
            return {1: _fork_call(5, 6), 2: None}.get(turn, DONE)
        if thread == "control":
            return None
        return None if thread == "fork0/branch-0" and turn == 1 else _commit(thread)

    agent, rows, tree, history, _ = _run_tree(monkeypatch, tmp_path, reply, OCTOPUS_PLAN_CONTROL="1")
    assert tree["main_end"] == "done" and tree["reminded"] == 1
    assert {"role": "user", "content": REMIND_NOTE.format(calls="done")} in tree["main"]
    branches = tree["forks"][0]["branches"]
    assert [(b["ended_with"], b["reminded"]) for b in branches] == [("commit", 1), ("commit", 0)]
    assert {"role": "user", "content": REMIND_NOTE.format(calls="abort or commit or done")} in branches[0]["messages"]
    assert tree["control"]["reminded"] == 1 and tree["control"]["end"] == "no tool call"
    assert {"role": "user", "content": REMIND_NOTE.format(calls="done")} in history["control"]
    record = episode_record(tmp_path, tree, rows, 1.0, {})
    assert record["reminded"] == {"main": 1, "branches": 1}
