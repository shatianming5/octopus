"""The SWE (Harbor) environment's model-visible text, harbor-free (octopus/rl reads it): the system prompt
(delivery), the repository tools (TOOLS, run in the container by octopus/envs/swe/container_tools.py), each strategy's
fork / commit / abort, the fork task format (a task and its acceptance check, a bundle's task id) and what a branch_fix
branch and the joining thread are told on top of the runtime's texts (octopus/runtime/core.py).
"""

from __future__ import annotations

import copy

from octopus.runtime import core
from octopus.runtime.core import _tool, decode_tool_call, fork_task_list, limits_note
from octopus.runtime.settings import auto_fork, auto_merge, branch_fix, bundle, call_limit, rewind_tool

# OCTOPUS_STRATEGY=branch_fix: every branch is one complete fix (locate, edit, verify in its private copy) of one
# independent problem or of one hypothesis about a problem; the main thread compares the branches' verified patches
# and merges them. Forking is the model's choice (DELIVERY) or absent (the no-fork control: delivery(linear=True)).
_GOAL = (
    "你的任务是在当前容器的产品源码中完成修复。"
    "解释原因、展示建议补丁或运行独立 Python 演示，都不等于完成任务。"
)
_FORK_TASKS = (
    "每条 task 是一个完整的修复任务（一个独立的问题，或针对同一问题的一个具体假设），"
    "至少两条且彼此不同；每条都给 acceptance：一条自包含的 shell 命令，从仓库根目录运行，"
    "问题修好时退出码为 0、否则非 0，会被 bash 原样执行，不依赖分支自己新建的文件"
    "（多行 Python 用 heredoc：python - <<'EOF' ... EOF，或运行已有测试；不要在引号里写 \\n 转义）。"
)
_MERGE_MANUAL = (
    "全部分支收口后，主线比较各分支的验收结果，用 git apply --3way 合并验收通过的分支 patch"
    "（针对同一问题的几个方案只合并最好的一个，修不同问题的逐个合并），"
    "不要自己重写分支已验证的修复；合并后再运行 acceptance 和测试确认。"
)
# OCTOPUS_AUTO_MERGE: the runtime applies the committed patches at the join (CursorRuntime._merge_committed)
_MERGE_AUTO = (
    "全部分支收口后，运行时按 commit 顺序用 git apply --3way 把每个 commit 的分支 patch 合并进 fork 它们的线程的仓库"
    "（主线的 /testbed，嵌套分支合并进父分支的副本），汇总里逐个写明 merged 或 conflict: <文件>；"
    "不要再手动 apply 这些 patch。冲突标记留在文件里由你解决（修不同问题的两边都保留），"
    "然后运行 acceptance 和测试确认。"
)
# bundle: a merged patch was committed after its task's acceptance ran; only what did not land needs main's work
_MERGE_AUTO_BUNDLE = (
    "分支收口后，运行时按 commit 顺序用 git apply --3way 把每个 commit 的分支 patch 合并进 fork 它们的线程的仓库"
    "（主线的 /testbed，嵌套分支合并进父分支的副本），汇总里逐个写明 merged 或 not applied；不要再手动 apply 这些 patch。"
    "汇总里补丁 merged 且 acceptance passed、由运行时检查过的任务已经修好，不要重新检查或重做（只标 as reported 的可以检查一次）；"
    "只处理补丁没有合并（冲突时运行时整个撤回该补丁，文件保持原样）、没有 commit 或 acceptance FAILED 的任务。"
    "主线在所有任务都修好后调用 done；分支修好自己的任务就 commit。"
)
_FORK_BRANCHES = (
    "每个分支在自己的私有副本里定位问题、直接修改产品源码，收尾前用 shell 原样运行 acceptance（不要改写）："
    "通过才 commit（acceptance_passed=true，summary 写清改了哪些文件、修复做了什么），做不到就 abort。"
    + _MERGE_MANUAL + "都没通过时可以再 fork 一轮新的方案。"
)
_ALL_CLOSED = "有分支时全部收口后才 done。"
_FORK_CHOICE = "是否 fork、何时 fork、分几个分支都由你决定。"
# OCTOPUS_AUTO_FORK (Gate 0): the environment's fork before main's first request (CursorRuntime.run)
_AUTO_FORK = "本次评测开局时运行时已为每个任务各 fork 出一个分支，主线在这些分支全部收口后收尾，之后仍可再 fork。"
_REPORT = "报告完成时，应已有实际落盘的产品源码修改，并说明验证结果；未能完成时如实说明阻塞。"

DELIVERY = (
    _GOAL
    + "所有工具从一开始就可用：你可以自己直接定位、修改产品源码并验证，也可以在任何时候 fork 出并行分支。"
    + _FORK_CHOICE
    + "先 view path=. 了解 /testbed 布局、用 reproduce 复现问题。"
    "fork 时列出任务和验收：" + _FORK_TASKS
    + "分支的私有副本包含你 fork 之前已做的修改。"
    + _FORK_BRANCHES
    + "不要改 site-packages。" + _ALL_CLOSED
    + _REPORT
)

DELIVERY_LINEAR = (
    _GOAL
    + "这是单线程会话，没有 fork：view、reproduce、edit、write、shell、test、done 从一开始就可用。"
    "先 view path=. 了解 /testbed 布局、用 reproduce 复现问题，再定位并直接修改产品源码，用 shell 或 test 验证。"
    "不要改 site-packages。验证完成后 done。"
    + _REPORT
)


_BUNDLE_TASKS = (
    "这个仓库里有若干个任务（见 /tasks/INDEX.md）：/tasks/<k>.md 描述任务 k，/tasks/INDEX.md 每行列一个任务，"
    "用 view 读取（例如 view path=/tasks/INDEX.md）。每个任务按最终 /testbed 由它自己的隐藏测试单独评分，修好几个算几个。"
)
_BUNDLE = _BUNDLE_TASKS + (
    "fork 的每条 task 必须带 task_id（整数 k），表示这条分支只修任务 k："
    "它的补丁会单独应用在初始仓库上，只用任务 k 的隐藏测试评分。"
    "fork 前会在当前代码的副本上先运行每条 acceptance：已经退出 0（查不出这个 bug）、命令本身跑不起来或因检查自身出错而失败的"
    "那条 task 照样启动，但它的分支会被告知这条检查不可靠，要先写一条在当前代码上失败的检查，修好后让它通过才能 commit；"
    "fork 结果里写明这些 task 和原因。这项检查不会让 fork 被拒；task_id 缺失或不对应任何任务时 fork 会被拒。"
)


def _calls_cap(linear: bool) -> str:
    """Main's call cap (call_limit) in a bundle instruction; each tool result's tag counts down."""
    cap = call_limit()
    if cap is None:
        return ""
    return (f"{'你' if linear else '主线'}最多调用 {cap} 次工具{'' if linear else '（分支各有自己的上限）'}，"
            "用完时会话结束，按当时的 /testbed 评分。")


def delivery(linear: bool = False) -> str:
    """The system message: of a forking run, or (linear) of the no-fork control."""
    text = DELIVERY_LINEAR if linear else DELIVERY
    if bundle():
        text += (_BUNDLE_TASKS if linear else _BUNDLE) + _calls_cap(linear)  # linear: the no-fork control
    if auto_merge():
        text = text.replace(_MERGE_MANUAL, _MERGE_AUTO_BUNDLE if bundle() else _MERGE_AUTO)
    if auto_fork() and not linear:
        text = text.replace(_FORK_CHOICE, _AUTO_FORK)
    return text + "\n" + limits_note(not linear, calls=call_limit() is not None)  # its own line


TOOLS = [
    _tool(
        "reproduce",
        "Run a Python snippet at /testbed. A finished call means that command ran; it does not mean the product bug is fixed.",
        {"hypothesis": {"type": "string"}, "code": {"type": "string"}},
        ["hypothesis", "code"],
    ),
    _tool(
        "rerun_reproduction",
        "Re-run the last reproduce snippet at /testbed after product files change. Success is command execution, not a product fix.",
        {},
        [],
    ),
    _tool(
        "edit",
        "Replace one unique string in a product file under /testbed. path is relative to /testbed or an absolute path inside it. The result names that path and whether the replacement landed.",
        {name: {"type": "string"} for name in ("path", "old", "new")},
        ["path", "old", "new"],
    ),
    _tool(
        "shell",
        "Run one bash command in /testbed. A command that writes a valid product patch is a complete fix. The result includes the exit code.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    _tool(
        "view",
        "Read a file or directory under /testbed. path is relative to /testbed or an absolute path inside it. "
        "If the path is missing, the result lists /testbed. Start with path=. when the layout is unknown.",
        {
            "path": {"type": "string"},
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        ["path"],
    ),
    _tool(
        "write",
        "Create a new file under /testbed. Fails if the path already exists. The result names the path.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _tool(
        "test",
        "Run a developer command argv in /testbed. The result includes the exit code. This is not the Harbor verifier.",
        {"args": {"type": "array", "items": {"type": "string"}}},
        ["args"],
    ),
    _tool(
        "done",
        "Stop the session. Product source on disk should already have changed, or name the blocker.",
        {},
        [],
    ),
]


# branch_fix strategy: every fork task carries an acceptance check, and a closing branch reports it.
ACCEPTANCE_PREFIX = "Acceptance (exits 0 when done): "
_BRANCH_FIX_TOOLS = {
    "fork": _tool(
        "fork",
        "Clone this conversation into N private branches, one fix each. Need at least two tasks. Each task "
        "names what to fix (an independent problem, or one hypothesis about a problem) and an acceptance check: one self-contained "
        "shell command that bash runs verbatim from the repo root, exiting 0 exactly when the task is done "
        "(the bug is fixed) and non-zero otherwise. It must not depend on files a branch creates; for "
        "multi-line Python use a heredoc (python - <<'EOF' ... EOF) or an existing test file, never \\n "
        "escapes inside quotes." + core.FORK_RULE,
        {
            "tasks": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}, "acceptance": {"type": "string"}},
                    "required": ["task", "acceptance"],
                },
            },
        },
        ["tasks"],
    ),
    "commit": _tool(
        "commit",
        "Close this branch: its changes are kept as a patch for the thread that forked it. Run the task's acceptance "
        "check first and report whether it passed; the runtime runs it too, and refuses the first commit it fails. "
        "When your task's check was marked unreliable, pass the check you wrote as acceptance. Available only after "
        "a successful fork.",
        {
            "branch_id": {"type": "string"},
            "summary": {"type": "string"},
            "acceptance_passed": {"type": "boolean"},
            "acceptance": {"type": "string"},
        },
        ["branch_id", "summary", "acceptance_passed"],
    ),
    "abort": _tool(
        "abort",
        "Close this branch without keeping its changes, saying why the fix attempt failed (what you tried, "
        "what the acceptance check or tests showed) so the thread that forked it can act on it. Available only after a "
        "successful fork.",
        {"branch_id": {"type": "string"}, "reason": {"type": "string"}},
        ["branch_id", "reason"],
    ),
}


# OCTOPUS_BUNDLE: every fork task names the bundle task it fixes; that branch's patch is graded alone on the initial
# tree with that task's hidden tests only (octopus/envs/swe/taskset.py bundle_verifier_shell)
_BUNDLE_FORK = copy.deepcopy(_BRANCH_FIX_TOOLS["fork"])
_BUNDLE_ITEMS = _BUNDLE_FORK["function"]["parameters"]["properties"]["tasks"]["items"]
_BUNDLE_ITEMS["properties"]["task_id"] = {"type": "integer", "description": "The bundle task k (/tasks/<k>.md) this branch fixes."}
_BUNDLE_ITEMS["required"] = _BUNDLE_ITEMS["required"] + ["task_id"]
# e10 held-out s40: main made 3.7 commit calls per forked episode, all no-ops (its closing call is done)
_BUNDLE_COMMIT = copy.deepcopy(_BRANCH_FIX_TOOLS["commit"])
_BUNDLE_COMMIT["function"]["description"] += (
    " Only a branch commits, itself: the main thread never commits (its branches close themselves) and ends with done.")
TASK_ID_NEEDS = "fork needs task_id (the integer k of /tasks/<k>.md) on every task: the bundle task that branch fixes"


def fork_checks(args: dict) -> list[tuple[str, str]] | None:
    """(task id, acceptance command) of each task of a bundle fork call; None when a task lacks either (the runtime
    refuses that call)."""
    try:
        tasks = fork_task_list(args.get("tasks"))
    except ValueError:
        return None
    if tasks is None:
        return None
    out = [(task_id(t.get("task_id")), str(t.get("acceptance") or "").strip()) if isinstance(t, dict) else (None, "")
           for t in tasks]
    return out if out and all(k and check for k, check in out) else None


def weak_acceptance(task: str, why: str) -> str:
    """A bundle fork task whose check the red check (HarborWorkspace._red_check) found unreliable, as its branch sees
    it: the check is shown, but not under ACCEPTANCE_PREFIX (acceptance_of finds none, so the runtime does not verify
    a commit with it), and the branch is told to write its own check that fails now and to pass it to commit, where
    the runtime verifies it both ways."""
    head, sep, check = task.rpartition("\n" + ACCEPTANCE_PREFIX)
    if not sep:
        return task
    return (f"{head}\nAcceptance check given at the fork (unreliable: {why}, so it cannot tell a fix from no fix): "
            f"{check}\n{OWN_CHECK}")


OWN_CHECK = ("First write your own check that fails on the current code because of the bug, then fix the task and make "
             "that check pass. Pass it to commit as acceptance: the runtime runs it on the code you forked from, where "
             "it must fail, and on your tree, where it must pass.")


def auto_task(k: str) -> str:
    """OCTOPUS_AUTO_FORK: the task of the branch the environment forks for bundle task k (its statement follows in the
    task message): no check is given, so the branch writes its own (check_close verifies it both ways)."""
    return f"Fix task {k}.\nNo acceptance check was given. {OWN_CHECK}"


def runtime_tools() -> list[dict]:
    """The runtime tools as SWE offers them: core.runtime_tools, with each strategy's fork / commit / abort
    (branch_fix: acceptance checks; a bundle's fork names its task, its commit is a branch's)."""
    tools = core.runtime_tools()
    if not branch_fix():
        return tools
    catalog = dict(_BRANCH_FIX_TOOLS, fork=_BUNDLE_FORK, commit=_BUNDLE_COMMIT) if bundle() else _BRANCH_FIX_TOOLS
    tools = [catalog.get(item["function"]["name"], item) for item in tools]
    return core.with_rewind_rule(tools) if rewind_tool() else tools


def task_text(item) -> str | None:
    """A fork task as the branch sees it: a plain string, or task + acceptance check."""
    if isinstance(item, str):
        return item if item.strip() else None
    if isinstance(item, dict):
        task, check = str(item.get("task") or "").strip(), str(item.get("acceptance") or "").strip()
        if task and check:
            return f"{task}\n{ACCEPTANCE_PREFIX}{check}"
    return None


def acceptance_of(task: str) -> str | None:
    """The acceptance command inside a task text (see task_text), if any."""
    head, sep, check = str(task).rpartition("\n" + ACCEPTANCE_PREFIX)
    return check.strip() if sep else None


def offered_tools() -> list[dict]:
    """Whether and when to fork is the model's choice: every working tool from the start."""
    return TOOLS + runtime_tools()


# branch_fix: what a branch is told after its commit / abort contract (HarborWorkspace.branch_rules)
BRANCH_FIX_RULES = (
    "\nThis task is one complete fix attempt: locate the bug for your hypothesis and fix it by editing "
    "the product source. Before closing, run the task's acceptance check with shell, verbatim as written "
    "(do not rewrite it). Commit only when it "
    "exits 0, with acceptance_passed=true and a summary of the files you changed and what the fix does. "
    "If you cannot make it pass, abort with the reason."
)


def _fork_tasks(message: dict) -> list:
    """The tasks of the fork call in an assistant message ([] without one)."""
    call = next((c for c in message.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "fork"), None)
    try:
        return fork_task_list(decode_tool_call(call)[1].get("tasks") if call else None) or []
    except ValueError:
        return []


def task_id(value) -> str | None:
    """A fork task's bundle task id as "k" (from an int or a digit string), else None."""
    ok = (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and value.strip().isdigit())
    return str(int(value)) if ok else None


def stated_task_ids(message: dict) -> list:
    """The bundle task id ("k") each task of the fork call names, None where it names none."""
    return [task_id(t.get("task_id")) if isinstance(t, dict) else None for t in _fork_tasks(message)]


def fork_tasks(message: dict) -> list[tuple[str, str | None]]:
    """(acceptance check as the forking thread wrote it, bundle task id) of each task of the fork call in a forking
    reply: tree.json's branch check and task_id (core.fork_record)."""
    checks = [str(t.get("acceptance") or "") if isinstance(t, dict) else "" for t in _fork_tasks(message)]
    return list(zip(checks, stated_task_ids(message)))


def join_note(reports: list[dict], patches: dict | None = None, merged: dict | None = None, main: bool = True) -> str:
    """core.join_note with the branch_fix strategy's guidance after it (HarborWorkspace.join_note). main: the
    joining thread is main (a bundle branch that forked closes with commit/abort, not done)."""
    lines = [core.join_note(reports, patches, merged)]
    if branch_fix() and merged is not None and bundle():
        lines.append(
            f"The runtime applied each committed branch patch to {'/testbed' if main else 'your working copy'} with "
            "git apply --3way in commit order (merged above); do not apply those again. A task whose patch merged with "
            "its acceptance passed as checked by the runtime is fixed: do not re-check or redo it (one passed only as "
            "reported may be checked once). Work only on tasks whose patch was not applied (a conflicting patch is "
            "taken back whole: your files are unchanged), was not committed, or whose acceptance FAILED. " + ("Call done once every task is fixed." if main else
                                     "Then commit your task, or abort it if it cannot be fixed.")
            + " If no branch passed, fork again with new fix attempts."
        )
    elif branch_fix() and merged is not None:
        lines.append(
            "The runtime applied each committed branch patch to /testbed with git apply --3way in commit order "
            "(merged above); do not apply those again. A patch that conflicted was not applied: "
            "read it and apply the parts you still need by hand. Then re-run the acceptance checks and the tests. If "
            "no branch passed, fork again with new fix attempts."
        )
    elif branch_fix():
        lines.append(
            "Compare the branches' acceptance results. Merge passing branch patches into /testbed with git apply "
            "--3way instead of rewriting their fixes: only the best one among attempts at the same problem, each of "
            "them when they fix different problems. Then re-run the acceptance checks and the tests. If no branch "
            "passed, fork again with new fix attempts."
        )
    return "\n".join(lines)
