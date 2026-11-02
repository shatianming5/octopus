"""The workspace of SWE episodes: a git repository at /testbed in a Harbor container, and GrokHarborAgent, the cursor
runtime (octopus/runtime/tree.py) on it with the v2 native repository tools (tool_choice stays auto)."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
from pathlib import Path

from octopus.runtime.tree import BranchCopyError, CursorRuntime, WorkspaceProvider
from octopus.envs.swe.prompt import (
    BRANCH_FIX_RULES, TASK_ID_NEEDS, TOOLS, acceptance_of, auto_task, delivery, fork_checks, fork_tasks, join_note,
    offered_tools, runtime_tools, task_id, task_text, weak_acceptance,
)
from octopus.runtime.core import (
    BRANCH_REPORT, apply_runtime, branch_key, fork_task_list, tree_branches, unlocked_tools, unparsed_tasks,
)
from octopus.runtime.settings import auto_merge, branch_fix, bundle
from harbor.environments.base import BaseEnvironment

TESTBED = "/testbed"
BRANCH_ROOT = "/tmp/branches"
# RL plan episodes: the plan verifier's inputs (plan.tsv: per branch key, fork-point snapshot, patch path; the branch
# patches; control.snap), written from memory after the policy's last tool call: it may delete /tmp/branches
PLAN_DIR = "/tmp/octopus-plan"
PATCH_LOG_LIMIT = 2_000_000  # bytes of a branch patch kept in tree.json


# Red check (bundle-e2b audit, docs/task-bundles.md): before a bundle fork's branches start, every task's acceptance check
# runs in its branch's copy of the forking thread's tree, which is then put back to its fork point. One that already exits 0 cannot tell a fix from no fix, and one that is not
# a working command (not found, or bash/python cannot parse it) fails on any tree: such a task still starts, its branch
# told to write a check that fails now first (dropping them cost forks: bundle-e4, a dropped task was solved 16% of the
# time by main vs 37% for a started one)
RED_TIMEOUT = 60  # s per check, the shell tool's own limit (container_tools._run)
# the check's own code does not parse: bash -c's syntax errors, or python -c / heredoc compile errors (a "<string>" or
# "<stdin>" location that is not a traceback frame: code compiled at run time, e.g. bottle's templates, has one)
_UNPARSED = ('p !~ /^(    |  File )/ && /^  File "<(string|stdin)>", line [0-9]+$/ {f = 1} '
             '/^bash: -c: line [0-9]+: (syntax error|unexpected EOF)/ {f = 1} {p = $0} END {exit !f}')
BROKEN = ("127", "syntax")
RED_OUT = 1500  # bytes of each check's output kept for own_failure and the notes
# bundle-e6 audit (2026-10-13, 831 checks replayed on the reference fix): 62% of the checks that fail on the bug also
# fail on the fix. A failure raised in the check's own line by one of these (or any ModuleNotFoundError, or pytest
# finding no such test) is the check's own in 94% of cases: such a task starts weak like a broken one
OWN_ERRORS = ("ImportError", "ModuleNotFoundError", "NameError", "TypeError", "AttributeError", "FileNotFoundError",
              "IndexError", "UnboundLocalError")


def red_check_command(roots: list[str], checks: list[str], point: str) -> str:
    """Each check in its branch's copy (roots[i], fresh from its fork-point commit; /testbed in it mapped to the copy,
    as a branch's shell is), all at once, RED_TIMEOUT s each; prints "<i> <exit code> <base64 of the output's last
    RED_OUT bytes>" per check ("<i> syntax ..." for a failing check whose own code does not parse), then puts each copy
    back to its fork point (the commit `point`): git reset --hard restores every tracked file, and the fork-point
    commit tracks every file of the tree that is not ignored (git add -A), so git clean -fd (no -x: ignored files are
    never removed) removes only untracked files the check made. Fails if a copy cannot be put back."""
    return " ".join(
        f"(cd {r} && timeout {RED_TIMEOUT} bash -lc {shlex.quote(check.replace('/testbed', r))} > {r}.red 2>&1; rc=$?; "
        f"[ $rc != 0 ] && awk {shlex.quote(_UNPARSED)} {r}.red && rc=syntax; "
        f"echo \"{i} $rc $(tail -c {RED_OUT} {r}.red | base64 -w0)\"; rm -f {r}.red; "
        f"cd {r} && git reset -q --hard {point} && git clean -fdq) & r{i}=$!;"
        for i, (r, check) in enumerate(zip(roots, checks))) + " true" + "".join(
        f" && wait $r{i}" for i in range(len(checks)))


def red_results(stdout: str, n: int) -> list[tuple[str | None, str]]:
    """Each check's (exit code or "syntax", output tail) from red_check_command's output ((None, "") where it printed
    none)."""
    found = {}
    for line in str(stdout or "").splitlines():
        m = re.fullmatch(r"(\d+) (\d+|syntax)(?: (\S*))?", line.strip())
        if m:
            found[m.group(1)] = (m.group(2), base64.b64decode(m.group(3) or "").decode(errors="replace"))
    return [found.get(str(i), (None, "")) for i in range(n)]


def own_failure(check: str, code: str | None, out: str) -> str | None:
    """The error line when a check fails because of itself, not the bug (OWN_ERRORS): pytest finding no such test
    (exit 4: no such file or directory, 5: no test selected), a ModuleNotFoundError, or one of OWN_ERRORS raised in the
    check's own line (its last traceback frame is "<string>" / "<stdin>": python -c, a heredoc)."""
    if code in (None, "0") or code in BROKEN:
        return None
    if re.search(r"\bpytest\b", check) and code in ("4", "5"):
        return "pytest " + ("found no such test file or directory" if code == "4" else "selected no test") + f" (exit {code})"
    frames = re.findall(r'^\s*File "([^"]+)", line \d+', out, re.M)
    last = next((l.strip() for l in reversed(out.splitlines()) if re.match(r"\w+(Error|Exception)\b", l.strip())), "")
    kind = last.split(":")[0]
    if kind == "ModuleNotFoundError" or (kind in OWN_ERRORS and frames and frames[-1] in ("<string>", "<stdin>")):
        return last[:300]
    return None


def _red_why(passing: list[str], broken: list[str], own: dict) -> str:
    return "; ".join(([f"the acceptance check of task {', '.join(passing)} already exits 0 on the current tree, before "
                       "any fix, so it cannot detect the bug"] if passing else []) + (
        [f"the acceptance check of task {', '.join(broken)} is not a working command (command not found, or its code "
         "does not parse), so it fails whatever the code does"] if broken else []) + [
        f"the acceptance check of task {k} fails because of an error in the check itself ({err}), not because of the "
        "bug" for k, err in own.items()])


def red_weak_note(passing: list[str], broken: list[str], own: dict | None = None) -> str:
    """The forking thread's note on the tasks of a bundle fork whose check the red check found unreliable (started);
    own: {task id: error line} of checks failing because of themselves (own_failure)."""
    return ("\nStarted with an unreliable acceptance check: " + _red_why(passing, broken, own or {}) + ". Those "
            "branches were told to write a check that fails on the current code first." +
            (" A task that is already fixed needs no branch." if passing else ""))


def weak_why(code: str, error: str | None = None) -> str:
    """Why the red check found a check unreliable (its weak code: "0", BROKEN, or "own" with own_failure's line)."""
    return ("it already exits 0 on the unfixed code" if code == "0" else "it is not a working command" if code in BROKEN
            else f"it fails because of an error in the check itself ({error}), not because of the bug")


def fork_point_command(check: str, slot: str) -> str:
    """A branch's own check on its fork point (the HEAD commit of its private copy, which /testbed maps to in its
    shell), RED_TIMEOUT s, in a scratch copy named by `slot` (unique per thread); its exit code is the check's."""
    s = f"/tmp/octopus-fork-point/{re.sub(r'[^A-Za-z0-9_.-]', '_', slot)}"
    return (f"rm -rf {s} && mkdir -p {s} && cp -a /testbed/. {s} && cd {s} && git reset -q --hard && git clean -fdq && "
            f"timeout {RED_TIMEOUT} bash -lc {shlex.quote(check.replace('/testbed', s))}; rc=$?; rm -rf {s}; exit $rc")


def gate_note(check: str | None, out: str, fork_point: bool) -> str:
    """The commit refusal a branch gets once when the runtime's acceptance run fails, or when it has no check to run (an
    unreliable task's branch that passed none): it may commit again as it is."""
    tail = out[-RED_OUT:]
    why = ("your task has no reliable acceptance check, so the runtime cannot verify your fix. Write a command that fails "
           "on the code you forked from because of the bug and exits 0 on your tree, and commit again with it as "
           "acceptance" if not check else
           f"your acceptance check also exits 0 on the code you forked from, so it cannot show that your change fixes "
           f"the task:\n{tail}\nWrite one that fails there because of the bug and passes with your fix, and commit "
           "again with it" if fork_point else
           f"the acceptance check `{check[:300]}` does not exit 0 on your tree:\n{tail}\nFix the code until it exits 0 "
           "and commit again")
    return (f"commit not accepted yet: {why}; abort if you cannot, or call commit again to keep your changes as they "
            f"are (the thread that forked you is told {'the check failed' if check else 'your fix was not verified'}).")


# Rewinds (core.run_branch_continuation): what a thread's dropped steps did. A failed command is a non-zero exit
# or a timeout ("exit N"; edit/view replies never start with "exit")
COMMANDS = {"shell", "test", "reproduce", "rerun_reproduction"}
FAILED = re.compile(r"exit (?!0\n)\S+")


def error_key(observed: str) -> str:
    """A failure without its addresses, timings and line numbers, so repeats of one error compare equal."""
    return re.sub(r"0x[0-9a-f]+|\d+(\.\d+)?s\b|line \d+", "", str(observed))[-300:]


def _last_failures(failures: list) -> str:
    return " | ".join(list(dict.fromkeys(f.strip().splitlines()[-1][:200] for f in reversed(failures) if f.strip()))[:3])


def dropped_work(steps: list[dict]) -> tuple[list, list]:
    """The files edit/write calls among run_branch_continuation steps changed, and their failed commands (error_key)."""
    edited = sorted({s["arguments"]["path"] for s in steps
                     if s["name"] in ("edit", "write") and isinstance(s["arguments"].get("path"), str)})
    return edited, [error_key(s["observed"]) for s in steps if s["name"] in COMMANDS and FAILED.match(str(s["observed"]))]


# run in the repo: prints HEAD and a stash commit of the working tree (none for a clean tree), touching neither
SNAPSHOT = "git rev-parse HEAD && git add -A && git stash create; git reset -q"


def scratch_file(path: str) -> bool:
    """A file a branch made for itself (its repro or acceptance script, its test data), not code anything imports:
    at the repository root or a test_ module. The join may keep a sibling's file of the same name instead of it."""
    return "/" not in path or os.path.basename(path).startswith("test_")


def restore_command(snapshot: str) -> str:
    """Back to a SNAPSHOT ("HEAD [stash]"), undoing edits, new files and any commits made since."""
    head, _, stash = snapshot.partition(" ")
    return f"git reset -q --hard {head} && git clean -fdq" + (f" && git stash apply -q {stash} && git reset -q" if stash else "")


def closing_needs(name: str, args: dict) -> str | None:
    """branch_fix: a commit reports whether its acceptance check passed, an abort why the fix attempt failed."""
    if branch_fix() and name == "commit" and not isinstance(args.get("acceptance_passed"), bool):
        return "commit needs acceptance_passed (true/false): run the task's acceptance check first"
    if branch_fix() and name == "abort" and not (isinstance(args.get("reason"), str) and args["reason"].strip()):
        return "abort needs branch_id and a reason: why the fix attempt failed"
    return None


class HarborWorkspace(WorkspaceProvider):
    """Main works in /testbed, each branch in its own git copy of its forking thread's tree under /tmp/branches;
    tools run in the container through container_tools.py, a branch's result is its patch, applied at the join with
    git apply --3way, snapshots are git stash commits. A bundle (OCTOPUS_BUNDLE) has its statements in /tasks."""
    isolated = True
    branch_report = BRANCH_REPORT
    @property
    def branch_close(self) -> str:
        """After the commit/abort contract (search's branch_close, #193): what a branch's done does. Only under
        OCTOPUS_AUTO_MERGE, where the join merges committed patches alone; without it main sees every branch's patch."""
        return ("done only closes this branch: the join does not merge its patch into the forking thread's tree; "
                "commit is what merges it." if auto_merge() else "")

    # branch_fix: an abort says why the fix attempt failed (closing_needs), and the branch is told what a fix attempt
    # is (BRANCH_FIX_RULES)
    abort_reason = property(lambda self: branch_fix())
    branch_rules = property(lambda self: BRANCH_FIX_RULES if branch_fix() else "")
    # a bundle: main's commit / abort are refused (its branches close themselves before main's next call: its fork
    # waits for them), and a no-fork control may start from any fork of the tree
    main_close_refused = property(lambda self: bundle())
    any_fork_control = property(lambda self: bundle())
    fork_tasks = staticmethod(fork_tasks)
    join_note = staticmethod(join_note)
    remerge_note = ("Merges undone with the repository: {}; these branches are joined already, apply a patch again "
                    "with git apply --3way to keep it.")
    tool_names = frozenset(t["function"]["name"] for t in TOOLS + runtime_tools())
    edits, unchanged = frozenset({"edit", "write"}), frozenset({"view"})
    dropped = staticmethod(dropped_work)

    def __init__(self, environment: BaseEnvironment | None = None):
        super().__init__(environment)
        self.tool_states: dict[str, dict] = {}  # per root: container_tools.py's state
        self.statements, self.initial = {}, None  # a bundle's (begin)

    def system(self, linear: bool) -> str:
        return delivery(linear=linear)

    def tools(self, port, linear: bool) -> list[dict]:
        return unlocked_tools(TOOLS, True) if linear else offered_tools()

    def exec(self, command: str, **kw):
        """Every container command of the agent's run (tool calls, branch copies, red checks, git, which runs the agent's
        .git/hooks) has no network off the host (udocker_nokill.c): an agent fetched upstream source, i.e. the fix."""
        return self.environment.exec(command, env={"OCTOPUS_NO_EGRESS": "1"}, **kw)

    def root(self, thread: str) -> str:
        return TESTBED if thread == "main" else f"{BRANCH_ROOT}/{branch_key(thread)}"

    async def setup(self) -> None:
        source = Path(__file__).with_name("container_tools.py").read_bytes()
        blob = base64.b64encode(source).decode()
        command = (
            "python3 -c \"import base64,pathlib; "
            f"pathlib.Path('/tmp/harbor_tools.py').write_bytes(base64.b64decode('{blob}'))\""
        )
        await self.exec(
            command, cwd="/tmp",
            timeout_sec=int(os.environ.get("HARBOR_SETUP_TIMEOUT", "180")),
        )

    async def begin(self) -> None:
        """OCTOPUS_BUNDLE: the task statements (tasks) and the initial tree, which bundle branches are graded on
        (export)."""
        if bundle():
            self.statements = await self.tasks()  # task id -> its statement
            self.initial = await self.snapshot(self.root("main"))

    async def tasks(self) -> dict:
        """The bundle's task statements, {"k": text of /tasks/<k>.md as its branch sees it} (not INDEX.md: its length is
        the task count)."""
        read = "import glob, json; print(json.dumps({p[7:-3]: open(p).read() for p in glob.glob('/tasks/[0-9]*.md')}))"
        result = await self.exec(f'python3 -c "{read}"', cwd="/", timeout_sec=120)
        return {k: f"The statement of task {k} (/tasks/{k}.md):\n{text.strip()}"
                for k, text in json.loads(result.stdout).items()}

    async def call(self, root: str, thread: str | None, name: str, args: dict) -> tuple[str, bool]:
        """A repository tool, run by container_tools.py in the container with root's tool state."""
        if root != TESTBED:  # a branch: its /testbed is its private copy
            args = {k: v.replace(TESTBED, root) if isinstance(v, str) else v for k, v in args.items()}
        payload = base64.b64encode(json.dumps({
            "name": name,
            "args": args,
            "state": self.tool_states.get(root, {}),
        }, ensure_ascii=False).encode()).decode()
        result = await self.exec(
            f"python3 /tmp/harbor_tools.py {payload} {root}",
            cwd=root,
            timeout_sec=int(os.environ.get("HARBOR_EXEC_TIMEOUT", "180")),
        )
        raw = ((result.stdout or "") + (result.stderr or "")).strip()
        parsed = None
        for line in reversed(raw.splitlines()):
            piece = line.strip()
            if not piece.startswith("{"):
                continue
            try:
                body = json.loads(piece)
            except json.JSONDecodeError:
                continue
            if "observation" in body:
                parsed = body
                break
        if parsed is None:
            return (raw[-4000:] or f"exit {result.return_code}"), False
        if isinstance(parsed.get("state"), dict):
            self.tool_states[root] = parsed["state"]
        return str(parsed.get("observation") or ""), bool(parsed.get("stop"))

    async def snapshot(self, root: str) -> str:
        """/testbed's (or a branch workdir's) HEAD and working tree (a stash commit, empty for a clean tree), without
        touching either."""
        result = await self.exec(f"cd {root} && {SNAPSHOT}", cwd="/", timeout_sec=120)
        return " ".join((result.stdout or "").split()[:2])

    async def restore(self, root: str, snapshot: str) -> None:
        """Back to a _snapshot, undoing edits, new files and any commits the agent made since."""
        await self.exec(f"cd {root} && {restore_command(snapshot)}", cwd="/", timeout_sec=120)

    def rewind_note(self, edited: list, failed: list, step: int | None = None, note=None, forks: str = "") -> str:
        if step is None:  # OCTOPUS_REWIND
            return ("[runtime] Rewound: the attempt from this point did not work, so its file changes were undone and its "
                    f"steps removed. It edited {', '.join(edited) or 'nothing'} and kept failing with: "
                    + _last_failures(failed) + ". The repository is back as it was here. Take a different approach.")
        return (f"[runtime] Rewound to step {step} at your request: the steps after it were removed and the repository is "
                f"back as it was after step {step}. They edited {', '.join(edited) or 'nothing'}; "
                + (f"{len(failed)} commands failed, last with: {_last_failures(failed)}." if failed else "no command failed.")
                + (f" {forks}" if forks else "") + f"\nYour note: {note}")

    def runtime(self, name: str, args: dict, port) -> tuple[str, bool]:
        """A fork task is a string or a task and its acceptance check (task_text); a bundle's names its task_id.
        branch_fix: closing_needs."""
        if name == "fork":
            try:
                tasks = fork_task_list(args.get("tasks"))
            except json.JSONDecodeError as e:
                keys = ['"task": "..."', '"acceptance": "..."'] + ['"task_id": 1'] * bundle()
                return unparsed_tasks(e, f"[{{{', '.join(keys)}}}, ...]") if branch_fix() else unparsed_tasks(e), False
            if bundle() and isinstance(tasks, list) and not all(isinstance(t, dict) and task_id(t.get("task_id")) for t in tasks):
                return TASK_ID_NEEDS, False
            if isinstance(tasks, list):
                args = dict(args, tasks=[task_text(t) for t in tasks])
        return apply_runtime(name, args, port, "tasks, each with task and acceptance" if branch_fix() else "tasks",
                             closing_needs)

    def fork_refusal(self, args: dict) -> str | None:
        """OCTOPUS_BUNDLE: a fork naming a task id the bundle does not have (bundle-e2b: 0.7% of branches named k=0,
        99, n+1)."""
        if bundle() and any(k not in self.statements for k, _ in fork_checks(args) or []):
            return TASK_ID_NEEDS
        return None

    def forked(self, fork: dict, args: dict) -> None:
        """OCTOPUS_BUNDLE: each branch also sees its task's own statement (the forking thread's text may be wrong);
        the fork's checks are red-checked in the branch copies (open_branches)."""
        if bundle():
            fork["checks"] = fork_checks(args)
            for branch, (k, _) in zip(fork["branches"], fork["checks"]):
                branch["task_id"] = k
                branch["statement"] = self.statements[k]

    def auto_fork(self, port) -> int:
        """OCTOPUS_AUTO_FORK: the environment's fork on main's port, one branch per bundle task in task order, each
        named by its task id and seeing its statement (as forked); returns the branch count."""
        tasks = sorted(self.statements, key=int)
        port.fork([auto_task(k) for k in tasks])
        for branch, k in zip(port.last_fork["branches"], tasks):
            branch.update(task_id=k, statement=self.statements[k])
        return len(tasks)

    async def check_close(self, start: dict, args: dict, closes: bool, call):
        """branch_fix: a commit first runs the task's acceptance check, or on a closing commit of a task whose check
        was unreliable the branch's own one, which must also fail on its fork point. Verified as soon as the branch
        says it is done: the first closing commit whose check fails, or of a fix branch with no check to run, is
        refused (gate_note)."""
        check = acceptance_of(start["task"]) if branch_fix() else None
        own = (str(args.get("acceptance") or "").strip() or None) if closes and not check else None
        verified, out, at_fork = None, "", False
        if check or own:
            out = str(await call("shell", {"command": check or own}))
            verified = re.match(r"exit 0(\n|$)", out) is not None
            if own and verified:  # a control thread has no session
                out = str(await call("shell", {"command": fork_point_command(own, start["session_id"] or start["branch_id"])}))
                at_fork = re.match(r"exit 0(\n|$)", out) is not None  # passes where it forked too: shows no fix
                verified = not at_fork
        fields = {} if verified is None else {"acceptance_verified": verified, **({"acceptance_own": own} if own else {})}
        if verified is False or (verified is None and branch_fix()):
            return gate_note(check or own, out, at_fork), fields
        return None, fields

    async def open_branches(self, source: str, roots: dict, fork: dict) -> tuple[dict, dict | None]:
        """A bundle fork's checks (forked) run in the copies (_red_check): a task whose check is unreliable still
        starts, its branch told why (weak_acceptance); the fork gets weak [index] and the note naming those tasks."""
        checks = fork.get("checks") or []
        heads, results = await self._open_branch_copies(roots, source, [c for _, c in checks])
        if not checks:
            return heads, None
        red = self._red_check(checks, results)
        for i, why in red["weak"].items():
            fork["branches"][i]["task"] = weak_acceptance(fork["branches"][i]["task"], why)
        fork.update(weak=sorted(red["weak"]), red_note=red["note"])
        return heads, {"task_ids": [k for k, _ in checks], **red["record"]}

    @staticmethod
    def _red_check(checks: list, results: list) -> dict:
        """OCTOPUS_BUNDLE: each fork task's acceptance check ran in its branch's copy of the forking thread's tree
        (_open_branch_copies); a check that already exits 0 cannot detect its bug and a broken one
        (BROKEN) fails on any tree, and one failing because of itself (own_failure: an error raised in
        its own line) fails on the fixed tree too: that task still starts, its branch told so (weak_why)."""
        exits = [code for code, _ in results]
        own = {i: own_failure(c, code, out) for i, ((_, c), (code, out)) in enumerate(zip(checks, results))}
        own = {i: line for i, line in own.items() if line}
        passing = [k for (k, _), code in zip(checks, exits) if code == "0"]
        broken = [k for (k, _), code in zip(checks, exits) if code in BROKEN]
        weak = {i: (code, None) for i, code in enumerate(exits) if code == "0" or code in BROKEN}
        weak |= {i: ("own", line) for i, line in own.items()}
        return {"weak": {i: weak_why(*why) for i, why in weak.items()},
                "note": red_weak_note(passing, broken, {checks[i][0]: line for i, line in own.items()}) if weak else "",
                "record": {"exits": exits, "weak": sorted(weak), "own": {str(i): line for i, line in own.items()}}}

    async def _open_branch_copies(self, workdirs: dict, source: str = TESTBED,
                                  checks: list[str] = ()) -> tuple[dict, list]:
        """One private working copy of `source` (/testbed, or a forking branch's own copy) per branch, with its fork
        point committed, so branches cannot see or clobber each other's edits and each one's changes diff cleanly.
        One command: the fork point is committed in the first copy, which the others then copy all at once, each
        refreshing its index (cp -a gives new inodes/ctimes; a stale index fails git apply --index/--3way). A bundle
        fork's red checks then run in the copies (red_check_command), which are put back to the fork point. Returns
        each copy's fork-point commit (the same) and each check's red_results. `source` is staged
        first (the thread may have edited: the join's git apply --3way needs index == tree)."""
        await self.exec(f"cd {source} && git add -A", cwd="/", timeout_sec=120)
        first, *rest = workdirs.values()
        copies = " ".join(f"(cp -a {first} {root} && git -C {root} update-index -q --refresh) & p{i}=$!;"
                          for i, root in enumerate(rest))
        waits = "".join(f" && wait $p{i}" for i in range(len(rest)))
        result = await self.exec(
            f"{{ rm -rf {' '.join(workdirs.values())} && mkdir -p {BRANCH_ROOT} && cp -a {source} {first} && cd {first}"
            " && git add -A && git -c user.name=octopus -c user.email=octopus@localhost"
            f" commit -q --allow-empty -m 'fork point'; }} || exit 1; {copies} true{waits} || exit 1; "
            + (f"h=$(git rev-parse HEAD); {red_check_command(list(workdirs.values()), checks, '$h')} || exit 1; "
               if checks else "") + "git rev-parse HEAD",
            cwd="/", timeout_sec=600 * len(workdirs) + (RED_TIMEOUT + 600 if checks else 0),
        )
        if result.return_code:  # e11: the policy's own shell had deleted the forking branch's workdir, `rm -rf
            # /tmp/branches`; cp failed on a dir main had made 0555
            raise BranchCopyError(f"branch copies of {source} failed: {(result.stderr or result.stdout or '')[-500:]}")
        lines = (result.stdout or "").strip().splitlines()
        return dict.fromkeys(workdirs.values(), lines[-1] if lines else ""), red_results(result.stdout, len(checks))

    async def capture(self, root: str, since: str | None = None) -> tuple[str, str, str]:
        """Save a branch's changes since its fork point as a patch; returns (patch path, diffstat,
        patch text). The text, its changes since `since` (a nested branch's depth-1 ancestor's fork point; None: since its
        own), is logged
        so each branch can be verified alone after the trial."""
        patch = f"{root}.patch"
        result = await self.exec(
            f"cd {root} && git add -A && git diff --cached --binary HEAD > {patch}"
            " && git diff --cached --shortstat HEAD",
            cwd="/", timeout_sec=300,
        )
        text = await self.exec(f"head -c {PATCH_LOG_LIMIT} {patch}" if not since else
                                      f"cd {root} && git diff --cached --binary {since} | head -c {PATCH_LOG_LIMIT}",
                                      cwd="/", timeout_sec=120)
        return patch, (result.stdout or "").strip(), text.stdout or ""

    async def apply(self, root: str, results: list[tuple[str, str]]) -> dict:
        """Each (branch id, patch) applied to root with git apply --3way, in order: {branch id: "merged" | "conflict:
        <files>"}. A patch that conflicts is taken back whole (git read-tree --reset -u to the tree before it), so the
        thread's files never hold conflict markers (files: the unmerged paths, or git's error when the patch did not
        apply). bundle-e4: markers left in the tree cost main's own tasks (solved 6% vs 15% by the no-fork control) and
        most of fork - control. The thread's tree is staged first (git apply --3way needs index == tree).
        A patch whose only conflicts are scratch files it creates (scratch_file) that root already has (a sibling's,
        merged before it) is applied without them instead: "merged: except its new files <files> ...", so its fixes in
        tracked files land (Gate 0 k 0/1: 41 of 66 taken-back patches conflicted only so)."""
        out = {}
        await self.exec(f"cd {root} && git add -A", cwd="/", timeout_sec=120)
        async def take(patch: str, exclude: list[str] = ()) -> list[str]:
            skip = "".join(f" --exclude={shlex.quote(f)}" for f in exclude)
            result = await self.exec(
                f"cd {root} && t=$(git write-tree) && if err=$(git apply --3way --whitespace=nowarn{skip} {patch} 2>&1);"
                " then echo merged; else echo conflict; u=$(git diff --name-only --diff-filter=U); "
                "git read-tree --reset -u $t; echo \"${u:-$(echo \"$err\" | tail -1)}\"; fi",
                cwd="/", timeout_sec=300)
            return (result.stdout or "").strip().splitlines()

        for branch_id, patch in results:
            lines = await take(patch)
            out[branch_id] = "merged" if lines[:1] == ["merged"] else f"conflict: {', '.join(lines[1:])}"
            if lines[:1] == ["merged"] or not lines[1:]:
                continue
            new = await self.exec(f"cd {root} && git apply --summary {patch} | sed -n 's/^ create mode [0-7]* //p'",
                                  cwd="/", timeout_sec=120)
            if all(f in (new.stdout or "").splitlines() and scratch_file(f) for f in lines[1:]) and \
                    (await take(patch, lines[1:]))[:1] == ["merged"]:
                out[branch_id] = f"merged: except its new files {', '.join(lines[1:])} (a sibling's of the same names kept)"
        return out

    def touched(self, root: str, name: str, args: dict, observed: str) -> tuple[str, bool] | None:
        """OCTOPUS_READ_EVENT: a view of an existing path (args' path, relative to root), or an edit / write as
        container_tools reports it done (its relative path)."""
        done = re.fullmatch(r"replaced 1 occurrence in (.+)|created (.+) \(\d+ bytes\)", observed)
        if name == "view" and isinstance(args.get("path"), str) and not observed.startswith("missing "):
            path = args["path"].replace(TESTBED, root)
            return os.path.normpath(os.path.relpath(path, root) if path.startswith("/") else path), False
        if name in ("edit", "write") and done:
            return done.group(1) or done.group(2), True
        return None

    async def changed(self, root: str) -> set:
        """OCTOPUS_READ_EVENT: the files of a closed branch's own patch (capture staged them; HEAD: its fork point)."""
        result = await self.exec(f"cd {root} && git diff --cached --name-only HEAD", cwd="/", timeout_sec=120)
        return set((result.stdout or "").split("\n")) - {""}

    async def conflict_by(self, ref: str, files: list[str], merged: list[str]) -> dict:
        """OCTOPUS_CONFLICT_BY: per file (git's unmerged paths), "sibling" when the branch's version (its workdir, next
        to its .patch) conflicts (git merge-file) with its fork-point version (the workdir's HEAD) plus the patches
        merged into the forking thread since the fork (merged, in order) alone, else "own" (the forking thread's own
        edits since the fork); "unclear" when one of those patches does not apply to the fork-point version or the name
        is no file (git's error when the patch did not apply at all)."""
        q, w = shlex.quote, ref[:-len(".patch")]
        lines = []
        for f in files:
            applies = " ".join(f"&& (cd $d/s && git apply --include={q(f)} {q(p)})" for p in merged)
            lines.append(
                f"if [ -e {q(w + '/' + f)} ] || git -C {q(w)} cat-file -e HEAD:{q(f)} 2>/dev/null; then "
                f"rm -rf $d/s && mkdir -p \"$(dirname $d/s/{q(f)})\" && {{ git -C {q(w)} show HEAD:{q(f)} 2>/dev/null; true; }}"
                f" > $d/o && cp $d/o $d/s/{q(f)} && {{ cat {q(w + '/' + f)} 2>/dev/null; true; }} > $d/b {applies} "
                f"&& {{ git merge-file -q -p $d/b $d/o $d/s/{q(f)} > /dev/null && echo own || echo sibling; }} "
                "|| echo unclear; else echo unclear; fi")
        result = await self.exec("d=$(mktemp -d); " + "; ".join(lines) + "; rm -rf $d", cwd="/", timeout_sec=300)
        return dict(zip(files, (result.stdout or "").split()))

    async def export(self, local: Path, forks: list[dict], read_patches: dict, control_patch: tuple | None,
                     control_snap: str | None) -> None:
        """PLAN_DIR from the in-memory fork records (a copy stays in `local`, the agent logs' plan/)."""
        local.mkdir(parents=True, exist_ok=True)
        initial = self.initial
        rows = []
        for name, fork, branch in tree_branches(forks):  # every node but the root, at any depth
            key, patch = branch_key(name), ""
            if branch.get("patch_text"):
                (local / f"{key}.patch").write_text(branch["patch_text"])
                patch = f"{PLAN_DIR}/{key}.patch"
            if bundle():  # alone on the initial tree, graded on the task it named (every task if none)
                rows.append(f"{key}\t{initial}\t{branch.get('task_id') or '-'}\t{patch or '-'}\n")
            else:
                rows.append(f"{key}\t{fork['fork_point']}\t{patch}\n")
        for name, fork, branch in tree_branches(forks):  # OCTOPUS_READ_CONTROL: each read control, graded like its reader
            if name not in read_patches:
                continue
            key, (task, text) = branch_key(f"readctl/{name}"), read_patches[name]
            if text:
                (local / f"{key}.patch").write_text(text)
            patch = f"{PLAN_DIR}/{key}.patch" if text else "-"
            rows.append(f"{key}\t{initial}\t{task or '-'}\t{patch}\n" if bundle() else
                        f"{key}\t{fork['fork_point']}\t{patch if text else ''}\n")
        if control_patch:  # a nested control: graded like a branch, on its branch's task
            task, text = control_patch
            if text:
                (local / "control.patch").write_text(text)
            rows.append(f"control\t{initial}\t{task or '-'}\t{PLAN_DIR + '/control.patch' if text else '-'}\n")
        (local / "plan.tsv").write_text("".join(rows))
        if bundle():  # each fork's checks, and a branch's own one, run by the verifier on the bug and on its fix
            (local / "checks.tsv").write_text("".join(
                f"{branch_key(name)}\t{kind}\t{initial}\t{branch.get('task_id')}\t"
                f"{base64.b64encode(check.encode()).decode()}\n" for name, _, branch in tree_branches(forks)
                if branch.get("task_id") for kind, check in (("fork", branch.get("check")), ("own", branch.get("acceptance_own")))
                if check))
        if control_snap:
            (local / "control.snap").write_text(control_snap)
        await self.exec(f"rm -rf {PLAN_DIR}", cwd="/", timeout_sec=120)
        await self.environment.upload_dir(local, PLAN_DIR)


class GrokHarborAgent(CursorRuntime):
    provider = HarborWorkspace

    @staticmethod
    def name() -> str:
        return "grok-harbor"
