"""The cursor runtime: an Octopus episode's fork tree, independent of the world its threads work in.

CursorRuntime runs the threads (fork / join / rewind / read_thread, each thread's history and mounts, the chat server
and its cursor protocol, budgets, call caps, controls), each thread in octopus/runtime/core.py's loop. Everything
about the environment is its WorkspaceProvider's (self.ws): the system prompt, tools and budget, the threads'
workspaces (copy, snapshot, restore), working-tool and runtime-tool calls, the check of a closing call, rewind notes,
a branch's result (capture, apply at the join, grade) and what the episode's verifier gets. An environment is a
provider plus a two-line agent class: octopus/envs/swe/agent.py (SWE: a repository in a Harbor container),
octopus/envs/search/agent.py (an offline search corpus, no workspace).
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from octopus.runtime.core import (
    BRANCH_REPORT, CLOSING, RUNTIME_NAMES, THREAD_TAG, RuntimePort, add_note, apply_runtime, branch_starts,
    branch_turns, closing_name, closing_step, cursor_delta, decode_tool_call, fork_record, handed, join_note,
    post_chat, read_card, read_offer, reply_tools, run_branch_continuation, step_number, task_line, thread_name,
    tree_branches, without_cursors,
)
from octopus.runtime.settings import (
    answer_reserve, auto_fork, auto_merge, call_limit, conflict_by, cursor_mode, max_reply, plan_episode, read_control,
    read_event, read_tool, rewind_tool, thread_budget, window_room,
)
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory

# Own pool: asyncio's default one (min(32, cpus+4)) is shared with the workspace's calls and fills at n=16 with forks.
_HTTP_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("OCTOPUS_HTTP_WORKERS", "128")),
    thread_name_prefix="grok-http",
)



# a shared token budget (search's OCTOPUS_SEARCH_BUDGET; SWE has none): completion tokens of the main thread and every
# branch count against it; every tool result starts with what is left, and at 0 the episode ends as it stands
BUDGET_TAG = "[token budget left: "
STEP_TAG = "[step "  # OCTOPUS_REWIND_TOOL without a budget tag
CONTEXT_TAG = "[context left: "  # thread_budget without a budget tag (with one: after its tokens left)


def read_deadline(stop: str, end_time: float | None, mounted_at: float, pool_out: float | None) -> tuple:
    """OCTOPUS_READ_CONTROL: the external deadline a reader faced from its mount request (at mounted_at), which its read
    control then faces from its own start: ([seconds, 0] or None, where it came from: "pool"). The episode's shared
    pool running out (pool_out: every running thread stops) or, for this reader alone, its own stop by an empty pool,
    at its end_time (main's end stops no branch: main's fork waits for it). stop: the reader's stop when it did not
    close itself, else ""; a reader that closed or stopped itself still faced the pool."""
    t = pool_out if pool_out is not None else end_time if "budget used up" in stop else None
    return (None, None) if t is None else ([max(0.0, t - mounted_at), 0], "pool")


def without_step_listing(result: str) -> str:
    """A mounting read's result without its step listing (single lines), keeping a note add_note put after it."""
    head, _, rest = result.partition("\nstep ")
    return head + (rest[rest.index("\n\n"):] if "\n\n" in rest else "")


# Nested forks: a branch may fork like the main thread while its depth (main 0, its branches 1, ...) is below
# OCTOPUS_FORK_DEPTH; OCTOPUS_MAX_THREADS caps the threads running at once (main + open branches), bounding the KV cache
FORK_DEPTH_REFUSAL = "Forking is not available at this depth; finish this task yourself."
# and while its branches would start with room in the serving window (settings.window_room) for the answer reserve and
# FORK_FLOOR more: one search result (10 documents of 512 tokens) and a reply's text, beside the fork's result and task
# message, which a branch's start adds to the forking reply's prompt and completion (a few hundred tokens)
FORK_FLOOR = 8192
FORK_WINDOW_REFUSAL = ("Forking is not available here: not enough context window is left at this depth; finish this "
                       "task yourself.")
ONE_FORK = "fork refused: one fork per reply, this reply's first fork call; put every task in that one fork's tasks."
# a bundle (and search before protocol v6, whose threads have no commit / abort now): main's commit closes nothing
# (e10 s40: 3.7 per forked episode, each answered as if recorded; search smoke 10-20: 13 calls in 10 of 40 episodes);
# its abort neither: its fork has joined its branches
MAIN_CLOSE = ("{} has no effect in the main thread: a branch commits or aborts itself. "
              "End the main thread with done.")


class BranchCopyError(RuntimeError):
    """A fork's branch workspaces could not be made (WorkspaceProvider._open_branch_copies): the fork is refused, not
    the episode."""


def fork_depth() -> float:
    """OCTOPUS_FORK_DEPTH=D: threads fork while their depth is below D (1: only main forks); unset: 2; 0: no depth
    limit (math.inf): the serving window ends deep chains (FORK_WINDOW_REFUSAL, settings.window_room)."""
    return int(os.environ.get("OCTOPUS_FORK_DEPTH", "").strip() or 2) or math.inf


def max_threads() -> int:
    return int(os.environ.get("OCTOPUS_MAX_THREADS", "16"))


def max_branches() -> int | None:
    """OCTOPUS_MAX_BRANCHES=K (the search default 10, Context-Folding's): at most K branches made in one episode, at
    every depth; a fork past it is refused. Unset or 0: no cap."""
    return int(os.environ.get("OCTOPUS_MAX_BRANCHES", "").strip() or 0) or None


def main_ending(steps: list[dict], budget_out: bool) -> str:
    """How the main thread ended (tree.json main_end): done, budget (budget_out: the shared token budget is gone), step
    cap (OCTOPUS_MAIN_MAX_CALLS), context (thread_budget's growth), no tool call, or
    request failed."""
    if any(s.get("name") == "done" and not s.get("rejected") for s in steps):
        return "done"
    if budget_out:
        return "budget"
    last = steps[-1] if steps else {"name": None, "text": ""}
    if last.get("name"):  # context: it grew past thread_budget (run_branch_continuation); else out of calls
        return "context" if last["name"] == "context" else "step cap"
    text = str(last.get("text") or "")
    return "step cap" if text.startswith("stopped: ") else "request failed" if text.startswith("request failed") \
        else "no tool call"


def closing_only(payload: dict) -> bool:
    """The reply that used up the budget: only its done/commit/abort calls run. Whether it had tool calls (one with
    none: its thread stopped itself, on its text)."""
    choice = payload["choices"][0]["message"]
    calls, text = reply_tools(choice)
    choice["tool_calls"] = [c for c in calls if (c.get("function") or {}).get("name") in CLOSING]
    choice["content"] = text
    return bool(calls)


class WorkspaceProvider:
    """What CursorRuntime needs from the world its threads work in (CursorRuntime.ws). A thread's workspace is named
    by its root (root()); a branch's result is (ref, summary, text): what apply takes, a short description ("" for no
    change), its full text for the logs. Operations a provider's episodes never reach (bundle tasks, verifier inputs,
    rewinds) it may leave out."""
    isolated: bool  # each branch works in its own copy and hands back a result (the join note lists it, the task names
    # the copy)
    branch_report: str = BRANCH_REPORT  # what a branch's commit summary carries (core.branch_task_message)
    remerge_note: str  # "{}": the merges a rewind undid (_rewound)
    tool_names: frozenset  # every tool the environment has, offered or not
    # a branch's task message (core.branch_task_message): a sentence after its commit/abort contract, whether
    # its abort carries a reason, a paragraph after them
    branch_close: str = ""
    abort_reason: bool = False
    branch_rules: str = ""
    # how a branch closes: its task message's contract in place of commit/abort's (None: theirs), and the calls that
    # close it (run_branch_continuation's closing)
    branch_contract: str | None = None
    branch_closing: frozenset = frozenset(CLOSING)
    fork_min = 2  # the fewest tasks a fork takes (apply_runtime's fewest)
    # OCTOPUS_READ_TOOL: a branch is offered read_thread (a roster of the threads it may read and the card: in its task
    # message, or before a later request) once a thread it may read has a readable step (CursorRuntime._readable), and
    # its read_thread is refused until then; False: always, in its task message
    read_when_readable = False
    # OCTOPUS_READ_TOOL: read_thread names a thread by its episode-unique name (core.thread_name, e.g. fork0/branch-1) in
    # its argument, a branch's roster and the refusals; a bare branch id only where one thread has it. False: by branch
    # id, a later fork's id shadowing an earlier one's
    read_by_name = False
    # a no-fork control may start from any accepted fork of the tree (_maybe_control), not only main's first
    any_fork_control: bool = False
    # rewinds (run_branch_continuation): working tools that edit (OCTOPUS_REWIND's attempt starts at the first) and
    # that never change the workspace (OCTOPUS_REWIND_TOOL takes no snapshot before them)
    edits: frozenset = frozenset()
    unchanged: frozenset = frozenset()

    def __init__(self, environment: BaseEnvironment | None = None):
        self.environment = environment  # the episode's Harbor environment (None without one)

    main_close_refused = False  # the main thread's commit / abort are refused (MAIN_CLOSE: a SWE bundle's)

    @property
    def read_card(self) -> str:
        """OCTOPUS_READ_TOOL: read_thread's card (a branch's task message, main's first fork result)."""
        return read_card()

    def system(self, linear: bool) -> str:
        """The system message: of a forking run, or (linear) of the no-fork control."""
        raise NotImplementedError

    def tools(self, port, linear: bool) -> list[dict]:
        """A request's tool list: what the thread's runtime port has unlocked, or (linear) the no-fork control's."""
        raise NotImplementedError

    def budget(self, instruction: str) -> dict | None:
        """The shared completion-token budget of the whole tree, {"limit", "used": 0}, or None (SWE: none)."""
        return None

    def fork_note(self) -> str:
        """What an accepted fork call's result says after the runtime port's ("fork started: branches ...")."""
        return ""

    def root(self, thread: str) -> str:
        """The root of a thread's workspace (core.thread_name; main: the episode's own)."""
        raise NotImplementedError

    async def setup(self) -> None:
        """Harbor's: prepare the environment before the episode."""
        raise NotImplementedError

    async def begin(self) -> None:
        """Before the episode's first request (SWE: a bundle's task statements and initial tree)."""

    async def call(self, root: str, thread: str | None, name: str, args: dict) -> tuple[str, bool]:
        """One working-tool call (not a runtime tool) in the workspace at root, by thread (main, a branch's thread
        name, control, readctl/...): (observation, stop)."""
        raise NotImplementedError

    async def snapshot(self, root: str) -> str:
        """The workspace's state, for restore (rewinds, fork points, controls), without changing it."""
        raise NotImplementedError

    async def restore(self, root: str, snapshot: str) -> None:
        raise NotImplementedError

    def runtime(self, name: str, args: dict, port) -> tuple[str, bool]:
        """One runtime-tool call (RUNTIME_NAMES) on the thread's port (apply_runtime); a provider turns its own fork
        task shape into the texts branches are told and adds what its closing calls need."""
        return apply_runtime(name, args, port, fewest=self.fork_min)

    def fork_refusal(self, args: dict) -> str | None:
        """Why a thread's fork call is refused before it runs (SWE: a bundle task id the episode does not have), or
        None."""
        return None

    def forked(self, fork: dict, args: dict) -> None:
        """A fork the runtime accepted (port.last_fork) and its call's arguments, before its branches start; a provider
        may add to it (SWE: a bundle fork's checks, each branch's task_id and statement)."""

    def fork_tasks(self, message: dict) -> list[tuple[str, str | None]]:
        """tree.json's (check, task id) of each task of the fork call in a forking reply (core.fork_record);
        none by default."""
        return []

    def join_note(self, reports: list[dict], patches: dict | None, merged: dict | None, main: bool) -> str:
        """What the joining thread is told once its fork's branches closed (core.join_note; main: it is
        main)."""
        return join_note(reports, patches, merged)

    async def check_close(self, start: dict, args: dict, closes: bool, call) -> tuple[str | None, dict]:
        """A commit a thread (start: run_branch_continuation's) is about to run, checked in its workspace
        first (closes: it would close the thread; call(name, args): a working-tool call there, unseen by the model).
        Returns the refusal it deserves (the loop applies it to a closing call, once per thread) and fields for its
        step; by default no check."""
        return None, {}

    def dropped(self, steps: list[dict]) -> tuple[list, list]:
        """Rewinds: what the steps a rewind drops did (the files they changed, their failures' error texts)."""
        return [], []

    def rewind_note(self, edited: list, failed: list, step: int | None = None, note=None, forks: str = "") -> str:
        """Rewinds: what replaces the dropped steps in the thread's history, an OCTOPUS_REWIND rewind's (step None; only a
        provider with edits has those) or a rewind call's to `step` with the model's note (forks: what it did to the
        thread's forks). By default the history alone (search: no workspace); SWE's names the repository too."""
        return (f"[runtime] Rewound to step {step} at your request: the steps after it were removed from your history."
                + (f" {forks}" if forks else "") + f"\nYour note: {note}")

    async def open_branches(self, source: str, roots: dict, fork: dict) -> tuple[dict, dict | None]:
        """One private workspace per branch (roots: branch id -> root) from the forking thread's at source, for the
        fork port.last_fork holds; raises BranchCopyError. Returns ({root: the base its result is taken against}, None
        or the fields of a tree.json red_checks record: a bundle fork's checks run in the copies, which may also rewrite
        the fork's branch tasks and set its "weak" (indices) and "red_note" (the forking thread's note on them))."""
        raise NotImplementedError

    async def capture(self, root: str, since: str | None = None) -> tuple:
        """A closed branch's result (ref, summary, text), text since `since` (None: since its fork point)."""
        raise NotImplementedError

    async def apply(self, root: str, results: list[tuple[str, str]]) -> dict:
        """Each (branch id, ref) applied to the workspace at root, in order: {branch id: "merged" | "conflict: ..."}; a
        result applied in part is "merged: <what was left out>" (every reader takes the "merged" prefix as merged)."""
        raise NotImplementedError

    def touched(self, root: str, name: str, args: dict, observed: str) -> tuple[str, bool] | None:
        """OCTOPUS_READ_EVENT: the file (relative to root) a working-tool call looked at (False) or changed (True), as
        its observation reports it; None for any other call."""
        raise NotImplementedError

    async def changed(self, root: str) -> set:
        """OCTOPUS_READ_EVENT: the files a closed branch's own result changes (since its fork point)."""
        raise NotImplementedError

    async def conflict_by(self, ref: str, files: list[str], merged: list[str]) -> dict:
        """OCTOPUS_CONFLICT_BY: for each file of a result apply took back (files: as its "conflict: ..." names them),
        "sibling" (it conflicts with its base plus the results merged since its fork, merged, alone), "own" (with the
        forking thread's own changes) or "unclear"."""
        raise NotImplementedError

    async def export(self, local: Path, forks: list[dict], read_patches: dict, control_patch: tuple | None,
                     control_snap: str | None) -> None:
        """RL plan episodes (OCTOPUS_PLAN_EPISODE): the fork tree's results and the controls', for the episode's verifier
        (a copy in local)."""
        raise NotImplementedError


class CursorRuntime(BaseAgent):
    capabilities = AgentCapabilities(atif=True)
    provider: type[WorkspaceProvider]  # the environment (an agent class sets it)
    # an agent before (or driven without) run, e.g. in tests: run and _read_controls rebind the containers it changes
    _pinned_endpoint = _initial_messages = _runtime = _dispatch_lock = None
    _budget = _event = _readctl = None
    _rewinds = _main_end = _pool_out_t = _control = _control_patch = _control_snap = None
    _control_running = _control_linear = False
    _read_patches, _red_checks = {}, []

    @functools.cached_property
    def ws(self) -> WorkspaceProvider:
        """The episode's workspace: setup (or run) binds it to the episode's environment (an agent driven piecewise,
        without either, gets one without)."""
        return self.provider()

    async def setup(self, environment: BaseEnvironment) -> None:
        self.ws = self.provider(environment)
        await self.ws.setup()

    def version(self) -> str | None:
        return "1"

    def _endpoint(self) -> str:
        return self._pinned_endpoint or os.environ["OCTOPUS_CHAT_URL"]  # RL: model_env; no default endpoint

    def _complete(self, messages: list[dict], session_id: str | None = None, cursor: str | None = None,
                  mount: dict | None = None) -> dict:
        if not self.model_name:
            raise RuntimeError("no model name: Harbor's -m (the search env: OCTOPUS_MODEL) is unset")
        body = {
            "model": self.model_name,  # Harbor's -m (RL: "policy"; the AReaL proxy serves its own adapter)
            "messages": messages,
            "tools": self._tools(),
            "tool_choice": "auto",
            "max_tokens": max_reply(),
        }
        if session_id:
            body["session_id"] = session_id
        if cursor:  # OCTOPUS_CURSOR: messages are those after this reply (_complete_async)
            body["cursor"] = cursor
        if mount:  # OCTOPUS_READ_TOOL: another thread's steps for the attention layers (_read)
            body["mount"] = mount
        # e.g. a model card's temperature, top_p; never the agent's own fields (model, messages, tools, ...)
        body = {**json.loads(os.environ.get("OCTOPUS_SAMPLING_JSON") or "{}"), **body}
        template_kwargs = os.environ.get("OCTOPUS_CHAT_TEMPLATE_KWARGS", "").strip()
        if template_kwargs:  # e.g. {"enable_thinking": false} for a student served by SGLang
            body["chat_template_kwargs"] = json.loads(template_kwargs)
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OCTOPUS_API_KEY", "local-proxy")
        if key:
            headers["Authorization"] = "Bearer " + key
        events = []
        try:
            payload = post_chat(self._endpoint(), body, headers, log=events.append)
            # SGLang's own chat endpoint (an agent pointed at SGLang directly; its responses carry metadata) confirms a
            # mount; one without the chat mount ignored it silently. Its row is then not a 200 (an infra failure). The
            # AReaL proxy's responses carry no metadata: its engine fails a mount its server ignored.
            if mount and "metadata" in payload and "octopus_mount" not in (payload["metadata"] or {}):
                events[-1]["http_code"] = "unmounted"
                raise RuntimeError(f"{self._endpoint()} did not mount {mount}: its SGLang lacks the chat mount "
                                   "(patches/apply_sglang_mount.py), or the request was aborted")
            return payload
        finally:
            log_path = Path(self.logs_dir) / "requests.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # thread: main, a branch's session = its thread name (core.thread_name), or control (RL plan
            # episodes: critical-path tokens, and which thread each completion belongs to)
            # (a read control's: readctl/<its reader's thread name>, _read_controls)
            running = self._control_running
            thread = (running if isinstance(running, str) else "control") if running else session_id or "main"
            with log_path.open("a") as handle:
                for event in events:
                    handle.write(json.dumps({
                        "tool_choice": body["tool_choice"],
                        "tools": [item["function"]["name"] for item in self._tools()],
                        "thread": thread,
                        **({"mount": mount} if mount else {}),
                        **event,
                    }) + "\n")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.ws.environment is not environment:  # setup's, when Harbor ran it on this environment
            self.ws = self.provider(environment)
        await self.ws.begin()
        max_steps = call_limit()  # main's calls
        messages = [dict(m) for m in self._initial_messages or []] or [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": instruction},
        ]
        steps = [Step(step_id=1, source="user", message=instruction)]
        self._runtime = RuntimePort()
        self._budget = self.ws.budget(instruction)
        self._red_checks = []  # every bundle fork's red check (tree.json red_checks)
        main = self._thread("main", 0, self._runtime, self.ws.root("main"), None, None)  # the root of the fork tree
        self._main_thread, self._usage = main, {}  # OCTOPUS_READ_TOOL: every thread; each reply's usage
        self._mounted = {}  # OCTOPUS_READ_TOOL: (tokens, source replies held) a reply's request mounted, by reply id
        # and by (from, to, ended) (_mount_range)
        if read_event():  # Gate 0: every sibling event, mounted or not (_evented; tree.json read_event)
            self._event = []
        forks: list[dict] = main["forks"]
        self._live = 1  # threads running now (OCTOPUS_MAX_THREADS): main + every branch, at any depth, not yet closed
        self._branches = 0  # branches made so far, at every depth (OCTOPUS_MAX_BRANCHES)
        self._rewinds = []
        self._control_snap = None  # main's workspace at a main control's end (plan verifier input)
        self._control_patch = None  # a nested control's (task id, patch text) (plan verifier input)
        self._fork_calls = []  # every accepted fork call of the tree: the control's candidates
        own, next_id = {"payload": {}}, [2]  # the reply being executed; the next trajectory step id

        def step(**fields) -> None:
            steps.append(Step(step_id=next_id[0], **fields))
            next_id[0] += 1

        def call_step(name, args, call_id, observed, message) -> None:
            step(source="agent", model_name=self.model_name, message=message,
                 tool_calls=[ToolCall(tool_call_id=call_id, function_name=name, arguments=args)] if name else [],
                 observation=Observation(results=[ObservationResult(source_call_id=call_id, content=observed)])
                 if name else None)

        def log(kind: str, **f) -> None:  # the main thread's trajectory and its tree.json rewinds
            if kind == "call":
                call_step(f["name"], f["args"], f["call_id"], f["observed"], f["text"] or f["name"])
            elif kind == "rewind":
                self._rewinds.append({**f, "step": next_id[0]})
                step(source="user", message=f["note"])
            else:  # "end", "remind": a reply without a tool call (and the reminder after it)
                step(source="agent", model_name=self.model_name, message=f["text"] or "(no tool call)")
                if f.get("note"):
                    step(source="user", message=f["note"])

        root = main["start"] = {"branch_id": "main", "session_id": None, "task": "", "messages": messages}

        async def complete(history, session_id=None):
            own["payload"] = await self._complete_async(history, calls_left=root["calls_left"],
                                                        step=root.get("step"), mount=self._take_mount(main, history),
                                                        context_left=root.get("context_left"), start=root)
            return own["payload"]

        async def dispatch(name, args, _thread_id):
            try:
                observed, call_stop = await self._thread_tool(main, name, args, messages, own["payload"])
            except Exception as exc:
                observed, call_stop = f"{type(exc).__name__}: {exc}", False
            return observed, call_stop

        async def join(history):
            joined = await self._join(main, history, own["payload"])
            return joined and joined[0]

        def log_reports(reports) -> None:  # at each join of main's forks
            for report in reports:  # the branches' steps go into the main trajectory
                for item in report["steps"]:
                    name, args = item.get("name"), item.get("arguments")
                    call_id = str(item.get("tool_call_id") or f"{report['branch_id']}-{next_id[0]}")
                    call_step(name, args if isinstance(args, dict) else {}, call_id, str(item.get("observed") or ""),
                              name or report["branch_id"])
        main["on_join"] = log_reports

        try:
            if auto_fork():  # Gate 0: the environment's fork before main's first request, which starts from its join
                n = self.ws.auto_fork(self._runtime)
                self._branches += n
                self._live += n
                main["fork"] = True
                add_note(messages, await join(messages))
            # the root thread: the same loop as a branch, closed by done, capped in steps (branches have their own cap)
            report = await run_branch_continuation(
                root, complete, dispatch,
                max_steps=None, closing={"done"}, budget=thread_budget(), join=join,
                refused=lambda *_: False, max_calls=max_steps,  # main's commit/abort close nothing
                snapshot=lambda: self.ws.snapshot(main["root"]),
                restore=lambda snap: self.ws.restore(main["root"], snap),
                default_id=lambda _steps, index: f"call-{next_id[0]}-{index}", log=log,
                rewound=lambda end: self._rewound(main, end), ws=self.ws)
            self._main_end = main_ending(report["steps"], bool(self._budget) and self._spendable() <= 0)
            await self._maybe_control(messages, forks, max_steps)
            if read_control():
                await self._read_controls(forks)
            if plan_episode() and forks:
                await self._write_plan_inputs(forks)
        finally:
            for task in main["tasks"]:  # cancelled (Harbor's timeout): its branches stop, not run on in the shared loop
                task.cancel()
            self._write_trajectory(steps)
            self._write_tree(messages, forks)
            context.metadata = {"n_steps": len(steps)}

    def _system(self, linear: bool = False) -> str:
        """The system message: of a forking run, or (linear) of the no-fork control."""
        return self.ws.system(linear)

    def _spendable(self) -> int:
        """Tokens every thread may still draw from the shared budget."""
        return self._budget["limit"] - self._budget["used"]

    async def _complete_async(self, messages: list[dict], session_id: str | None = None, calls_left: int | None = None,
                              thread: str | None = None, step: int | None = None, mount: dict | None = None,
                              me: str | None = None, context_left: int | None = None, start: dict | None = None
                              ) -> dict:
        """One request of a thread (thread: a branch's thread name). In a bundle every new tool result is first tagged
        with the tokens the threads may still draw (_spendable) and, for a capped thread, its calls left
        (run_branch_continuation's start["calls_left"]). Under OCTOPUS_CURSOR only the messages after the history's last marked reply are tagged and sent, from its
        cursor, and the payload gets its own cursor unless the budget cut its reply (core.with_cursor).
        step (OCTOPUS_REWIND_TOOL): the thread's last step, whose results these are: "| step N" ends the budget tag, or
        without one each result starts with "[step N]". mount (OCTOPUS_READ_TOOL): the read this request carries.
        Under OCTOPUS_READ_TOOL the tag starts with the thread's name as read_thread takes it, "[main | ..." or
        "[branch-1 | ...": me (a nested control's branch id), else main or the last part of thread.
        context_left (thread_budget: run_branch_continuation's start["context_left"]): the thread's own growth
        left, "context left: N" in the tag (after the shared budget's tokens left, or leading it without one).
        start (OCTOPUS_READ_TOOL: the thread's run_branch_continuation start): start["mounted"], the mounted tokens its
        context holds after this reply (_mounts), which prompt_tokens leave out (settings.window_room)."""
        budget = self._budget
        full = messages
        cursor, messages = cursor_delta(messages) if cursor_mode() else (None, messages)
        context = None if context_left is None else f"context left: {context_left}"
        if budget:  # on the event loop: threads that run concurrently update it one at a time
            left = self._spendable()
            if left <= 0:
                self._pool_out()
                raise RuntimeError("token budget used up")
            head, prefix = f"{BUDGET_TAG}{left}" + ("" if context is None else f" | {context}"), BUDGET_TAG
        elif context is not None:
            head, prefix = f"[{context}", CONTEXT_TAG
        if budget or context is not None:
            tag = head + ("" if calls_left is None else f" | calls left: {calls_left}") + (
                "" if step is None else f" | step {step}") + "]"
        elif step is not None:
            tag, prefix = f"{STEP_TAG}{step}]", STEP_TAG
        if budget or context is not None or step is not None:
            named = read_tool()
            if named:
                tag = f"[{me or (thread or 'main').split('/')[-1]} | {tag[1:]}"
            for m in messages:  # each tool result once, before it is first sent: the history stays a fixed prefix
                content = str(m.get("content"))
                if m.get("role") == "tool" and not (THREAD_TAG.match(content) if named else content.startswith(prefix)):
                    m["content"] = f"{tag}\n{m['content']}"
        # Harbor runs all trials on one event loop; a blocking HTTP call here stalls every trial.
        payload = await asyncio.get_running_loop().run_in_executor(
            _HTTP_POOL, self._complete, messages, session_id, cursor, *([mount] if mount else []),
        )
        if read_tool():  # a read's size: the tokens between the turns it spans (_mount_range)
            self._usage[payload.get("id")] = payload.get("usage") or {}
            if mount:  # which prompt_tokens leave out: a later range holding this reply mounts them too (_mount_range)
                self._mounted[payload.get("id")] = self._mounted[mount["from"], mount["to"], "ended" in mount]
        served = dict((payload.get("choices") or [{}])[0].get("message") or {})  # before a budget cut rewrites it
        if budget:
            used = int((payload.get("usage") or {}).get("completion_tokens") or 0)
            budget["used"] += used
            self._pool_out()  # before the raise below
            if self._spendable() <= 0:
                payload["budget_out"] = True  # its thread's last reply: one without a tool call is not reminded (core)
                had = closing_only(payload)
                # a branch whose calls were all dropped is stopped by the budget, as at its next request (main's end
                # reads the budget itself, and main collects its forks at this reply); a text-only reply stopped itself
                if had and thread and not payload["choices"][0]["message"]["tool_calls"]:
                    raise RuntimeError("token budget used up")
        if cursor_mode() and served == payload["choices"][0]["message"]:  # its cursor holds what the history will
            payload["cursor"] = payload["id"]
        if start is not None and read_tool():
            start["mounted"] = self._mounts(full + [payload])
        return payload

    def _pool_out(self) -> None:
        """OCTOPUS_READ_CONTROL: the first time the episode's shared token pool is found spent: every running thread's
        deadline (read_deadline). Not a control's own budget."""
        if not read_control() or self._pool_out_t is not None or self._control_running:
            return
        if self._budget and self._spendable() <= 0:
            self._pool_out_t = time.time()

    def _tools(self):
        return self.ws.tools(self._runtime, self._control_linear)

    def _thread(self, name: str, depth: int, port, root: str, base: str | None, point: str | None) -> dict:
        """One thread of the fork tree: main is the root (depth 0, the episode's own workspace and runtime port), every
        branch a thread of the same kind with its own port and private workspace. base: the commit its descendants'
        patches are graded against (its depth-1 ancestor's fork point), point: the snapshot those patches are graded on.
        tasks: every branch task it started (cancelled when the thread ends: a cancelled thread's branches stop too);
        fork_call: its current reply called fork (only that first fork call may fork: fork_record reads it)."""
        return {"name": name, "depth": depth, "port": port, "root": root, "base": base, "point": point,
                "forks": [], "fork": False, "fork_call": False, "tasks": [], "started": []}

    async def _thread_tool(self, thread: dict, name: str, args: dict, messages: list, payload: dict):
        """One tool call of a thread at any depth. fork is refused at OCTOPUS_FORK_DEPTH and where its branches would
        start with too little of the serving window (FORK_WINDOW_REFUSAL); an accepted fork, within OCTOPUS_MAX_THREADS,
        is run by _join after the reply's tool calls."""
        if name == "fork" and thread["depth"] >= fork_depth():
            return FORK_DEPTH_REFUSAL, False
        if name == "fork" and window_room(self._context(thread, payload)) < answer_reserve() + FORK_FLOOR:
            return FORK_WINDOW_REFUSAL, False
        if name == "fork" and thread["fork_call"]:  # _join starts one fork per reply
            return ONE_FORK, False
        if name == "fork":
            thread["fork_call"] = True
        # a bundle's branches have closed themselves before main's next call (its fork waits for them)
        if not thread["depth"] and name in ("commit", "abort") and self.ws.main_close_refused:
            return MAIN_CLOSE.format(name), False
        if name == "read_thread" and read_tool():
            return self._read(thread, args, messages, payload), False
        if name == "fork" and (refusal := self.ws.fork_refusal(args)):
            return refusal, False
        port = thread["port"]
        before = port.last_fork
        observed, stop = await self._dispatch(name, args, root=thread["root"], port=port, thread=thread["name"])
        if self._event is not None and thread["depth"]:
            self._touch(thread, name, args, str(observed))
        if name == "fork" and port.last_fork and port.last_fork is not before:
            ids = [b["branch_id"] for b in port.last_fork["branches"]]
            if self._live + len(ids) > max_threads():
                port.last_fork = before
                return (f"fork refused: {self._live} threads are running and {len(ids)} more would pass the limit "
                        f"of {max_threads()}; fork fewer branches or finish this task yourself"), False
            cap = max_branches()
            if cap and self._branches + len(ids) > cap:
                port.last_fork = before
                left = cap - self._branches
                return (f"fork refused: {len(ids)} more branches would pass the episode's limit of {cap} branches; "
                        f"{left} left: " + ("fork fewer branches or finish this task yourself"
                                            if left >= self.ws.fork_min else
                                            "no more forks are possible, finish this task yourself")), False
            self._branches += len(ids)
            self._live += len(ids)  # counted now: a sibling's fork admitted before these start must see them
            thread["fork"] = True
            observed += self.ws.fork_note()
            self.ws.forked(port.last_fork, args)
        return observed, stop

    async def _join(self, thread: dict, messages: list, payload: dict):
        """After a reply's tool calls: if the thread forked, start its branches (_start_fork), wait for them all to
        close and return (join note, branch reports) (_collect)."""
        thread["fork_call"] = False
        thread["reads"] = 0  # read_thread calls of the reply (_read)
        if not thread["fork"]:
            return None
        thread["fork"] = False
        try:
            fork = await self._start_fork(thread, messages, payload)
        except BranchCopyError as exc:  # no branch started: the thread is told why and works on
            self._live -= len(thread["port"].last_fork["branches"])
            self._branches -= len(thread["port"].last_fork["branches"])  # OCTOPUS_MAX_BRANCHES: none was made
            return f"The fork failed, no branch started ({exc}). Fork again or do these tasks yourself.", []
        # main's first fork result, or its next one after a rewind removed that
        card = f"\n{self.ws.read_card}" if read_tool() and not thread["depth"] and not any(
            self.ws.read_card in str(m.get("content")) for m in messages) else ""
        note, reports = await self._collect(thread, fork, messages, payload)
        return note + card, reports

    async def _start_fork(self, thread: dict, messages: list, payload: dict) -> dict:
        """Start the branches of the fork the thread just made, one asyncio task each; returns the running fork. Its
        tree.json record takes its place in the thread's "forks" now (fork_time) and is filled when it is collected.

        Each branch starts from the thread's conversation so far plus its task, in a private copy of the thread's
        workspace (_workspace of its name), under the name core.thread_name gives its path (its session id
        and requests.jsonl thread; branch_key of it names its patch and plan.tsv / branches.json key)."""
        port, root, depth = thread["port"], thread["root"], thread["depth"]
        info = port.last_fork
        k = len(thread["forks"])
        names = {b["branch_id"]: thread_name(thread["name"], k, b["branch_id"]) for b in info["branches"]}
        workdirs = {bid: self.ws.root(name) for bid, name in names.items()}
        # RL plan episodes: the tree a depth-1 branch patch is graded on alone (and the no-fork control restores);
        # deeper branches keep their ancestor's
        point = (thread["point"] if depth else await self.ws.snapshot(root)
                 if plan_episode() or os.environ.get("OCTOPUS_PLAN_CONTROL") else None)
        # and a forking branch's own tree here: a no-fork control from this fork restores it (_maybe_control)
        own_point = await self.ws.snapshot(root) if depth and plan_episode() else None
        heads, red = await self.ws.open_branches(root, workdirs, info)
        if red:  # tree.json red_checks (and the fork's weak and red_note: _collect)
            self._red_checks.append({"thread": thread["name"], **red})
        # its branches may read their siblings and every thread `thread` may read (_others) but one a branch id of this
        # fork shadows (read_by_name off)
        offer = read_tool() and (not self.ws.read_when_readable or self._readable(thread, messages))
        others = [(k, t["start"]["task"]) for k, t in self._others(thread).items() if k not in names] if offer else []
        starts = branch_starts(port, messages, workdirs if self.ws.isolated else None, self.ws.branch_report,
                               self.ws.read_card if offer else None, self.ws.branch_close, self.ws.abort_reason,
                               self.ws.branch_rules, self.ws.branch_contract, offer,
                               names if self.ws.read_by_name else None, others)
        children = []
        for start in starts:
            bid = start["branch_id"]
            start["session_id"] = names[bid]  # the port names every fork's branches alike
            if read_tool():  # its own message dicts: a result its forking reply left untagged gets its name, not a
                start["messages"] = [dict(m) for m in start["messages"]]  # sibling's (THREAD_TAG)
            if depth + 1 >= fork_depth():  # fork is not offered here
                start["messages"][-1]["content"] += "\n" + FORK_DEPTH_REFUSAL
            children.append(self._thread(names[bid], depth + 1, RuntimePort(bid), workdirs[bid],
                                         thread["base"] or heads[workdirs[bid]], point))
            children[-1]["task_at"] = len(start["messages"]) - 1  # its task message
            if "on_join" in thread:  # main's trajectory takes every branch's steps, at any depth
                children[-1]["on_join"] = thread["on_join"]
            step = thread["start"].get("step")  # OCTOPUS_REWIND_TOOL: its history ends with the forking reply's step
            children[-1]["start"], start["step"] = start, None if step is None else step + 1
        if read_event():  # a branch's siblings: the children of every fork of this thread (_event_for)
            for c in children:
                c["parent"] = thread
        record = {"fork_time": time.time()}
        thread["forks"].append(record)
        # history: the thread's messages up to its forking reply's tool results, which every branch starts from
        # (fork_record, the control's history: a later OCTOPUS_REWIND may drop them from messages)
        fork = {"k": k, "info": info, "record": record, "history": list(messages), "point": point,
                "step": thread["start"].get("step"),  # OCTOPUS_REWIND_TOOL: the steps before the forking reply
                "calls_left": thread["start"].get("calls_left"),  # the thread's before it (rewound calls spent too)
                "rewinds_used": thread["start"].get("rewinds_used"),  # OCTOPUS_REWIND_MAX: its rewinds before it
                "reminded": thread["start"].get("reminded"),  # core.REMIND_NOTE: its reminders before it
                "own_point": own_point, "response_id": payload.get("id"), "ids": list(names),
                "closed": [],  # branch ids in the order they closed: a committed branch closes at its commit
                "children": {s["branch_id"]: c for s, c in zip(starts, children)},
                "merge_seq": len(vars(self).get("_merge_log", {}).get(root, []))}  # results merged into root before it

        async def run(start, child):
            began = child["began"] = time.time()
            report, fields = await self._branch(start, child, thread["base"])
            fork["closed"].append(start["branch_id"])
            return report, dict(fields, start_time=began, end_time=time.time())
        fork["tasks"] = {s["branch_id"]: asyncio.ensure_future(run(s, c)) for s, c in zip(starts, children)}
        thread["started"].append(fork)
        thread["tasks"] += fork["tasks"].values()
        return fork

    async def _collect(self, thread: dict, fork: dict, messages: list, payload: dict) -> tuple[str, list]:
        """Wait for every branch of the thread's fork, merge their committed patches (OCTOPUS_AUTO_MERGE) and return
        (the join note, their reports). The fork's record is filled then (join_index: the thread's message then,
        join_time, join_response_id: the forking reply; each branch's join_time / join_response_id alike)."""
        results = [await fork["tasks"][b] for b in fork["ids"]]
        now = time.time()
        for _, fields in results:
            fields.update(join_time=now, join_response_id=payload.get("id"))
        fork["results"] = dict(zip(fork["ids"], results))
        reports = [report for report, _ in results]
        patches = {b: (f["patch"], f["diffstat"]) for b, (_, f) in fork["results"].items()} if self.ws.isolated else None
        merged = None
        if auto_merge():
            merged = await self._merge_committed(thread["root"], [
                {"branch_id": b, "ended_with": closing_name(r["steps"]), "patch": f["patch"], "diffstat": f["diffstat"]}
                for b, (r, f) in fork["results"].items()], fork["closed"], fork["merge_seq"],
                fork.setdefault("conflict_by", {}))
            # where in the thread's history each merge happened (OCTOPUS_REWIND_TOOL: a rewind before it undoes it)
            fork["merged_at"] = {b: len(messages) for b, how in merged.items() if how.startswith("merged")}
        info, record = fork["info"], fork["record"]
        note = self.ws.join_note(reports, patches, merged, not thread["depth"]) + info.get("red_note", "")
        if "on_join" in thread:
            thread["on_join"](reports)
        record.update(fork_record(fork["history"], reports, self.ws.fork_tasks),
                      fork_point=fork["point"], response_id=fork["response_id"], join_index=len(messages),
                      join_time=now, join_response_id=payload.get("id"), weak=info.get("weak") or [])
        if fork["own_point"]:
            record["workdir_point"] = fork["own_point"]
        for branch, own, (_, fields) in zip(record["branches"], info["branches"], results):
            branch.update(fields, merge=(merged or {}).get(branch["branch_id"]))
            if branch["task_id"] is None:  # no forking reply names it (OCTOPUS_AUTO_FORK): the environment's
                branch["task_id"] = own.get("task_id")
            if branch["branch_id"] in fork.get("conflict_by", {}):  # OCTOPUS_CONFLICT_BY
                branch["conflict_by"] = fork["conflict_by"][branch["branch_id"]]
        path = f"fork{fork['k']}" if thread["depth"] == 0 else f"{thread['name']}/fork{fork['k']}"
        self._fork_calls.append({"thread": thread, "path": path, "messages": fork["history"], "record": record,
                                 "step": fork["step"], "calls_left": fork["calls_left"],
                                 "rewinds_used": fork["rewinds_used"], "reminded": fork["reminded"]})
        return note, reports

    async def _merge_committed(self, root: str, branches: list[dict], closed: list[str], since: int = 0,
                               by: dict | None = None) -> dict:
        """OCTOPUS_AUTO_MERGE: the forking thread's workspace (root) takes each committed branch result that changed
        something, in commit order (WorkspaceProvider.apply). Returns {branch_id: "merged" | "conflict: ..."}. since:
        how many results were merged into root before this fork (_merge_log); OCTOPUS_CONFLICT_BY labels each file of a
        conflicting one into by (WorkspaceProvider.conflict_by)."""
        order = sorted(branches, key=lambda b: closed.index(b["branch_id"]))
        ready = [(b["branch_id"], b["patch"]) for b in order if b["diffstat"] and b["ended_with"] == "commit"]
        applied = await self.ws.apply(root, ready)
        log = vars(self).setdefault("_merge_log", {}).setdefault(root, [])  # every result merged into root, in order
        for bid, ref in ready:
            if applied[bid].startswith("merged"):
                log.append(ref)
            elif conflict_by():
                by[bid] = await self.ws.conflict_by(ref, [f for f in applied[bid][len("conflict: "):].split(", ") if f],
                                                    log[since:])
        return applied

    async def _branch(self, start: dict, thread: dict, graded_base: str | None) -> tuple:
        """A branch thread until it closes; returns its report and its record fields: its patch since its fork point
        (for the join), its patch since graded_base (graded: a deeper branch's includes its ancestors' changes up to
        its fork), and its own forks."""
        own = {"payload": {}}
        name = thread["name"]

        async def complete(messages, session_id=None):
            if self.ws.read_when_readable and read_tool() and not self._offered(thread, messages) and self._readable(
                    thread, messages):  # read_thread is offered from this request on
                roster = [(k, task_line(t["start"]["task"])) for k, t in self._others(thread).items()]
                add_note(messages, read_offer(roster, self.ws.read_card, self.ws.read_by_name, now=True))
            mount = self._take_mount(thread, messages)
            # OCTOPUS_READ_CONTROL: the state before the request carrying its own first read (_read_controls); an
            # evented mount (OCTOPUS_READ_EVENT, below) never starts a read control
            cap = read_control() and mount and "read_ctl" not in thread and self._read_state(thread, own["payload"])
            if self._event is not None:
                mount = self._evented(thread, messages, mount)
            own["payload"] = await self._complete_async(messages, session_id, start["calls_left"], name,
                                                        start.get("step"), mount,
                                                        context_left=start.get("context_left"), start=start)
            if cap:  # its history as sent (tags added), and its tree: unchanged while it waits for its own reply
                sent, i = [dict(m) for m in messages], max(j for j, m in enumerate(messages) if m.get("role") == "assistant")
                j = [i + 1 + n for n, c in enumerate(messages[i].get("tool_calls") or [])  # its tool results follow it
                     if (c.get("function") or {}).get("name") == "read_thread"][cap["call"]]
                # the mounting read's result without its step listing (in the request's delta: the cursor is at most
                # the read reply), so the read call is credited with listing and mount
                sent[j]["content"] = without_step_listing(sent[j]["content"])
                thread["read_ctl"] = dict(cap, step=start.get("step"), rewinds_used=start.get("rewinds_used"),
                                          reminded=start.get("reminded"),
                                          messages=sent,
                                          snap=await self.ws.snapshot(thread["root"]),
                                          grown=int((own["payload"].get("usage") or {}).get("prompt_tokens") or 0)
                                          - cap["first_prompt"])
            return own["payload"]

        async def join(messages):
            joined = await self._join(thread, messages, own["payload"])
            return joined and joined[0]

        def before_turn(turns: int) -> None:
            thread["turns"] = turns  # replies so far (read records)
        try:
            # branch_turns(): the branch's turns, the "calls left" of its tool-result tags
            report = await run_branch_continuation(
                start, complete,
                lambda name, args, _bid: self._thread_tool(thread, name, args, start["messages"], own["payload"]),
                branch_turns(), closing=self.ws.branch_closing, budget=thread_budget(), join=join,
                before_turn=before_turn,
                snapshot=lambda: self.ws.snapshot(thread["root"]),
                restore=lambda snap: self.ws.restore(thread["root"], snap),
                rewound=lambda end: self._rewound(thread, end), ws=self.ws)
        finally:
            for task in thread["tasks"]:
                task.cancel()
            self._live -= 1
            thread["ended"] = True  # OCTOPUS_READ_EVENT: no more events for it
        patch, stat, text = await self.ws.capture(thread["root"], graded_base)
        if self._event is not None and closing_name(report["steps"]) == "commit":
            self._event_for(thread, "commit", await self.ws.changed(thread["root"]))  # its own result's files
        return report, {"workdir": thread["root"], "patch": patch, "diffstat": stat, "patch_text": text,
                        "forks": thread["forks"]}

    def _tree(self, reader: dict) -> tuple[dict, set]:
        """For _read: every thread of the episode by its read_thread name (read_by_name: its thread name; else its
        branch id, a later fork's shadowing an earlier one's), and the ids of the reader's forking threads (main and
        its forking ancestors), which wait for its fork (branch_starts): none of their steps can be read while it
        runs."""
        key = (lambda t: t["name"]) if self.ws.read_by_name else (lambda t: t["name"].split("/")[-1])  # noqa: E731
        threads, todo, parent = {}, [self._main_thread], {}
        while todo:  # the fork tree; by branch id, an id that several forks gave names the latest
            t = todo.pop(0)
            threads[key(t)] = t
            kids = [c for f in t["started"] for c in f["children"].values()]
            parent.update({id(c): t for c in kids})
            todo += kids
        waiting, t = set(), reader
        while (t := parent.get(id(t))) is not None:
            waiting.add(id(t))
        return threads, waiting

    def _others(self, reader: dict) -> dict:
        """The threads the reader may read (_read's rule), by read_thread name: all but itself and those waiting for its
        fork (_tree). A forking thread's are its new branches' but for their siblings, which start with them."""
        threads, waiting = self._tree(reader)
        return {k: t for k, t in threads.items() if t is not reader and id(t) not in waiting}

    def _readable(self, reader: dict, messages: list) -> bool:
        """WorkspaceProvider.read_when_readable: whether one of the reader's others (_others) has a readable step
        (_steps) that its history (messages) does not hold; a branch it forks now could read that step too."""
        own = self._own(messages)
        return any(first <= last for *_, first, last in (self._steps(t, own) for t in self._others(reader).values()))

    def _offered(self, thread: dict, messages: list) -> bool:
        """Whether a branch's history holds read_thread's card from its task message on (read_when_readable: else its
        read_thread is refused); a card it inherited (main's first join note) offered it nothing."""
        card = self.ws.read_card
        return any(card in str(m.get("content")) for m in messages[thread.get("task_at", 0):])

    def _mounts(self, messages: list) -> int:
        """The mounted tokens a thread's context holds: what the requests of the replies in its history mounted
        (_mount_range's count, transitive reads included), which prompt_tokens leave out."""
        return sum(self._mounted[m["cursor"]][0] for m in messages if m.get("cursor") in self._mounted)

    @staticmethod
    def _context(thread: dict, payload: dict) -> int:
        """A thread's context through its reply `payload` (settings.window_room): its prompt and completion, and its
        mounts (start["mounted"]: its history's and its pending read's)."""
        usage = payload.get("usage") or {}
        return (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
                + thread["start"].get("mounted", 0))

    def _rewound(self, thread: dict, end: int) -> str:
        """OCTOPUS_REWIND_TOOL: what a rewind of the thread to history length `end` did to its forks: the merges past
        that point it undid with the workspace (a join at `end` itself stays: the step that holds its note ends after
        it), their tree.json merge then "undone: ...". OCTOPUS_READ_TOOL: a read that the removed reply mounted goes
        with it (its mount is not sent); a read whose step `end` ends, whose mount rode on a request the rewind
        removed, is mounted again with the next request (_take_mount); a branch's read control is dropped when the
        rewind lands at or before its read (its history from the mount request on is gone), and no later read gets
        one; a later one's removed messages still count for its forked_after. OCTOPUS_READ_EVENT: the siblings' events of the thread's steps past `end` are dropped (those step numbers
        name other replies from then on)."""
        thread.pop("mount", None)
        thread.pop("read_call", None)
        sent = thread["sent_mounts"] = {k: m for k, m in thread.get("sent_mounts", {}).items() if k <= end}
        if end in sent:
            thread["mount"] = sent[end]
        if thread.get("read_ctl") and end <= len(thread["read_ctl"]["messages"]):
            thread["read_ctl"] = None
        elif thread.get("read_ctl"):  # what it removed after the read still counts (forked_after)
            thread["read_ctl"].setdefault("rewound", []).extend(thread["start"]["messages"][end:])
        if self._event is not None and "parent" in thread:
            base, marks = thread["start"]["marks"]
            to = base + sum(m["end"] <= end for m in marks) - 1
            for b in (c for f in thread["parent"]["started"] for c in f["children"].values() if c.get("events")):
                b["events"] = [e for e in b["events"] if e["source"] is not thread or e["step"] == "commit"
                               or e["step"] <= to]
        undone = []
        for f in thread["started"]:
            for b in [b for b in f["ids"] if f.get("merged_at", {}).get(b, -1) > end]:
                del f["merged_at"][b]
                record = next(x for x in f["record"]["branches"] if x["branch_id"] == b)
                record["merge"] = f"undone: {record['merge']}"
                undone.append(f"{b} ({f['results'][b][1]['patch']})")
        return self.ws.remerge_note.format(", ".join(undone)) if undone else ""

    @staticmethod
    def _take_mount(thread: dict, history: list) -> dict | None:
        """The read mount (OCTOPUS_READ_TOOL) the thread's next request carries, or None; kept by the history length it
        is sent at, for a rewind back to there (_rewound)."""
        mount = thread.pop("mount", None)
        if mount:
            thread.setdefault("sent_mounts", {})[len(history)] = mount
        return mount

    @staticmethod
    def _steps(t: dict, own: set) -> tuple:
        """For _read: a thread t's (the step before its first, its step marks, its history, held: its readable steps
        whose reply the reader holds (own, _own: the source replies it mounted; never mounted twice), its first
        readable step: the first neither held nor rewritten (a rewritten reply has no cursor), its last readable step:
        the last whose next reply is back, or its last step once it has ended, read through its reply alone
        (_mount_range)); no readable step when first > last, e.g. before its loop starts."""
        base, marks = t.get("start", {}).get("marks") or (0, [])
        msgs = t["start"]["messages"] if marks else []

        def reply(k: int) -> int | None:  # the history index of step k's reply
            return next((j for j in range(marks[k - 1 - base]["end"], len(msgs)) if msgs[j].get("role") == "assistant"),
                        None)
        last = base + len(marks) - 1
        while last > base and not t.get("ended") and reply(last + 1) is None:
            last -= 1
        held = {k for k in range(base + 1, last + 1) if msgs[reply(k)].get("cursor") in own}
        first = base + 1
        while first <= last and (first in held or msgs[reply(first)].get("cursor") is None):
            first += 1
        return base, marks, msgs, held, first, last

    @staticmethod
    def _unheld(held: set, lo: int, hi: int) -> list:
        """For _read and _evented: the ranges of steps lo..hi to mount, in order: from each step not held, up to the
        step before the next held one (a step is never mounted twice)."""
        return [(k, min([h - 1 for h in held if k < h <= hi] + [hi])) for k in range(lo, hi + 1) if k not in held]

    def _read(self, thread: dict, args: dict, messages: list, payload: dict) -> str:
        """OCTOPUS_READ_TOOL: read_thread(thread, from_step, to_step) of another thread of the episode (both steps omitted:
        its latest readable steps, at most 16; else the range's readable part, an omitted end open), listed one line per
        step in the result (_step_line). Its steps a..b run from its reply of step a to its reply after step b
        (docs/cross-branch-read.md); both must be kept as the server sent them (with cursors). The reader's next request
        carries their response ids as its mount, which the proxy turns into a token range. Refused, with nothing
        mounted: a control, a reply that also forks or rewinds, an unknown thread, the reader itself, a thread that waits
        for the reader's fork (its forking threads; not listed), a range with no
        readable step (steps that are not the source's own or whose next reply is not back, the reader holds already
        or whose reply was rewritten), a second read in the same reply. A range may hold a read of the source's own:
        the reader then also attends to what that read mounted. Refusals name the other threads and their readable
        steps. thread["read_call"]: which read_thread call of its reply mounted
        (OCTOPUS_READ_CONTROL's credit goes on that call)."""
        k = thread["reads"] = thread.get("reads", 0) + 1
        if self._control_running:
            return "read_thread is not available to the control. Nothing was read."
        if self.ws.read_when_readable and thread["depth"] and not self._offered(thread, messages):  # none offered yet
            return "read_thread is not available in this branch. Nothing was read."
        name, lo, hi = args.get("thread"), step_number(args.get("from_step")), step_number(args.get("to_step"))
        given = args.get("from_step") is not None, args.get("to_step") is not None
        latest = not any(given)  # the thread's latest readable steps
        code = "view" in self.ws.tool_names  # the code environment (its read card names rewind, on or off); search
        if (not isinstance(name, str) or given[0] and lo is None or given[1] and hi is None
                or all(given) and lo > hi):
            return ("read_thread needs thread (another thread: "
                    + ("its name, such as fork0/branch-1" if self.ws.read_by_name else "a branch id")
                    + ") and, if given, the integers from_step <= to_step" + (" (for files use view)" if code else "")
                    + ". Nothing was read.")
        if thread.get("mount"):
            return ("One read_thread per reply: the first one of this reply is mounted with your next reply. Nothing "
                    "else was read.")
        calls = {c.get("function", {}).get("name")
                 for c in reply_tools((payload.get("choices") or [{}])[0].get("message") or {})[0]}
        if calls & {"fork", "rewind"}:  # the forked branches or the rewound history would not hold this read
            return ("read_thread cannot share a reply with fork" + (" or rewind" if code or rewind_tool() else "")
                    + ": call it in a reply of its own. Nothing was read.")
        threads, waiting = self._tree(thread)  # its forking threads are left out of the roster
        me = thread["name"] if self.ws.read_by_name else thread["name"].split("/")[-1]
        own = self._own(messages + [payload])  # the reply being run: its mount is held too
        spans = {k: self._steps(t, own) for k, t in threads.items() if t is not thread and id(t) not in waiting}
        yet = lambda t: "" if t.get("ended") else " yet"  # noqa: E731  an ended thread gets no more steps
        others = ("Other threads: " + ", ".join(f"{k} (readable steps {a} to {z})" if a <= z else
                                               f"{k} (no readable step{yet(threads[k])})"
                                               for k, (*_, a, z) in spans.items())
                  + "." if spans else "There is no other readable thread." if waiting
                  else "There is no other thread yet: fork first.")
        name = name.strip()
        src = threads.get(name)
        if src is None and self.ws.read_by_name:  # a bare branch id: the one thread that has it
            hits = [n for n in threads if n.split("/")[-1] == name]
            if len(hits) > 1:
                return f"{name} is ambiguous: {', '.join(hits)} have it; name one of them. Nothing was read."
            src, name = (threads[hits[0]], hits[0]) if hits else (None, name)
        if src is thread:
            return f"You are {me}: read_thread reads another thread, never your own. {others} Nothing was read."
        if src is None:
            return f"No thread {name} in this episode. {others} Nothing was read."
        if id(src) in waiting:
            return f"{name} waits for your fork to close: none of its steps can be read. {others} Nothing was read."
        base, marks, msgs, held, start, last = spans[name]
        readable = f"readable steps {start} to {last}" if start <= last else f"no readable step{yet(src)}"
        span = lambda s: msgs[marks[s - 1 - base]["end"]:marks[s - base]["end"]]  # noqa: E731  step s's reply and results
        # ponytail: fixed 16-step cap; size it in tokens if long steps crowd the reader's context
        want = (max(base + 1, last - 15), last) if latest else (base + 1 if lo is None else lo, last if hi is None else hi)
        a, b = max(want[0], base + 1), min(want[1], last)  # its readable part; an omitted end: the first or last step
        if a > b:  # none of it readable
            return (f"{name} has {readable}" + (": it has ended and gets no more" if src.get("ended") else
                                                f" (a step is readable once {name}'s next reply is back)")
                    + ". Nothing was read.")
        ranges = self._unheld(held, a, b)  # the first that mounts: from the first step not held, cut before a held one
        if not ranges:
            return f"{name}'s step {a} is in your own history already ({name} has {readable}). Nothing was read."
        got = next(((r, g) for r in ranges if not isinstance(g := self._mount_range(name, base, marks, msgs, *r), str)),
                   None)
        if got is None:  # rewritten replies: before its first readable step, or one at an edge
            x = ranges[0][0]
            return (f"{name}'s step {x} cannot be mounted: its reply was rewritten ({name} has {readable}). "
                    "Nothing was read.") if x < start else self._mount_range(name, base, marks, msgs, *ranges[0])
        (lo, hi), (mount, n) = got
        if window_room(self._context(thread, payload) + n) < answer_reserve():  # its next request would pass the window
            return (f"{name}'s steps {lo} to {hi} hold {n} tokens, more than your context window has left: read fewer "
                    "steps. Nothing was read.")
        thread["mount"], thread["read_call"] = mount, k - 1  # OCTOPUS_READ_CONTROL's credit goes on this call
        thread["start"]["mounted"] = thread["start"].get("mounted", 0) + n  # its next request carries it
        return (f"Mounted {name}'s steps {lo} to {hi} ({n} tokens; {name} has {readable}): "
                "from your next reply on you attend to them; their full text is not copied here."
                + "".join(f"\nstep {s}: {self._step_line(span(s))}" for s in range(lo, hi + 1)))

    @staticmethod
    def _step_line(span: list) -> str:
        """For _read: one step's calls as tool(key arg) -> the first line of its result (after the result's tag)."""
        calls = [decode_tool_call(c) for m in span if m.get("role") == "assistant" for c in reply_tools(m)[0]]
        results = [c.split("\n", 1)[-1] if THREAD_TAG.match(c := str(m.get("content"))) else c
                   for m in span if m.get("role") == "tool"]
        key = lambda a: str(next((a[k] for k in ("path", "command") if k in a), next(iter(a.values()), "")))  # noqa: E731
        return "; ".join(f"{n}({key(a).split(chr(10), 1)[0][:60]}) -> {r.split(chr(10), 1)[0][:60]}" for (n, a, _), r in zip(calls, results))

    def _mount_range(self, name: str, base: int, marks: list, msgs: list, lo: int, hi: int):
        """Readable steps lo..hi of thread `name` (_steps' base, marks, history) as (mount, its tokens): from step lo's
        reply to the reply after step hi, or, hi being the last step of an ended thread (no reply after it), through
        step hi's reply ("ended", delta_render.read_range); or _read's refusal when a reply at their edge was
        rewritten."""
        def reply(k: int) -> int | None:  # the first reply at or after history index k
            return next((j for j in range(k, len(msgs)) if msgs[j].get("role") == "assistant"), None)
        i, j = reply(marks[lo - 1 - base]["end"]), reply(marks[hi - base]["end"])
        ended = j is None
        if ended:
            j = reply(marks[hi - 1 - base]["end"])
        first, after = msgs[i].get("cursor"), msgs[j].get("cursor")
        if first is None or after is None:
            return (f"{name}'s steps {lo} to {hi} cannot be mounted: a reply at their edge was rewritten. Nothing was "
                    "read.")
        # the source's prompt tokens between them (ended: and the last reply's output; "from"'s generation prompt, a
        # few tokens, is not counted then), and what its replies in (first, after] mounted: their requests'
        # placeholders sit inside the range (a transitive read), and prompt_tokens leave them out
        use, to = self._usage.get(first, {}), self._usage.get(after, {})
        inner = [self._mounted.get(m.get("cursor"), (0, set())) for m in msgs[i + 1:j + 1]
                 if m.get("role") == "assistant"]
        n = (to.get("prompt_tokens") or 0) - (use.get("prompt_tokens") or 0) + sum(k for k, _ in inner)
        n += (to.get("completion_tokens") or 0) if ended else 0
        # the replies whose steps it holds (steps lo..hi), and what those mounted: a reader holding them reads them
        # never again (_own)
        held = {m["cursor"] for m in msgs[i:j + ended] if m.get("role") == "assistant" and m.get("cursor")}
        self._mounted[first, after, ended] = n, held.union(*(c for _, c in inner))
        return {"thread": name, "from": first, "to": after, **({"ended": True} if ended else {})}, n

    def _own(self, messages: list) -> set:
        """For _steps: the source replies a reader holds: the ones the replies in its history (messages) mounted
        (_mount_range), transitively."""
        # ponytail: a reader mounting a source S directly holds S's steps, but a later mount of T's reply that itself
        # mounted those S steps still brings them again (stacked twice through a third thread); a held check on each
        # mount's transitive content (self._mounted) closes it if it shows up
        return set().union(*(self._mounted[m["cursor"]][1] for m in messages if m.get("cursor") in self._mounted))

    def _touch(self, thread: dict, name: str, args: dict, observed: str) -> None:
        """OCTOPUS_READ_EVENT: a file a branch's call looked at or changed (WorkspaceProvider.touched) goes into its
        touched set; a change is also an event of its current step (_event_for)."""
        got = self.ws.touched(thread["root"], name, args, observed)
        if got is None:
            return
        f, changed = got
        thread.setdefault("touched", set()).add(f)
        if changed:
            base, marks = thread["start"]["marks"]
            self._event_for(thread, base + len(marks), {f})  # the step of the reply being run

    def _event_for(self, src: dict, step, files: set) -> None:
        """OCTOPUS_READ_EVENT: src's step (a number, or "commit" with its own result's files) changed these files:
        queued for each sibling still running (a child of any fork of src's forking thread) that looked at or changed
        one of them before, once per (sibling, step); _evented mounts it."""
        for b in (c for f in src["parent"]["started"] for c in f["children"].values()):
            hit = files & b.get("touched", set())
            if b is src or b.get("ended") or not hit:
                continue
            ev = next((e for e in b.setdefault("events", []) if e["source"] is src and e["step"] == step), None)
            if ev:
                ev["files"] = sorted(hit | set(ev["files"]))
            else:
                b["events"].append({"source": src, "step": step, "files": sorted(hit)})

    def _evented(self, thread: dict, messages: list, mount: dict | None) -> dict | None:
        """OCTOPUS_READ_EVENT (Gate 0, docs/cross-branch-read.md): the branch's oldest queued event (_event_for) whose
        step is readable now (_steps) is mounted with this request as if read_thread had been called: no call, no text.
        A changing step mounts that step alone; a commit its source's latest readable steps, at most 16, through the
        commit reply (the source has ended: _steps), later where a rewritten edge sits in between. A step the branch
        holds already is never mounted again (_unheld): a held step's event is "no mountable step"; a commit mounts
        the first run of steps not held and stays queued, one mount per request, until its steps are all held. A read
        of the branch's own pending keeps its mount; the events wait. Each mount is recorded (self._event: reader,
        source, step, files, turn = the reader's replies before, and steps / tokens / from / to when mounted, else
        why not)."""
        if mount is not None:
            return mount
        own = self._own(messages)
        for ev in list(thread.get("events") or []):
            src = ev["source"]
            name = src["name"].split("/")[-1]
            base, marks, msgs, held, _, last = self._steps(src, own)
            if ev["step"] != "commit" and ev["step"] > last:
                continue  # readable once its next reply is back
            a, b = (max(base + 1, last - 15), last) if ev["step"] == "commit" else (ev["step"], ev["step"])
            ranges = self._unheld(held, a, b)  # a step the branch holds already: none
            got = next(((lo, hi, g) for lo, hi in ranges
                        if not isinstance(g := self._mount_range(name, base, marks, msgs, lo, hi), str)), None)
            # one mount per request: a commit whose steps are not all held after this one stays queued
            if got is None or not any(k not in held and not got[0] <= k <= got[1] for k in range(a, b + 1)):
                thread["events"].remove(ev)
            rec = {"reader": thread["name"], "source": src["name"], "step": ev["step"], "files": ev["files"],
                   "turn": thread.get("turns", 0)}
            if got is None:
                self._event.append(dict(rec, mounted=False, why="no mountable step"))
                continue
            lo, hi, (m, n) = got
            self._event.append(dict(rec, mounted=True, steps=[lo, hi], tokens=n, **{"from": m["from"], "to": m["to"]}))
            return m
        return mount

    def _events_left(self) -> list:
        """OCTOPUS_READ_EVENT: the events no request carried (their reader ended first or the step never became
        readable), as records with mounted false."""
        out, todo = [], [self._main_thread]
        while todo:
            t = todo.pop(0)
            todo += [c for f in t["started"] for c in f["children"].values()]
            out += [{"reader": t["name"], "source": e["source"]["name"], "step": e["step"], "files": e["files"],
                     "mounted": False, "why": "pending at end"} for e in t.get("events") or []]
        return out

    async def _maybe_control(self, messages: list, forks: list[dict], max_steps: int | None) -> None:
        """After a forking plan episode: one no-fork continuation from one accepted fork call, chosen uniformly at random
        (seeded by the trial's name) among all of the tree's where the provider allows it (any_fork_control, a bundle:
        main's first, main's later ones, a branch's at any depth), else main's first. It starts where the forking
        thread stood just before its forking reply: its tree, its history, the calls (main, under
        OCTOPUS_MAIN_MAX_CALLS) or turns (a branch, under OCTOPUS_BRANCH_MAX_TURNS) it had left, the context growth (thread_budget) it had
        left and, with a shared token budget (search's OCTOPUS_SEARCH_BUDGET), what was left of it then (every
        thread's tokens before that reply; budget_before: the tokens it had counted). tree.json control:
        {kind (main-first / main-later / nested), path, depth, response_id, calls_before, budget_before, end,
        budget_used}.

        main's fork: main's workspace back to that fork point (earlier joins' merges in it), the linear system message
        and tools, closed by done; its tree is graded on every task (control.snap). A branch's fork: the branch's
        workspace back to its tree there, fork refused as at OCTOPUS_FORK_DEPTH, closed by commit/abort like the branch; its patch
        is graded on the branch's task (plan.tsv row "control")."""
        if not os.environ.get("OCTOPUS_PLAN_CONTROL") or not forks:
            return
        pool = sorted(self._fork_calls, key=lambda c: c["path"])  # join order varies between runs; paths do not
        if not self.ws.any_fork_control:  # single-task plan episodes keep main's first fork
            pool = [c for c in pool if c["path"] == "fork0"]
        call = random.Random(Path(self.logs_dir).parent.name).choice(pool)
        thread, record = call["thread"], call["record"]
        depth, idx = thread["depth"], record.get("fork_index")
        state = record.get("workdir_point") if depth else record.get("fork_point")
        if not state or not isinstance(idx, int) or idx < 1:
            return
        rows = [json.loads(x) for x in (Path(self.logs_dir) / "requests.jsonl").read_text().splitlines()]
        before = list(itertools.takewhile(lambda r: r.get("id") != record["response_id"], rows))
        ctrl = [dict(m) for m in call["messages"][:idx]]
        main_budget, self._budget = self._budget, self._control_budget(before)
        self._control = {"kind": "nested" if depth else "main-first" if call["path"] == "fork0" else "main-later",
                         "path": call["path"], "depth": depth, "response_id": record["response_id"],
                         "budget_before": self._budget and self._budget["used"]}
        # OCTOPUS_REWIND_TOOL: the steps before its fork, and the rewinds its thread ran (OCTOPUS_REWIND_MAX)
        own, start = {"payload": {}}, {"step": call["step"], "rewinds_used": call["rewinds_used"],
                                       "reminded": call["reminded"]}

        async def complete(history, *_):
            own["payload"] = await self._complete_async(history, calls_left=start["calls_left"],
                                                        step=start.get("step"),  # main's: tagged main
                                                        context_left=start.get("context_left"), start=start)
            return own["payload"]

        async def dispatch(name, args, _thread_id):
            try:
                observed, stop = await self._dispatch(name, args)
            except Exception as exc:
                observed, stop = f"{type(exc).__name__}: {exc}", False
            return observed, stop
        self._control_running, began = True, time.time()
        try:
            if depth == 0:
                root = self._main_thread["root"]
                final = await self.ws.snapshot(root)
                await self.ws.restore(root, state)
                ctrl[0] = dict(ctrl[0], content=self._system(linear=True))  # the system message of a run without fork
                card = "\n" + self.ws.read_card if read_tool() else None
                for m in ctrl:  # and other tools: its first request is a full one (OCTOPUS_CURSOR), without the
                    m.pop("cursor", None)  # read_thread definition of main's first fork result (it cannot read)
                    if card and isinstance(m.get("content"), str):
                        m["content"] = m["content"].replace(card, "")
                # main's calls before the forking reply, those a rewind removed from its history too
                calls = (sum(m.get("role") == "tool" for m in ctrl) if max_steps is None
                         else max_steps - call["calls_left"])
                # thread_budget: what main had left of it at the forking reply, as its calls
                mains = [r for r in before if r.get("thread") == "main" and r.get("id")]
                grown = int(rows[len(before)].get("prompt_tokens") or 0) - int(
                    (mains or rows[len(before):])[0].get("prompt_tokens") or 0)
                growth = max(1, thread_budget() - grown)
                self._control_linear = True
                start.update(branch_id="control", session_id=None, task="", messages=ctrl)
                report = await run_branch_continuation(
                    start, complete, dispatch,
                    max_steps=None, closing={"done"}, budget=growth,
                    refused=lambda *_: False,  # as main's
                    max_calls=None if max_steps is None else max_steps - calls,
                    snapshot=lambda: self.ws.snapshot(root), restore=lambda snap: self.ws.restore(root, snap),
                    ws=self.ws)
                self._control_linear = False
                self._control.update(calls_before=calls,
                                     end=main_ending(report["steps"], bool(self._budget) and self._spendable() <= 0))
                self._control_snap = await self.ws.snapshot(root)
                await self.ws.restore(root, final)
            else:
                branch = next(b for name, _, b in tree_branches(forks) if name == thread["name"])
                at = thread["task_at"]  # its task message, told what a branch at the depth limit is told
                ctrl[at] = dict(ctrl[at], content=ctrl[at]["content"] + "\n" + FORK_DEPTH_REFUSAL)
                for m in ctrl[at:]:  # the branch's replies after it followed the old text: resent (OCTOPUS_CURSOR)
                    m.pop("cursor", None)
                turns = [r for r in before if r.get("thread") == thread["name"] and r.get("id")]
                # the branch's prompt growth so far counts against its thread_budget(), as its turns against its cap
                grown = int(rows[len(before)].get("prompt_tokens") or 0) - int((turns or rows[len(before):])[0].get(
                    "prompt_tokens") or 0)
                report, text = await self._rerun_branch(thread, branch, state, ctrl, start, len(turns),
                                                        grown)
                self._control_patch = (branch.get("task_id"), text)
                self._control.update(calls_before=len(turns), end=closing_name(report["steps"]))
            self._control["budget_used"] = self._budget and self._budget["used"]
            self._control["reminded"] = report["reminded"]  # core.REMIND_NOTEs in its history
            self._control["wall_s"] = round(time.time() - began, 3)  # its own run's wall (dev_eval's row totals)
        finally:
            self._control_running, self._control_linear, self._budget = False, False, main_budget

    def _control_budget(self, before: list, used: int | None = None) -> dict | None:
        """A control's own token budget (None without one); the real one stays the episode's. before: the
        requests.jsonl rows before its first request. used: the tokens the budget had counted then (a read
        control's), in place of before's sum."""
        if not self._budget:
            return None
        used = sum(int(r.get("completion_tokens") or 0) for r in before) if used is None else used
        return {"limit": self._budget["limit"], "used": used}

    async def _rerun_branch(self, thread: dict, branch: dict, state: str, ctrl: list,
                            start: dict, turns: int, grown: int, deadline: tuple | None = None) -> tuple:
        """A branch thread run again (a nested control, a read control) from history ctrl and its workdir back at
        state, fork refused as at OCTOPUS_FORK_DEPTH, closed by commit/abort like the branch, under the turns
        (branch_turns() less `turns`) and prompt growth (thread_budget() less `grown`) it had left. Returns its
        report and its patch since the branch's base, graded like the branch's (it rewrites the closed branch's .patch
        file; the branch's own patch is kept in memory). deadline (a read control): [seconds, 0] from its start, after
        which it stops before its next request."""
        own = {"payload": {}}

        def before_turn(turns: int) -> str | None:
            return "deadline" if time.time() - began >= deadline[0] else None

        async def complete(history, *_):  # tagged with its branch's name (OCTOPUS_READ_TOOL)
            own["payload"] = await self._complete_async(history, calls_left=start["calls_left"],
                                                        step=start.get("step"), me=start["branch_id"],
                                                        context_left=start.get("context_left"), start=start)
            return own["payload"]
        await self.ws.restore(thread["root"], state)
        capped = dict(thread, depth=fork_depth())  # its fork calls are refused
        start.update(branch_id=branch["branch_id"], session_id=None, task=branch["task"], messages=ctrl)
        began = time.time()
        report = await run_branch_continuation(
            start, complete, lambda name, args, _bid: self._thread_tool(capped, name, args, ctrl, own["payload"]),
            branch_turns() - turns, closing=self.ws.branch_closing, budget=thread_budget() - grown,
            before_turn=before_turn if deadline else None,
            snapshot=lambda: self.ws.snapshot(thread["root"]),
            restore=lambda snap: self.ws.restore(thread["root"], snap), ws=self.ws)
        _, _, text = await self.ws.capture(thread["root"], thread["base"])
        return report, text

    def _read_state(self, thread: dict, read: dict) -> dict:
        """OCTOPUS_READ_CONTROL: what a branch's read control needs from just before the request carrying the branch's
        first read: the read reply's id, which of its read_thread calls mounted, its tool calls as the server parsed
        them (names in order: where training finds that call, group.call_spans), the requests.jsonl rows logged so far
        (its budget: _control_budget; complete lines only, the HTTP threads may be appending one), the branch's replies
        so far, its first request's prompt size and the time."""
        rows = [json.loads(x) for x in (Path(self.logs_dir) / "requests.jsonl").read_text().split("\n")[:-1]]
        budget = self._budget
        return {"rid": read.get("id"), "call": thread.get("read_call", 0), "rows": len(rows), "t": time.time(),
                "calls": [(c.get("function") or {}).get("name")
                          for c in reply_tools((read.get("choices") or [{}])[0].get("message") or {})[0]],
                "used": budget and budget["used"],  # what the reader's mount request was told was left
                "turns": thread.get("turns", 0),
                "first_prompt": next((int(r.get("prompt_tokens") or 0) for r in rows
                                      if r.get("thread") == thread["name"] and r.get("id")), 0)}

    async def _read_controls(self, forks: list[dict]) -> None:
        """OCTOPUS_READ_CONTROL (docs/rl-plan-episodes.md "Read control"): after the episode, for each branch whose own
        read_thread mounted (its first valid read; main's reads have no control), one continuation from the same state
        as the request that carried that mount: the same history (the read reply and the read's tool result, the same
        cursor), the branch's workdir back to its tree then, its budget then (_control_budget: the shared pool left at
        the read) and its turns and prompt growth left, but no mount. It cannot fork (nor read). It faces the deadline
        its reader faced (read_deadline), on its own clock from its start. Every pair runs whatever the reader did after
        its read. Its patch is graded
        like the branch's, on its task (plan.tsv row readctl-<branch key>); its requests are requests.jsonl thread
        readctl/<thread name>. Its copy of the read result keeps the tag and the "Mounted ..." sentence, not the step
        listing, so the read call is credited with both. tree.json read_controls: {of, read_rid, read_call (which read_thread call of the read
        reply mounted), read_calls (the read reply's tool calls as the server parsed them, names in order), deadline [seconds, 0] |
        None, deadline_from (pool), cut (the deadline or the pool stopped it), forked_after (the reader
        called fork after its read), control_forked (the control called fork, refused), end, accepted (its closing's
        acceptance_verified), summary (core.handed: its commit's summary, its done's answer), budget_before,
        budget_used}."""
        records = {name: b for name, _, b in tree_branches(forks)}
        rows = [json.loads(x) for x in (Path(self.logs_dir) / "requests.jsonl").read_text().splitlines()]
        self._readctl, self._read_patches, todo = [], {}, [self._main_thread]
        while todo:
            t = todo.pop(0)
            todo += [c for f in t["started"] for c in f["children"].values()]
            cap, b = t.get("read_ctl"), records.get(t["name"])
            if not cap or b is None:
                continue
            deadline, why = read_deadline(
                "" if b.get("ended_with") in ("commit", "abort") else str(b.get("stop") or ""), b.get("end_time"),
                cap["t"], self._pool_out_t)
            # the reader's history from its mount request on, and what its rewinds removed of it
            after = t["start"]["messages"][len(cap["messages"]):] + cap.get("rewound", [])
            rec = {"of": t["name"], "read_rid": cap["rid"], "read_call": cap["call"], "read_calls": cap["calls"],
                   "deadline": deadline, "deadline_from": why,
                   "forked_after": any((c.get("function") or {}).get("name") == "fork"
                                       for m in after for c in m.get("tool_calls") or [])}
            self._readctl.append(rec)
            # the pool left at the read: the reader drew on the shared pool, and the pool's end is its deadline
            main_budget, self._budget = self._budget, self._control_budget(rows[:cap["rows"]], cap["used"])
            self._control_running, rec["budget_before"] = f"readctl/{t['name']}", self._budget and self._budget["used"]
            try:
                report, text = await self._rerun_branch(t, b, cap["snap"], cap["messages"],
                                                        {"step": cap["step"], "rewinds_used": cap["rewinds_used"],
                                                         "reminded": cap["reminded"]},
                                                        cap["turns"], cap["grown"], deadline)
                last = report["steps"][-1] if report["steps"] else {}
                # its read-time pool spent: the reader's shared pool would have run out there (a pool stop)
                spent = "budget used up" in str(last.get("text"))
                closing = closing_step(report["steps"]) or {}  # as a branch's record (fork_record): accepted
                rec.update(end=closing_name(report["steps"]), cut=last.get("text") == "stopped: deadline" or spent,
                           control_forked=any(x.get("name") == "fork" for x in report["steps"]),
                           accepted=closing.get("acceptance_verified"),
                           summary=handed(closing), reminded=report["reminded"],
                           budget_used=self._budget and self._budget["used"])
                self._read_patches[t["name"]] = (b.get("task_id"), text)
            finally:
                self._control_running, self._budget = False, main_budget

    async def _dispatch(self, name: str, args: dict, root: str | None = None, port=None, thread: str = "main"):
        """One tool call; runtime tools go to `port` (default: the main thread's), working tools to the workspace at
        root (default: main's) as the thread's (a control's: its own). Working tools of concurrent threads run side by
        side (each thread its own workspace); runtime tools (fork bookkeeping) stay one at a time."""
        if name not in RUNTIME_NAMES:
            return await self._dispatch_locked(name, args, root, port, thread)
        if self._dispatch_lock is None:
            self._dispatch_lock = asyncio.Lock()
        async with self._dispatch_lock:
            return await self._dispatch_locked(name, args, root, port, thread)

    async def _dispatch_locked(self, name: str, args: dict, root: str | None = None, port=None, thread: str = "main"):
        allowed = {item["function"]["name"] for item in self._tools()}
        if name not in allowed:
            return f"{name} is not a tool. Available tools: {', '.join(sorted(allowed))}.", False
        if name in RUNTIME_NAMES:
            return await asyncio.to_thread(self.ws.runtime, name, args, port or self._runtime or RuntimePort())
        running = self._control_running  # a control's calls are its own (readctl/... or control)
        return await self.ws.call(self.ws.root("main") if root is None else root,
                                  running if isinstance(running, str) else "control" if running else thread, name, args)

    async def _write_plan_inputs(self, forks: list[dict]) -> None:
        """RL plan episodes: the fork tree's results and the controls' for the episode's verifier (a copy in plan/)."""
        await self.ws.export(Path(self.logs_dir) / "plan", forks, self._read_patches, self._control_patch,
                             self._control_snap)

    def _write_tree(self, messages: list, forks: list[dict]) -> None:
        """The conversation as a tree for branch-aware SFT: the main thread plus, per fork, the
        prefix every branch resumed from and each branch's own turns after it."""
        path = Path(self.logs_dir) / "tree.json"
        path.write_text(json.dumps({
            "schema": "octopus-tree-v1",
            "tools": self._tools(),
            "main": without_cursors(messages),
            "rewinds": without_cursors(self._rewinds),  # main's (a branch's are in its fork record)
            "forks": without_cursors(forks),
            "budget": self._budget,
            "main_end": self._main_end,
            "reminded": self._main_thread["start"].get("reminded", 0),  # main's REMIND_NOTEs (a branch's: its record)
            "control": self._control,
            "red_checks": self._red_checks,
            **({"read_event": self._event + self._events_left()}  # OCTOPUS_READ_EVENT (_evented)
               if self._event is not None else {}),
            **({"read_controls": self._readctl} if self._readctl else {}),
        }, ensure_ascii=False) + "\n")

    def _write_trajectory(self, steps: list[Step]) -> None:
        trajectory = Trajectory(
            schema_version="ATIF-v1.8",
            session_id=self.session_id,
            agent=Agent(name=self.name(), version=self.version() or "1", model_name=self.model_name,
                        tool_definitions=self._tools()),
            steps=steps,
        )
        path = Path(self.logs_dir) / "trajectory.json"
        path.write_text(json.dumps(trajectory.to_json_dict(), ensure_ascii=False, indent=2) + "\n")
