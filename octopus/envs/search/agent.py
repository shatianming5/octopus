"""Offline search environment: the Octopus agent (CursorRuntime: fork / read_thread, per-thread growth caps and answer
reserve, call caps, main-first control, requests.jsonl / tree.json / trajectory.json) on a question, not a repo.

The working tools are search(query) and open(docid, offset), HTTP calls to the retrieval service (part A, API in
OCTOPUS_SEARCH_URL: POST /search, POST /open), and done(answer, evidence), which closes main (its graded answer) and
a branch alike (protocol v6: no commit / abort in search). search is BrowseComp-Plus's (searcher/tools.py) with CLM's
top 10 (SEARCH_K): documents, each with its first 512 tokens; open is our paged form of its optional get_document
(docs/search-env.md). There is no container: SearchWorkspace gives the runtime nothing to copy, snapshot or merge.
A trial dir: agent/{requests.jsonl, tree.json, trajectory.json} (the runtime's), verifier/answers.json (the graded
answers of main and the control, and under OCTOPUS_READ_CONTROL each reader's and read control's graded finding:
octopus/envs/search/reward.search_outcome builds the episode record from them) and result.json (exception_info when the
retrieval service, the episode or the judge failed: AReaL rejects it).

    python -m octopus.envs.search.agent QUESTION_JSON JOB_DIR   # one episode; QUESTION_JSON = {qid, question, answer}

Settings (env, defaults.DEFAULTS; docs/search-env.md "Limits"): OCTOPUS_SEARCH_URL, OCTOPUS_SEARCH_OPEN_CHARS (32768
characters, about CLM's 8192 tokens at 4 characters a token),
OCTOPUS_SEARCH_BUDGET (completion tokens of the whole tree; 0, the default: none), OCTOPUS_THREAD_BUDGET (tokens a
thread, main or branch, may add to its context; settings.thread_budget, 32768 as in SWE), OCTOPUS_ANSWER_RESERVE (the
last of it kept for done; settings.answer_reserve, 2048), OCTOPUS_MAX_BRANCHES
(branches per episode, 10), OCTOPUS_FORK_DEPTH (1: only main forks; 0: no depth limit),
OCTOPUS_MAIN_MAX_CALLS (main's calls, 100), OCTOPUS_BRANCH_MAX_TURNS
(a branch's turns, 50), OCTOPUS_READ_TOOL (no default: the env
file's or the caller's, the one the rollout worker sees too; 1: read_thread, a mount needs OCTOPUS_CURSOR=1 and
mount-capable servers, else every read is refused), OCTOPUS_REWIND_TOOL (no default either: 1, the search training
env file's, offers rewind, history only: REWIND; OCTOPUS_REWIND_MAX, the env file's 2, caps each thread's),
OCTOPUS_PLAN_CONTROL, OCTOPUS_LINEAR (a single agent: no fork in
the system prompt or the tool list; rewind stays), and the model endpoint (OCTOPUS_CHAT_URL, OCTOPUS_MODEL, ...) and judge
(octopus/envs/search/reward.py) settings.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from octopus.runtime import core
from octopus.runtime.tree import CursorRuntime, WorkspaceProvider, fork_depth, max_branches
from octopus.runtime.core import (
    READ_TOOL, REWIND_TOOL, _tool, branch_key, branch_turns, read_card, rewind_capped, runtime_tools, tree_branches,
    unlocked_tools,
)
from octopus.runtime.settings import rewind_tool
from octopus.envs.search.defaults import DEFAULTS
from octopus.envs.search.reward import grade
from harbor.models.agent.context import AgentContext

SEARCH_K = 10  # hits per search (CLM's zero-shot BrowseComp-Plus setting; BCP's own default is 5)
SEARCH_TOOLS = [
    _tool("search", "Search the document corpus. Returns the top 10 results, best first: docid, title, url and the "
          "document's first 512 tokens.", {"query": {"type": "string"}}, ["query"]),
    _tool("open", "Read a document by docid, a window of its text from offset (characters, default 0); the result "
          "names the total length and the next offset.",
          {"docid": {"type": "string"}, "offset": {"type": "integer"}}, ["docid"]),
    _tool("done", "Finish with your answer: a short string (a name, number, date or phrase), no explanation. In the "
          "main thread it is the final answer and ends the session; in a branch it closes the branch and hands the "
          "thread that forked it your answer (what you found for your task, or that you found nothing) and its "
          "evidence.",
          {"answer": {"type": "string"},
           "evidence": {"type": "string", "description": "Optional: docids and short quotes that support the answer."}},
          ["answer"]),
]
# search's fork: one task or more (WorkspaceProvider.fork_min); commit / abort are not search tools; its last sentence
# (core.FORK_ALONE) names rewind under OCTOPUS_REWIND_TOOL (SearchWorkspace.tools)
FORK = _tool("fork", "Start parallel branches, one per task. Each task is one branch's assignment: a self-contained "
             "sub-question or line of search. Every branch sees this conversation so far plus its task, searches and "
             "reads on its own, and closes with done (its answer and evidence). A reply whose fork starts cannot also "
             "call done: call it in a later reply.",
             {"tasks": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, ["tasks"])
# OCTOPUS_REWIND_TOOL: search's rewind, history only (no workspace: core.REWIND_TOOL's restores a repository)
REWIND = dict(REWIND_TOOL, function=dict(REWIND_TOOL["function"], description=(
    "Go back to the end of an earlier step of this thread, the step N in each tool result's tag: every later step "
    "(this call too) is removed from your history and your note stays in their place. Use it to leave a dead end or "
    "results you no longer need, keeping what they taught you in the note. Calls and budget tokens already spent stay "
    "spent; your context left grows back by the removed steps.")))
# OCTOPUS_READ_TOOL: read_thread offered only through its card (SearchWorkspace.read_card), not the tool list
READ_THREAD = ("Read another thread of this episode, never your own (a branch), without copying its text: the readable "
               "steps among from_step to to_step are mounted into your attention from your next reply on and stay "
               "there. A step is a reply and the tool results after it, numbered as the step N in that thread's tool "
               "result tags; a branch's steps continue the numbering of the thread it forked from. A step is readable "
               "once that thread's next reply is back or it has ended, unless your own history holds it already; "
               "without from_step and to_step, its latest readable steps are mounted. One read_thread per reply, not "
               "in a reply that calls fork")  # read_card ends it: "." or (OCTOPUS_REWIND_TOOL) " or rewind."

# read_thread's arguments, its thread named as search names it (read_by_name)
READ_PARAMETERS = dict(READ_TOOL["function"]["parameters"], properties=dict(
    READ_TOOL["function"]["parameters"]["properties"],
    thread={"type": "string", "description": "The other thread: its name, such as fork0/branch-1 (as the fork's "
                                             "findings name its branches) or main; a branch id such as branch-1 where "
                                             "one thread has it."}))

_TASK = ("You answer a question by searching an offline document corpus. search(query) returns the top 10 documents, "
         "each with its first 512 tokens; open(docid, offset) reads a document. The answer is a short string (a name, "
         "number, date or phrase): give it with done(answer), which ends the session. ")
_FORK = ("You may work alone, or fork parallel branches at any time: fork(tasks) starts one branch per task, each with "
         "this conversation so far plus its task; branches search and read on their own and close with done (their "
         "answer and its evidence). Fork when it helps; whether, when and how many is your choice. ")
_SYNC = "fork waits for every branch to close and returns their findings. "
_LINEAR = "This is a single-thread session without fork. "


def branch_limits() -> str:
    """The branch limits in effect (OCTOPUS_MAX_BRANCHES, OCTOPUS_FORK_DEPTH, OCTOPUS_BRANCH_MAX_TURNS), one text for main
    and its branches."""
    cap, turns, depth = max_branches(), branch_turns(), fork_depth()
    return ("Limits: " + (f"at most {cap} branches in the session; " if cap else "")
            + ("a branch cannot fork; " if depth <= 1 else f"forks nest at most {depth} levels below the main thread; "
               if depth < math.inf else "") + f"a branch has at most {turns} turns. ")


def limits_note(shared: bool) -> str:
    """core.limits_note with the search budget (OCTOPUS_SEARCH_BUDGET); main's calls are always capped."""
    return core.limits_note(shared, int(os.environ.get("OCTOPUS_SEARCH_BUDGET") or 0))


def post_json(url: str, body: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def search_text(results: list[dict]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(f"[{i}] docid={r['docid']} | {r.get('title') or ''} | {r.get('url') or ''}\n{r.get('snippet') or ''}"
                       for i, r in enumerate(results, 1))


def open_text(doc: dict) -> str:
    start, text, total = int(doc.get("offset") or 0), doc.get("text") or "", int(doc.get("total_length") or 0)
    end = start + len(text)
    more = f"; next offset {end}" if end < total else "; end of document"
    return f"docid={doc['docid']} | {doc.get('title') or ''} | {doc.get('url') or ''}\n[characters {start}-{end} of {total}{more}]\n{text}"


class SearchWorkspace(WorkspaceProvider):
    """No workspace: a thread's tools are calls to the retrieval service, and a branch hands back only its finding (its
    done's answer and evidence: the join note). A thread's root is its name; main's done (and the control's) is graded,
    a branch's closes it. answers: those graded answers; retrieval_error: the service failed (run_trial rejects the
    episode)."""
    isolated = False
    branch_contract = ("When it is done, call done with your answer and its evidence (docids, short quotes); if you "
                       "found nothing, say so in your answer.")
    branch_closing = frozenset({"done"})
    fork_min = 1
    read_when_readable = True  # read_thread offered once a thread a branch may read has a readable step
    read_by_name = True  # a later fork's branch reads an earlier fork's ended branch by its name (fork0/branch-1)
    tool_names = frozenset(t["function"]["name"] for t in SEARCH_TOOLS + runtime_tools()) - {"commit", "abort"}

    def __init__(self, environment=None):
        super().__init__(environment)
        self.answers, self.retrieval_error = {}, None

    @property
    def read_card(self) -> str:  # one text for every thread
        return read_card(READ_THREAD + (" or rewind." if rewind_tool() else "."), READ_PARAMETERS)

    def system(self, linear: bool) -> str:
        if linear or os.environ.get("OCTOPUS_LINEAR"):  # the no-fork control, a single-agent run: no branches
            return _TASK + _LINEAR + limits_note(False)
        return _TASK + _FORK + _SYNC + branch_limits() + limits_note(True)

    def tools(self, port, linear: bool) -> list[dict]:
        catalog = unlocked_tools(SEARCH_TOOLS, linear or bool(os.environ.get("OCTOPUS_LINEAR")))
        fork = dict(FORK, function=dict(FORK["function"], description=FORK["function"]["description"].replace(
            "call done:", "call done or rewind:"))) if rewind_tool() else FORK
        return [{"fork": fork, "rewind": rewind_capped(REWIND)}.get(t["function"]["name"], t) for t in catalog
                if t["function"]["name"] not in ("commit", "abort")]

    def join_note(self, reports: list[dict], patches: dict | None, merged: dict | None, main: bool) -> str:
        """Each branch named by its thread name (session_id: core.thread_name), as read_thread takes it."""
        return core.join_note([dict(r, branch_id=r["session_id"]) for r in reports], close="done")

    def fork_note(self) -> str:
        return "; their findings follow when they finish."

    def budget(self, instruction: str) -> dict | None:
        limit = int(os.environ.get("OCTOPUS_SEARCH_BUDGET") or 0)
        return {"limit": limit, "used": 0} if limit else None  # 0: no episode budget (DEFAULTS)

    def root(self, thread: str) -> str:
        return thread

    async def snapshot(self, root: str) -> str:
        return "search"  # nothing to restore: the control only needs a fork point

    async def restore(self, root: str, snapshot: str) -> None:
        pass

    async def open_branches(self, source: str, roots: dict, fork: dict) -> tuple[dict, None]:
        return dict.fromkeys(roots.values()), None

    async def capture(self, root: str, since: str | None = None) -> tuple:
        return None, "", ""

    async def apply(self, root: str, results: list) -> dict:
        return {}

    async def call(self, root: str, thread: str | None, name: str, args: dict) -> tuple[str, bool]:
        if name == "done":
            answer = str(args.get("answer") or "").strip()
            if thread in ("main", "control"):  # a branch's (a read control's too) closes it, ungraded
                self.answers[thread] = answer
            return f"Final answer recorded: {answer}", True
        base = os.environ["OCTOPUS_SEARCH_URL"].rstrip("/")
        if name == "search":
            body = {"query": str(args.get("query") or ""), "k": SEARCH_K}
        else:
            try:
                body = {"docid": str(args.get("docid") or ""), "offset": max(int(args.get("offset") or 0), 0),
                        "length": int(os.environ.get("OCTOPUS_SEARCH_OPEN_CHARS", "32768"))}
            except (TypeError, ValueError):
                return "open: offset is an integer", False
        try:
            out = await asyncio.to_thread(post_json, f"{base}/{name}", body)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:  # a request the service refused (an unknown docid): the model's mistake
                return f"{name} failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:200]}", False
            self.retrieval_error = f"{name} HTTP {exc.code}"
            return f"retrieval service error: HTTP {exc.code}", False
        except Exception as exc:  # the service is down: the episode is rejected (run_trial), never trained on
            self.retrieval_error = f"{name}: {type(exc).__name__}: {exc}"
            return "retrieval service error", False
        return (search_text(out.get("results") or []) if name == "search" else open_text(out)), False


class SearchAgent(CursorRuntime):
    provider = SearchWorkspace

    @staticmethod
    def name() -> str:
        return "octopus-search"


async def run_trial(data: dict, job_dir: Path, endpoint: str | None = None) -> Path:
    """One episode on data = {qid, question, answer} in a new trial dir of job_dir (the code environment's layout).
    octopus/envs/search/reward.grade grades main's and the control's done answers (a thread that ended without done has
    none: incorrect); a judge failure is the episode's error.
    endpoint pins the policy's chat URL (one server per episode)."""
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    trial = job_dir / f"{str(data['qid'])[:40]}__{uuid.uuid4().hex[:7]}"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    agent = SearchAgent(logs_dir=trial / "agent", model_name=os.environ.get("OCTOPUS_MODEL"))
    if endpoint:
        agent._pinned_endpoint = endpoint
    error = None
    try:
        await agent.run(data["question"], None, AgentContext())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    answers, error = agent.ws.answers, error or agent.ws.retrieval_error
    if not error:
        tree = json.loads((trial / "agent/tree.json").read_text())
        try:
            y, how = await asyncio.to_thread(grade, data["question"], answers.get("main"), data["answer"])
            graded = {"main": {"answer": answers.get("main"), "correct": y, "how": how}}
            # OCTOPUS_READ_CONTROL (reward.search_read_rewards): a reader and its read control are graded on their own
            # result, the answer their done hands main (summary: core.handed), under their leaf keys; no other leaf
            # costs a judge call, nor a pair read_values drops (the reader forked after its read, the control called fork)
            readers = {c["of"]: c for c in tree.get("read_controls") or []
                       if "skipped" not in c and not c.get("forked_after") and not c.get("control_forked")}
            found = {branch_key(name): b.get("summary")
                     for name, _, b in tree_branches(tree["forks"]) if name in readers}
            found.update({branch_key(f"readctl/{of}"): c.get("summary") for of, c in readers.items()})
            for key, summary in found.items():
                correct, how = await asyncio.to_thread(grade, data["question"], summary, data["answer"])
                graded[key] = {"answer": summary, "correct": correct, "how": how}
            if tree.get("control"):
                cy, chow = await asyncio.to_thread(grade, data["question"], answers.get("control"),
                                                   data["answer"])
                graded["control"] = {"answer": answers.get("control"), "correct": cy, "how": chow}
        except Exception as exc:  # the judge failed (endpoint down): rejected like a retrieval error
            error = f"judge: {type(exc).__name__}: {exc}"
        else:
            (trial / "verifier/answers.json").write_text(json.dumps(graded))
    info = {"exception_type": "SearchEpisodeError", "exception_message": error} if error else None
    (trial / "result.json").write_text(json.dumps({"task_name": str(data["qid"]), "exception_info": info}))
    return trial


if __name__ == "__main__":
    asyncio.run(run_trial(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])))
