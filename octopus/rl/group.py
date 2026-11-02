"""The group step of tree episodes: one outcome advantage for the whole tree (pure outcome reward), and the optional
read counterfactual (OCTOPUS_READ_CONTROL); the episode record every environment builds (episode_record, with its
critical-path cost) and each exported leaf's advantage (assign_threads). Environment-agnostic: an environment's
grades come in.

Env -> trainer contract: one record per episode of a group, built by its environment (episode_record from the
agent's logs and the environment's grades: the code environment's octopus/envs/swe/reward.plan_outcome, search's
octopus/envs/search/reward.search_outcome). What group_step, assign_threads and plan_group_stats read of it:
  y              the tree's outcome in [0, 1] (pass bit, a bundle's solved fraction, the answer's grade)
  C, forked      its critical-path cost (completion tokens) and whether it forked, recorded only
  control        None, or main's no-fork control from a fork point {y, C, kind} (rl.plan_control, an evaluation's):
                 recorded, never trained
  branches       [{key, passed}]: every branch thread (octopus.runtime.core.branch_key of its name) and its own
                 grade, None where the environment grades none (search adds summary and searched, its own
                 search/open calls: search's rl.branch_credit gate)
  read_controls  [{of, read_rid, read_call, read_calls, forked_after, control_forked, passed}] (OCTOPUS_READ_CONTROL):
                 per reader (of: its branch key) its read control and the control's own grade
  requests       [(response id, thread, prompt tokens, completion tokens)] in order: the leaves' threads
An environment's read_rewards reads the pair's grades (search: passed) or more of its own record (code: the fields
octopus/envs/swe/reward.code_read_rewards names). plan_group_stats also reads C_wall where present (the episode's and its control's) and a
code episode's tasks, branch_control and a bundle branch's task, ended_with and landed. Consumers derive the rest from
y (the episode's reward 2y−1, whether it passed y == 1).
"""

from __future__ import annotations

import collections
import itertools
import json
import re
from pathlib import Path

from octopus.runtime.core import branch_key, thread_name, token_index, tool_format, tree_branches


def leave_one_out(values: list[float]) -> list[float]:
    """A_i = x_i − mean_{j≠i} x_j, as the mean of the differences: exactly 0 when every value is equal (a 1/6 solved
    fraction too). One value → 0 (no baseline). No std."""
    n = len(values)
    return [sum(x - v for v in values) / (n - 1) for x in values] if n > 1 else [0.0] * n


BRANCH_CREDITS = ("outcome", "gate")


def branch_ok(branch: dict) -> bool:
    """rl.branch_credit gate, the code environment's ok: the branch committed, its patch was merged at the join, changed
    files and passed its own grade. Needs the code environment's fields (plan_outcome): a record without them raises,
    never counts as not ok."""
    missing = [k for k in ("ended_with", "merge", "diffstat", "passed") if k not in branch]
    if missing:
        raise ValueError(f"rl.branch_credit=gate needs {missing} on every branch (the code environment's plan_outcome)")
    return (branch["ended_with"] == "commit" and str(branch["merge"] or "").startswith("merged")
            and bool(branch["diffstat"]) and bool(branch["passed"]))


def search_branch_ok(branch: dict) -> bool:
    """rl.branch_credit gate, search's ok (search protocol v6): the branch closed with done, a non-empty answer (its
    summary: core.handed), and its subtree (itself or its branches at any depth) made at least one search or open call.
    Needs search's fields (search_outcome: summary,
    searched): a record without them raises."""
    missing = [k for k in ("ended_with", "summary", "searched") if k not in branch]
    if missing:
        raise ValueError(f"rl.branch_credit=gate needs {missing} on every search branch (search's search_outcome)")
    return branch["ended_with"] == "done" and bool(str(branch["summary"] or "").strip()) and branch["searched"] > 0


def code_gate(a: float, branch: dict) -> float:
    """rl.branch_credit=gate in the code environment (plan_group_step): an ok branch (branch_ok) keeps only A_main's
    good side, max(A_main, 0), any other only its bad side, min(A_main, 0)."""
    return max(a, 0.0) if branch_ok(branch) else min(a, 0.0)


def search_gate(a: float, branch: dict) -> float:
    """rl.branch_credit=gate in search (search_group_step): an ok branch (search_branch_ok) keeps A_main, any other only
    its bad side, min(A_main, 0)."""
    return a if search_branch_ok(branch) else min(a, 0.0)


def tree_advantage(episodes: list[dict], branch_credit: str = "outcome", gate=None) -> None:
    """The tree gives one outcome: every branch thread (cut by main or starved too) takes its episode's A_main.
    branch_credit (rl.branch_credit, a trainer setting) "gate": each branch takes gate(A_main, branch), the episodes'
    environment's rule (its group step passes it: code_gate, search_gate). Main's A is unchanged."""
    if branch_credit not in BRANCH_CREDITS:
        raise ValueError(f"rl.branch_credit {branch_credit!r}: one of {BRANCH_CREDITS}")
    if branch_credit == "gate" and gate is None:
        raise ValueError("rl.branch_credit=gate needs the environment's gate rule")
    for episode in episodes:
        for branch in episode.get("branches") or []:
            a = episode["A_main"]
            branch["A_branch"] = gate(a, branch) if branch_credit == "gate" else a


def read_values(episode: dict, read_rewards) -> dict:
    """OCTOPUS_READ_CONTROL (docs/rl-plan-episodes.md "Read control"): {a reader's read reply id: {"calls": the reply's
    tool calls as the server parsed them (names in order), "call": which of them is the read_thread call that mounted, "v": r_reader −
    r_rc}}. read_rewards(episode, c, b) -> (r_reader, r_rc): the environment's rewards of read control c (a branch's
    continuation from its first valid read without the mount, under the deadline its reader faced) and of its reader,
    branch b, by the same rules. No value where the reader forked after its read or the control called fork (refused):
    such a pair also differs in what forking was worth."""
    out = {}
    for c in episode.get("read_controls") or []:
        if c.get("forked_after") or c.get("control_forked"):
            continue
        b = next(x for x in episode["branches"] if x["key"] == c["of"])
        r_reader, c["r"] = read_rewards(episode, c, b)
        calls = c["read_calls"]
        out[c["read_rid"]] = {"calls": calls, "v": r_reader - c["r"],
                              "call": [i for i, name in enumerate(calls) if name == "read_thread"][c["read_call"]]}
    return out


def group_step(episodes: list[dict], read_rewards=None, branch_credit: str = "outcome", gate=None) -> list[dict]:
    """Fill R = 2y−1 and A_main (leave-one-out over the group's mains, no std) on each episode, every branch's A_branch
    = A_main (tree_advantage; branch_credit "gate": the environment's gate rule's side), and with read_rewards (read_values' hook) each episode's read credit "read_adv". A main
    control (an evaluation's, rl.plan_control) is graded and never trained: no R, no A, not in the baseline."""
    for episode in episodes:
        episode["R"] = 2.0 * float(episode["y"]) - 1.0
    for episode, adv in zip(episodes, leave_one_out([e["R"] for e in episodes])):
        episode["A_main"] = adv
    tree_advantage(episodes, branch_credit, gate)
    if read_rewards:
        for episode in episodes:
            episode["read_adv"] = read_values(episode, read_rewards)
    return episodes


_JSON = json.JSONDecoder()


def tool_calls(text: str) -> list[tuple[str, int, int]]:
    """(name, start, end) of each tool call in a completion's text, in the run's tool-call format (TOOL_FORMATS "call"):
    the regex match, or a JSON call's object. An object that is not valid JSON or whose name is not a string is no call
    (the tool-call parsers skip it too)."""
    out, end = [], 0
    for m in re.finditer(tool_format()["call"], text, re.S):
        if name := m.groupdict().get("name"):
            out.append((name, m.start(), m.end()))
        elif m.start() >= end:  # not inside the previous JSON call
            try:
                call, end = _JSON.raw_decode(text, m.start("json"))
            except json.JSONDecodeError:
                continue
            if isinstance(call.get("name"), str):
                out.append((call["name"], m.start("json"), end))
    return out


def call_spans(ids: list[int], token_text, calls: list[str]) -> list[tuple[int, int]]:
    """Token ranges [s, e) of a reply's tool calls as the server parsed them (`calls`: their names in order, the one
    definition of what the reply called) in its completion's ids, one per call: the format's matches (tool_calls) only
    locate them. None ([]) unless the matches are exactly those calls: a call the server's parser took from text the
    format does not match (SGLang's parser, AReaL's fallback, on a reply cut off inside its last call) or a match that was no call (a call written in prose)
    leaves no span rather than one on the wrong call."""
    text, at = token_index([token_text(i) for i in ids])
    found = tool_calls(text)
    return [(at(s), at(e - 1) + 1) for _, s, e in found] if [name for name, _, _ in found] == calls else []


def fork_spans(forks: list, parent: str = "main") -> dict:
    """{branch thread ("fork<k>/<branch id>", "<thread>/fork<k>/<branch id>"): (forking response id, the response id
    that collected it)} for every branch of a tree.json fork list at any depth whose fork records both (a fork that
    waited for its branches: the same id). A branch's own join_response_id when it has one (a fork's branches may be
    joined over several replies), else its fork's."""
    out = {}
    for k, fork in enumerate(forks):
        for branch in fork.get("branches") or []:
            name = thread_name(parent, k, branch["branch_id"])
            if fork.get("response_id") and fork.get("join_response_id"):
                out[name] = (fork["response_id"], branch.get("join_response_id") or fork["join_response_id"])
            out.update(fork_spans(branch.get("forks") or [], name))
    return out


def critical_path(requests: list[dict], spans: dict | None = None, unit: str = "tokens",
                  ends: dict | None = None) -> float:
    """The tree's longest chain of dependent work from main, in completion tokens, (unit "calls") model replies or (unit
    "wall") seconds: a request costs the time from its start to its thread's next request (model, tool, any wait;
    the thread's last request: to its end in `ends` {thread: wall time}, a branch's tree.json end_time, so the tools
    after its final reply count; else its own model time). A thread's own work in order, and per branch it forked the
    branch's critical path. With the branch's span (fork_spans) it runs from the forking reply while the thread works
    on, and the thread resumes at the reply collecting it after the longer of the two (main and branches overlapping
    count once; a fork's branches may be collected at different replies); without one its fork waits (adds its longest
    branch). In seconds a blocking fork (forked and collected in one reply, as every
    branch's own fork) adds nothing when that reply's cost ran to the thread's next request or end, after its
    branches (not main's last reply forking and calling done: model time only).

    requests.jsonl `thread`: "main", a branch "fork<k>/<branch id>" (nested: "<parent thread>/fork<k>/<branch id>"),
    or "control" (control is its own episode).
    """
    rows = collections.defaultdict(list)  # thread -> its requests in order
    for row in requests:
        if (row.get("thread") or "main") != "control":
            rows[row.get("thread") or "main"].append(row)
    own = collections.defaultdict(list)  # thread -> [(response id, cost, wall: ran to a next request or end)] in order
    for thread, rs in rows.items():
        for row, nxt in zip(rs, rs[1:] + [None]):
            end = (nxt.get("t") if nxt else (ends or {}).get(thread)) if unit == "wall" else None
            if unit == "wall":
                cost = float(end or 0) - float(row.get("t") or 0) if nxt or end else float(row.get("elapsed_s") or 0)
            else:
                cost = float(bool(row.get("id"))) if unit == "calls" else float(row.get("completion_tokens") or 0)
            own[thread].append((row.get("id"), cost, bool(nxt or end) and unit == "wall"))
    spans = spans or {}

    def path(thread: str) -> float:
        longest = {}  # a branch with a span: its own path; else its fork's longest branch
        for child in list(own):
            rest = child if thread == "main" else child[len(thread) + 1:] if child.startswith(thread + "/") else ""
            if child != "main" and rest.count("/") == 1:  # fork<k>/<branch id> directly under this thread
                key = child if child in spans else child.rpartition("/")[0]
                longest[key] = max(longest.get(key, 0.0), path(child))
        t, began = 0.0, {}
        for rid, cost, waited in own.get(thread, []):
            t += cost
            start = {}  # fork<k> -> when its branches start at this reply: together, after earlier forks' waits here
            for fork, (forked_at, joined_at) in spans.items():
                if fork in longest and rid == forked_at:
                    began[fork] = start.setdefault(fork.rpartition("/")[0], t)
                if fork in longest and fork in began and rid == joined_at:
                    branches = longest.pop(fork)
                    if not (waited and forked_at == joined_at):  # wall: a blocking fork's reply cost holds them
                        t = max(t, began[fork] + branches)
        return t + sum(longest.values())  # forks never collected in this thread's replies wait at its end
    return path("main")


def join_post(requests: list[dict], forks: list) -> tuple[float, float]:
    """Main's join idle J and post-join work P (seconds). From main's first forking reply:
    a reply collecting branches (a fork's or a branch's join_response_id: a fork's branches may be joined over several
    replies) waits J from the end of its model call to main's next request (main's last reply: to the latest
    join_time); a reply outside every fork's window is post-join work, its model call to the next request.

    A fork's window runs from its forking reply to its first collecting reply, or later to main's first reply starting
    after a branch collected later (a leftover, by an explicit join or main's end) had ended, when that leftover merged
    a committed patch: leftovers that contributed nothing (aborted, stopped, uncommitted, conflicting) do not extend it.
    A later reply that collects only leftovers of forks whose windows ended by then is post-join work too (its model
    call; its wait stays J); one that also collects a fork's first branches, or a leftover still running when it
    started, is not. So aborting a leftover and joining it explicitly counts the same as main's end sweeping it up."""
    main = [r for r in requests if (r.get("thread") or "main") == "main" and r.get("id")]
    idx = {r["id"]: i for i, r in enumerate(main)}
    t, el = [float(r.get("t") or 0) for r in main], [float(r.get("elapsed_s") or 0) for r in main]
    windows, joined, clear = [], {}, {}  # clear: collecting reply -> its model call is post-join work
    for f in forks:
        a, b = idx.get(f.get("response_id")), idx.get(f.get("join_response_id"))
        if a is None or b is None:
            continue
        for c in [f, *(f.get("branches") or [])]:
            if c.get("join_response_id") in idx:
                i = idx[c["join_response_id"]]
                joined[i] = max(joined.get(i, 0.0), float(c.get("join_time") or 0))
        got = [(idx.get(br.get("join_response_id") or f["join_response_id"]), br) for br in f.get("branches") or []]
        got = [(i, br) for i, br in got if i is not None] or [(b, {})]
        c1 = min(i for i, _ in got)  # its first collecting reply
        w = max([c1] + [next((k for k in range(len(main)) if t[k] >= float(br.get("end_time") or 0)), len(main))
                           for i, br in got if i > c1 and br.get("ended_with") == "commit"
                           and (br.get("merge") or "merged").startswith("merged")])
        windows.append((a, w))
        for i in {i for i, _ in got}:
            clear[i] = clear.get(i, True) and i != c1 and w <= i
    first = min((idx[f["response_id"]] for f in forks if f.get("response_id") in idx), default=len(main))
    J = P = 0.0
    for i in range(first, len(main)):
        end = t[i + 1] if i + 1 < len(main) else max(t[i] + el[i], joined.get(i, 0.0))
        inside = any(a <= i < b for a, b in windows)
        if i in joined:
            J += end - t[i] - el[i]
            if clear.get(i) and not inside:
                P += el[i]
        elif not inside:
            P += max(el[i], end - t[i])
    return J, P


def control_cost(requests: list[dict], fork_response_id: str | None, spans: dict | None = None,
                 unit: str = "tokens", ends: dict | None = None) -> float:
    """A main control's C: the tree's critical path before the forking reply it replaces (the control's own history:
    main's work before its first fork, plus earlier forks' branches before a later one) plus the control's work. In
    wall seconds main's last request before it runs to the forking reply's start (its tools count, as in the forking
    tree) and earlier branches to their `ends`."""
    before = list(itertools.takewhile(lambda r: r.get("id") != fork_response_id, requests))
    ends = {**(ends or {}), "main": requests[len(before)].get("t")}
    return critical_path(before, spans, unit, ends) + critical_path([dict(r, thread="main") for r in requests
                                                                 if r.get("thread") == "control"], unit=unit)


def assign_threads(leaves: list, episode: dict, token_text=None) -> list[dict]:
    """Per exported leaf of one plan episode, as the rollout worker gets it (AReaL deserialize_interactions:
    interaction_id and _cache tensors only): {reward, adv, own_start, spans}.

    episode["requests"]: requests.jsonl rows in order, (response id, thread, prompt_tokens, completion_tokens); thread
    is main / control (main's no-fork control, rl.plan_control, or a nested one) or a read control "readctl/..." (graded,
    never trained) / a branch's name (core.thread_name: fork<k>/<branch id>, nested <parent>/fork<k>/<branch
    id>; graded under branch_key of it). A thread's requests extend one prefix, so a completion sits at [prompt_tokens,
    prompt_tokens + completion_tokens) of its thread's leaf. A rewound thread (OCTOPUS_REWIND) is several leaves
    sharing the turns before the rewind point: in request order, each leaf owns the thread's requests after the
    previous leaf's last one up to its own last one, and every leaf gets the thread's reward and advantage. own_start =
    the prompt length of the leaf's first owned request: the tokens before it are another thread's (the forking
    thread's turns up to the fork, which a branch inherits) or an earlier leaf's of this thread, and get loss mask 0.
    A thread's requests after its last leaf have no leaf of their own: the last of them forked, and only its branches
    extend it (the fork was the thread's last request: its call cap, a deadline, a failed next request). The first leaf
    of its first branch to start then owns them too (own_start lowered to them), each completion at its own thread's
    advantage (a span). spans = [(start, end, value added to adv)]: those, and with token_text (token id -> its text)
    the read credit (read_values) on the read_thread call that mounted. Every request's completion is owned by exactly
    one leaf and lies before the requests of the leaf's own thread. Raises (the group is rejected) when a leaf has no
    requests.jsonl row, a branch thread was not graded, or a request has no owner."""
    requests = episode["requests"]
    thread_of = {rid: thread for rid, thread, _, _ in requests}
    branches = {b["key"]: b for b in episode.get("branches") or []}
    owner, last = {}, {}  # request index -> the leaf that trains it; thread -> its previous leaf's last request index
    ends = {leaf.interaction_id for leaf in leaves}
    if missing := ends - set(thread_of):
        raise RuntimeError(f"no requests.jsonl row for completion {sorted(missing)[0]}")
    for i, (rid, thread, _, _) in enumerate(requests):
        if rid in ends:
            owner.update({j: rid for j in range(last.get(thread, -1) + 1, i + 1) if requests[j][1] == thread})
            last[thread] = i

    def adopted(i: int) -> str:  # the owner of request i, a thread's request after its last leaf
        thread = requests[i][1]
        final = max(j for j, r in enumerate(requests) if r[1] == thread)
        child = next((j for j in range(final + 1, len(requests))
                      if re.fullmatch(rf"{re.escape(thread) + '/' if thread != 'main' else ''}fork\d+/[^/]+",
                                      requests[j][1])), None)
        if child is None:
            raise RuntimeError(f"request {requests[i][0]} of {thread} is in no leaf and no branch extends it")
        return owner[child] if child in owner else adopted(child)
    for i in range(len(requests)):
        if i not in owner and not requests[i][1].startswith("readctl/") and requests[i][1] != "control":
            owner[i] = adopted(i)
    if len(owner) != len(requests):
        raise RuntimeError(f"requests in no leaf: {[r[0] for i, r in enumerate(requests) if i not in owner]}")

    def advantage(thread: str) -> float:
        if thread == "main":
            return episode["A_main"]
        b = branches.get(branch_key(thread))
        if b is None:
            raise RuntimeError(f"branch thread {thread} has no graded branch")
        return b["A_branch"]
    out = []
    for leaf in leaves:
        thread = thread_of[leaf.interaction_id]
        ids = leaf._cache["input_ids"].reshape(-1).tolist()
        if thread.startswith("readctl/") or thread == "control":
            out.append({"reward": episode["R"], "adv": 0.0, "spans": [], "own_start": len(ids)})
            continue
        row = {"reward": episode["R"], "adv": advantage(thread)}
        own = [(*requests[j], advantage(requests[j][1])) for j in sorted(owner) if owner[j] == leaf.interaction_id]
        start = next(p for _, t, p, _, _ in own if t == thread)  # its own thread's first request
        if any(p + c > start for _, t, p, c, _ in own if t != thread):
            raise RuntimeError(f"leaf {leaf.interaction_id}: an adopted completion lies past its own first prompt")
        row["own_start"] = own[0][2]
        spans = [(p, p + c, a - row["adv"]) for _, t, p, c, a in own if t != thread]  # adopted: their thread's A
        for rid, _, prompt, completion, a in own if token_text else []:
            read = (episode.get("read_adv") or {}).get(rid)
            if read:
                found = call_spans(ids[prompt:prompt + completion], token_text, read["calls"])
                spans += [(prompt + x, prompt + y, read["v"] - a) for x, y in found[read["call"]:read["call"] + 1]]
        row["spans"] = spans
        out.append(row)
    return out


def episode_record(trial: Path, tree: dict, requests: list[dict], y, grades: dict, control_y=None) -> dict:
    """group.py's record of one finished episode: what the agent logged (tree: its agent/tree.json; requests: its
    requests.jsonl rows) and the environment's grades. y: the tree's outcome; grades: {leaf key: its own grade} of the
    branches (core.branch_key of the thread) and read controls (branch_key of readctl/<reader thread>) the
    environment graded, every other leaf None; control_y: main's control's y (None: no main control). C: the
    critical path in completion tokens. Also: C_wall (critical path in seconds, plan_group_stats),
    each branch's and read control's ended_with (code_read_rewards, plan_group_stats); recorded only: main's join idle
    J and post-join work P (join_post), how main ended (tree.json main_end), the token budget (search's; SWE: null),
    the reminders main and the branches got (reminded)."""
    forks = tree.get("forks") or []
    spans = fork_spans(forks)
    ends = {name: b["end_time"] for name, _, b in tree_branches(forks) if b.get("end_time")}  # wall: branch tails
    meta = tree.get("control") or {}  # the agent's record: kind, path (<forking thread>/fork<k>), response_id, end
    control = None if control_y is None else {
        "y": control_y, "C": control_cost(requests, meta["response_id"], spans, ends=ends), "kind": meta["kind"],
        "C_wall": control_cost(requests, meta["response_id"], spans, "wall", ends)}
    J, P = join_post(requests, forks)
    return {
        "y": y, "C": critical_path(requests, spans, ends=ends),
        "forked": bool(forks), "C_wall": critical_path(requests, spans, "wall", ends), "J": J, "P": P,
        "branches": [{"key": branch_key(name), "branch_id": b["branch_id"], "ended_with": b.get("ended_with"),
                      "passed": grades.get(branch_key(name))} for name, _, b in tree_branches(forks)],
        "control": control,
        "read_controls": [{"of": branch_key(c["of"]), "read_rid": c["read_rid"], "read_call": c.get("read_call", 0),
                           "read_calls": c["read_calls"],
                           **{k: c.get(k, False) for k in ("forked_after", "control_forked")},
                           "ended_with": c.get("end"), "passed": grades.get(branch_key(f"readctl/{c['of']}"))}
                          for c in tree.get("read_controls") or [] if "skipped" not in c],
        "budget": tree.get("budget"),
        "requests": [(r["id"], r.get("thread") or "main", int(r.get("prompt_tokens") or 0),
                      int(r.get("completion_tokens") or 0)) for r in requests if r.get("id")],
        "trial": str(trial), "ended_with": tree.get("main_end"),
        # core.REMIND_NOTEs: main's, and the branches' at every depth (a reply without a tool call, reminded once)
        "reminded": {"main": tree.get("reminded", 0), "branches": sum(b.get("reminded", 0) for _, _, b in
                                                                      tree_branches(forks))},
    }


def plan_group_stats(episodes: list[dict]) -> dict:
    """After plan_group_step: mean |A| per thread type; per control kind (main-first, main-later, nested) the pairs of
    forking vs no fork from the same point ([y, control y]; nested: [the forking branch passed, the control passed])
    and their mean difference; for bundle branches the share naming no task and the share of committed branches whose
    patch solved their task that landed in the final tree; the read credit. C_wall: each episode's critical path in
    seconds (forked, unforked, and its control's), recorded only."""
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    rows = [b for e in episodes for b in e.get("branches") or []]
    bundle = [b for e in episodes if e.get("tasks") is not None for b in e.get("branches") or []]
    kept = [b for b in bundle if b.get("task") and b.get("ended_with") == "commit" and b.get("passed")]
    pairs = collections.defaultdict(list)
    for e in episodes:
        if e.get("control"):
            pairs[e["control"]["kind"]].append([round(float(e["y"]), 4), round(float(e["control"]["y"]), 4)])
        if e.get("branch_control"):
            c = e["branch_control"]
            b = next(x for x in e["branches"] if x["key"] == c["of"])
            pairs["nested"].append([float(bool(b["passed"])), float(bool(c["passed"]))])
    return {"fork_vs_control": dict(pairs),
            "fork_minus_control": {kind: mean([a - b for a, b in p]) for kind, p in pairs.items()},
            "abs_A_main": mean([abs(e["A_main"]) for e in episodes]),
            "abs_A_branch": mean([abs(b["A_branch"]) for b in rows]),
            "no_task_id": mean([float(not b.get("task")) for b in bundle]),
            "committed_passing_landed": mean([float(bool(b.get("landed"))) for b in kept]),
            "read_adv": mean([x["v"] for e in episodes for x in (e.get("read_adv") or {}).values()]),  # r_reader − r_rc
            "read_adv_n": sum(len(e.get("read_adv") or {}) for e in episodes),
            "C_wall": {"forked": [round(e["C_wall"]) for e in episodes if e["forked"] and "C_wall" in e],
                       "unforked": [round(e["C_wall"]) for e in episodes if not e["forked"] and "C_wall" in e],
                       "control": [round(e["control"]["C_wall"]) for e in episodes
                                   if e.get("control") and "C_wall" in e["control"]]}}
