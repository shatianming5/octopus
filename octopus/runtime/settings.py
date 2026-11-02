"""The episode switches, read from the environment where they are used (no state): one module for the runtime core and
both environments."""

from __future__ import annotations

import json
import os


def branch_fix() -> bool:
    """OCTOPUS_STRATEGY=branch_fix (SWE): every branch is one complete fix (locate, edit, verify in its private copy) of
    one independent problem or of one hypothesis about a problem; the main thread compares the branches' verified
    patches and merges them (octopus/envs/swe/prompt.py)."""
    return os.environ.get("OCTOPUS_STRATEGY", "").strip() == "branch_fix"


def plan_episode() -> bool:
    """OCTOPUS_PLAN_EPISODE=1 (RL plan episodes, set by octopus/envs/swe/rl.py): the fork tree and controls are
    exported for the verifier (CursorRuntime._write_plan_inputs)."""
    return bool(os.environ.get("OCTOPUS_PLAN_EPISODE"))


def cursor_mode() -> bool:
    """OCTOPUS_CURSOR=1 (docs/cursor-protocol.md): each reply in a thread's history keeps its response id ("cursor"),
    and a request sends only the messages after the last such reply, naming it; the server holds the rest."""
    return os.environ.get("OCTOPUS_CURSOR", "").strip() == "1"


def bundle() -> bool:
    """OCTOPUS_BUNDLE=1: the task is a bundle (one repo, n bugs in one tree, statements in /tasks;
    octopus/envs/swe/data/build_bundles.py). Set per episode by octopus/envs/swe/rl.py from the
    task dir (tests/tasks.json), never by a run's environment."""
    return bool(os.environ.get("OCTOPUS_BUNDLE"))


def auto_merge() -> bool:
    """OCTOPUS_AUTO_MERGE=1 (default in bundle mode): at each join the runtime applies every committed branch patch to
    the forking thread's tree. 0: the model merges; a bundle branch is then credited by its task in the final tree."""
    return os.environ.get("OCTOPUS_AUTO_MERGE", "1" if bundle() else "0") == "1"


def call_limit() -> int | None:
    """Main's call cap (and the main control's): OCTOPUS_MAIN_MAX_CALLS (None: unset); a branch has its own
    OCTOPUS_BRANCH_MAX_TURNS."""
    cap = os.environ.get("OCTOPUS_MAIN_MAX_CALLS", "").strip()
    return int(cap) if cap else None


def rewind_tool() -> bool:
    """OCTOPUS_REWIND_TOOL=1: every thread may call rewind(to_step, note): its repository back to the end of one of its
    own earlier steps, the steps since dropped from its history, the note kept (core.run_branch_continuation).
    Each tool result's tag then names its step ("| step N" at its end)."""
    return os.environ.get("OCTOPUS_REWIND_TOOL", "").strip() == "1"


def rewind_max() -> int:
    """OCTOPUS_REWIND_MAX=N (with OCTOPUS_REWIND_TOOL): each thread runs at most N rewind calls (a control goes on from
    the count of the thread it replays; a rewind does not give one back); a further one is refused (core.REWIND_SPENT),
    the reserve notes stop offering rewind, and the rewind tool's description names the cap. 0 or unset: no cap. The
    OCTOPUS_REWIND rule's rewinds are not counted."""
    return int(os.environ.get("OCTOPUS_REWIND_MAX", "").strip() or 0)


def thinking() -> bool:
    """OCTOPUS_CHAT_TEMPLATE_KWARGS, the one thinking switch: set and without "enable_thinking": false, as the Qwen3.5
    template reads it (off only when defined and false); unset is off (the Grok API sends none). The agent sends it with
    every request; direct SGLang renders with it and the AReaL proxy lets it override grpo.yaml's
    chat_template_kwargs. Runtime notes then go as tool results (core.add_note, docs/cursor-protocol.md)."""
    kwargs = os.environ.get("OCTOPUS_CHAT_TEMPLATE_KWARGS", "").strip()
    return bool(kwargs) and json.loads(kwargs).get("enable_thinking") is not False


def thread_budget() -> int:
    """Every thread's growth cap, SWE and search alike: at most N tokens added to its context after its first request
    (run_branch_continuation's budget; the prefix it starts from is reused KV and does not count), N =
    OCTOPUS_THREAD_BUDGET or 32768. Main (a linear run's too) and each branch get N, the main-first control what main had
    left of it at the fork, a branch control what its branch had left; every thread's tool results start with its own
    growth left ("context left: M")."""
    return int(os.environ.get("OCTOPUS_THREAD_BUDGET", "").strip() or 0) or 32_768


def max_reply() -> int:
    """OCTOPUS_MAX_TOKENS: one reply's cap, every request's max_tokens (run_config.episode_env sets it from the run's
    max_new_tokens), 32768 when unset."""
    return int(os.environ.get("OCTOPUS_MAX_TOKENS", "32768"))


def window_room(context: int) -> int:
    """The tokens a thread whose context holds `context` tokens may still add with one reply (max_reply) left inside
    the serving context window, OCTOPUS_CONTEXT_WINDOW or 262144 (rl_run.sh's and eval_serve.sh's SGLang
    --context-length): a request past the window fails (HTTP 400, the thread's "request failed"). A thread's growth
    cap is at most the room its first request leaves (run_branch_continuation), and a fork whose branches would start
    with little room is refused (CursorRuntime._thread_tool), so that deep threads end by context instead."""
    return (int(os.environ.get("OCTOPUS_CONTEXT_WINDOW", "").strip() or 0) or 262_144) - context - max_reply()


def answer_reserve() -> int:
    """OCTOPUS_ANSWER_RESERVE=R: the last R tokens of a thread's growth cap (thread_budget) are kept for its closing
    call. Once a reply leaves the thread less than R of growth, only its closing calls (main: done; a branch: what
    closes it) and rewind (OCTOPUS_REWIND_TOOL) run and every other call is refused with core.RESERVE_NOTE; tool
    results that would pass into it are clipped at it (core.clip_results, counted with the policy's tokenizer: R needs OCTOPUS_TOKENIZER). 0 or unset: off, the
    runtime's default (SWE); search's defaults.DEFAULTS sets 2048."""
    reserve = int(os.environ.get("OCTOPUS_ANSWER_RESERVE", "").strip() or 0)
    if reserve and not os.environ.get("OCTOPUS_TOKENIZER"):  # core.clip_results counts the policy's tokens
        raise RuntimeError("OCTOPUS_ANSWER_RESERVE needs OCTOPUS_TOKENIZER, the policy's tokenizer dir")
    return reserve


def read_tool() -> bool:
    """OCTOPUS_READ_TOOL=1 (docs/cross-branch-read.md; needs OCTOPUS_CURSOR=1 and mount-capable SGLang servers): every
    thread but the controls may call read_thread(thread, from_step, to_step): those steps of another thread of the
    episode are mounted into its attention layers from its next request on. Tool results then carry step tags, as for
    rewind, led by the thread's name ([main | step 3])."""
    return os.environ.get("OCTOPUS_READ_TOOL", "").strip() == "1"


def read_event() -> bool:
    """OCTOPUS_READ_EVENT=1 (Gate 0 measurement, eval only, docs/cross-branch-read.md; needs OCTOPUS_READ_TOOL=1): when a
    branch's step edits a file a sibling has viewed or edited before, or its commit's patch holds one, the sibling's
    next request once that step is readable carries a mount of it, with no tool call and no text
    (CursorRuntime._evented)."""
    return os.environ.get("OCTOPUS_READ_EVENT", "").strip() == "1"


def conflict_by() -> bool:
    """OCTOPUS_CONFLICT_BY=1 (Gate 0 measurement): each file of a branch patch the join took back for a conflict is
    labelled sibling (it conflicts with the sibling patches merged since its fork alone), own (with the forking thread's
    own edits) or unclear (tree.json branch conflict_by; HarborWorkspace.conflict_by)."""
    return os.environ.get("OCTOPUS_CONFLICT_BY", "").strip() == "1"


def read_control() -> bool:
    """OCTOPUS_READ_CONTROL=1 (docs/rl-plan-episodes.md "Read control"; with OCTOPUS_READ_TOOL): each branch's first
    valid read gets a paired continuation from the same state without the mount (CursorRuntime._read_controls), and
    the read_thread call is credited with the reader's reward less that control's (group.read_values)."""
    return os.environ.get("OCTOPUS_READ_CONTROL", "").strip() == "1"


def auto_fork() -> bool:
    """OCTOPUS_AUTO_FORK=1 (Gate 0 measurement, SWE bundle eval only, never a training env): before main's first request
    the environment forks one branch per bundle task (CursorRuntime.run calls HarborWorkspace.auto_fork; the system
    prompt says so, prompt.delivery); main's history starts with that fork's join note, and it may fork again.
    That fork has no response_id: dev_eval's C / C_wall / J / P leave its branches out."""
    return os.environ.get("OCTOPUS_AUTO_FORK", "").strip() == "1"
