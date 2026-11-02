"""Plan-episode RL of the code environment: one episode is the whole tree (main thread + any branches).

Its episode record (plan_outcome: group.episode_record with the SWE verifier's grades) and the code group step
(group.group_step with the code read counterfactual, code_read_rewards); the group's stats are
octopus/rl/group.py's plan_group_stats. HarborEpisodeAgent (octopus/envs/swe/rl.py) imports these.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from octopus.rl.group import code_gate, episode_record, group_step
from octopus.runtime.core import branch_key, tree_branches

DISHONEST_COMMIT = -1.5  # a single-task commit whose patch failed its own acceptance check and the hidden tests


def code_read_rewards(episode: dict, c: dict, b: dict, solvable: bool) -> tuple[float, float]:
    """The code environment's read_rewards (group.read_values' hook): (r_reader, r_rc), the own rewards of reader
    branch b and of its read control c (graded in b's place). Single task (solvable: the group solved it): commit +1 if
    its patch passed, DISHONEST_COMMIT if it failed its own acceptance check, else −1; abort −1 if solvable, else 0;
    unclosed −1. Bundle: commit +1 when its patch solved its task, else −1; abort −1
    when its own patch solved its task, else 0; unclosed −1."""

    def single(leaf: dict) -> float:
        if leaf.get("ended_with") == "commit":
            passed = bool(leaf.get("passed"))
            return DISHONEST_COMMIT if not passed and leaf.get("accepted") is False else 1.0 if passed else -1.0
        return -1.0 if leaf.get("ended_with") != "abort" or solvable else 0.0

    def bundle(leaf: dict) -> float:
        if leaf.get("ended_with") == "commit":
            return 1.0 if leaf.get("passed") else -1.0
        return (-1.0 if leaf.get("passed") else 0.0) if leaf.get("ended_with") == "abort" else -1.0

    if episode.get("tasks") is None:
        return single(b), single(c)
    return bundle(b), bundle(c)


def plan_group_step(episodes: list[dict], branch_credit: str = "outcome") -> list[dict]:
    """A group of code episodes (plan_outcome's records): group.group_step with the code read counterfactual
    (code_read_rewards; solvable: any branch of the group passed) and gate rule (code_gate)."""
    solvable = any(b.get("passed") for e in episodes for b in e.get("branches") or [])
    return group_step(episodes, read_rewards=lambda e, c, b: code_read_rewards(e, c, b, solvable),
                      branch_credit=branch_credit, gate=code_gate)


def plan_outcome(trial: Path, requests: list[dict]) -> dict:
    """The code environment's episode record: episode_record with the SWE verifier's grades (verifier/reward.txt: y;
    branches.json: each branch's and read control's passed, the control's), plus what code_read_rewards and
    plan_group_stats read: per branch its task, accepted and (a bundle) whether its task is solved in
    the final tree; per read control accepted; a bundle's
    per-task vector (the verifier wrote tasks.json: y = solved fraction); a control from a branch's fork
    (branch_control, graded, never trained); check validity (checks, own_valid), recorded only. Raises when the
    verifier did not grade every branch, read control and the control."""
    tasks_f = trial / "verifier/tasks.json"
    tasks = json.loads(tasks_f.read_text()) if tasks_f.exists() else None
    text = (trial / "verifier/reward.txt").read_text().strip()
    y = float(text) if tasks is not None else text == "1"
    tree = json.loads((trial / "agent/tree.json").read_text()) if (trial / "agent/tree.json").exists() else {}
    forks = tree.get("forks") or []
    has_control = any(r.get("thread") == "control" for r in requests)
    vf = trial / "verifier/branches.json"
    body = json.loads(vf.read_text()) if vf.exists() else {}
    rows = {b["key"]: b for b in body.get("branches") or []}
    nested = rows.pop("control", None)  # a nested control's patch, graded like a branch (plan.tsv row "control")
    readctl = {k: rows.pop(k) for k in [k for k in rows if k.startswith("readctl-")]}  # read controls (likewise)
    task_of = {key: b.get("task") for key, b in rows.items()}
    nodes = list(tree_branches(forks))  # every node of the fork tree but the root (main), at any depth
    keys = [branch_key(name) for name, _, _ in nodes]
    graded_control = bool(body.get("control") or nested)
    ran_rc = {branch_key(f"readctl/{c['of']}"): c for c in tree.get("read_controls") or [] if "skipped" not in c}
    if sorted(keys) != sorted(rows) or has_control != graded_control or sorted(ran_rc) != sorted(readctl):
        raise RuntimeError(f"plan verifier graded {sorted(rows)} control={graded_control} {sorted(readctl)}, "
                           f"episode has {sorted(keys)} control={has_control} {sorted(ran_rc)}: {vf}")
    ctrl = body.get("control") or {}  # a bundle's control: its tree's solved fraction, like y
    control_y = sum(ctrl["tasks"].values()) / len(ctrl["tasks"]) if "tasks" in ctrl else bool(ctrl.get("passed"))
    grades = {k: b.get("passed", False) for k, b in {**rows, **readctl}.items()}
    out = episode_record(trial, tree, requests, y, grades, control_y if has_control and not nested else None)
    cf = trial / "verifier/checks.json"
    ran = json.loads(cf.read_text()) if cf.exists() else {}
    valid = lambda r: r.get("fix_applied") and r.get("bug") != "0" and r.get("fix") == "0"  # noqa: E731
    by_fork, own_valid = collections.defaultdict(list), []
    for row, key, (name, fork, b) in zip(out["branches"], keys, nodes):
        if fork.get("response_id"):  # the forking reply's checks, in task order (None: not run)
            r = ran.get(f"{key}:fork")
            by_fork[fork["response_id"]].append(None if r is None else 1.0 if valid(r) else 0.0)
        if f"{key}:own" in ran:
            own_valid.append(bool(valid(ran[f"{key}:own"])))
        row.update(accepted=b.get("accepted"), task=task_of[key], merge=b.get("merge"), diffstat=b.get("diffstat"),
                   landed=bool(tasks.get(task_of[key])) if tasks is not None and task_of[key] else None)
    for row, c in zip(out["read_controls"], ran_rc.values()):
        row.update(accepted=c.get("accepted"))
    out.update(tasks=tasks, branch_control={"of": branch_key(tree["control"]["path"].rpartition("/")[0]),
                                            "passed": bool(nested.get("passed"))} if nested else None,
               # check validity (bundle verifier, reference fixes), recorded only: fork reply id -> 1 / 0 per task, a
               # fork whose checks all ran; each branch's own check
               checks={rid: ls for rid, ls in by_fork.items() if None not in ls}, own_valid=own_valid)
    return out
