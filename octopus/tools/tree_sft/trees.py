"""Turn Harbor `agent/tree.json` logs into fork trees for branch-aware SFT.

A tree is one fork of one trajectory:

    root      main messages every branch resumed from (the checkpoint freeze); its assistant turns
              (pre-fork exploration) are trained only in the trajectory's first tree
              ("train_root"), since later trees' roots repeat main turns an earlier join trained
    children  "join":   main thread from the freeze up to the next fork's freeze (or the end): the
                        checkpoint/fork decision, the fork result, the join note, and the turns after it
              "branch": one per cloned branch: its task (user) message and its own turns

Every assistant turn of a trajectory is trained once: pre-fork exploration in the first tree's root,
the fork decision, join and main-thread fix in the join child, and each branch's execution in its
branch child. (Main turns between a skipped single-branch fork and the next kept fork stay context.)
"""

from __future__ import annotations

import json
from pathlib import Path


def fork_trees(tree: dict, min_branches: int = 2) -> list[dict]:
    main, forks = tree["main"], tree["forks"]
    out = []
    for i, fork in enumerate(forks):
        branches = [b for b in fork["branches"] if b["messages"]]
        if len(branches) < min_branches:
            continue
        start = fork["prefix_len"]
        end = forks[i + 1]["prefix_len"] if i + 1 < len(forks) else len(main)
        join = main[start:end]
        if not any(m.get("role") == "assistant" for m in join):
            continue
        out.append({
            "fork_no": i,
            "tools": tree["tools"],
            "root": main[:start],
            "train_root": not out,
            "children": [{"kind": "join", "messages": join}]
            + [{"kind": "branch", "branch_id": b["branch_id"], "messages": b["messages"]} for b in branches],
        })
    return out


def collect(trial_dirs: list[Path], min_reward: float | None = 1.0) -> list[dict]:
    """Trees from trials whose verifier reward is at least `min_reward` (None keeps every trial)."""
    trees = []
    for trial in trial_dirs:
        log = trial / "agent" / "tree.json"
        result = trial / "result.json"
        if not log.is_file() or not result.is_file():
            continue
        res = json.loads(result.read_text())
        reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if min_reward is not None and (reward is None or reward < min_reward):
            continue
        task = Path(res["config"]["task"]["path"]).name
        for t in fork_trees(json.loads(log.read_text())):
            t.update({"task": task, "trial": trial.name, "reward": reward})
            trees.append(t)
    return trees
