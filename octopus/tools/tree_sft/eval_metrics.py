"""Branch-behaviour metrics for Harbor runs of a student (or teacher) model, from agent/tree.json.

usage: python -m octopus.tools.tree_sft.eval_metrics --dataset DATASET_DIR --job NAME=JOB_DIR [--job ...]

Per model (job):
  success          share of trials with reward 1
  fork_rate        share of trials with at least one fork into >= 2 branches
  branch_close     share of branches that ended with commit/abort (not the step cap or silence)
  branch_overlap   mean Jaccard overlap of sibling branches' actions (lower = cleaner split)
  bug_coverage     share of forks where some branch viewed/edited a file the reference fix touches
  join_redo        share of the main thread's post-join file views that re-open a file a branch viewed
  merge_ok         share of main-thread `git apply` of branch patches that succeeded
File-level proxies (reference fix files from solution/fix.patch; no extra test runs):
  branch_hit       share of branches that viewed/edited a reference-fix file
  branch_fix       share of editing branches whose edits touch a reference-fix file
  select_ok        forks where some branch hit a fix file and the main thread changed code after the
                   join: share whose post-join changes (edits + applied branch patches) touch a fix file
  select_branch    main-thread `git apply` of a branch patch: share taken from a branch that hit a fix file
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def _args(call: dict) -> dict:
    raw = call["function"].get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
        return out if isinstance(out, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _path(p: str) -> str:
    p = re.sub(r"^(/testbed|/tmp/branches/[^/]+)/?", "", str(p or "")).lstrip("./")
    return p or "."


def action(call: dict) -> str:
    name, a = call["function"]["name"], _args(call)
    if name in {"view", "edit", "write"}:
        return f"{name}:{_path(a.get('path'))}"
    if name == "shell":
        return f"shell:{' '.join(str(a.get('command', '')).split())[:200]}"
    if name == "test":
        return f"test:{' '.join(map(str, a.get('args') or []))[:200]}"
    if name == "reproduce":
        return f"reproduce:{hash(str(a.get('code', ''))) & 0xffffffff}"
    return name


def fix_files(dataset: Path, task: str) -> set[str]:
    patch = dataset / task / "solution" / "fix.patch"
    if not patch.is_file():
        return set()
    return {m.group(1) for m in re.finditer(r"^\+\+\+ b/(\S+)", patch.read_text(errors="replace"), re.M)}


def job_metrics(job: Path, dataset: Path) -> dict:
    s = defaultdict(float)
    for trial in sorted(p for p in job.iterdir() if p.is_dir()):
        res_f, tree_f = trial / "result.json", trial / "agent" / "tree.json"
        if not res_f.is_file():
            continue
        res = json.loads(res_f.read_text())
        task = Path(res["config"]["task"]["path"]).name
        reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        s["trials"] += 1
        s["success"] += reward == 1.0
        if not tree_f.is_file():
            continue
        tree = json.loads(tree_f.read_text())
        main, bug = tree["main"], fix_files(dataset, task)
        forks = [f for f in tree["forks"] if len(f["branches"]) >= 2]
        s["forked"] += bool(forks)
        for fi, f in enumerate(forks):
            s["forks"] += 1
            acts, touched, hit_patches, edits_by_patch = [], set(), set(), {}
            for b in f["branches"]:
                s["branches"] += 1
                s["branch_closed"] += b.get("ended_with") in {"commit", "abort"}
                calls = [c for m in b["messages"] for c in (m.get("tool_calls") or [])]
                acts.append({action(c) for c in calls})
                seen = {_path(_args(c).get("path")) for c in calls if c["function"]["name"] in {"view", "edit", "write"}}
                edited = {_path(_args(c).get("path")) for c in calls if c["function"]["name"] in {"edit", "write"}}
                touched |= seen
                if bug:
                    s["branches_with_bug_files"] += 1
                    s["branch_hit"] += bool(bug & seen)
                    if edited:
                        s["editing_branches"] += 1
                        s["branch_fix"] += bool(bug & edited)
                if b.get("patch"):
                    edits_by_patch[b["patch"]] = edited
                    if bug & seen:
                        hit_patches.add(b["patch"])
            # selection: what the main thread changed between this join and the next fork (or the end)
            end = forks[fi + 1]["prefix_len"] if fi + 1 < len(forks) else len(main)
            changed, applied = set(), []
            for m in main[f["join_index"]:end]:
                for c in m.get("tool_calls") or []:
                    name, a = c["function"]["name"], _args(c)
                    if name in {"edit", "write"}:
                        changed.add(_path(a.get("path")))
                    if name == "shell" and "git apply" in str(a.get("command", "")):
                        for p in re.findall(r"/tmp/branches/\S+?\.patch", str(a.get("command", ""))):
                            applied.append(p)
                            changed |= edits_by_patch.get(p, set())
            if bug and bug & touched and changed:
                s["selections"] += 1
                s["select_ok"] += bool(bug & changed)
            for p in applied:
                if p in edits_by_patch:
                    s["branch_applies"] += 1
                    s["select_branch"] += p in hit_patches
            for x, y in combinations(acts, 2):
                if x | y:
                    s["pairs"] += 1
                    s["overlap"] += len(x & y) / len(x | y)
            s["bug_covered"] += bool(bug & touched)
            s["forks_with_bug_files"] += bool(bug)
            for i, m in enumerate(main[f["join_index"]:], start=f["join_index"]):
                for c in m.get("tool_calls") or []:
                    name = c["function"]["name"]
                    if name == "view":
                        s["join_views"] += 1
                        s["join_redo"] += _path(_args(c).get("path")) in touched
                    if name == "shell" and "git apply" in str(_args(c).get("command", "")):
                        out = next((x.get("content", "") for x in main[i + 1:i + 4] if x.get("role") == "tool"), "")
                        s["merges"] += 1
                        s["merge_ok"] += str(out).startswith("exit 0")
    ratio = lambda a, b: round(s[a] / s[b], 3) if s[b] else None
    return {
        "trials": int(s["trials"]), "success": ratio("success", "trials"), "fork_rate": ratio("forked", "trials"),
        "forks": int(s["forks"]), "branch_close": ratio("branch_closed", "branches"),
        "branch_overlap": ratio("overlap", "pairs"), "bug_coverage": ratio("bug_covered", "forks_with_bug_files"),
        "join_redo": ratio("join_redo", "join_views"), "merge_ok": ratio("merge_ok", "merges"), "merges": int(s["merges"]),
        "branch_hit": ratio("branch_hit", "branches_with_bug_files"), "branch_fix": ratio("branch_fix", "editing_branches"),
        "editing_branches": int(s["editing_branches"]), "select_ok": ratio("select_ok", "selections"),
        "selections": int(s["selections"]), "select_branch": ratio("select_branch", "branch_applies"),
        "branch_applies": int(s["branch_applies"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--job", action="append", required=True, help="NAME=JOB_DIR")
    args = ap.parse_args()
    for spec in args.job:
        name, path = spec.split("=", 1)
        print(json.dumps({"model": name, **job_metrics(Path(path), Path(args.dataset))}))


if __name__ == "__main__":
    main()
