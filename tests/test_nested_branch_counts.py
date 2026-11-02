"""The analysis scripts count branches at every depth of tree.json's fork tree (a branch's own forks in its "forks"):
dev_eval's usage, scripts/alpha/search_audit.py, scripts/alpha/pc_stats.py and scripts/rl_analysis/steps.py."""
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parents[1]
NESTED = {"branch_id": "branch-0.0", "ended_with": "context", "merge": "conflict: f.py",
          "rewinds": [{"note": "nn", "messages": [{"role": "assistant"}]}]}
FORKS = [{"branches": [{"branch_id": "branch-0", "ended_with": "done", "merge": "merged",
                        "rewinds": [{"note": "n", "messages": []}], "forks": [{"branches": [NESTED]}]}]}]


def _trial(path: Path) -> Path:
    (path / "agent").mkdir(parents=True)
    (path / "agent/tree.json").write_text(json.dumps({"main": [], "forks": FORKS, "main_end": "done"}))
    (path / "agent/requests.jsonl").write_text(json.dumps({"id": "main#1", "thread": "main"}) + "\n")
    (path / "result.json").write_text("{}")
    return path


def _script(*args) -> str:
    """A script as its callers run it (search_step3_check.sh, pc_orch.sh): plain python, no PYTHONPATH, elsewhere."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, check=True, env=env,
                          cwd=args[-1].parent).stdout


def test_dev_eval_usage_counts_nested_forks_and_branches(monkeypatch, tmp_path):
    config = types.ModuleType("octopus.rl.config")  # AReaL's run config, not needed for usage
    for name in ("add_run_args", "eval_config", "plan_row", "workflow_kwargs"):
        setattr(config, name, None)
    monkeypatch.setitem(sys.modules, "octopus.rl.config", config)
    monkeypatch.setitem(sys.modules, "octopus.eval.dev_eval", None)
    del sys.modules["octopus.eval.dev_eval"]  # imported on the stand-in, removed again after the test
    from octopus.eval.dev_eval import usage
    row = usage(str(_trial(tmp_path / "t")))
    assert (row["forks"], row["branches"]) == (2, 2)


def test_search_audit_counts_nested_branches(tmp_path):
    _trial(tmp_path / "search-x/t")
    out = _script(ROOT / "scripts/alpha/search_audit.py", tmp_path)
    assert "  branches 2\n" in out and "  branch_end=context 1\n" in out and "  rewind_calls 2\n" in out


def test_pc_stats_counts_nested_branch_ends(tmp_path):
    rows = tmp_path / "a.jsonl"
    rows.write_text(json.dumps({"task_dir": "d", "sample": 0, "y": 1, "trial": str(_trial(tmp_path / "t")),
                                "control": {"kind": "main-first", "y": 0}}))
    assert "{'done': 1, 'context': 1}" in _script(ROOT / "scripts/alpha/pc_stats.py", rows, rows)


def test_steps_counts_nested_merges(monkeypatch, tmp_path):
    _trial(tmp_path / "jobs/job/t")
    monkeypatch.setattr("octopus.envs.swe.reward.plan_outcome", lambda trial, reqs: {
        "y": 1, "forked": True, "branches": [], "control": None, "checks": {}, "own_valid": []})
    (tmp_path / "main.log").write_text("")
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(sys, "argv", ["steps.py", str(tmp_path / "jobs"), str(tmp_path / "main.log"), str(cache)])
    runpy.run_path(str(ROOT / "scripts/rl_analysis/steps.py"), run_name="__main__")
    row = json.loads(cache.read_text())
    assert (row["merged"], row["conflict"]) == (1, 1)
