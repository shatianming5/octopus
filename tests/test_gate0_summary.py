"""scripts/eval/gate0.py on two hand-made arms of two bundles each."""
import json
import subprocess
import sys
from pathlib import Path


def _arm(tmp_path, name, merges, conflict_by=None, mounted=(), y=0.5, unforked=()):
    rows = []
    for k, merge in enumerate(merges):  # one episode per bundle: two committed branches, the second's merge varies, and
        # an uncommitted one an older run merged (no landing); an unforked one hit main's context limit
        agent = tmp_path / name / f"t{k}/agent"
        agent.mkdir(parents=True)
        if k in unforked:
            (agent / "tree.json").write_text(json.dumps({"main": [], "forks": [], "main_end": "context"}))
            (agent / "requests.jsonl").write_text("")
            rows.append({"task_dir": f"/d/bundle{k}", "sample": 0, "trial": str(agent.parent), "y": y})
            continue
        branches = [{"branch_id": "branch-0", "ended_with": "commit", "merge": "merged", "messages": []},
                    {"branch_id": "branch-2", "ended_with": None, "merge": "conflict: x", "messages": [],
                     "conflict_by": {"x": "unclear"}},
                    {"branch_id": "branch-1", "ended_with": "commit", "merge": merge, "messages": [{"role": "assistant", "tool_calls": [
                        {"function": {"name": "read_thread", "arguments": "{}"}}]}],
                     **({"conflict_by": conflict_by} if merge != "merged" and conflict_by else {})}]
        (agent / "tree.json").write_text(json.dumps({"main": [], "forks": [{"branches": branches}],
                                                     "read_event": [{"mounted": m} for m in mounted]}))
        (agent / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"id": "a", "mount": {"thread": "branch-0"}}, {"id": "b"}, {"http_code": 503, "mount": {"thread": "x"}}]))
        rows.append({"task_dir": f"/d/bundle{k}", "sample": 0, "trial": str(agent.parent), "y": y,
                     "branch_credit": [{"task": "1", "solved": True}, {"task": "2", "solved": merge == "merged"}]})
    rows.insert(0, {"task_dir": "/d/bundle1", "sample": 0, "error": "infra"})  # superseded by its retry
    rows.append({"task_dir": "/d/bundle0", "sample": 1, "error": "infra"})  # its retry is not in yet
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    return f"{name}={path}"


def test_gate0_summary_landed_split_and_paired_difference(tmp_path):
    off = _arm(tmp_path, "off", ["conflict: f.py", "merged"], {"f.py": "sibling", "g.py": "own"})
    event = _arm(tmp_path, "event", ["merged", "merged", "conflict: f.py"], mounted=(True, False), y=0.75,
                 unforked={1})  # bundle2: event only
    out = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "scripts/eval/gate0.py"), off, event,
                          "--boot", "200"], capture_output=True, text=True, check=True).stdout
    assert "off: 2 episodes, 2 bundles, 1 infra-error rows left out" in out
    assert "event: left out, not in every arm: bundle2" in out
    assert "verdict event - off: y +0.250 [+0.250, +0.250] over 2 bundles: PASS (CI lower end > 0)" in out
    assert "off: 2 episodes, 0 unforked\n" in out and "event: 2 episodes, 1 unforked (context 1)\n" in out
    lines = {line.split(":")[0].strip(): line for line in out.splitlines() if " landed " in line}
    assert "landed 0.750 " in lines["off"] and "pass 0.750 " in lines["off"] and "reads 1.000 " in lines["off"]
    assert "mounts 1.000 " in lines["off"] and "sibling 0.500 " in lines["off"] and "own 0.500 " in lines["off"]
    assert "commit 0.667 " in lines["off"] and "y_forked 0.500 " in lines["off"] and "fork 1.000 " in lines["off"]
    assert "landed 1.000 " in lines["event"] and "forced 0.500 " in lines["event"] and "pass 1.000 " in lines["event"]
    assert "fork 0.500 " in lines["event"] and "y_forked 0.750 " in lines["event"]
    assert "landed +0.250 [+0.250, +0.500]" in lines["event - off"]  # a bundle1-only resample: event has no landing
