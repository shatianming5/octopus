"""scripts/rl_analysis/read_audit.py on a synthetic run: tool results in main and a branch, mount rows, the read-call
spread, the train log's read-leaf lines and the proxy's engine errors (AReaL timestamps with colour codes,
traceback tails), and a local and a --remote SGLang log (through a port), each in its step; no episodes at all. A
mount's result after a join-any notice is still a valid mount."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "scripts/rl_analysis/read_audit.py"


def engine_errors() -> list[str]:
    """The engine's four read errors as raised (patches/areal-cursor.patch: each message spans the line naming it and the
    next), with x for every field."""
    lines = (ROOT / "patches/areal-cursor.patch").read_text().splitlines()
    out = []
    for start in ("reads another thread but cannot continue", "mounts cursor {src.cursor}",
                  "or a cursor it mounts is gone from", "ignored the mount of"):
        i = next(k for k, line in enumerate(lines) if start in line)
        text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', lines[i] + lines[i + 1]))
        out.append(re.sub(r"\{[^}]*\}", "x", text))
    return out


def test_read_audit_counts_each_kind_in_its_step(tmp_path):
    trial = tmp_path / "jobs/rl-x:0-0/rl-x__abc"
    (trial / "agent").mkdir(parents=True)
    call = lambda i: {"id": f"c{i}", "function": {"name": "read_thread", "arguments": "{}"}}  # noqa: E731
    tree = {"messages": [
        {"role": "assistant", "tool_calls": [call(1)]},
        {"role": "tool", "tool_call_id": "c1", "content": "[token budget left: 9 | step 3]\n[branch-1 finished: commit]"
                                                           "\nMounted branch-0's steps 1 to 2 (191 tokens; branch-0 has "
                                                           "readable steps 1 to 2): ..."},  # after a join-any notice
        {"role": "assistant", "tool_calls": [call(2)]},
        {"role": "tool", "tool_call_id": "c2", "content": "[main | step 4]\nYou are main: read_thread reads another "
                                                           "thread, never your own. There is no other thread yet: fork "
                                                           "first. Nothing ..."}]}  # OCTOPUS_READ_TOOL's named tag
    branch = tree["messages"][:2] + [  # inherited history: counted once; the branch's own calls count too
        {"role": "assistant", "tool_calls": [call(3), {"id": "c4", "function": {"name": "view", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "No thread branch-9: ..."},
        {"role": "assistant", "tool_calls": [call(5)]},  # search's ambiguous bare id
        {"role": "tool", "tool_call_id": "c5", "content": "branch-0 is ambiguous: fork0/branch-0, fork1/branch-0 have "
                                                           "it; name one of them. Nothing was read."}]
    tree.update(forks=[{"branches": [{"messages": branch}]}], main_end="call budget")  # 4 of 5 calls read: a loop
    (trial / "agent/tree.json").write_text(json.dumps(tree))
    other = tmp_path / "jobs/rl-x:0-1/rl-x__def"  # a call-budget end without reads: no loop
    (other / "agent").mkdir(parents=True)
    (other / "agent/tree.json").write_text(json.dumps({"main_end": "call budget", "messages": [
        {"role": "assistant", "tool_calls": [{"id": "v1", "function": {"name": "view", "arguments": "{}"}}]}]}))
    rows = [{"mount": {"from": "a"}, "http_code": 200, "elapsed_s": 3.0}, {"http_code": 200, "elapsed_s": 412.5},
            {"http_code": None, "error": "TimeoutError: timed out", "elapsed_s": 900.1}]  # a long reply, a timeout
    (trial / "agent/requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0.5}}}))
    (other / "result.json").write_text("{}")
    at = lambda s: datetime.strptime(f"20261018-{s}", "%Y%m%d-%H:%M:%S").timestamp()  # noqa: E731
    for t in (trial, other):
        os.utime(t / "result.json", (at("10:00:00"),) * 2)  # before step 1
    train = tmp_path / "train.log"
    train.write_text(
        "[octopus] read leaves 5 with reads 1\n"
        "\x1b[92m(AReaL)\x1b[0m \x1b[92m20261018-10:30:00.100 StatsLogger INFO: Epoch 1/5 Step 1/9 Train step 1/9 done.\x1b[0m\n"
        "\x1b[91m(AReaL) 20261018-10:31:00.000 ProxyRolloutServer ERROR: AReaL client request failed\x1b[0m\n"
        + "".join(f"Traceback (most recent call last):\nRuntimeError: {e}\n" for e in engine_errors()) +
        "[octopus] read leaves 4 with reads 0\n"
        "(AReaL) 20261018-11:00:00.000 StatsLogger INFO: Epoch 1/5 Step 2/9 Train step 2/9 done.\n")
    sglang = tmp_path / "sglang-g0.log"
    sglang.write_text('[2026-10-18 10:10:00] INFO:     127.0.0.1:1 - "POST /generate HTTP/1.1" 404 Not Found\n'
                      "[2026-10-18 10:20:00] octopus-mount: the KV of a mounted read is evicted, self.rid='r1'\n")
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "sglang-g3.log").write_text(2 * '[2026-10-18 10:25:00] INFO:     1.2.3.4:5 - "POST /generate HTTP/1.1" 503'
                                          ' x\n' + "Traceback (most recent call last):\nnoise\n")
    bin_ = tmp_path / "bin"  # ssh [OPTIONS] HOST CMD logs its arguments and runs CMD here
    bin_.mkdir()
    (bin_ / "ssh").write_text(f'#!/bin/bash\necho "${{@:1:$#-1}}" >> {tmp_path}/ssh.args\nexec bash -c "${{@: -1}}"\n')
    (bin_ / "ssh").chmod(0o755)
    run = lambda jobs, *more: subprocess.run(  # noqa: E731
        [sys.executable, str(AUDIT), str(jobs), str(train), str(sglang), *more], capture_output=True, text=True,
        check=True, env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}).stdout
    out = run(tmp_path / "jobs", "--remote", f"u@gpu-x:2244:{remote}/sglang-g*.log")
    assert (tmp_path / "ssh.args").read_text() == "-p 2244 u@gpu-x\n"  # one call, through the port
    step1, rest = out.split("step 2\n")
    step2, unstamped = rest.split("step ?\n")
    assert "'read_thread calls': 4" in step1 and "'valid': 1" in step1 and "'self': 1" in step1
    assert "'no-thread': 1" in step1 and "'ambiguous': 1" in step1 and "other: " not in step1
    spread = "read_thread calls per episode: median 2 mean 2.0 top-10 episodes 100% of 4"
    assert spread in step1 and f"all steps: {spread}" in unstamped
    assert "'generate 503': 2" in step1 and "'traceback': 1" in unstamped  # the remote log's counts
    assert "'(of these, no other thread yet)': 1" in step1 and "n 1 median 191" in step1
    assert "mount requests: {'sent': 1" in step1 and "'exported': 5, 'with reads': 1" in step1
    assert "'generate 404': 1" in step1 and "'engine: cannot hop': 0" in step1
    assert "'mount rebuilds': 1" in step1  # since slice 6 a recovered rebuild, not a failure
    assert "long waits: {'requests > 300 s': 2, 'requests timed out': 1, 'episodes with a timeout': 1}" in step1
    assert "mount requests: {'sent': 1, 'failed (http != 200)': 0}" in step1
    assert "'exported': 4, 'with reads': 0" in step2
    for k in ("read dropped", "other server", "cannot hop", "mount ignored"):  # each message counts once, in step 2
        assert f"'engine: {k}': 1" in step2, k
    (tmp_path / "empty").mkdir()  # log lines but no finished episode: no spread line, no error
    out = run(tmp_path / "empty")
    assert "'generate 404': 1" in out and "read_thread calls per episode" not in out and "all steps" not in out
