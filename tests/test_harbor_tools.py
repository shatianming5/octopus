"""The Harbor agent offers the v2 native tools together, and those tools run."""

import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from octopus.envs.swe.prompt import DELIVERY, TOOLS
from octopus.envs.swe.container_tools import apply
from octopus.runtime.core import post_chat


def test_native_catalog_is_offered_together():
    names = [item["function"]["name"] for item in TOOLS]
    assert names == [
        "reproduce", "rerun_reproduction", "edit", "shell",
        "view", "write", "test", "done",
    ]
    assert all(item["type"] == "function" for item in TOOLS)


def test_edit_view_write_and_done(tmp_path: Path):
    target = tmp_path / "port.py"
    target.write_text("scheme = 'http'\ndefault_port = 80\n")
    viewed, _, stop = apply(tmp_path, "view", {"path": "port.py"}, {})
    assert "default_port = 80" in viewed and stop is False
    edited, _, stop = apply(tmp_path, "edit", {
        "path": "port.py", "old": "default_port = 80", "new": "default_port = 443",
    }, {})
    assert "port.py" in edited and "1 occurrence" in edited and stop is False
    assert "default_port = 443" in target.read_text()
    target.write_text("default_port = 443\ndefault_port = 443\n")
    rejected, _, _ = apply(tmp_path, "edit", {
        "path": "port.py", "old": "default_port = 443", "new": "default_port = 80",
    }, {})
    assert "exactly once" in rejected
    created, _, _ = apply(tmp_path, "write", {"path": "notes.txt", "content": "hi\n"}, {})
    assert "notes.txt" in created and "3 bytes" in created
    assert (tmp_path / "notes.txt").read_text() == "hi\n"
    text, _, stop = apply(tmp_path, "done", {}, {})
    assert text == "done" and stop is True


def test_reproduce_rerun_shell_and_escape(tmp_path: Path):
    seen, state, stop = apply(tmp_path, "reproduce", {
        "hypothesis": "the port follows the scheme",
        "code": "print(80)",
    }, {})
    assert seen.startswith("exit 0") and seen.strip().endswith("80") and stop is False
    again, _, _ = apply(tmp_path, "rerun_reproduction", {}, state)
    assert again.startswith("exit 0") and again.strip().endswith("80")
    missing, _, _ = apply(tmp_path, "rerun_reproduction", {}, {})
    assert "reproducer" in missing
    echoed, _, _ = apply(tmp_path, "shell", {"command": "printf ready"}, {})
    assert echoed.startswith("exit 0") and echoed.strip().endswith("ready")
    escaped, _, _ = apply(tmp_path, "view", {"path": "../outside"}, {})
    assert "escapes" in escaped
    (tmp_path / "furl").mkdir()
    (tmp_path / "furl" / "furl.py").write_text("ok\n")
    missing, _, _ = apply(tmp_path, "view", {"path": "furl.py"}, {})
    assert "missing furl.py" in missing and "furl/" in missing


def test_view_clips_long_files(tmp_path: Path):
    target = tmp_path / "big.py"
    target.write_text("\n".join("x" * 80 for _ in range(250)))
    viewed, _, _ = apply(tmp_path, "view", {"path": "big.py"}, {})
    assert "...[clipped]..." in viewed
    assert len(viewed) < 4000


@pytest.mark.parametrize("code", [403, 504])
def test_post_chat_retries_transient_code_then_succeeds(monkeypatch, code):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "http://x", code, "transient", None, io.BytesIO(b'{"error":"transient"}'),
            )
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    assert post_chat("http://x", {"a": 1}, {"Content-Type": "application/json"}) == {"ok": True}
    assert calls["n"] == 2


def test_delivery_contract_and_tool_meanings():
    assert "落盘" in DELIVERY
    assert "view path=." in DELIVERY
    assert "checkpoint" not in DELIVERY and "fork" in DELIVERY
    assert "commit" in DELIVERY and "abort" in DELIVERY
    assert "所有工具从一开始就可用" in DELIVERY
    assert "全部收口后才 done" in DELIVERY
    assert "所有工具始终可用" not in DELIVERY
    assert "不必 fork" not in DELIVERY
    catalog = Path("docs/harbor-collection.md").read_text()
    for name in (
        "reproduce", "edit", "shell", "view", "done",
        "fork", "commit", "abort",
    ):
        assert name in catalog
    assert "DEFAULT_PORTS" not in DELIVERY
    assert "scheme setter" not in DELIVERY.lower()
    text = " ".join(item["function"]["description"] for item in TOOLS)
    assert "does not mean the product bug is fixed" in text
    assert "valid product patch is a complete fix" in text
    assert "/testbed" in text


def test_post_chat_stops_after_six_403s(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "http://x", 403, "forbidden", None, io.BytesIO(b'{"error":"forbidden"}'),
        )

    events = []
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    try:
        post_chat("http://x", {}, {"Authorization": "secret"}, log=events.append)
    except RuntimeError as exc:
        assert "HTTP 403" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert calls["n"] == 6
    assert len(events) == 6
    assert all("secret" not in json.dumps(event) for event in events)
    assert all(event["http_code"] == 403 for event in events)


def test_post_chat_does_not_retry_400(monkeypatch):
    def fake_urlopen(request, timeout=0):
        raise urllib.error.HTTPError(
            "http://x", 400, "bad", None, io.BytesIO(b'{"error":"bad"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        post_chat("http://x", {}, {})
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "bad" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_post_chat_retries_dropped_connection_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    assert post_chat("http://x", {}, {}) == {"ok": True}
    assert calls["n"] == 3


def test_post_chat_timeout_comes_from_env(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout=0):
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OCTOPUS_HTTP_TIMEOUT", "900")
    post_chat("http://x", {}, {})
    assert seen["timeout"] == 900


def test_branch_fix_strategy_switches_prompt_and_messages(monkeypatch):
    from octopus.envs.swe import prompt as rt
    from octopus.runtime.core import branch_task_message, limits_note
    from test_rl_plan import _import_agent
    ws = _import_agent(monkeypatch).HarborWorkspace()

    def told():  # a branch's task message with the SWE provider's parts
        return branch_task_message("branch-0", "t", 2, close=ws.branch_close, reason=ws.abort_reason,
                                   rules=ws.branch_rules)

    monkeypatch.delenv("OCTOPUS_STRATEGY", raising=False)
    assert "complete fix attempt" not in told()
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    assert rt.delivery() == rt.DELIVERY + "\n" + limits_note(True, calls=False)  # no OCTOPUS_MAIN_MAX_CALLS
    assert "git apply --3way" in rt.DELIVERY
    assert "complete fix attempt" in told()
    note = rt.join_note([{"branch_id": "branch-0", "task": "t", "steps": [{"name": "commit", "arguments": {"summary": "s"}}]}])
    assert "Merge passing branch patches" in note and "each of them when they fix different problems" in note


def test_branch_fix_fork_tasks_carry_acceptance_and_commit_reports_it(monkeypatch):
    from octopus.envs.swe import prompt as rt
    from octopus.runtime.core import RUNTIME_TOOLS, RuntimePort
    from test_rl_plan import _import_agent
    apply_runtime = _import_agent(monkeypatch).HarborWorkspace().runtime

    monkeypatch.delenv("OCTOPUS_STRATEGY", raising=False)
    assert rt.runtime_tools() is RUNTIME_TOOLS  # default strategy: tools unchanged
    monkeypatch.setenv("OCTOPUS_STRATEGY", "branch_fix")
    fork = next(t for t in rt.runtime_tools() if t["function"]["name"] == "fork")
    assert fork["function"]["parameters"]["properties"]["tasks"]["items"]["required"] == ["task", "acceptance"]

    port = RuntimePort()
    tasks = [{"task": "fix the parser", "acceptance": "python -c 'import m; assert m.f() == 1'"},
             {"task": "fix the cache", "acceptance": "pytest -q tests/test_cache.py"}]
    apply_runtime("fork", {"tasks": tasks}, port)
    text = port.last_fork["branches"][0]["task"]
    assert text.startswith("fix the parser\n") and rt.acceptance_of(text) == "python -c 'import m; assert m.f() == 1'"
    assert rt.acceptance_of("plain task") is None
    missing = apply_runtime("fork", {"tasks": [{"task": "a"}, {"task": "b", "acceptance": "true"}]}, RuntimePort())[0]
    assert missing.startswith("fork needs at least two tasks, each with task and acceptance")

    assert "acceptance_passed" in apply_runtime("commit", {"branch_id": "branch-0", "summary": "s"}, port)[0]
    note = rt.join_note([{"branch_id": "branch-0", "task": text, "steps": [
        {"name": "commit", "arguments": {"summary": "fixed m.f", "acceptance_passed": True}}]}])
    assert "committed (acceptance passed, as reported): fixed m.f" in note


def test_fork_modes_offer_tools_and_instructions(monkeypatch):
    from octopus.envs.swe.prompt import DELIVERY_LINEAR, delivery, offered_tools
    from octopus.runtime.core import limits_note

    def names():
        return [item["function"]["name"] for item in offered_tools()]

    def notice(name):
        return "after a successful fork" in next(t for t in offered_tools() if t["function"]["name"] == name)["function"]["description"]

    assert {"edit", "shell", "done", "fork", "commit", "abort"} <= set(names())
    assert not notice("edit") and notice("commit") and delivery() == DELIVERY + "\n" + limits_note(True, calls=False)

    from octopus.runtime.core import unlocked_tools  # the no-fork control's: no fork tools at all
    linear = {t["function"]["name"]: t["function"]["description"] for t in unlocked_tools(TOOLS, True)}
    assert "edit" in linear and "fork" not in linear and "commit" not in linear
    assert "after a successful fork" not in linear["edit"]
    assert delivery(linear=True) == DELIVERY_LINEAR + "\n" + limits_note(False, calls=False)
