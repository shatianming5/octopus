"""The repository tools of the SWE environment as they run in the container: GrokHarborAgent copies this file to
/tmp/harbor_tools.py and runs each call there (main), so it imports nothing of octopus. Their names and parameters
(the v2 native catalog: reproduce, rerun_reproduction, edit, shell, view, write, test, done) are
octopus/envs/swe/prompt.py's TOOLS.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path


_LIMIT = 4000


def _clip(text: str) -> str:
    if len(text) <= _LIMIT:
        return text
    return text[:2000] + "\n...[clipped]...\n" + text[-1500:]


def _inside(root: Path, raw: str) -> Path:
    root = root.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError("path escapes the repository")
    if ".git" in path.relative_to(root).parts:
        raise ValueError("path is inside .git")
    return path


def _python() -> str:
    conda = Path("/opt/miniconda3/envs/testbed/bin/python")
    return str(conda) if conda.is_file() else "python3"


def _run(argv: list[str], cwd: Path, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        return "exit timeout\n" + (_clip(text) or "(no output)")
    text = (result.stdout or "") + (result.stderr or "")
    return f"exit {result.returncode}\n" + (_clip(text) or "(no output)")


def _view_range(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if (not isinstance(raw, list) or len(raw) != 2
            or any(type(item) is not int for item in raw)):
        raise ValueError("view_range must be two integers")
    return raw


def _list_top(root: Path) -> str:
    names = []
    for item in sorted(root.iterdir()):
        if item.name.startswith("."):
            continue
        names.append(item.name + ("/" if item.is_dir() else ""))
        if len(names) == 40:
            break
    return "\n".join(names) or "(empty)"


def _view(root: Path, args: dict) -> str:
    raw = str(args.get("path") or ".")
    tasks = raw.startswith("/tasks")  # a bundle's task statements
    path = _inside(Path("/tasks"), raw) if tasks else _inside(root, raw)
    if not path.exists():
        rel = path if tasks else path.relative_to(root)
        return f"missing {rel}\n/testbed contains:\n" + _list_top(root)
    if path.is_dir():
        names = []
        for item in sorted(path.iterdir()):
            if item.name.startswith("."):
                continue
            names.append(item.name + ("/" if item.is_dir() else ""))
            if len(names) == 200:
                break
        return "\n".join(names) or "(empty directory)"
    lines = path.read_text(errors="replace").splitlines()
    start, end = 1, min(len(lines), 200) if lines else 0
    interval = _view_range(args.get("view_range"))
    if interval is not None:
        start, end = interval
        if start < 1:
            raise ValueError("view_range start must be positive")
        if end == -1:
            end = min(len(lines), start + 199)
        elif end < start:
            raise ValueError("view_range end is before start")
        else:
            end = min(end, start + 199)
    shown = [
        f"{number}:{lines[number - 1]}"
        for number in range(start, min(end, len(lines)) + 1)
    ]
    return _clip("\n".join(shown) or "(no lines)")


def _edit(root: Path, args: dict) -> str:
    path = _inside(root, str(args.get("path") or ""))
    old, new = args.get("old"), args.get("new")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise ValueError("edit needs path, nonempty old, and new")
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise ValueError(f"edit old text must match exactly once (found {count})")
    path.write_text(text.replace(old, new, 1))
    rel = path.relative_to(root) if path != root else Path(".")
    return f"replaced 1 occurrence in {rel}"


def _write(root: Path, args: dict) -> str:
    path = _inside(root, str(args.get("path") or ""))
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("write needs path and content")
    if path.exists():
        raise ValueError("write creates a new file only")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    rel = path.relative_to(root) if path != root else Path(".")
    return f"created {rel} ({len(content)} bytes)"


def apply(root, name: str, args: dict | None, state: dict | None = None):
    """Run one repository tool. Returns observation text, next state, and whether to stop."""
    root = Path(root)
    state = dict(state or {})
    args = dict(args or {})
    try:
        if name == "done":
            return "done", state, True
        if name == "view":
            return _view(root, args), state, False
        if name == "edit":
            return _edit(root, args), state, False
        if name == "write":
            return _write(root, args), state, False
        if name == "shell":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return "shell needs a command", state, False
            return _run(["bash", "-lc", command], root), state, False
        if name == "test":
            argv = args.get("args")
            if (not isinstance(argv, list) or not argv
                    or any(not isinstance(item, str) for item in argv)):
                return "test needs args, a nonempty list of strings", state, False
            return _run(list(argv), root), state, False
        if name == "reproduce":
            code, hypothesis = args.get("code"), args.get("hypothesis")
            if (not isinstance(code, str) or not code.strip()
                    or not isinstance(hypothesis, str) or not hypothesis.strip()):
                return "reproduce needs hypothesis and code", state, False
            state["repro_code"] = code
            state["hypothesis"] = hypothesis
            return _run([_python(), "-c", code], root), state, False
        if name == "rerun_reproduction":
            code = state.get("repro_code")
            if not isinstance(code, str) or not code:
                return "Define and execute a reproducer first.", state, False
            return _run([_python(), "-c", code], root), state, False
        return f"unknown tool {name}", state, False
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"{type(exc).__name__}: {exc}", state, False


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    payload = json.loads(base64.b64decode(argv[1]))
    root = Path(argv[2] if len(argv) > 2 else "/testbed")
    observation, state, stop = apply(
        root, payload["name"], payload.get("args") or {}, payload.get("state") or {},
    )
    sys.stdout.write(json.dumps(
        {"observation": observation, "state": state, "stop": stop},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
