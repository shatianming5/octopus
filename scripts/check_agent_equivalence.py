"""Equivalence of GrokHarborAgent and the search agent against another revision (a refactor's base): scripted
episodes run through REF's octopus/ (benchmarks/ before the step-3 repackage, with the episode env under its old
names, OLD_ENV; git archive, in a temp dir) and this tree's must send the same chat requests
(system prompt, tool list, every message), run the same container commands (SWE) or retrieval calls (search) and leave
the same records (requests.jsonl, tree.json (main's rewinds too), trajectory.json, the plan inputs; search: answers.json,
result.json).

    python scripts/check_agent_equivalence.py [REF]   # default origin/main; exit 1 on any difference

A fake chat model answers from a script per thread; every HTTP and to_thread call runs inline, so the threads
interleave alike in both runs. The container is a real git repository: commands run in bash with /testbed and /tmp/
mapped into a temp dir (bundle task statements are canned). Episodes:
  swe      one task: cursor, read_thread, rewind past a merge, fork, auto merge, control
  bundle   three tasks: fork, main's refused abort, a branch reading its sibling, a conflicting patch, read control,
           control
  plain    one task, no strategy (string tasks), no read, thinking on (notes as tool results), main's commit, an abort
  search          read_thread, read control, main-first control
  search-noread   fork without read_thread: main's commit (no search tool), branches' done, no control
  search-linear   a single agent (OCTOPUS_LINEAR), no read
Episode inputs keep what revisions before the shared budget's and will_pass's removal need (a bundle instruction's
"budget of N tokens", each fork task's will_pass); later revisions ignore both.
Times (*_time and wall_s keys) are dropped from the records before comparing, and the source text of the container
tools uploaded to /tmp/harbor_tools.py from the commands (their tool results are compared). Harbor is stubbed (not installed here).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

HARBOR = ("harbor", "harbor.agents", "harbor.agents.base", "harbor.agents.capabilities", "harbor.environments",
          "harbor.environments.base", "harbor.models", "harbor.models.agent", "harbor.models.agent.context",
          "harbor.models.trajectories", "harbor.models.trajectories.agent", "harbor.models.trajectories.observation",
          "harbor.models.trajectories.observation_result", "harbor.models.trajectories.step",
          "harbor.models.trajectories.tool_call", "harbor.models.trajectories.trajectory")
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


def call(name, **args):
    return [{"id": f"{name}-call", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def commit(branch, summary):
    return call("commit", branch_id=branch, summary=summary, acceptance_passed=True)


def task(k, text, check):
    return {"task": text, "acceptance": check, "will_pass": True, **({"task_id": k} if k else {})}


EPISODES = {
    "swe": ({"OCTOPUS_REWIND_TOOL": "1", "OCTOPUS_AUTO_MERGE": "1"},
            "Fix the bug in f.py.", {
        "main": [call("shell", command="echo m > main.txt"),
                 call("fork", tasks=[task(None, "Fix f", "grep -q fixed f.py"), task(None, "Add g", "test -f g.py")]),
                 call("read_thread", thread="branch-0"), call("shell", command="cat f.py"),
                 call("rewind", to_step=1, note="redo"), call("done")],
        "fork0/branch-0": [call("shell", command="sed -i s/one/fixed/ f.py"), call("shell", command="true"),
                           commit("branch-0", "fixed f")],
        "fork0/branch-1": [call("shell", command="echo g > g.py"), call("shell", command="true"),
                           call("read_thread", thread="branch-0"), commit("branch-1", "g")],
        "control": [call("shell", command="echo c >> f.py"), call("done")]}),
    "bundle": ({"OCTOPUS_BUNDLE": "1", "OCTOPUS_READ_CONTROL": "1"},
               "Fix the bugs. You have a budget of 100000 tokens.", {
        "main": [call("fork", tasks=[task(1, "Task 1", "grep -q fixed f.py"), task(2, "Task 2", "test -f two.txt"),
                                     task(3, "Task 3", "grep -q other f.py")]),
                 call("abort", branch_id="branch-2", reason="not needed"), call("done")],
        "fork0/branch-0": [call("shell", command="sed -i s/one/fixed/ f.py"), call("shell", command="true"),
                           commit("branch-0", "one")],
        "fork0/branch-1": [call("shell", command="echo two > two.txt"), call("shell", command="true"),
                           call("read_thread", thread="branch-0"), commit("branch-1", "two")],
        "fork0/branch-2": [call("shell", command="sed -i s/one/other/ f.py"), call("shell", command="true"),
                           call("shell", command="true"), commit("branch-2", "other")],
        "control": [call("shell", command="echo c >> f.py"), call("done")],
        "readctl/fork0/branch-1": [commit("branch-1", "two, unread")]}),
    "plain": ({"OCTOPUS_STRATEGY": "", "OCTOPUS_READ_TOOL": "", "OCTOPUS_PLAN_EPISODE": "", "OCTOPUS_PLAN_CONTROL": "",
               "OCTOPUS_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": true}', "OCTOPUS_MAIN_MAX_CALLS": "9"},
              "Fix the bug in f.py.", {
        "main": [call("shell", command="echo m > main.txt"), call("fork", tasks=["Fix f", "Add g"]),
                 call("commit", branch_id="branch-0", summary="main's"), call("done")],
        "fork0/branch-0": [call("shell", command="sed -i s/one/fixed/ f.py"),
                           call("commit", branch_id="branch-0", summary="fixed f")],
        "fork0/branch-1": [call("shell", command="echo g > g.py"), call("abort", branch_id="branch-1")]}),
}
# search episodes (octopus.envs.search.agent.run_trial): env, question, script; the rollout worker's env has no SWE
# strategy, plan episode or bundle
SEARCH = {"OCTOPUS_STRATEGY": "", "OCTOPUS_PLAN_EPISODE": "", "OCTOPUS_SEARCH_URL": "http://search"}
SEARCH_EPISODES = {
    "search": ({"OCTOPUS_READ_CONTROL": "1"}, {"qid": "q1", "question": "Who wrote the first program?", "answer": "Ada"}, {
        "main": [call("search", query="first program"), call("fork", tasks=["Find the author", "Check the year"]),
                 call("done", answer="Ada")],
        "fork0/branch-0": [call("search", query="author"), call("open", docid="d1", offset=0),
                           call("done", answer="Ada", evidence="d1")],
        "fork0/branch-1": [call("search", query="year"), call("search", query="1843"),
                           call("read_thread", thread="branch-0"), call("done", answer="1843", evidence="d1")],
        "readctl/fork0/branch-1": [call("done", answer="1843, unread")],
        "control": [call("search", query="author"), call("done", answer="Ada")]}),
    "search-noread": ({"OCTOPUS_READ_TOOL": "", "OCTOPUS_PLAN_CONTROL": ""},
                      {"qid": "q2", "question": "Who wrote the first program?", "answer": "Ada"}, {
        "main": [call("fork", tasks=["Find the author", "Check the year"]),
                 call("commit", branch_id="branch-0", summary="main's"), call("done", answer="Babbage")],
        "fork0/branch-0": [call("search", query="author"), call("done", answer="Ada", evidence="d1")],
        "fork0/branch-1": [call("done", answer="1843")]}),
    "search-linear": ({"OCTOPUS_READ_TOOL": "", "OCTOPUS_PLAN_CONTROL": "", "OCTOPUS_LINEAR": "1"},
                      {"qid": "q3", "question": "Who wrote the first program?", "answer": "Ada"}, {
        "main": [call("search", query="first program"), call("open", docid="d1"), call("done", answer="Ada")]}),
}
# RL's episode env (octopus/envs/swe/rl.py, run_config.episode_env); OCTOPUS_FORK_OPTIONAL and OCTOPUS_MODEL: what
# revisions before #211 read for its forking mode and request model
COMMON = {"OCTOPUS_STRATEGY": "branch_fix", "OCTOPUS_FORK_OPTIONAL": "1", "OCTOPUS_PLAN_EPISODE": "1",
          "OCTOPUS_PLAN_CONTROL": "1", "OCTOPUS_CURSOR": "1", "OCTOPUS_READ_TOOL": "1", "OCTOPUS_HTTP_ATTEMPTS": "1",
          "OCTOPUS_MODEL": "policy", "OCTOPUS_CHAT_URL": "http://proxy/v1/chat/completions"}
# the names a revision before the step-3 repackage (its code in benchmarks/) reads these settings under
OLD_ENV = {"OCTOPUS_MAIN_MAX_CALLS": "GROK_HARBOR_MAX_STEPS", "OCTOPUS_CHAT_TEMPLATE_KWARGS": "GROK_CHAT_TEMPLATE_KWARGS",
           "OCTOPUS_HTTP_ATTEMPTS": "GROK_HTTP_ATTEMPTS", "OCTOPUS_MODEL": "GROK_MODEL", "OCTOPUS_CHAT_URL": "GROK_CHAT_URL"}
OLD_LAYOUT = False  # child(): the revision's code is benchmarks/


def episode_env(env: dict) -> dict:
    return {OLD_ENV.get(k, k) if OLD_LAYOUT else k: v for k, v in env.items()}
STUB_TEMPLATE = ('{%- if tools %}{{- "<tools>" }}{%- for tool in tools %}{{- "\\n" + tool | tojson }}{%- endfor %}'
                 '{{- "\\n</tools>\\n\\n" }}{%- endif %}{%- for m in messages %}{{- m.content }}{%- endfor %}')


class Inline(concurrent.futures.ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return plain(vars(x)) if hasattr(x, "__dict__") else x


def stub_harbor() -> None:
    class Stub:
        session_id = None

        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def to_json_dict(self):
            return plain(self)
    for name in HARBOR:
        module = types.ModuleType(name)
        for attr in ("BaseAgent", "AgentCapabilities", "BaseEnvironment", "AgentContext", "Agent", "Observation",
                     "ObservationResult", "Step", "ToolCall", "Trajectory"):
            setattr(module, attr, Stub)
        sys.modules[name] = module


class Container:
    """bash on a git repository; container paths under `root`, outputs mapped back."""

    def __init__(self, root: Path, commands: list):
        self.root, self.commands = str(root), commands

    def path(self, text: str) -> str:
        return re.sub(r"(?<![\w./+=-])(/testbed|/tmp/)", lambda m: self.root + m.group(1), text)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None):
        # the upload of the container tools is compared by what their calls return, not by their source text
        self.commands.append([re.sub(r"b64decode\('[A-Za-z0-9+/=]+'\)", "b64decode(<harbor_tools.py>)", command), cwd,
                              timeout_sec])
        if "glob.glob('/tasks/" in command:
            return SimpleNamespace(stdout=json.dumps({str(k): f"Statement of task {k}.\n" for k in (1, 2, 3)}),
                                   stderr="", return_code=0)
        r = subprocess.run(["bash", "-c", self.path(command)], cwd=self.path(cwd or "/"), capture_output=True,
                           text=True, env={**os.environ, **GIT_ENV})
        return SimpleNamespace(stdout=r.stdout.replace(self.root, ""), stderr=r.stderr.replace(self.root, ""),
                               return_code=r.returncode)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, self.path(target_dir), dirs_exist_ok=True)


def episode(mod, name: str, work: Path) -> dict:
    env, instruction, script = EPISODES[name]
    saved = dict(os.environ)
    os.environ.update(episode_env({**COMMON, **env}))
    root = work / "root"
    shutil.rmtree(work, ignore_errors=True)
    (root / "testbed").mkdir(parents=True)
    (root / "tmp").mkdir()
    for f, text in {"f.py": "one = 1\n", "test_f.py": "def test_f():\n    assert True\n"}.items():
        (root / "testbed" / f).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", "-C", str(root / "testbed"), *args], check=True, env={**os.environ, **GIT_ENV})
    bodies, commands, turns = [], [], {}
    agent = mod.GrokHarborAgent.__new__(mod.GrokHarborAgent)
    agent.logs_dir = work / "trial" / "agent"
    agent.logs_dir.mkdir(parents=True)
    agent.session_id, agent.model_name = "s", "policy"  # Harbor's -m (RL: -m policy)

    def post_chat(url, body, headers, log):
        running = getattr(agent, "_control_running", False)
        thread = body.get("session_id") or (running if isinstance(running, str) else "control" if running else "main")
        turn = turns[thread] = turns.get(thread, 0) + 1
        bodies.append(json.dumps(body, sort_keys=True))
        calls = (script.get(thread) or [])[turn - 1: turn]
        rid, used = f"{thread}#{turn}", {"prompt_tokens": 10 * turn, "completion_tokens": 5}
        log({"id": rid, "http_code": 200, **used})
        return {"id": rid, "usage": used, "choices": [{
            "message": {"content": "" if calls else "no tool call", "tool_calls": calls[0] if calls else []}}]}
    runtime = sys.modules[mod.GrokHarborAgent._complete.__module__]  # where the chat client lives in that revision
    runtime.post_chat, runtime._HTTP_POOL = post_chat, Inline()
    container = Container(root, commands)

    async def run():
        asyncio.get_running_loop().set_default_executor(Inline())
        await agent.setup(container)
        await agent.run(instruction, container, SimpleNamespace())
    error = None
    try:
        asyncio.run(run())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        os.environ.clear()
        os.environ.update(saved)
    files = {str(p.relative_to(work)): p.read_text() for p in sorted(work.rglob("*"))
             if p.is_file() and ("trial" in p.parts or "octopus-plan" in p.parts)}
    return {"error": error, "bodies": bodies, "commands": commands, "files": files}


def search_episode(smod, name: str, work: Path) -> dict:
    env, question, script = SEARCH_EPISODES[name]
    saved = dict(os.environ)
    os.environ.update(episode_env({**COMMON, **SEARCH, **env}))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bodies, calls, turns, held = [], [], {}, {}

    def post_chat(url, body, headers, log):
        running = getattr(held.get("agent"), "_control_running", False)
        thread = body.get("session_id") or (running if isinstance(running, str) else "control" if running else "main")
        turn = turns[thread] = turns.get(thread, 0) + 1
        bodies.append(json.dumps(body, sort_keys=True))
        reply = (script.get(thread) or [])[turn - 1: turn]
        rid, used = f"{thread}#{turn}", {"prompt_tokens": 10 * turn, "completion_tokens": 5}
        log({"id": rid, "http_code": 200, **used})
        return {"id": rid, "usage": used, "choices": [{
            "message": {"content": "" if reply else "no tool call", "tool_calls": reply[0] if reply else []}}]}

    def post_json(url, body, timeout=60):  # the retrieval service
        calls.append([url, body])
        if url.endswith("/search"):
            return {"results": [{"docid": "d1", "title": "Notes", "url": "u1", "snippet": "Ada wrote it in 1843."}]}
        return {"docid": body["docid"], "title": "Notes", "url": "u1", "offset": body["offset"],
                "text": "Ada Lovelace wrote it.", "total_length": 22}
    runtime = sys.modules[smod.SearchAgent._complete.__module__]
    runtime.post_chat, runtime._HTTP_POOL, smod.post_json = post_chat, Inline(), post_json
    run_agent = smod.SearchAgent.run

    async def run(self, *args):
        held["agent"] = self
        return await run_agent(self, *args)
    smod.SearchAgent.run = run

    async def main():
        asyncio.get_running_loop().set_default_executor(Inline())
        return await smod.run_trial(question, work)
    error = None
    try:
        asyncio.run(main())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        smod.SearchAgent.run = run_agent
        os.environ.clear()
        os.environ.update(saved)
    files = {str(p.relative_to(work)): p.read_text() for p in sorted(work.rglob("*")) if p.is_file()}
    return {"error": error, "bodies": bodies, "commands": calls, "files": files}


def untimed(x):
    if isinstance(x, dict):
        return {k: untimed(v) for k, v in x.items() if not k.endswith("_time") and k != "wall_s"}
    return [untimed(v) for v in x] if isinstance(x, list) else x


def normal(text: str, path: str) -> list[str]:
    if path.endswith(".json"):
        return json.dumps(untimed(json.loads(text)), indent=1, sort_keys=True).splitlines()
    return text.splitlines()


def child(code: str, out: str, work: str) -> None:
    global OLD_LAYOUT
    OLD_LAYOUT = not (Path(code) / "octopus").is_dir()
    sys.path.insert(0, code)
    for key in [k for k in os.environ if k.startswith(("OCTOPUS_", "GROK_", "SGLANG_", "HARBOR_"))]:
        del os.environ[key]
    tokenizer = Path(work) / "tokenizer"  # read_thread's card: a chat template showing tools as Qwen3.5's does
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "chat_template.jinja").write_text(STUB_TEMPLATE)
    COMMON["OCTOPUS_TOKENIZER"] = str(tokenizer)
    stub_harbor()
    import uuid
    uuid.uuid4 = lambda: SimpleNamespace(hex="0" * 32)  # search trial dirs are named by it
    import importlib
    mod, smod = map(importlib.import_module, ("benchmarks.grok_harbor_agent", "benchmarks.search.agent") if OLD_LAYOUT
                    else ("octopus.envs.swe.agent", "octopus.envs.search.agent"))
    assert mod.__file__.startswith(code) and smod.__file__.startswith(code), (mod.__file__, smod.__file__)
    runs = {name: episode(mod, name, Path(work) / name) for name in EPISODES}
    runs |= {name: search_episode(smod, name, Path(work) / name) for name in SEARCH_EPISODES}
    Path(out).write_text(json.dumps(runs))


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    here = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "old"
        old.mkdir()
        package = "octopus" if subprocess.run(["git", "-C", str(here), "cat-file", "-e", f"{ref}:octopus"],
                                              capture_output=True).returncode == 0 else "benchmarks"
        archive = subprocess.run(["git", "-C", str(here), "archive", ref, package], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(old)], input=archive, check=True)
        runs = {}
        for side, code in (("old", old), ("new", here)):
            out = Path(tmp) / f"{side}.json"
            subprocess.run([sys.executable, __file__, "--child", str(code), str(out), str(Path(tmp) / "work")],
                           check=True, cwd=tmp)
            runs[side] = json.loads(out.read_text())
    bad = 0
    for name in runs["old"]:
        a, b = runs["old"][name], runs["new"][name]
        print(f"{name}: {len(a['bodies'])} requests, {len(a['commands'])} commands, {len(a['files'])} files, "
              f"error {a['error']}")
        pairs = [("error", [str(a["error"])], [str(b["error"])]), ("bodies", a["bodies"], b["bodies"]),
                 ("commands", [json.dumps(c) for c in a["commands"]], [json.dumps(c) for c in b["commands"]]),
                 ("files", sorted(a["files"]), sorted(b["files"]))]
        pairs += [(f, normal(a["files"][f], f), normal(b["files"].get(f, ""), f)) for f in sorted(a["files"])]
        for what, x, y in pairs:
            if x != y:
                bad += 1
                print(f"  DIFFERS: {what}")
                print("\n".join(list(difflib.unified_diff(x, y, "old", "new", lineterm="", n=1))[:40]))
    print("identical" if not bad else f"{bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:5])
    else:
        sys.exit(main())
