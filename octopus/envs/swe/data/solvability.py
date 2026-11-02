"""Statement solvability of bundle tasks: can the graded FAIL_TO_PASS tests be met from the task statement plus the
bug tree? The reverse of statement_coverage.py (is every part of the fix described): a test that asserts an exact
literal (an escape string, a repr, an error message) that neither the statement nor the tree gives, or a statement
that calls the bug's current behaviour the expected one, makes a task unsolvable no matter how good the agent is.

Per task, Grok lists each requirement the F2P tests put on the fix (statement, F2P test source, reference fix in the
prompt) and classes it: a = stated or implied; b = to be found in the repository; c = missing / contradicts /
ambiguous. Each b/c requirement comes with needles (exact source strings); they are grepped in the bundle's bug tree
(the healthcheck's git steps replayed on a shared clone of the image's golden repo), skipping the lines that the
bundle's reference fixes remove. Label = the worst requirement:
  contradicts   a "contradicts" requirement (a hit does not resolve it: in the pilot a contradiction's needles hit
                unrelated code, e.g. glom sf15 `target = {}`)
  unsolvable    a b or "missing" requirement whose needles are nowhere in the tree (found -> derivable)
  uncertain     an "ambiguous" requirement whose needles are nowhere in the tree
  solvable      otherwise

The judge only shortlists: every non-solvable task must be reviewed by hand (statement, test, fix, grep in the tree)
before it is used to drop anything. Its main false positives are needles spelled differently from the tree (the
requirement is already met by unchanged code or docs), its own miscalculations and misread tests. Coupled-a audit
(2026-10-21, 890 tasks): 53 flagged, 30 kept after review (23 overturned to solvable); on the controls precision 6/7,
recall 6/7; 12 bundles dropped from the dataset afterwards.

usage: python -m octopus.envs.swe.data.solvability judge DATASET_DIR OUT.jsonl [BUNDLE:K ...] [--workers N]
The API comes from GROK_MODELS_BASE_URL / XAI_API_KEY (environment only), the model from REWRITE_MODEL (grok-4.6).
Trees go to $SOLV_TREES/<dataset>/<bundle> (default /mnt/disk6/user/coupled-solv/trees).
"""
import ast, collections, concurrent.futures, hashlib, json, os, re, subprocess, sys, threading, tomllib
from pathlib import Path

from octopus.envs.swe.data.statement_coverage import statement

N = Path("/home/user/harbor-udocker")
TREES = Path(os.environ.get("SOLV_TREES", "/mnt/disk6/user/coupled-solv/trees"))
RANK = ("solvable", "uncertain", "unsolvable", "contradicts")


def test_source(task_dir: Path, ids: list, cap: int = 12000) -> str:
    """The F2P tests' source from tests/hidden_full: each test function (with decorators), plus the test file's
    top-level helpers, constants and fixtures it names and its class's setUp; a non-Python id (doctest file, lexer
    golden file) as the file's head."""
    out, seen = [], set()
    by_file = collections.defaultdict(list)
    for i in ids:
        path, _, rest = i.partition("::")
        by_file[path].append(re.sub(r"\[.*$", "", rest))
    for path, names in by_file.items():
        src_path = task_dir / "tests/hidden_full" / path
        if not src_path.is_file():
            out.append(f"# {path}: not in the hidden test files")
            continue
        text = src_path.read_text(errors="replace")
        if not path.endswith(".py"):
            out.append(f"# {path} (graded as a whole file{', ' + ', '.join(set(names)) if any(names) else ''}):\n{text[:3000]}")
            continue
        tree = ast.parse(text)
        seg = lambda n: "\n".join(text.split("\n")[min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno]) - 1:n.end_lineno])
        top = collections.defaultdict(list)  # name -> the top-level statements defining or assigning it
        for n in tree.body:
            if hasattr(n, "name") and not n.name.startswith(("test", "Test")):
                top[n.name].append(n)
            for t in ast.walk(n) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr)) else []:
                if isinstance(t, ast.Name) and (isinstance(n, ast.Expr) or isinstance(t.ctx, ast.Store)):
                    top[t.id].append(n)
        parts, used = [], set()
        for name in dict.fromkeys(names):
            node, chain = None, name.split("::")
            scope = tree.body
            for part in chain:
                node = next((n for n in scope if getattr(n, "name", None) == part), None)
                scope = getattr(node, "body", [])
            if node is None:
                parts.append(f"# {path}::{name}: not found")
                continue
            key = (path, node.lineno)
            if key in seen:
                continue
            seen.add(key)
            if isinstance(node, ast.ClassDef):  # a whole class graded: all of it
                parts.append(seg(node))
            else:
                if len(chain) > 1:
                    cls = next(n for n in tree.body if getattr(n, "name", None) == chain[0])
                    parts += [seg(n) for n in cls.body if getattr(n, "name", None) in ("setUp", "setup_method", "setUpClass")]
                parts.append(seg(node))
            used |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {a.arg for a in ast.walk(node) if isinstance(a, ast.arg)}
        helpers = [seg(h) for n in sorted(used) if n in top for h in top[n]]
        out.append(f"# {path}\n" + "\n\n".join(dict.fromkeys(helpers + parts)))
    return "\n\n".join(out)[:cap]


def bug_lines(task_dir: Path) -> dict:
    """{path: {stripped line}} the bundle's reference fixes remove (the bug lines themselves)."""
    out, path = collections.defaultdict(set), None
    for p in sorted((task_dir / "solution").glob("fix_*.patch")):
        for line in p.read_text(errors="replace").split("\n"):
            if line.startswith("+++ "):
                path = line[4:].removeprefix("b/")
            elif line.startswith("-") and not line.startswith("--- ") and line[1:].strip():
                out[path].add(line[1:].strip())
    return out


_locks = collections.defaultdict(threading.Lock)


def bug_tree(task_dir: Path) -> Path:
    """The bundle's initial tree without its statements: the healthcheck's steps up to the statements, run on a
    shared clone of the image's golden repo (built once per dataset and bundle)."""
    dest = TREES / task_dir.parent.name / task_dir.name
    with _locks[dest]:
        if (dest / ".solv_ready").exists():
            return dest
        cfg = tomllib.loads((task_dir / "task.toml").read_text())["environment"]
        key = hashlib.sha1(cfg["docker_image"].encode()).hexdigest()[:16]
        golden = N / "golden" / (N / "golden/.golden" / f"{key}.cid").read_text().split()[0] / "ROOT/testbed"
        steps = re.search(r"then (.*?) && mkdir -p /tasks ", cfg["healthcheck"]["command"]).group(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(golden), str(dest)], check=True,
                           timeout=600)
        subprocess.run(["bash", "-c", steps], cwd=dest, check=True, capture_output=True, timeout=600)
        (dest / ".solv_ready").touch()
        return dest


def grep(tree: Path, needle: str, skip: dict) -> list:
    """['path:line: text'] of the needle in the tree (not .git), skipping the bug lines."""
    if len(needle.strip()) < 4:  # ponytail: too short to be evidence; a needle list per requirement covers the rest
        return []
    r = subprocess.run(["grep", "-rnIF", "--exclude-dir=.git", "--exclude=.solv_ready", "-e", needle, "."], cwd=tree,
                       capture_output=True, text=True, errors="replace", timeout=120)
    hits = []
    for line in r.stdout.splitlines():
        path, no, text = (line.split(":", 2) + ["", ""])[:3]
        if text.strip() not in skip.get(path.removeprefix("./"), ()):
            hits.append(f"{path.removeprefix('./')}:{no}: {text.strip()[:160]}")
    return hits


PROMPT = """You audit one task of a bug-fixing benchmark. A bug was injected into a Python repository. The developer
gets the buggy repository and ONLY this task statement:
<statement>
{statement}
</statement>

They can read and run all of the repository's remaining code, docs and tests, but the hidden tests below are deleted
from their copy. Their fix is graded by these hidden tests, which fail on the buggy code and must pass after the fix:
<hidden_tests>
{tests}
</hidden_tests>

The reference fix (buggy code -> original code), shown only to you, never to the developer:
<reference_fix>
{fix}
</reference_fix>

Question: could a competent developer make the hidden tests pass from the statement plus the repository alone? A hard
task is fine; what matters is information that is missing or contradicted. List the concrete requirements the hidden
tests put on the fix: only what the fix must CHANGE (exact values, strings, formats, messages, reprs, return values,
raised exceptions, behaviours), not things that already hold on the buggy code. Class each requirement:
- "a": the statement states or clearly implies it, or it follows from restoring the evident correct behaviour the
  statement points at (the code's own logic, its documented or conventional semantics, an example in the statement).
- "b": it needs a specific value the statement does not give and does not determine, which the developer would have
  to look up in the repository (other code, docs, docstrings, examples).
- "c": it cannot be inferred. kind "missing": an exact literal, format, message or repr that neither the statement
  nor a reasonable reading gives and that you do not expect anywhere else in the repository; "contradicts": the
  statement asks for different behaviour than the tests require (e.g. it describes the buggy behaviour as the
  expected one); "ambiguous": the statement allows several behaviours and the tests pick one with no clear reason.
For "b" and "c" give 1-3 needles: short exact single-line strings as they would be written in the repository's source
or docs (distinctive, at least 4 characters, not a common token like "None" or "self") whose presence somewhere else
in the repository would tell the developer what the tests require.

Answer with one JSON object and nothing else:
{{"requirements": [{{"test": "<test name>", "requires": "<what the fix must make true>", "class": "a" | "b" | "c",
"kind": "missing" | "contradicts" | "ambiguous" (c only), "needles": ["..."] (b and c only), "why": "<one sentence>"}}],
"summary": "<one sentence>"}}"""


def label(reqs: list) -> str:
    worst = 0
    for r in reqs:
        found = bool(r.get("hits"))
        if r["class"] == "a":
            continue
        if r["class"] == "c" and r.get("kind") == "contradicts":
            worst = 3
        elif r["class"] == "c" and r.get("kind") == "ambiguous":
            worst = max(worst, 0 if found else 1)
        else:  # b, or c missing
            worst = max(worst, 0 if found else 2)
    return RANK[worst]


def judge_task(ds: Path, bundle: str, k: int) -> dict:
    from octopus.envs.swe.data.acceptance_rewrite import ask
    d = ds / bundle
    rec, text = {"dataset": ds.name, "bundle": bundle, "k": k}, None
    try:  # one broken task (tree, test source, API, answer) is recorded with its error; a rerun retries it
        ids = json.loads((d / "tests/tasks.json").read_text())[str(k)]["f2p"]
        text = ask(PROMPT.format(statement=statement(d, k)[:8000], tests=test_source(d, ids),
                                 fix=(d / f"solution/fix_{k}.patch").read_text(errors="replace")[:6000]))
        got = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        reqs = [r for r in got["requirements"] if r.get("class") in ("a", "b", "c")]
        if not reqs:  # nothing judged is not "solvable"
            raise ValueError("no requirement with class a, b or c")
        tree, skip = None, bug_lines(d)
        for r in reqs:
            if r["class"] != "a":
                tree = tree or bug_tree(d)
                r["hits"] = {n: grep(tree, n, skip)[:5] for n in r.get("needles") or [] if isinstance(n, str)}
                r["hits"] = {n: h for n, h in r["hits"].items() if h}
        return rec | {"label": label(reqs), "summary": got.get("summary", ""), "requirements": reqs}
    except Exception as exc:
        return rec | {"error": f"{type(exc).__name__}: {exc}"[:300]} | ({"raw": text} if text else {})


def judge(ds: str, out: str, only: list, workers: int = 16):
    """Label every task of DATASET_DIR (its .manifest.jsonl), or the BUNDLE:K ones; resumable (OUT is appended)."""
    ds = Path(ds)
    done = {(r["dataset"], r["bundle"], r["k"]) for r in map(json.loads, open(out)) if "error" not in r} if Path(out).exists() else set()
    todo = [(r["bundle"], t["k"]) for r in map(json.loads, open(f"{ds}.manifest.jsonl")) for t in r["tasks"]
            if (ds.name, r["bundle"], t["k"]) not in done and (not only or f"{r['bundle']}:{t['k']}" in only)]
    print(len(todo), "tasks", flush=True)
    with open(out, "a") as f, concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(judge_task, ds, *t) for t in todo]):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["bundle"], rec["k"], rec.get("error") or rec["label"], flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 16
    plain = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--workers")]
    if cmd == "judge":
        judge(plain[0], plain[1], plain[2:], workers)
