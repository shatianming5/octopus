"""Task bundles: one Harbor task = one SWE-smith repo with 4-8 single-bug instances applied to ONE tree.

Separate per-instance trees would leak: every instance of a repo starts from the same original commit, so another
task's tree holds this task's correct code and hidden tests. A bundle tree is the original commit with every bundle
bug patch applied and every bundle task's "Remove F2P Tests" change applied, then a fresh git history. Instance branch
<ref> (in the image's /testbed repo): <ref>~2 original, <ref>~1 "Bug Patch", <ref> "Remove F2P Tests".

Bundle task dir (Harbor):
  task.toml             healthcheck = the setup below, run with the image's own git objects (commit shas), then
                        statements to /tasks/<k>.md + /tasks/INDEX.md, then `rm -rf .git` and one fresh commit
  instruction.md        N tasks in /tasks, one repo
  tests/test.sh         INFRA_RESTORE, put back the hidden test files (original content), run check.py
  tests/check.py        per task k: pytest on k's FAIL_TO_PASS ids + k's PASS_TO_PASS ids in k's hidden files (other
                        tasks' FAIL_TO_PASS and ids not passing on the reference left out) -> /logs/verifier/tasks.json
                        {k: 0/1}, reward.txt = fraction solved
  tests/tasks.json      {k: {f2p, p2p}};  tests/hidden_full/<path>: the hidden test files at the original commit
  solution/fix_<k>.patch  reference fix of task k; solve.sh applies every fix_*.patch present
Repos are split train/eval (seeded, ~15% eval, stable per repo): `split` in NAME.manifest.jsonl, NAME.splits.json.

usage (Harbor/udocker host, `source scripts/env.sh`, PYTHONPATH=<code dir>):
  build_bundles.py baseline NAME KEY...|all          harbor python: tests passing on the original commit, per repo
  build_bundles.py build PARQUET_DIR NAME KEY...|all [per_repo=10] [--sizes 1,1,2,...] [--only IDS_FILE] [--tag b]
                     [--rule samefile]
                                     python with pyarrow: datasets/NAME (sizes drawn from --sizes instead of 8-16/6;
                                     only the instance ids in IDS_FILE; bundle ids KEY.<tag>NN; samefile: every task's
                                     fix changes one .py file, tasks collide, plan_samefile)
  build_bundles.py accept NAME [conc=16 [conc_after HH:MM]]   harbor python: one container per bundle, reference
                                     fixes none / k only / all / 2 random subsets (every pair for <= 4 tasks),
                                     twice -> NAME.acceptance-1c.json
  build_bundles.py audit NAME [n=20] [conc=32]   harbor python: n accepted bundles through `harbor run -a oracle`
  build_bundles.py timing NAME MANIFEST [conc=16]  harbor python: none/all once, test.sh wall time, <= 300 s gate
  build_bundles.py matrix DATASET_DIR OUT BUNDLE...  harbor python: accept's matrices through OCTOPUS_HARBOR_ENV (backend check)
  build_bundles.py check BUNDLE_DIR...           rewrite tests/check.py with the current grader
  build_bundles.py clean BUNDLE_DIR...           add clean_command (copies of the original code outside /testbed) to the
                                                 healthcheck of bundles built before it
  build_bundles.py repair PARQUET_DIR NAME SRC     python with pyarrow: repair round for SRC's rejected bundles
                                                 -> datasets/<shard>-rep<round> (then accept it like a shard)
  build_bundles.py repairs SHARD                 <shard>.repairs.json from the shard's repair rounds
  build_bundles.py drop-budget BUNDLE_DIR|MANIFEST...  drop the old token budget from instruction.md / manifest rows
  build_bundles.py reduce PARQUET_DIR NAME OUT LABELS [REFRESH]   python with pyarrow: NAME's bundles with each task cut
                                     to what its statement describes (statement_coverage judge LABELS) -> datasets/OUT
  build_bundles.py refill PARQUET_DIR SRC CUT OUT [small=6 [target=8]]   python with pyarrow: CUT's small or rejected
                                     train bundles refilled from unused tasks -> datasets/OUT (then judge, reduce, ...)
  build_bundles.py refresh NAME OUT [conc=16]   harbor python: graded ids on NAME's and OUT's initial trees ->
                                     OUT.refresh.jsonl; reduce again with it (to a new OUT), then accept
KEY is the repo key of an image, e.g. mahmoud__glom.fb3c4e76 (lower case)."""
import ast, asyncio, base64, collections, contextlib, glob, hashlib, io, itertools, json, os, random, re, shutil, subprocess, sys, tarfile, tempfile, time, tomllib
from pathlib import Path

from octopus.envs.swe.taskset import HARBOR_ENV, INFRA_RESTORE, branch_setup, task_config_text, write_task_tree
from octopus.envs.swe.data.regen_tasks import N, git, golden_repo

KINDS = re.compile(r"\.(lm_rewrite|func_basic|func_pm_\w+|class_\w+)__\w+$")
# no `-p no:cacheprovider`: plugins such as pytest-snail (hydra) need config.cache and abort pytest without it
# -v prints each result as it happens: a suite cut by the timeout (pandas, mypy) or one that crashes before the
# summary (modin) still yields the ids that passed
PYTEST = "/opt/miniconda3/envs/testbed/bin/python -m pytest -v -rA --tb=no --continue-on-collection-errors"

CHECK = r"""import importlib.util, json, subprocess, sys, tempfile
from pathlib import Path

# Each task is scored from its own test ids: every FAIL_TO_PASS id must PASS (skipped is not solved: a fix that makes
# the package skip its tests solves nothing), and every PASS_TO_PASS id in its hidden test files must pass or skip.
# Other tasks' FAIL_TO_PASS and ids that do not pass on the reference are left out at build time. All tasks' ids go
# into ONE pytest run; if that run crashes, times out, has collection errors or leaves an id without a result, each
# task gets its own run under the same rule (a task whose own run is broken scores 0).
RUNNER = '''
import json, os, sys, pytest
res, dur, broken = {}, {}, []
class Record:
    def pytest_collectreport(self, report):
        if report.failed:
            broken.append(report.nodeid)
    def pytest_runtest_logreport(self, report):
        dur[report.nodeid] = dur.get(report.nodeid, 0) + report.duration
        if report.failed:
            res[report.nodeid] = "failed"
        elif report.when == "call" or report.skipped:
            res.setdefault(report.nodeid, report.outcome)
rc = pytest.main([*sys.argv[3:], *json.load(open(sys.argv[1]))], plugins=[Record()])
with open(sys.argv[2], "w") as f:
    json.dump({"rc": int(rc), "res": res, "dur": dur, "broken": broken}, f)
sys.stdout.flush()
os._exit(0)  # results are written: do not wait for threads a test suite leaves running (django__daphne hangs here)
'''
tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())
flags = ["-q", "--tb=line", *(["--no-cov"] if importlib.util.find_spec("pytest_cov") else [])]


def run(ids, timeout):
    # -> (outcomes, {id: its result nodes}, "") or (None, None, why the run cannot be used)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "ids.json").write_text(json.dumps(ids))
    try:
        subprocess.run([sys.executable, "-c", RUNNER, str(tmp / "ids.json"), str(tmp / "out.json"), *flags],
                       timeout=timeout)
        out = json.loads((tmp / "out.json").read_text())
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    except (OSError, ValueError):
        return None, None, "pytest crashed"
    # the id itself, or the tests under it when it names a parametrized test or a class
    found = {i: [i] if i in out["res"] else [n for n in out["res"] if n.startswith(i + "[") or n.startswith(i + "::")]
             for i in ids}
    why = (f"pytest exit {out['rc']}" if out["rc"] not in (0, 1) else "collection errors" if out["broken"]
           else "ids without a result" if not all(found.values()) else "")
    return (None, None, why) if why else (out, found, "")


def score(out, found, t):
    f2p = [n for i in t["f2p"] for n in found[i]]
    p2p = [n for i in t["p2p"] for n in found[i]]
    ok = all(out["res"][n] == "passed" for n in f2p) and all(out["res"][n] in ("passed", "skipped") for n in p2p)
    bad = [n for n in f2p if out["res"][n] != "passed"] + [n for n in p2p if out["res"][n] not in ("passed", "skipped")]
    return (int(ok), round(sum(out["dur"].get(n, 0) for n in f2p + p2p), 1),
            {"f2p_skipped": sum(out["res"][n] == "skipped" for n in f2p), "not_passed": bad[:20]})


ids = list(dict.fromkeys(i for t in tasks.values() for i in t["f2p"] + t["p2p"]))
out, found, why = run(ids, 900)
solved, seconds, details = {}, {}, {}
for k, t in tasks.items():
    if why:
        print(f"=== task {k} (one run unusable: {why})", flush=True)
        o, f, w = run(t["f2p"] + t["p2p"], 100)
        solved[k], seconds[k], details[k] = score(o, f, t) if o else (0, 0.0, {"own_run": w})
    else:
        solved[k], seconds[k], details[k] = score(out, found, t)
res = Path("/logs/verifier")
res.mkdir(parents=True, exist_ok=True)
(res / "tasks.json").write_text(json.dumps(solved))
(res / "seconds.json").write_text(json.dumps(seconds))
(res / "details.json").write_text(json.dumps(details))
(res / "grader.txt").write_text(f"per task: {why}\n" if why else "one run\n")
(res / "reward.txt").write_text(f"{sum(solved.values()) / len(solved)}\n")
"""

TEST_SH = ("#!/bin/bash\nset +e\ncd /testbed\n" + INFRA_RESTORE
           + "cp -a /tests/hidden_full/. /testbed/\n/opt/miniconda3/envs/testbed/bin/python /tests/check.py\n")

SOLVE_SH = """#!/bin/bash
# applies every reference fix present, in task order
cd /testbed && for p in $(ls /solution/fix_*.patch 2>/dev/null | sort -V); do git apply --whitespace=nowarn "$p" || exit 1; done
"""

INSTRUCTION = """The repository in /testbed has {n} tasks. Each one is described in /tasks/<k>.md \
(k = 1..{n}); /tasks/INDEX.md lists them, one line per task. Fix as many tasks as you can by changing the code in \
/testbed; each task is graded on its own by hidden tests.
"""


def local_images() -> dict:
    """repo key (lower case) -> local image name with a golden container, from `udocker images` (HF names
    jyangballin/, local swebench/)."""
    out = subprocess.run([str(N / "bin/udocker"), "images"], capture_output=True, text=True).stdout
    images = [l.split()[0] for l in out.splitlines() if "swesmith.x86_64." in l]
    images = [i for i in images if (N / "golden/.golden" / f"{hashlib.sha1(i.encode()).hexdigest()[:16]}.cid").exists()]
    return {i.split("swesmith.x86_64.")[1].split(":")[0].lower().replace("_1776_", "__"): i for i in images}


def branches(repo: Path, key: str) -> list:
    refs = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split()
    return [r.removeprefix("origin/") for r in refs if r.lower().startswith(f"origin/{key}.") and KINDS.search(r)]


def fix_lines(patch: str) -> set:
    return {l[1:].strip() for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")
            and len(l[1:].strip()) >= 30}


def touched_functions(repo, orig, bug) -> list:
    out = set()
    for f in git(repo, "diff", "--name-only", orig, bug).split():
        try:
            tree = ast.parse(git(repo, "show", f"{orig}:{f}")) if f.endswith(".py") else None
        except (subprocess.CalledProcessError, SyntaxError):
            tree = None
        if tree is None:
            continue
        defs = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for m in re.finditer(r"^@@ -(\d+)", git(repo, "diff", "-U0", orig, bug, "--", f), re.M):
            inner = [d for d in defs if d[0] <= int(m.group(1)) <= d[1]]
            out.add(f"{f}:{max(inner)[2] if inner else '<module>'}")
    return sorted(out)


def candidate(repo: Path, row: dict) -> dict:
    """Git facts of one instance branch plus its SWE-smith row (FAIL_TO_PASS, PASS_TO_PASS, problem_statement)."""
    ref = f"origin/{row['instance_id']}"
    orig, bug, rm = (git(repo, "rev-parse", f"{ref}{s}").strip() for s in ("~2", "~1", ""))
    patch = git(repo, "diff", "--binary", orig, bug)
    status = [l.split("\t") for l in git(repo, "diff", "--name-status", bug, rm).splitlines()]
    fix = git(repo, "diff", "--binary", bug, orig)
    lines = fix_lines(fix)
    bug_files = git(repo, "diff", "--name-only", orig, bug).split()
    functions = touched_functions(repo, orig, bug)
    # fix lines already in the instance's own SWE-smith tree (shuffled or untouched lines of the buggy function, the
    # same code elsewhere in the repo): what every single-instance task shows too, so not a bundle leak
    found = subprocess.run(["git", "-C", str(repo), "grep", "-h", "-F", "-f", "-", bug], input="\n".join(lines),
                           capture_output=True, text=True).stdout if lines else ""
    return {"inherent": lines & {l.strip() for l in found.splitlines()},
        "instance_id": row["instance_id"], "statement": row["problem_statement"].strip(),
        "kind": KINDS.search(row["instance_id"]).group(1), "orig": orig, "bug": bug, "rm": rm, "fix": fix,
        "bug_files": bug_files, "functions": functions,
        "whole": [f for f in bug_files if not any(u.startswith(f"{f}:") for u in functions)],
        "deleted": [p[-1] for p in status if p[0] == "D"], "modified": [p[-1] for p in status if p[0] != "D"],
        "f2p": list(row["FAIL_TO_PASS"]), "p2p": list(row["PASS_TO_PASS"]),
        "patch_lines": sum(1 for l in patch.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")),
    }


def setup_steps(c: dict) -> list:
    """Shell steps putting task c's bug in and its F2P tests out; same commands in the container and on the host."""
    steps = [f"git diff --binary {c['orig']} {c['bug']} | git apply --whitespace=nowarn"]
    if c["modified"]:
        steps.append(f"git diff --binary {c['bug']} {c['rm']} -- {' '.join(c['modified'])} | git apply --whitespace=nowarn")
    if c["deleted"]:
        steps.append(f"rm -f {' '.join(c['deleted'])}")
    return steps


# SWE-smith's "Remove F2P Tests" also deletes a source module whose doctests are FAIL_TO_PASS ids (tldextract's
# tldextract/tldextract.py): the agent then gets a package that does not import
TEST_FILE = re.compile(r"(^|/)(tests?|testing)/|(^|/)(test_[^/]*|[^/]*_tests?\.py|conftest\.py|tests?\.py)$")


# a comment the bug patch adds that says what it changed ("# Changed from self.plural to self.plural_adj")
CHANGE_NOTE = re.compile(r"\b(chang(e|ed|es|ing)|swap\w*|introduc\w*|incorrect\w*|off-by-one|bug\w*|instead|invert\w*|"
                         r"revers\w*|wrong\w*|original\w*|should|subtle|mistake\w*|typo|intention\w*|intend\w*|"
                         r"redundant|misplac\w*|remov\w*|reorder\w*|trimm\w*|without)\b", re.I)


def change_notes(fix: str) -> list:
    """'#' comments on the bug side of a reference fix, not on its fixed side, that name a change."""
    lines = fix.splitlines()
    fixed = "\n".join(l[1:] for l in lines if l.startswith("+") and not l.startswith("+++"))
    notes = [m.group(1).strip() for l in lines if l.startswith("-") and not l.startswith("---")
             for m in [re.search(r"(?:^|\s)#\s*(.+)$", l[1:])] if m]
    return [n for n in notes if n not in fixed and CHANGE_NOTE.search(n)]


def ineligible(c: dict, orig: str, passed: set) -> str:
    hidden = set(c["deleted"]) | set(c["modified"])
    if c["orig"] != orig:
        return "different original commit"
    if not c["statement"] or not c["f2p"]:
        return "no statement or no FAIL_TO_PASS"
    if hidden & set(c["bug_files"]):
        return "hidden tests inside a bug file"
    if any(p.endswith(".py") and not TEST_FILE.search(p) for p in hidden):
        return "a hidden file is not a test file"
    if any(t.split("::")[0] not in hidden for t in c["f2p"]):
        return "FAIL_TO_PASS outside the hidden files"
    if not set(c["f2p"]) <= passed:
        return "FAIL_TO_PASS not passing on the reference"
    if any(l in c["statement"] for l in fix_lines(c["fix"])):
        return "statement holds reference-fix code"
    if change_notes(c["fix"]):
        return "bug patch adds a comment naming the change"
    if sum(l.startswith("+") and not l.startswith("+++") for l in c["fix"].splitlines()) >= 250:
        return "bug patch deletes >= 250 lines"  # class_rm_funcs of whole classes: next to unsolvable
    return ""


def leaks(tree: Path, tasks: list) -> list:
    """Reference-fix lines (>= 30 chars, not already in the task's own instance tree) or FAIL_TO_PASS test
    definitions of any bundle task left in the tree."""
    lines, found = set(), []
    for f in git(tree, "ls-files", "-co", "--exclude-standard").splitlines():
        if (tree / f).is_file():
            lines.update(l.strip() for l in (tree / f).read_text(errors="ignore").splitlines())
    for c in tasks:
        found += [f"{c['instance_id']}: fix line in tree: {l[:60]}" for l in (fix_lines(c["fix"]) & lines) - c["inherent"]]
        for t in c["f2p"]:
            path, func = t.split("::")[0], t.split("::")[-1].split("[")[0]
            if (tree / path).is_file() and re.search(rf"def {re.escape(func)}\b", (tree / path).read_text(errors="ignore")):
                found.append(f"{c['instance_id']}: F2P test still in tree: {t}")
    return found


def sh(tree: Path, cmd: str) -> bool:
    return subprocess.run(["bash", "-c", cmd], cwd=tree, capture_output=True).returncode == 0


def fixes_apply(tree: Path, tasks: list) -> bool:
    """Every reference fix applies on the bundle tree alone, and all of them in order 1..n (as solve.sh applies them)."""
    run = lambda c, *check: subprocess.run(["git", "apply", *check, "--whitespace=nowarn"], cwd=tree, input=c["fix"],
                                           text=True, capture_output=True).returncode == 0
    ok = all(run(c, "--check") for c in tasks) and all(run(c) for c in tasks)
    sh(tree, "git reset -q --hard && git clean -fdq")
    return ok


NEAR = 20  # samefile: lines between two tasks' bug hunks that still count as near


def hunks(fix: str, source: str, new_side: bool) -> list:
    """[start, end, functions, classes] per run of changed lines of a one-file fix, in the lines of its new side (the
    original commit, for a task's own fix) or old side (the bundle tree, for a fix on it); source = that side's text.
    functions / classes: the innermost one (name@line) around each changed line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    defs = [(n.lineno, n.end_lineno, f"{n.name}@{n.lineno}", isinstance(n, ast.ClassDef)) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))] if tree else []
    # a run of changed lines spans its lines on this side (as git diff -U0 on it); a run with none of them (only the other
    # side's lines) is the line before which they go
    side, runs, at, block = "+" if new_side else "-", [], None, None
    for l in fix.splitlines() + ["@@"]:
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", l)
        if block and (l.startswith("@@") or l[:1] == " "):
            runs.append([block[1][0], block[1][-1]] if block[1] else [block[0] - 1] * 2)
            block = None
        if m:  # a side with 0 lines gives the line before them (git diff -U0 of an insertion or deletion)
            at = {"-": int(m.group(1)) + (m.group(2) == "0"), "+": int(m.group(3)) + (m.group(4) == "0")}
        elif at and l[:1] == " ":
            at["-"], at["+"] = at["-"] + 1, at["+"] + 1
        elif at and l[:1] in ("-", "+"):
            block = block or [at[side], []]
            if l[0] == side:
                block[1].append(at[side])
            at[l[0]] += 1
    out = []
    for s, e in runs:
        inner = [[d for d in defs if d[0] <= line <= d[1] and d[3] == cls] for line in range(s, e + 1) for cls in (False, True)]
        out.append([s, e, *(sorted({max(ds)[2] for ds in inner[i::2] if ds}) for i in (0, 1))])
    return out


def collide(a: dict, b: dict, scope: bool = False) -> bool:
    """Bug hunks at most NEAR lines apart (with `scope` also: in one function or one class)."""
    return any(max(s1, s2) - min(e1, e2) <= NEAR or scope and (set(f1) & set(f2) or set(c1) & set(c2))
               for s1, e1, f1, c1 in a["hunks"] for s2, e2, f2, c2 in b["hunks"])


def shipped_hunks(tree: Path, tasks: list) -> list:
    """Each task's fix applied alone on `tree` (the bundle tree as shipped, committed): its hunks from git diff -U0, in
    tree lines, into c["hunks"]. A fix's @@ lines need not match the tree (it was made on its own instance's bug). The one
    measure for reduce and finalize. -> tasks whose fix does not apply."""
    bad = []
    for c in tasks:
        f = c["bug_files"][0]
        ap = subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=tree, input=c["fix"], text=True, capture_output=True)
        u0 = subprocess.run(["git", "diff", "-U0", "--", f], cwd=tree, capture_output=True, text=True).stdout
        sh(tree, "git reset -q --hard && git clean -fdq")
        if ap.returncode:
            bad.append(c)
        c["hunks"] = hunks(u0, (tree / f).read_text(), False)
    return bad


def coupled(tasks: list) -> bool:
    """The samefile rule on a whole bundle: one bug file, tasks connected through collide(scope), one pair within NEAR."""
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j in set(range(len(tasks))) - seen:
            if collide(tasks[i], tasks[j], True):
                seen.add(j)
                todo.append(j)
    return (len({tuple(c["bug_files"]) for c in tasks}) == 1 and len(seen) == len(tasks) > 1
            and any(collide(a, b) for a, b in itertools.combinations(tasks, 2)))


def conflicts(c: dict, picked: list, samefile: bool = False) -> bool:
    """Tasks of a bundle touch different functions (a file without function info counts whole), no bug file is
    another task's hidden test file, and FAIL_TO_PASS sets are disjoint (a test that two bugs break needs both fixes,
    so fixing one task alone would not solve it). With `samefile` the different-functions rule becomes: every task's
    bug patch changes the same one file and c collides with a picked task: the second task within NEAR lines of the
    first (so every bundle has such a pair), later ones also through one function or class."""
    files = {f for p in picked for f in p["bug_files"]}
    hidden = {f for p in picked for f in p["deleted"] + p["modified"]}
    if (set(c["bug_files"]) & hidden or set(c["deleted"] + c["modified"]) & files
            or set(c["f2p"]) & {t for p in picked for t in p["f2p"]} or c["statement"] in {p["statement"] for p in picked}):
        return True
    if samefile:
        return bool(picked) and (c["bug_files"] != picked[0]["bug_files"] or not any(collide(c, p, len(picked) > 1) for p in picked))
    funcs = {u for p in picked for u in p["functions"]}
    whole = {f for p in picked for f in p["whole"]}
    return bool(set(c["functions"]) & funcs or set(c["whole"]) & files or set(c["bug_files"]) & whole)


COMMIT = "git add -A && git -c user.name=b -c user.email=b@b commit -qm step"


def mark_hard(cands: list):
    """hard = above-median rank sum of patch lines and FAIL_TO_PASS count."""
    rank = lambda key: {id(c): i for i, c in enumerate(sorted(cands, key=key))}
    lines, f2ps = rank(lambda c: c["patch_lines"]), rank(lambda c: len(c["f2p"]))
    score = sorted(lines[id(c)] + f2ps[id(c)] for c in cands)
    for c in cands:
        c["hard"] = lines[id(c)] + f2ps[id(c)] > score[len(score) // 2] if score else False


def fill(tree: Path, picked: list, pool: list, size: int, drops: collections.Counter, samefile: bool = False) -> list:
    """Extend `picked` (already applied and committed on `tree`) from `pool` up to `size` tasks: no conflict, kinds
    spread, easy and hard alternating; a task whose patch or test removal does not apply, that leaks, or that leaves
    some reference fix (alone or all together) not applying is dropped from the pool (with `samefile` it goes back to
    the pool at the end: it collides with the picked tasks, so another bundle may take it). Taken tasks leave the pool."""
    picked, back = list(picked), []
    while len(picked) < size:
        fits = [c for c in pool if not conflicts(c, picked, samefile)]
        want = [c for c in fits if c["hard"] == bool(len(picked) % 2)] or fits
        if not want:
            break
        kinds = collections.Counter(p["kind"] for p in picked)
        c = min(want, key=lambda c: kinds[c["kind"]])
        pool.remove(c)
        if not sh(tree, " && ".join(setup_steps(c))):
            drops["patch or test removal does not apply on the bundle tree"] += 1
            back.append(c)
            sh(tree, "git reset -q --hard && git clean -fdq")
            continue
        sh(tree, COMMIT)
        why = ("leak (fix line or F2P test left in the tree)" if leaks(tree, picked + [c])
               else "a reference fix no longer applies on the bundle tree" if not fixes_apply(tree, picked + [c])
               else "")
        if not why:
            picked.append(c)
            continue
        drops[why] += 1
        back.append(c)
        sh(tree, "git reset -q --hard HEAD~1 && git clean -fdq")
    if samefile:
        pool += back
    return picked


def plan_bundles(repo: Path, orig: str, cands: list, n_bundles: int, seed, drops: collections.Counter,
                 sizes: list = ()) -> list:
    """Up to n_bundles seeded bundles, no task used twice: each draws a size from 8-16 and is filled as far as it
    goes (kept with >= 8 tasks); once the pool cannot fill 8, bundles of 6 ("6-fallback") until it cannot fill 6.
    With `sizes` each bundle draws its size from that list instead (repeats weight it) and is kept only when full;
    the first one that cannot be filled ends the repo.
    Within a bundle no two tasks conflict, kinds are spread, easy and hard alternate (hard = above-median rank sum of
    patch lines and FAIL_TO_PASS count). A task whose patch or test removal does not apply on the bundle tree so far,
    that leaks, or that leaves some reference fix (alone or all together) not applying is swapped for another.
    Returns [(tasks, rule, drawn size)]."""
    rng = random.Random(seed)
    pool = list(cands)
    rng.shuffle(pool)
    mark_hard(pool)
    bundles, rule = [], "8-16"
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        while len(bundles) < n_bundles:
            size = rng.choice(sizes) if sizes else rng.randint(8, 16) if rule == "8-16" else 6
            sh(tree, f"git checkout -q --force {orig} && git clean -fdq")
            picked = fill(tree, [], pool, size, drops)
            if sizes:
                if len(picked) < size:
                    break
                bundles.append((picked, "sizes", size))
                continue
            if len(picked) < (8 if rule == "8-16" else 6):
                pool += picked
                if rule == "6-fallback":
                    break
                rule = "6-fallback"
                continue
            bundles.append((picked, rule, size))
    return bundles


def plan_samefile(repo: Path, orig: str, cands: list, n_bundles: int, seed, drops: collections.Counter,
                  sizes: list) -> list:
    """--rule samefile: up to n_bundles seeded bundles of tasks whose bug patches change one and the same .py file,
    each task colliding with a task picked before it (conflicts). Files take turns in a seeded order, one bundle per
    turn: a size drawn from `sizes`, the file's next task not yet tried first, filled from the file's pool, kept only
    when full (else its tasks go back to the pool). No task is used twice. Returns [(tasks, "samefile", drawn size)]."""
    rng = random.Random(seed)
    pool = [c for c in cands if len(c["bug_files"]) == 1 and c["bug_files"][0].endswith(".py")]
    rng.shuffle(pool)
    mark_hard(pool)
    files = collections.defaultdict(list)
    for c in pool:
        c["hunks"] = hunks(c["fix"], git(repo, "show", f"{c['orig']}:{c['bug_files'][0]}"), True)
        files[c["bug_files"][0]].append(c)
    order = sorted(files)
    rng.shuffle(order)
    firsts = {f: list(files[f]) for f in order}
    bundles = []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        while len(bundles) < n_bundles and any(firsts.values()):
            for f in order:
                if not firsts[f] or len(bundles) == n_bundles:
                    continue
                first = firsts[f].pop(0)
                if first not in files[f]:  # in an earlier bundle, or dropped
                    continue
                size = rng.choice(sizes)
                files[f].remove(first)
                sh(tree, f"git checkout -q --force {orig} && git clean -fdq")
                picked = fill(tree, [], [first], 1, drops)
                if picked:
                    picked = fill(tree, picked, files[f], size, drops, samefile=True)
                if len(picked) == size:
                    bundles.append((picked, "samefile", size))
                else:
                    files[f] += picked
    return bundles


def tasks_tar(tasks: list) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        files = {f"{k}.md": c["statement"] + "\n" for k, c in enumerate(tasks, 1)}
        files["INDEX.md"] = "".join(f"{k}. {c['statement'].splitlines()[0][:150]}\n" for k, c in enumerate(tasks, 1))
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


# what pip vendors (pip/_vendor): its own wheels (ensurepip's bundled pip, virtualenv's embedded pip) carry these
PIP_VENDORED = {"cachecontrol", "certifi", "chardet", "colorama", "distlib", "distro", "idna", "msgpack", "packaging",
                "platformdirs", "pygments", "pyparsing", "pyproject_hooks", "requests", "resolvelib", "rich", "six",
                "tenacity", "tomli", "truststore", "typing_extensions", "urllib3", "webencodings"}


def clean_globs(pkg: str) -> list:
    """Paths (shell globs) a bundle's healthcheck deletes because they hold the original code outside /testbed: pip's
    wheel cache, conda's package archives, copies of the repo's package that tools ship (pip/_vendor,
    setuptools/_vendor) or that the base env installs, also inside conda's extracted packages (lib/.../site-packages,
    noarch site-packages, info/test; deleting all of /opt/miniconda3/pkgs, 20k files, takes over 300 s under udocker),
    and for a package pip vendors, pip's own wheels. The testbed env's own site-packages/<pkg> is not among them (the
    package imports from /testbed: baseline import_path)."""
    out = ["/root/.cache/pip", "/opt/miniconda3/pkgs/*.conda", "/opt/miniconda3/pkgs/*.tar.bz2"]
    for name in (pkg, f"{pkg}.py"):
        for sp in ("/opt/miniconda3/lib/python3*/site-packages", "/opt/miniconda3/pkgs/*/lib/python3*/site-packages",
                   "/usr/lib/python3/dist-packages"):
            out += [f"{sp}/{name}", f"{sp}/*/_vendor/{name}"]
        out += [f"/opt/miniconda3/envs/*/lib/python3*/site-packages/*/_vendor/{name}",
                f"/opt/miniconda3/pkgs/*/site-packages/{name}", f"/opt/miniconda3/pkgs/*/info/test/{name}"]
    if pkg in PIP_VENDORED:
        for env in ("/opt/miniconda3", "/opt/miniconda3/envs/*", "/opt/miniconda3/pkgs/*"):
            out += [f"{env}/lib/python3*/ensurepip/_bundled/pip-*.whl",
                    f"{env}/lib/python3*/site-packages/virtualenv/seed/wheels/embed/pip-*.whl"]
    return out


def package(path: str) -> str:
    return path.removeprefix("src/").split("/")[0].removesuffix(".py")


def clean_command(pkgs) -> str:
    """Delete clean_globs of every package, and the /testbed __pycache__ dirs (compiled original code). The globs only in
    a SWE-smith image (its testbed env): a replay of the healthcheck on a host tree touches nothing outside it."""
    return (f"if [ -d /opt/miniconda3/envs/testbed ]; then rm -rf {' '.join(g for p in sorted(pkgs) for g in clean_globs(p))}; fi"
            " && find /testbed -name __pycache__ -type d -prune -exec rm -rf {} +")


def setup_command(tasks: list, bake: str = "") -> str:
    """Healthcheck (like branch_setup): the bundle tree from the image's git objects, then `bake` (reduce: the
    undescribed part of each task's bug fixed), statements, fresh history."""
    steps = [f"git checkout -q --force {tasks[0]['orig']}"] + [s for c in tasks for s in setup_steps(c)] + (
        [f"printf %s {base64.b64encode(bake.encode()).decode()} | base64 -d | git apply --whitespace=nowarn"]
        if bake else []) + [
        f"mkdir -p /tasks && printf %s {tasks_tar(tasks)} | base64 -d | tar -xz -C /tasks",
        clean_command({package(f) for c in tasks for f in c["bug_files"] if f.endswith(".py")}),
        "rm -rf .git && git init -q && git add -A",
        'git -c user.name=octopus -c user.email=octopus@localhost commit -qm "Initial commit"',
        "touch .git/octopus_ready"]
    return f"cd /testbed && if [ ! -f .git/octopus_ready ]; then {' && '.join(steps)}; fi && test -f .git/octopus_ready"


def write_bundle(root: Path, bundle_id: str, image: str, repo: Path, tasks: list, bake: str = "") -> Path:
    spec = {"instance_id": bundle_id, "image": image, "verifier_shell": TEST_SH,
            "instruction": INSTRUCTION.format(n=len(tasks))}
    dest = write_task_tree(root, spec, check_py=CHECK)
    config = task_config_text(spec).replace(branch_setup(bundle_id), setup_command(tasks, bake))
    config = config.replace("[agent]\ntimeout_sec = 300.0", "[agent]\ntimeout_sec = 3600.0")
    (dest / "task.toml").write_text(config.replace("[verifier]\ntimeout_sec = 180.0", "[verifier]\ntimeout_sec = 1800.0"))
    f2p_all = {t for c in tasks for t in c["f2p"]}
    graded = {}
    for k, c in enumerate(tasks, 1):
        hidden = set(c["deleted"]) | set(c["modified"])
        p2p = [t for t in c["p2p"] if t.split("::")[0] in hidden and t in c["passed"] and t not in f2p_all]
        graded[str(k)] = {"f2p": c["f2p"], "p2p": p2p}
        c["p2p_graded"] = len(p2p)
        for path in hidden:
            target = dest / "tests/hidden_full" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(["git", "-C", str(repo), "show", f"{c['orig']}:{path}"],
                                              check=True, capture_output=True).stdout)
        (dest / f"solution/fix_{k}.patch").parent.mkdir(exist_ok=True)
        (dest / f"solution/fix_{k}.patch").write_text(c["fix"])
    (dest / "tests/tasks.json").write_text(json.dumps(graded))
    (dest / "solution/solve.sh").write_text(SOLVE_SH)
    (dest / "solution/solve.sh").chmod(0o755)
    return dest


def split_of(key: str, seed: int = 0) -> str:
    """Repo-level split, seeded and stable per repo whatever else is built: ~15% of repos are eval only."""
    return "eval" if random.Random(f"{seed}:{key}").random() < 0.15 else "train"


def build_repo(job: tuple) -> tuple:
    """Bundles of one repo, written under datasets/NAME; returns (manifest rows, candidate counts)."""
    name, key, image, rows, per_repo, seed, sizes, tag, rule = job
    base = N / "datasets" / f"{name}.baseline" / f"{key}.json"
    if not base.exists():
        return [], {f"no baseline {base}": 1}
    baseline = json.loads(base.read_text())
    passed = set(baseline["passed"])
    repo = golden_repo(image)
    drops = collections.Counter()
    cands = eligible(repo, key, rows, baseline, drops)
    out = []
    plan = plan_samefile if rule == "samefile" else plan_bundles
    for b, (tasks, size_rule, drawn) in enumerate(plan(repo, baseline["orig"], cands, per_repo, f"{seed}:{key}", drops, sizes)):
        bundle_id = f"{key}.{tag}{b:02d}"
        write_bundle(N / "datasets" / name, bundle_id, image, repo, tasks)
        out.append(manifest_row(bundle_id, key, image, tasks, size_rule, drawn, seed))
    return out, dict(drops)


def eligible(repo: Path, key: str, rows: list, baseline: dict, drops: collections.Counter) -> list:
    have, passed, cands = set(branches(repo, key)), set(baseline["passed"]), []
    for r in rows:
        if r["instance_id"] not in have:
            drops["branch not in golden"] += 1
            continue
        c = candidate(repo, r)
        why = ineligible(c, baseline["orig"], passed)
        drops[why or "eligible"] += 1
        if not why:
            c["passed"] = passed
            cands.append(c)
    return cands


def manifest_row(bundle_id, key, image, tasks, rule, drawn, seed=0, **extra) -> dict:
    return {"bundle": bundle_id, "repo": key, "split": split_of(key, seed), "image": image, "n": len(tasks),
            "check_sha256": hashlib.sha256(CHECK.encode()).hexdigest(), "size_rule": rule, "size_drawn": drawn,
            **extra, "tasks": [
                {"k": k, "instance_id": c["instance_id"], "kind": c["kind"], "hard": c["hard"],
                 "patch_lines": c["patch_lines"], "f2p": len(c["f2p"]), "p2p_graded": c["p2p_graded"],
                 "fix_lines_inherent": len(c["inherent"]),
                 "files": c["bug_files"], "functions": c["functions"], "hidden_files": c["deleted"] + c["modified"]}
                for k, c in enumerate(tasks, 1)]}


def drop_budget(paths: list):
    """Bundles built with the old shared token budget: its sentence out of bundle dirs' instruction.md, its "budget"
    key out of manifest rows (.jsonl). Idempotent; check.py and test.sh never read it."""
    for path in map(Path, paths):
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.open()]
            path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "budget"}) + "\n" for r in rows))
        else:
            f = path / "instruction.md"
            f.write_text(re.sub(r" You have a budget of \d+ tokens for the whole repository\.", "", f.read_text()))


def build(parquets: str, name: str, keys: list, per_repo: int = 10, seed: int = 0,
          sizes: list = (), only: str = "", tag: str = "b", rule: str = ""):
    """Bundles for every repo key; `sizes` as in plan_bundles, `only` a file of the instance ids that may be used,
    bundle ids KEY.<tag>NN; rule "samefile" plans with plan_samefile."""
    import concurrent.futures, pyarrow.parquet as pq
    dest = N / "datasets" / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick a new name")
    images = local_images()
    keys = sorted(images) if keys == ["all"] else keys
    cols = ["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]
    allowed = set(Path(only).read_text().split()) if only else None
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{parquets}/*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            key = r["instance_id"].lower().rsplit(".", 1)[0]
            if key in keys and KINDS.search(r["instance_id"]) and (allowed is None or r["instance_id"] in allowed):
                rows[key].append(r)
    dest.mkdir(parents=True)
    splits = {"train": {}, "eval": {}}
    jobs = [(name, k, images[k], rows[k], per_repo, seed, sizes, tag, rule) for k in keys]
    with open(f"{dest}.manifest.jsonl", "w") as manifest, concurrent.futures.ProcessPoolExecutor(32) as pool:
        for key, (bundles, drops) in zip(keys, pool.map(build_repo, jobs)):
            splits[split_of(key, seed)].setdefault(key, [])
            for row in bundles:
                splits[row["split"]].setdefault(key, []).append(row["bundle"])
                manifest.write(json.dumps(row) + "\n")
            print(json.dumps({"repo": key, "bundles": len(bundles), "candidates": drops}), flush=True)
    Path(f"{dest}.splits.json").write_text(json.dumps(splits, indent=1))


# test files left out of a repo's baseline, so none of their ids is ever graded (and a task with a FAIL_TO_PASS id there is
# not eligible): nikola's test_dev_server_auto stops its event loop with loop.call_later(0.01, loop.stop), which under CPU
# contention fires before run_forever starts, so that test hangs and every later id of the same pytest run is lost
UNRELIABLE = {"getnikola__nikola.0f4c230e": {"tests/integration/test_dev_server_auto.py"}}


async def baseline_repo(name: str, key: str, image: str, sem) -> None:
    """Tests passing on the original commit (the image's clean /testbed), over every candidate's hidden test files;
    plus where the package imports from, and any second copy of it under site-packages."""
    from octopus.envs.swe.data.validate_tasks import make_env
    out = N / "datasets" / f"{name}.baseline" / f"{key}.json"
    if out.exists() and json.loads(out.read_text())["passed"]:  # an empty one is redone
        return
    async with sem:
        repo = golden_repo(image)
        refs = [f"origin/{b}" for b in branches(repo, key)]
        files = sorted({f for r in refs for f in git(repo, "diff", "--name-only", f"{r}~1", r).split()} - UNRELIABLE.get(key, set()))
        bug_files = [f for r in refs[:50] for f in git(repo, "diff", "--name-only", f"{r}~2", f"{r}~1").split()]
        pkg = collections.Counter(f.removeprefix("src/").split("/")[0].removesuffix(".py") for f in bug_files).most_common(1)[0][0]
        env = make_env(image)
        start = time.time()
        try:
            await env.start(False)
            orig = (await env.exec("cd /testbed && git rev-parse HEAD")).stdout.strip()
            cov = (await env.exec("/opt/miniconda3/envs/testbed/bin/python -c 'import pytest_cov'")).return_code == 0
            run = await env.exec(f"cd /testbed && timeout 1700 {PYTEST} {'--no-cov' if cov else ''} {' '.join(files)} 2>&1",
                                 timeout_sec=1800)
            where = (await env.exec(f"cd / && /opt/miniconda3/envs/testbed/bin/python -c 'import {pkg}, os; "
                                    f"print(os.path.realpath({pkg}.__file__))'")).stdout.strip()
            copies = (await env.exec(f"find /opt/miniconda3/envs/testbed/lib -path '*-packages/{pkg}/*.py' | head -3")).stdout.split()
        finally:
            await env.stop(True)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", run.stdout)  # addopts such as --color=yes (cantools, moto, dask)
        passed = sorted(set(re.findall(r"^PASSED (.+?)\s*$", plain, re.M))
                        | set(re.findall(r"^(\S.*?::.*?) PASSED(?:\s+\[\s*\d+%\])?\s*$", plain, re.M)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"image": image, "orig": orig, "rc": run.return_code, "seconds": round(time.time() - start),
                                   "files": files, "package": pkg, "import_path": where, "site_copies": copies,
                                   "passed": passed}))
        print(json.dumps({"repo": key, "rc": run.return_code, "files": len(files), "passed": len(passed),
                          "import_path": where, "site_copies": copies, "seconds": round(time.time() - start)}), flush=True)


async def baseline(name: str, keys: list, conc: int = 8):
    images = local_images()
    sem = asyncio.Semaphore(conc)
    keys = sorted(images) if keys == ["all"] else keys
    done = await asyncio.gather(*(baseline_repo(name, k, images[k], sem) for k in keys), return_exceptions=True)
    for key, exc in zip(keys, done):
        if exc:  # that repo gets no baseline, so no bundles; the others go on
            print(json.dumps({"repo": key, "baseline_error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)


def variant_cmd(fixes) -> str:
    """One acceptance variant in a set-up container: back to the bundle's initial commit, the reference fixes applied
    as solve.sh does, then the bundle's own verifier with /tests and /logs/verifier laid out as Harbor does."""
    apply = "".join(f" && git apply --whitespace=nowarn /solution/fix_{k}.patch" for k in fixes)
    return ("cd /testbed && git reset -q --hard && git clean -fdq && rm -rf /logs/verifier && mkdir -p /logs/verifier"
            f"{apply} && bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1;"
            " for f in tasks seconds; do cat /logs/verifier/$f.json; echo; done; cat /logs/verifier/grader.txt;"
            " cat /logs/verifier/details.json")


def variants(row: dict) -> dict:
    """Acceptance variants -> the fixes applied: none, each task alone, all, and two seeded random subsets of about
    half the tasks (tasks sharing files must not interact in groups either); every pair instead for <= 4 tasks."""
    ks = [str(t["k"]) for t in row["tasks"]]
    subsets = ({f"p{a}{b}": [a, b] for a, b in itertools.combinations(ks, 2)} if len(ks) <= 4 else
               {f"s{i}": sorted(random.Random(f"{row['bundle']}:{i}").sample(ks, len(ks) // 2), key=int) for i in (1, 2)})
    return {"none": [], **{k: [k] for k in ks}, "all": ks} | subsets


async def accept_bundle(src: Path, row: dict, slot) -> tuple:
    """Two rounds of every variant in one container: {variant: (tasks.json, seconds.json, grader, details.json)}."""
    from octopus.envs.swe.data.validate_tasks import make_env
    d = src / row["bundle"]
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    runs = [{}, {}]
    async with slot():
        env = make_env(row["image"])
        try:
            await env.start(False)
            done = await env.exec(setup, timeout_sec=300)
            if done.return_code:
                row = row | {"error": f"setup rc {done.return_code}: {(done.stdout + done.stderr)[-300:]}"}
            else:
                await env.upload_dir(d / "tests", "/tests")
                await env.upload_dir(d / "solution", "/solution")
                for run in runs:
                    for v, fixes in variants(row).items():
                        out = (await env.exec(variant_cmd(fixes), timeout_sec=1900)).stdout.strip().splitlines()
                        try:
                            run[v] = (json.loads(out[0]), json.loads(out[1]), out[2], json.loads(out[3]))
                        except (IndexError, json.JSONDecodeError):
                            run[v] = (None, {}, "", {})
        except Exception as exc:  # a container that does not start: the bundle has no results, the others go on
            row = row | {"error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            await env.stop(True)
    return row, runs


def judge(row: dict, runs: list, base: dict) -> dict:
    """Task k solved exactly when fix k is in (alone, in a subset, all), both rounds equal, every task's tests < 60 s,
    the import check passes."""
    matrix, why = {}, []
    for v, fixes in variants(row).items():
        want = {str(t["k"]): int(str(t["k"]) in fixes) for t in row["tasks"]}
        got = [r.get(v, (None, {}, "", {})) for r in runs]
        matrix[v] = got[0][0]
        if got[0][0] is None or got[1][0] is None:
            why.append(f"{v}: no verifier result")
        elif got[0][0] != got[1][0]:
            why.append(f"{v}: flaky {got[0][0]} vs {got[1][0]}")
        elif got[0][0] != want:
            kind = {"none": "nofix not 0", "all": "all fixes not 1", "s1": "interaction (subset)",
                    "s2": "interaction (subset)"}.get(v, "interaction")
            why.append(f"{v}: {kind} {got[0][0]}")
        if v == "all" and max([*got[0][1].values(), *got[1][1].values()], default=0) >= 60:
            why.append(f"slow: {got[0][1]}")
    if row.get("error"):
        why.insert(0, row["error"])
    if not base["import_path"].startswith("/testbed/") or base["site_copies"]:
        why.append(f"leak: package imports from {base['import_path']} / site-packages copies {base['site_copies']}")
    every = [x for r in runs for x in r.values()]
    return {"accepted": not why, "why": why, "matrix": matrix, "fixes": variants(row),
            # both rounds raw, per variant
            "rounds": [{v: dict(zip(("tasks", "seconds", "grader", "details"), x)) for v, x in r.items()} for r in runs],
            "max_seconds": max((s for x in every for s in x[1].values()), default=None),
            "graders": dict(collections.Counter(x[2] for x in every)),
            # tasks whose FAIL_TO_PASS has a skipped id with every fix in
            "f2p_skipped_tasks": sorted(k for r in runs for k, d in r.get("all", (0, 0, 0, {}))[3].items()
                                        if d.get("f2p_skipped"))}


async def accept(name: str, conc: int = 16, conc_after: int = 0, after: str = "99:99"):
    """Accept every bundle of datasets/NAME (one container each, `conc` at once; `conc_after` from local time HH:MM
    `after` on); accepted bundles are copied to datasets/NAME-valid, the report goes to NAME.acceptance-1c.json."""
    src, valid = N / "datasets" / name, N / "datasets" / f"{name}-valid"
    rows = [json.loads(l) for l in open(f"{src}.manifest.jsonl")]
    running = 0

    class Slot:
        async def __aenter__(self):
            nonlocal running
            while running >= (conc if time.strftime("%H:%M") < after else conc_after):
                await asyncio.sleep(10)
            running += 1

        async def __aexit__(self, *exc):
            nonlocal running
            running -= 1

    # each judgement is appended as it lands, and a restart skips judged bundles: a shard can be stopped and its
    # unjudged bundles moved elsewhere without judging any bundle twice
    log = Path(f"{src}.acceptance-1c.jsonl")
    report = {r["bundle"]: r["report"] for r in map(json.loads, log.open())} if log.exists() else {}
    valid.mkdir(exist_ok=True)
    for done in asyncio.as_completed([accept_bundle(src, row, Slot) for row in rows if row["bundle"] not in report]):
        row, runs = await done
        base = json.loads((N / "datasets" / f"{name}.baseline" / f"{row['repo']}.json").read_text())
        report[row["bundle"]] = judge(row, runs, base) | {"repo": row["repo"], "split": row.get("split")}
        with log.open("a") as f:
            f.write(json.dumps({"bundle": row["bundle"], "report": report[row["bundle"]]}) + "\n")
        if report[row["bundle"]]["accepted"] and not (valid / row["bundle"]).exists():
            shutil.copytree(src / row["bundle"], valid / row["bundle"])
        print(json.dumps({"bundle": row["bundle"], **{k: report[row["bundle"]][k] for k in ("accepted", "why")}}), flush=True)
    Path(f"{src}.acceptance-1c.json").write_text(json.dumps(report, indent=1))
    print(f"{sum(r['accepted'] for r in report.values())}/{len(report)} bundles accepted -> {valid}; {src}.acceptance-1c.json")


def audit(name: str, n: int = 20, conc: int = 32, same_as: str = ""):
    """Independent check of the training path: n bundles of NAME-valid (repaired ones included), seeded and
    stratified over size (6, 8-11, 12-16) and accept shard, or the bundles of an earlier audit file (same_as); each
    variant as its own Harbor task run by the real `harbor run -a oracle` (solve.sh applies the fixes kept in that
    copy), one round. Every result must equal the single-container round 1 -> NAME.audit.json."""
    src, var = N / "datasets" / f"{name}-valid", N / "datasets" / f"{name}-audit"
    report = json.loads(Path(N / "datasets" / f"{name}.acceptance-1c.json").read_text())
    rows = {r["bundle"]: r for r in map(json.loads, open(f"{src}.manifest.jsonl"))}
    strata = collections.defaultdict(list)
    for b in sorted(rows):
        n_tasks = len(rows[b]["tasks"])
        strata[("6" if n_tasks < 8 else "8-11" if n_tasks < 12 else "12-16", report[b].get("shard"))].append(b)
    rng = random.Random(0)
    for group in strata.values():
        rng.shuffle(group)
    picked = ([g[i] for i in range(max(map(len, strata.values()))) for g in strata.values() if i < len(g)][:n]
              if not same_as else json.loads(Path(same_as).read_text())["bundles"])
    shutil.rmtree(var, ignore_errors=True)
    for b in picked:
        for v, fixes in variants(rows[b]).items():
            d = var / f"{b}--{v}"
            shutil.copytree(src / b, d)
            config = d / "task.toml"  # one Harbor task name per variant
            config.write_text(re.sub(r'^(name = "[^"]*)"', rf'\1--{v}"', config.read_text(), count=1, flags=re.M))
            for t in rows[b]["tasks"]:
                if str(t["k"]) not in fixes:
                    (d / f"solution/fix_{t['k']}.patch").unlink()
    job = f"audit-{name}"
    shutil.rmtree(N / "jobs" / job, ignore_errors=True)
    subprocess.run(["harbor", "run", "-p", str(var), "-a", "oracle", "-e", HARBOR_ENV,
                    "-n", str(conc), "-o", str(N / "jobs"), "--job-name", job, "--environment-build-timeout-multiplier", "6",
                    "-y"], stdout=open(N / "jobs" / f"{job}.console.log", "w"), stderr=subprocess.STDOUT)
    got = {}
    for rj in (N / "jobs" / job).glob("*/result.json"):
        tj = rj.parent / "verifier/tasks.json"
        got[Path(json.loads(rj.read_text())["config"]["task"]["path"]).name] = json.loads(tj.read_text()) if tj.exists() else None
    out = {b: {v: {"harbor": got.get(f"{b}--{v}"), "single_container": report[b]["matrix"][v]}
               for v in variants(rows[b])} for b in picked}
    mismatch = [f"{b}--{v}" for b in out for v, x in out[b].items() if x["harbor"] != x["single_container"]]
    Path(N / "datasets" / f"{name}.audit.json").write_text(json.dumps({"bundles": picked, "mismatch": mismatch, "results": out}, indent=1))
    print(json.dumps({"bundles": len(picked), "variants": sum(map(len, out.values())), "mismatch": mismatch,
                      "strata": {"/".join(map(str, k)): sum(b in picked for b in g) for k, g in strata.items()}}))


def diagnose(row: dict, rep: dict) -> tuple:
    """Culprit tasks of a rejected bundle, from its two raw acceptance rounds -> ({k: [evidence]}, [whole-bundle
    reasons a task removal cannot clear]). Each wrong cell blames the task whose result is wrong: a task solved
    without its own fix (with every bug in, or by other tasks' fixes) has its failure caused by another bug; a task
    left unsolved with its own fix in needs another task's fix; a task that differs between rounds is flaky; a task
    whose tests take >= 60 s with every fix in is slow."""
    files = {str(t["k"]): set(t["files"]) for t in row["tasks"]}
    blame, fatal, rounds = collections.defaultdict(list), [], rep.get("rounds") or []
    for w in rep["why"]:
        if w.startswith("leak") or not re.match(r"^(none|all|[sp]\d+|\d+|slow): ", w):
            fatal.append(w)
    if len(rounds) < 2:
        return {}, fatal or rep["why"]
    for v, fixes in rep["fixes"].items():
        a, b = (r.get(v, {}).get("tasks") for r in rounds)
        if a is None or b is None:
            fatal.append(f"{v}: no verifier result")
            continue
        for k in files:
            want = int(k in fixes)
            shared = sorted({f for j in fixes if j != k for f in files[j] & files[k]})
            failing = (rounds[0][v].get("details") or {}).get(k, {}).get("not_passed")
            if a[k] != b[k]:
                blame[k].append(f"flaky: variant {v} gives {a[k]} then {b[k]}")
            elif a[k] == want:
                continue
            elif v == "none":
                blame[k].append("its FAIL_TO_PASS passes with every bug in")
            elif want:
                blame[k].append(f"unsolved with fixes {fixes} in (own fix included); not passing: {failing}")
            else:
                blame[k].append(f"solved by fixes {fixes} without its own; shared bug files: {shared}")
    for r in rounds:
        for k, sec in (r.get("all", {}).get("seconds") or {}).items():
            if sec >= 60:
                blame[k].append(f"slow: {sec} s with every fix in")
    return dict(blame), fatal


def assemble(tree: Path, orig: str, tasks: list) -> bool:
    """`tasks` applied in order on a fresh tree, each committed, passing the leak and fix checks."""
    sh(tree, f"git checkout -q --force {orig} && git clean -fdq")
    for c in tasks:
        if not sh(tree, " && ".join(setup_steps(c)) + " && " + COMMIT):
            return False
    return not leaks(tree, tasks) and fixes_apply(tree, tasks)


def repair_repo(job: tuple) -> tuple:
    """Repair one repo's rejected bundles in order: drop the diagnosed culprits, refill to the drawn size from the
    repo's eligible tasks that no bundle has used (8-16 bundles that cannot get back to 8 become 6-fallback bundles
    if 6 hold, the way the build falls back), rebuild the dir. -> (manifest rows, plan entries)."""
    name, out, rnd, key, bad, report, prows, used = job
    baseline = json.loads((N / "datasets" / f"{name}.baseline" / f"{key}.json").read_text())
    image, orig = bad[0]["image"], baseline["orig"]
    repo = golden_repo(image)
    drops = collections.Counter()
    cands = eligible(repo, key, prows, baseline, drops)
    mark_hard(cands)
    by_id = {c["instance_id"]: c for c in cands}
    pool = [c for c in cands if c["instance_id"] not in used]
    random.Random(f"{out}:{key}").shuffle(pool)
    rows_out, plans = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        for row in bad:
            rep = report[row["bundle"]]
            blame, fatal = diagnose(row, rep)
            ids = {str(t["k"]): t["instance_id"] for t in row["tasks"]}
            entry = {"bundle": row["bundle"], "failure": rep["why"],
                     "culprits": {k: {"instance_id": ids[k], "evidence": ev} for k, ev in blame.items()}}
            keep = [by_id.get(t["instance_id"]) for t in row["tasks"] if str(t["k"]) not in blame]
            why = ("; ".join(fatal) if fatal else "no culprit task found" if not blame
                   else "a kept task is no longer eligible" if None in keep
                   else "kept tasks do not assemble" if not assemble(tree, orig, keep) else "")
            if not why:
                rule = row["size_rule"]
                picked = fill(tree, keep, pool, row["size_drawn"] if rule == "8-16" else 6, drops)
                new = picked[len(keep):]
                if rule == "8-16" and 6 <= len(picked) < 8:  # the pool cannot fill 8: fall back to 6, as the build does
                    picked, rule = picked[:6], "6-fallback"
                    why = "" if assemble(tree, orig, picked) else "6 kept tasks do not assemble"
                elif len(picked) < (8 if rule == "8-16" else 6):
                    why = f"the repo's unused tasks refill it to {len(picked)} only"
                pool += [c for c in new if why or c not in picked]  # tasks not taken go back
            if why:
                plans.append(entry | {"action": f"unrepairable: {why}"})
                continue
            new_id = re.sub(r"\.r\d+$", "", row["bundle"]) + f".r{rnd}"
            write_bundle(N / "datasets" / out, new_id, image, repo, picked)
            removed = [ids[k] for k in blame]
            added = [c["instance_id"] for c in picked if c["instance_id"] not in ids.values()]
            rows_out.append(manifest_row(new_id, key, image, picked, rule, row["size_drawn"], repair_of=row["bundle"],
                                         removed=removed, added=added))
            plans.append(entry | {"action": f"removed {len(removed)}, added {len(added)} -> {new_id} ({len(picked)} tasks, "
                                            f"{rule})", "new_bundle": new_id, "removed": removed, "added": added})
    return rows_out, plans


def repair(parquets: str, name: str, src: str):
    """Repair round for the rejected bundles of datasets/SRC (a shard of build NAME, or its previous repair round):
    writes datasets/<shard>-rep<round> + manifest + .plan.json, to be accepted like any shard. At most 2 rounds.
    Tasks used by any bundle of NAME or of an earlier repair are never reused."""
    import concurrent.futures, pyarrow.parquet as pq
    D = N / "datasets"
    m = re.match(r"(.*)-rep(\d+)$", src)
    shard, rnd = (m.group(1), int(m.group(2)) + 1) if m else (src, 1)
    if rnd > 2:
        raise SystemExit("at most 2 repair rounds")
    out = f"{shard}-rep{rnd}"
    report = json.loads((D / f"{src}.acceptance-1c.json").read_text())
    rows = [json.loads(l) for l in open(D / f"{src}.manifest.jsonl")]
    bad = [r for r in rows if not report[r["bundle"]]["accepted"]]
    used = {t["instance_id"] for f in [D / f"{name}.manifest.jsonl", *D.glob(f"{name}*-rep*.manifest.jsonl")]
            for r in map(json.loads, open(f)) for t in r["tasks"]}
    keys = sorted({r["repo"] for r in bad})
    prows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{parquets}/*.parquet")):
        for r in pq.read_table(f, columns=["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]).to_pylist():
            key = r["instance_id"].lower().rsplit(".", 1)[0]
            if key in keys and KINDS.search(r["instance_id"]):
                prows[key].append(r)
    (D / out).mkdir()
    (D / f"{out}.baseline").mkdir()
    for key in keys:
        shutil.copy(D / f"{name}.baseline" / f"{key}.json", D / f"{out}.baseline" / f"{key}.json")
    jobs = [(name, out, rnd, k, [r for r in bad if r["repo"] == k], report, prows[k], used) for k in keys]
    manifest, plans = [], []
    with concurrent.futures.ProcessPoolExecutor(16) as pool:
        for rows_out, plan in pool.map(repair_repo, jobs):
            manifest += rows_out
            plans += plan
    (D / f"{out}.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (D / f"{out}.plan.json").write_text(json.dumps(plans, indent=1))
    print(json.dumps({"rejected": len(bad), "rebuilt": len(manifest), "unrepairable": len(plans) - len(manifest),
                      "out": out}))


def repair_report(shard: str):
    """<shard>.repairs.json: per rejected bundle, every repair round's diagnosis, action and acceptance result."""
    D = N / "datasets"
    first = json.loads((D / f"{shard}.acceptance-1c.json").read_text())
    table = {b: {"failure": r["why"], "rounds": [], "result": "unrepairable"} for b, r in first.items() if not r["accepted"]}
    origin = {b: b for b in table}
    for rnd in (1, 2):
        plan_f, acc_f = D / f"{shard}-rep{rnd}.plan.json", D / f"{shard}-rep{rnd}.acceptance-1c.json"
        if not plan_f.exists():
            break
        acc = json.loads(acc_f.read_text()) if acc_f.exists() else {}
        for e in json.loads(plan_f.read_text()):
            b0 = origin[e["bundle"]]
            nb = e.get("new_bundle")
            res = acc.get(nb, {}) if nb else {}
            table[b0]["rounds"].append({"culprits": e["culprits"], "action": e["action"],
                                        "accepted": res.get("accepted"), "why": res.get("why")})
            if nb:
                origin[nb] = b0
                if res.get("accepted"):
                    table[b0]["result"] = f"repaired as {nb}"
    (D / f"{shard}.repairs.json").write_text(json.dumps(table, indent=1))
    rep = [t for t in table.values() if t["result"].startswith("repaired")]
    print(json.dumps({"rejected": len(table), "repaired": len(rep), "unrepairable": len(table) - len(rep)}))


async def time_bundle(d: Path, row: dict, sem) -> dict:
    """Variants none and all once in one container, each timed over the bundle's whole test.sh."""
    from octopus.envs.swe.data.validate_tasks import make_env
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    ks, out = [str(t["k"]) for t in row["tasks"]], {}
    async with sem:
        env = make_env(row["image"])
        try:
            await env.start(False)
            if (await env.exec(setup, timeout_sec=300)).return_code == 0:
                await env.upload_dir(d / "tests", "/tests")
                await env.upload_dir(d / "solution", "/solution")
                for v, fixes in (("none", []), ("all", ks)):
                    start = time.time()
                    lines = (await env.exec(variant_cmd(fixes), timeout_sec=1900)).stdout.strip().splitlines()
                    wall = round(time.time() - start, 1)
                    try:
                        out[v] = {"tasks": json.loads(lines[0]), "wall": wall, "grader": lines[2]}
                    except (IndexError, json.JSONDecodeError):
                        out[v] = {"tasks": None, "wall": wall, "grader": ""}
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            await env.stop(True)
    why = [out["error"]] if "error" in out else []
    for v in ("none", "all"):
        want = {k: int(v == "all") for k in ks}
        got = out.get(v, {}).get("tasks")
        if got != want:
            why.append(f"{v}: {got} (want {want})")
    if out.get("all", {}).get("wall", 0) > TIMING_LIMIT:
        why.append(f"slow (verifier wall time): {out['all']['wall']} s with every fix in")
    return {"bundle": row["bundle"], "ok": not why, "why": why, **out}


TIMING_LIMIT = 300  # s, grading with every fix in (the longest grading a bundle sees)


async def timing(name: str, manifest: str, conc: int = 16):
    """Timing gate over datasets/NAME bundles listed in MANIFEST: none and all once, wall time of the whole test.sh;
    both must give their expected row, and `all` must finish within TIMING_LIMIT -> MANIFEST.timing.jsonl (resumable)."""
    log = Path(f"{manifest}.timing.jsonl")
    done = {json.loads(l)["bundle"] for l in log.open()} if log.exists() else set()
    sem = asyncio.Semaphore(conc)
    rows = [r for r in map(json.loads, open(manifest)) if r["bundle"] not in done]
    for fut in asyncio.as_completed([time_bundle(N / "datasets" / name / r["bundle"], r, sem) for r in rows]):
        res = await fut
        with log.open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps({k: res.get(k) for k in ("bundle", "ok", "why")}), flush=True)


async def matrix(dataset: str, out: str, bundles: list):
    """Acceptance matrices of BUNDLE... of DATASET (a bundle dir with its .manifest.jsonl) through the OCTOPUS_HARBOR_ENV
    backend, all bundles at once -> OUT {bundle: {variant: [round 1 tasks.json, round 2]}}; run it once per backend
    and the two files must be equal (octopus/envs/swe/singularity.py against udocker)."""
    rows = {r["bundle"]: r for r in map(json.loads, open(f"{dataset}.manifest.jsonl"))}
    done = await asyncio.gather(*(accept_bundle(Path(dataset), rows[b], contextlib.nullcontext) for b in bundles))
    report = {row["bundle"]: {"error": row.get("error"), **{v: [r.get(v, (None,))[0] for r in runs] for v in variants(row)}}
              for row, runs in done}
    Path(out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({b: {"error": r["error"], "variants": len(r) - 1} for b, r in report.items()}))


def tree_diff(old: dict, new: dict) -> str:
    """git diff of {path: content} -> {path: content} (paths a/<p>, b/<p>, as the reference fixes have them)."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(p for p in old if old[p] != new[p]):
            for side, text in (("a", old[p]), ("b", new[p])):
                (Path(tmp) / side / p).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / side / p).write_text(text)
            d = subprocess.run(["git", "diff", "--no-index", "--binary", f"a/{p}", f"b/{p}"], cwd=tmp,
                               capture_output=True, text=True).stdout
            out.append(d.replace(f"a/a/{p}", f"a/{p}").replace(f"b/b/{p}", f"b/{p}"))
    return "".join(out)


def compiles(src: str) -> bool:
    try:
        compile(src, "<cut>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def reindent(lines: list, like: list) -> list:
    """Grok's partial code at the indentation of the unit it replaces (its first non-blank line's)."""
    first = lambda ls: next((l for l in ls if l.strip()), "")
    have, want = (lambda l: l[: len(l) - len(l.lstrip())])(first(lines)), (lambda l: l[: len(l) - len(l.lstrip())])(first(like))
    if have == want or not all(l.startswith(have) for l in lines if l.strip()):
        return lines
    return [want + l[len(have):] if l.strip() else l for l in lines]


def cut(tree: Path, tasks: list, labels: dict) -> tuple:
    """Cut `tasks` (assembled on `tree`) to what their statements describe: a unit labelled not described is fixed in
    the tree, a mixed one takes Grok's partial code, a described one stays the bug. -> (bake patch on the tree,
    {instance id: fix on the baked tree}, {instance id: why dropped}, ids cut). A task is kept as it is when it has no
    labels, its fix is not a plain text edit, or its labels were made for another fix; dropped when nothing of it is
    described or its cut code does not compile."""
    from octopus.envs.swe.data.statement_coverage import edit, units
    plan, dropped = {}, {}
    for c in tasks:
        lab, us = labels.get(c["instance_id"]) or {}, units(c["fix"], c["kind"])
        v = lab.get("verdict")
        if us is None or v is None or lab.get("units") != us:
            plan[c["instance_id"]] = None
        elif all(x["label"] == "not described" for x in v.values()):
            dropped[c["instance_id"]] = "the statement describes none of the fix"
        else:
            plan[c["instance_id"]] = {u["id"]: u["plus"] if v[u["id"]]["label"] == "not described"
                                      else reindent(v[u["id"]]["partial"].rstrip("\n").split("\n"), u["plus"] or u["minus"])
                                      for u in us if v[u["id"]]["label"] != "described"}
    for c in tasks:  # the units must rebuild the fix exactly (as git applies it), or the task stays as it is
        if plan.get(c["instance_id"]) is not None:
            paths = sorted({u["path"] for u in units(c["fix"], c["kind"])})
            texts = {p: (tree / p).read_text() for p in paths}
            subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=tree, input=c["fix"], text=True, capture_output=True)
            fixed = {p: (tree / p).read_text() for p in paths}
            sh(tree, "git reset -q --hard && git clean -fdq")
            try:
                same = edit(texts, c["fix"], {u["id"]: u["plus"] for u in units(c["fix"], c["kind"])}, c["kind"]) == fixed != texts
            except ValueError:
                same = False
            if not same:
                plan[c["instance_id"]] = None
    while True:
        kept = [c for c in tasks if c["instance_id"] in plan]
        t0 = {p: (tree / p).read_text() for c in kept if plan[c["instance_id"]] is not None
              for p in {u["path"] for u in units(c["fix"], c["kind"])}}
        tb, broken = dict(t0), set()
        for c in kept:
            if plan[c["instance_id"]]:
                try:
                    tb = edit(tb, c["fix"], plan[c["instance_id"]], c["kind"])
                except ValueError:
                    broken.add(c["instance_id"])
        bad = {p for p in tb if p.endswith(".py") and tb[p] != t0[p] and not compiles(tb[p])}
        broken |= {c["instance_id"] for c in kept if plan[c["instance_id"]] and {u["path"] for u in units(c["fix"], c["kind"])} & bad}
        if not broken:
            break
        for i in broken:
            plan.pop(i)
            dropped[i] = "the cut code does not compile, or the fix does not locate in the tree"
    fixes = {}
    for c in kept:
        if plan[c["instance_id"]] is None:
            fixes[c["instance_id"]] = c["fix"]
            continue
        g = dict(t0)
        for d in kept:
            if plan[d["instance_id"]] is not None:
                g = edit(g, d["fix"], {u["id"]: u["plus"] for u in units(d["fix"], d["kind"])} if d is c else plan[d["instance_id"]],
                         d["kind"])
        fixes[c["instance_id"]] = tree_diff(tb, g)
    return tree_diff(t0, tb), fixes, dropped, {i for i, p in plan.items() if p}


def reduce_repo(job: tuple) -> tuple:
    """One repo's bundles of datasets/NAME rebuilt under datasets/OUT with every task cut to its statement (cut).
    Dropped as well: tasks no longer eligible (a hidden source module), and with REFRESH (refresh's per bundle test
    results) tasks with no FAIL_TO_PASS id failing on the cut tree, and the cut task owning an id that the cuts made
    fail (every cut task of the bundle when only an uncut task owns it); a samefile bundle whose fixes on the cut tree
    are no longer coupled. FAIL_TO_PASS ids that pass on the cut tree become PASS_TO_PASS.
    -> (manifest rows, {bundle: {instance id: what happened}})."""
    from octopus.envs.swe.data.statement_coverage import units
    name, out, key, rows, prows, labels, refresh, base_name, statements = job
    baseline = json.loads((N / "datasets" / f"{base_name}.baseline" / f"{key}.json").read_text())
    image, orig, passed = rows[0]["image"], baseline["orig"], set(baseline["passed"])
    repo = golden_repo(image)
    cands, why = {}, {}
    for r in prows:
        c = candidate(repo, r)
        w = ineligible(c, orig, passed)
        if w:
            why[r["instance_id"]] = w
        else:
            cands[r["instance_id"]] = c | {"passed": passed}
    rows_out, report = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        for row in rows:
            rep = report[row["bundle"]] = {}
            res = refresh.get(row["bundle"])
            if res and res.get("error"):
                rep["refresh"], res = f"refresh failed: {res['error'][:200]}", None
            tasks = []
            for t in row["tasks"]:
                if t["instance_id"] not in cands:
                    rep[t["instance_id"]] = f"dropped: {why.get(t['instance_id'], 'no SWE-smith row')}"
                    continue
                c = dict(cands[t["instance_id"]], hard=t["hard"])
                c["statement"] = statements.get(t["instance_id"], c["statement"])  # snippet repairs
                tasks.append(c)
            while True:  # a dropped task's bug leaves the tree: assemble and cut again without it
                if not tasks or not assemble(tree, orig, tasks):
                    rep["bundle"] = "dropped: no task left" if not tasks else "dropped: kept tasks do not assemble"
                    break
                bake, fixes, dropped, cut_ids = cut(tree, tasks, labels)
                if res:
                    # an id the cuts made fail: drop the cut task it belongs to (one of its FAIL_TO_PASS, or in its
                    # hidden test files); one that only an uncut task owns drops every cut task of the bundle
                    owns = lambda c, t: t in c["f2p"] or t.split("::")[0] in set(c["deleted"]) | set(c["modified"])
                    for t in res["broken"]:
                        culprits = [c["instance_id"] for c in tasks if c["instance_id"] in cut_ids and owns(c, t)]
                        for i in culprits or cut_ids:
                            dropped.setdefault(i, f"dropped: the cut made ids fail that passed before ({t})")
                    for c in tasks:
                        if c["instance_id"] in cut_ids and not [t for t in c["f2p"] if res["v3"].get(t) != "passed"]:
                            dropped[c["instance_id"]] = "dropped: no FAIL_TO_PASS id fails on the cut tree"
                if not dropped:
                    break
                for i, w in dropped.items():
                    rep[i] = w if w.startswith("dropped") else f"dropped: {w}"
                tasks = [c for c in tasks if c["instance_id"] not in dropped]
            if "bundle" in rep:
                continue
            for c in tasks:
                if c["instance_id"] in cut_ids:
                    if res:
                        now = [t for t in c["f2p"] if res["v3"].get(t) == "passed"]
                        c["f2p"], c["p2p"] = [t for t in c["f2p"] if t not in now], c["p2p"] + now
                    lines = sum(1 for l in fixes[c["instance_id"]].splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
                    rep[c["instance_id"]] = f"cut: fix lines {c['patch_lines']} -> {lines}"
                    c["patch_lines"] = lines
                else:
                    rep[c["instance_id"]] = "kept"
                c["fix"] = fixes[c["instance_id"]]
            leak = [c for c in tasks if any(l in c["statement"] for l in fix_lines(c["fix"]))]
            if leak:  # a repaired statement must not hold reference-fix code (as ineligible)
                rep["bundle"] = f"dropped: statement holds reference-fix code ({leak[0]['instance_id']})"
                continue
            if bake:
                subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=tree, input=bake, text=True, check=True)
                sh(tree, COMMIT)
            for c in tasks:  # a fix line already in its own cut files is in view anyway, as an instance's inherent lines
                if c["instance_id"] in cut_ids:
                    own = {l.strip() for p in {u["path"] for u in units(c["fix"], c["kind"])} if (tree / p).is_file()
                           for l in (tree / p).read_text(errors="ignore").splitlines()}
                    c["inherent"] = c["inherent"] | (fix_lines(c["fix"]) & own)
            if row["size_rule"] == "samefile":
                if shipped_hunks(tree, tasks) or not coupled(tasks):
                    rep["bundle"] = "dropped: no longer a coupled same-file bundle"
                    continue
            problems = leaks(tree, tasks) + ([] if fixes_apply(tree, tasks) else ["a cut fix does not apply"])
            if problems:
                rep["bundle"] = f"dropped: {problems[:3]}"
                continue
            write_bundle(N / "datasets" / out, row["bundle"], image, repo, tasks, bake)
            rows_out.append(manifest_row(row["bundle"], key, image, tasks, row["size_rule"], row["size_drawn"],
                                         cut_from=name, cut=rep))
    return rows_out, report


def reduce(parquets: str, name: str, out: str, labels_path: str, refresh_path: str = "", base_name: str = "bundles-v2",
           statements_path: str = ""):
    """datasets/OUT: datasets/NAME's bundles with every task cut to what its statement describes (reduce_repo),
    from statement_coverage judge labels (LABELS.jsonl) and optionally refresh's results; same splits.
    -> OUT.manifest.jsonl, OUT.splits.json, OUT.reduce.json."""
    import concurrent.futures, pyarrow.parquet as pq
    if (N / "datasets" / out).exists():
        raise SystemExit(f"datasets/{out} exists; pick a new name")
    rows = [json.loads(l) for l in open(N / "datasets" / f"{name}.manifest.jsonl")]
    used = {t["instance_id"] for r in rows for t in r["tasks"]}
    prows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{parquets}/*.parquet")):
        for r in pq.read_table(f, columns=["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]).to_pylist():
            if r["instance_id"] in used:
                prows[r["instance_id"].lower().rsplit(".", 1)[0]].append(r)
    labels = {r["instance_id"]: r for r in map(json.loads, open(labels_path)) if "verdict" in r}
    refresh = {r["bundle"]: r for r in map(json.loads, open(refresh_path))} if refresh_path and refresh_path != "-" else {}
    statements = json.loads(Path(statements_path).read_text()) if statements_path else {}  # statement_coverage overrides
    by = collections.defaultdict(list)
    for r in rows:
        by[r["repo"]].append(r)
    (N / "datasets" / out).mkdir(parents=True)
    splits, report = {"train": {}, "eval": {}}, {}
    jobs = [(name, out, key, rs, prows[key], labels, refresh, base_name, statements) for key, rs in sorted(by.items())]
    with open(N / "datasets" / f"{out}.manifest.jsonl", "w") as manifest, concurrent.futures.ProcessPoolExecutor(16) as pool:
        for (key, rs), (bundles, rep) in zip(sorted(by.items()), pool.map(reduce_repo, jobs)):
            report |= rep
            for row in bundles:
                splits[row["split"]].setdefault(key, []).append(row["bundle"])
                manifest.write(json.dumps(row) + "\n")
            print(json.dumps({"repo": key, "bundles": f"{len(bundles)}/{len(rs)}",
                              "tasks": dict(collections.Counter(v.split(":")[0] for b in rep.values() for v in b.values()))}),
                  flush=True)
    Path(N / "datasets" / f"{out}.splits.json").write_text(json.dumps(splits, indent=1))
    Path(N / "datasets" / f"{out}.reduce.json").write_text(json.dumps(report, indent=1))


def refill_repo(job: tuple) -> tuple:
    """One repo's train bundles of datasets/CUT (a reduce output) left with fewer than SMALL tasks, or rejected by its
    accept (minus diagnose's culprits), refilled to TARGET tasks from the repo's eligible SWE-smith tasks that no bundle
    of datasets/SRC uses (fill: no conflict, kinds spread, easy and hard alternating). Written with their original
    bugs under datasets/OUT (same bundle id), to go through judge (the new tasks), reduce, refresh and accept like SRC.
    -> (manifest rows, {bundle: what happened})."""
    out, key, rows, report, prows, used, base_name, target = job
    baseline = json.loads((N / "datasets" / f"{base_name}.baseline" / f"{key}.json").read_text())
    image, orig = rows[0]["image"], baseline["orig"]
    repo = golden_repo(image)
    drops = collections.Counter()
    cands = eligible(repo, key, prows, baseline, drops)
    mark_hard(cands)
    by_id = {c["instance_id"]: c for c in cands}
    pool = [c for c in cands if c["instance_id"] not in used]
    random.Random(f"{out}:{key}").shuffle(pool)
    rows_out, plans = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        for row in rows:
            rep = report[row["bundle"]]
            blame, fatal = diagnose(row, rep) if not rep["accepted"] else ({}, [])
            keep = [by_id.get(t["instance_id"]) for t in row["tasks"] if str(t["k"]) not in blame]
            if fatal or None in keep or not assemble(tree, orig, keep):
                plans[row["bundle"]] = f"unrefillable: {fatal[:2] or 'kept tasks do not assemble'}"
                continue
            picked = fill(tree, keep, pool, max(target, len(keep)), drops)
            new = picked[len(keep):]
            plans[row["bundle"]] = {"kept": len(keep), "blamed": sorted(blame), "added": [c["instance_id"] for c in new],
                                    "pool_left": len(pool)}
            write_bundle(N / "datasets" / out, row["bundle"], image, repo, picked)
            rows_out.append(manifest_row(row["bundle"], key, image, picked, row["size_rule"], row["size_drawn"],
                                         refill_of=row["bundle"], added=[c["instance_id"] for c in new]))
    return rows_out, plans


def refill(parquets: str, src: str, cut: str, out: str, small: int = 6, target: int = 8, base_name: str = "bundles-v2"):
    """datasets/OUT: CUT's train bundles under SMALL tasks or rejected by CUT's accept, refilled (refill_repo)
    -> OUT.manifest.jsonl, OUT.splits.json, OUT.refill.json; then judge their new tasks, reduce, refresh, accept."""
    import concurrent.futures, pyarrow.parquet as pq
    if (N / "datasets" / out).exists():
        raise SystemExit(f"datasets/{out} exists; pick a new name")
    rows = [json.loads(l) for l in open(N / "datasets" / f"{cut}.manifest.jsonl")]
    report = {r["bundle"]: r["report"] for r in map(json.loads, open(N / "datasets" / f"{cut}.acceptance-1c.jsonl"))}
    todo = [r for r in rows if r["split"] == "train" and r["bundle"] in report
            and (not report[r["bundle"]]["accepted"] or len(r["tasks"]) < small)]
    used = {t["instance_id"] for r in map(json.loads, open(N / "datasets" / f"{src}.manifest.jsonl")) for t in r["tasks"]}
    keys = {r["repo"] for r in todo}
    prows = collections.defaultdict(list)
    cols = ["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]
    for f in sorted(glob.glob(f"{parquets}/*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            key = r["instance_id"].lower().rsplit(".", 1)[0]
            if key in keys and KINDS.search(r["instance_id"]):
                prows[key].append(r)
    by = collections.defaultdict(list)
    for r in todo:
        by[r["repo"]].append(r)
    (N / "datasets" / out).mkdir(parents=True)
    splits, plans = {"train": {}, "eval": {}}, {}
    jobs = [(out, key, rs, report, prows[key], used, base_name, target) for key, rs in sorted(by.items())]
    with open(N / "datasets" / f"{out}.manifest.jsonl", "w") as manifest, concurrent.futures.ProcessPoolExecutor(16) as pool:
        for (key, rs), (bundles, plan) in zip(sorted(by.items()), pool.map(refill_repo, jobs)):
            plans |= plan
            for row in bundles:
                splits[row["split"]].setdefault(key, []).append(row["bundle"])
                manifest.write(json.dumps(row) + "\n")
            print(json.dumps({"repo": key, "bundles": f"{len(bundles)}/{len(rs)}",
                              "added": sum(len(p["added"]) for p in plan.values() if isinstance(p, dict))}), flush=True)
    Path(N / "datasets" / f"{out}.splits.json").write_text(json.dumps(splits, indent=1))
    Path(N / "datasets" / f"{out}.refill.json").write_text(json.dumps(plans, indent=1))


RUNNER_PY = re.search(r"RUNNER = '''(.*?)'''", CHECK, re.S).group(1)


async def refresh_tree(d: Path, ids: list, image: str, slot) -> dict:
    """Every id in `ids` on bundle dir d's initial tree, hidden tests restored as the verifier does: {id: outcome}."""
    from octopus.envs.swe.data.validate_tasks import make_env
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    async with slot:
        env = make_env(image)
        try:
            await env.start(False)
            done = await env.exec(setup, timeout_sec=300)
            if done.return_code:
                return {"error": f"setup rc {done.return_code}: {(done.stdout + done.stderr)[-300:]}"}
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "ids.json").write_text(json.dumps(ids))
                (Path(tmp) / "runner.py").write_text(RUNNER_PY)
                await env.upload_dir(tmp, "/tmp/refresh")
            await env.upload_dir(d / "tests", "/tests")
            r = await env.exec("cd /testbed && cp -a /tests/hidden_full/. /testbed/ && /opt/miniconda3/envs/testbed/bin/python "
                               "/tmp/refresh/runner.py /tmp/refresh/ids.json /tmp/refresh/out.json -q --tb=no > /dev/null "
                               "2>&1; cat /tmp/refresh/out.json", timeout_sec=1900)
            return json.loads(r.stdout)["res"]
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            await env.stop(True)


async def refresh(name: str, out: str, conc: int = 16):
    """For each bundle of datasets/OUT (a reduce output) with a cut task: every graded id of datasets/NAME's bundle on
    the old and on the cut initial tree -> OUT.refresh.jsonl {bundle, v2, v3, broken (passed before, not now)};
    then run reduce again with it."""
    rows = [json.loads(l) for l in open(N / "datasets" / f"{out}.manifest.jsonl")]
    log = N / "datasets" / f"{out}.refresh.jsonl"
    done = {json.loads(l)["bundle"] for l in log.open()} if log.exists() else set()
    sem = asyncio.Semaphore(conc)

    async def one(row):
        old = N / "datasets" / name / row["bundle"]
        ids = sorted({i for t in json.loads((old / "tests/tasks.json").read_text()).values() for i in t["f2p"] + t["p2p"]})
        v2, v3 = await asyncio.gather(refresh_tree(old, ids, row["image"], sem),
                                      refresh_tree(N / "datasets" / out / row["bundle"], ids, row["image"], sem))
        rec = {"bundle": row["bundle"], "v2": v2, "v3": v3,
               "broken": [i for i in ids if v2.get(i) == "passed" and v3.get(i) != "passed"]}
        if "error" in v2 or "error" in v3:
            rec["error"] = v2.get("error") or v3.get("error")
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("bundle", "error")} | {"broken": len(rec["broken"])}), flush=True)

    await asyncio.gather(*(one(r) for r in rows if r["bundle"] not in done
                           and any(str(v).startswith("cut") for v in r["cut"].values())))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "baseline":
        asyncio.run(baseline(args[0], args[1:]))
    elif cmd == "build":  # PARQUET_DIR NAME KEY...|all [per_repo] [--sizes 1,2,3] [--only IDS_FILE] [--tag b] [--rule samefile]
        opts = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
        plain = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
        keys = [a for a in plain[2:] if not a.isdigit()]
        nums = [int(a) for a in plain[2:] if a.isdigit()]
        build(plain[0], plain[1], keys, *nums, sizes=[int(x) for x in opts.get("--sizes", "").split(",") if x],
              only=opts.get("--only", ""), tag=opts.get("--tag", "b"), rule=opts.get("--rule", ""))
    elif cmd == "accept":
        asyncio.run(accept(args[0], *[int(a) if a.isdigit() else a for a in args[1:]]))
    elif cmd == "audit":
        audit(args[0], *[int(a) if a.isdigit() else a for a in args[1:]])
    elif cmd == "matrix":  # DATASET_DIR OUT_JSON BUNDLE...
        asyncio.run(matrix(args[0], args[1], args[2:]))
    elif cmd == "timing":  # NAME MANIFEST [conc]
        asyncio.run(timing(args[0], args[1], *map(int, args[2:])))
    elif cmd == "repair":  # PARQUET_DIR NAME SRC
        repair(*args)
    elif cmd == "repairs":  # SHARD
        repair_report(args[0])
    elif cmd == "drop-budget":  # bundle dirs and/or manifest .jsonl files
        drop_budget(args)
    elif cmd == "reduce":  # PARQUET_DIR NAME OUT LABELS.jsonl [REFRESH.jsonl|- [BASELINE_NAME [STATEMENTS.json]]]
        reduce(*args)
    elif cmd == "refill":  # PARQUET_DIR SRC CUT OUT [small=6 [target=8]]
        refill(*args[:4], *map(int, args[4:6]))
    elif cmd == "refresh":  # NAME OUT [conc]
        asyncio.run(refresh(args[0], args[1], *map(int, args[2:])))
    elif cmd == "clean":  # add clean_command to the healthcheck of the given bundle dirs (built before it existed)
        for d in map(Path, args):
            text = (d / "task.toml").read_text()
            pkgs = {package(f) for p in d.glob("solution/fix_*.patch") for f in re.findall(r"^\+\+\+ b/(.+\.py)$", p.read_text(), re.M)}
            text = re.sub(r" && if \[ -d /opt/miniconda3/envs/testbed \]; then rm -rf [^;]*; fi && find /testbed -name __pycache__ "
                          r"-type d -prune -exec rm -rf \{\} \+", "", text)  # an earlier clean step is replaced
            assert text.count(" && rm -rf .git && git init -q") == 1, d
            (d / "task.toml").write_text(text.replace(" && rm -rf .git && git init -q", f" && {clean_command(pkgs)} && rm -rf .git && git init -q"))
    elif cmd == "check":  # rewrite tests/check.py of the given bundle dirs with the current grader
        for d in args:
            Path(d, "tests/check.py").write_text(CHECK)
