"""Bundle builder on a toy SWE-smith repo: original commit + 6 instance branches (Bug Patch, Remove F2P Tests), three
bugs per module in different functions. Too few tasks for 8-16, so one 6-fallback bundle; the healthcheck builds one
tree with all 6 bugs and no hidden tests, and grading gives task k = 1 exactly when fix k is applied."""
import collections, json, shutil, subprocess, sys, tomllib

from octopus.envs.swe.data import build_bundles as bb

X = "abcdef"
GOOD = {x: f"value of {x} that the original code returns, long enough" for x in X}
MODULE = {x: "pkg/m.py" if x in "abc" else "pkg/n.py" for x in X}
TESTS = {f"tests/test_{x}.py": f"from pkg.{MODULE[x][4]} import f{x}\ndef test_f{x}(): assert f{x}() == {{{x}!r}}\n"
         f"def test_{x}_other(): pass\n" for x in X}
TESTS["tests/test_shared.py"] = ("from pkg.m import fa, fb\ndef test_fa2(): assert fa() == {a!r}\n"
                                 "def test_fb2(): assert fb() == {b!r}\ndef test_shared_other(): pass\n")
REMOVE = {x: [f"tests/test_{x}.py"] for x in "def"} | {"a": ["tests/test_a.py", "tests/test_shared.py"],
                                                         "b": ["tests/test_b.py", "tests/test_shared.py"], "c": []}
F2P = {x: [f"tests/test_{x}.py::test_f{x}"] for x in X}
F2P["a"].append("tests/test_shared.py::test_fa2")
F2P["b"].append("tests/test_shared.py::test_fb2")
ALL = [f"{p}::{n}" for p, src in TESTS.items() for n in src.split("def ")[1:] for n in [n.split("(")[0]]]


def module(path, bug=None, extra=False):
    pad = lambda x: "    # padding\n" * 2 + ("    # padding, changed by the bug too\n" if extra and x == bug else "    # padding\n")
    return "".join(f"def f{x}():\n{pad(x)}    return {('bug' if x == bug else GOOD[x])!r}\n\n\n# spacer\n# spacer\n\n"
                   for x in X if MODULE[x] == path)


def sh(cwd, cmd):
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, check=True, capture_output=True, text=True).stdout


def commit(repo, msg):
    sh(repo, f"git add -A && git -c user.name=t -c user.email=t@t commit -qm '{msg}' && git rev-parse HEAD")


def toy_repo(tmp_path, extra=""):
    """The toy image repo -> (golden, orig, SWE-smith rows); task `extra`'s bug also changes a padding line."""
    golden = tmp_path / "golden"
    (golden / "pkg").mkdir(parents=True)
    (golden / "tests").mkdir()
    (golden / "pkg/__init__.py").write_text("")
    for path in ("pkg/m.py", "pkg/n.py"):
        (golden / path).write_text(module(path))
    for path, src in TESTS.items():
        (golden / path).write_text(src.format(**GOOD))
    sh(golden, "git init -q")
    commit(golden, "Initial commit")
    orig = sh(golden, "git rev-parse HEAD").strip()
    rows = []
    for x in X:
        iid = f"toy__repo.abc.func_basic__{x}"
        sh(golden, f"git checkout -q -b {x} {orig}")
        (golden / MODULE[x]).write_text(module(MODULE[x], bug=x, extra=x == extra))
        commit(golden, "Bug Patch")
        for path in REMOVE[x]:
            (golden / path).unlink()
        if x == "c":  # a modified test file: only the F2P test goes
            (golden / "tests/test_c.py").write_text("def test_c_other(): pass\n")
        commit(golden, "Remove F2P Tests")
        sh(golden, f"git update-ref refs/remotes/origin/{iid} HEAD")
        rows.append({"instance_id": iid, "problem_statement": f"f{x} in {MODULE[x]} returns the wrong value.",
                     "FAIL_TO_PASS": F2P[x], "PASS_TO_PASS": [t for t in ALL if t not in F2P[x]]})
    sh(golden, f"git checkout -q {orig}")
    return golden, orig, rows


def grading_matrix(tmp_path, dest, tree, ks):
    """{fixes applied: tasks.json} for none, each fix alone and all, in the healthcheck tree (the acceptance variants)."""
    tests = tmp_path / "tests_dir"
    shutil.copytree(dest / "tests", tests, dirs_exist_ok=True)
    logs = tmp_path / "logs"
    (tests / "check.py").write_text((tests / "check.py").read_text().replace("/logs/verifier", str(logs)))
    (tests / "test.sh").write_text(bb.TEST_SH.replace("/opt/miniconda3/envs/testbed/bin/python", sys.executable)
                                   .replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
    out = {}
    for fixes in [(), *((k,) for k in ks), tuple(ks)]:
        cmd = (bb.variant_cmd(fixes).replace("/logs/verifier", str(logs)).replace("/solution/", f"{dest}/solution/")
               .replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
        out[fixes] = json.loads(sh(tmp_path, cmd).splitlines()[0])
    return out


def healthcheck(tmp_path, golden, dest):
    tree, statements = tmp_path / "testbed", tmp_path / "tasks"
    shutil.copytree(golden, tree)
    setup = tomllib.loads((dest / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    sh(tmp_path, setup.replace("/testbed", str(tree)).replace("/tasks", str(statements)))
    return tree


def test_bundle_tree_and_per_task_grading(tmp_path):
    golden, orig, rows = toy_repo(tmp_path)
    assert bb.branches(golden, "toy__repo.abc") == [r["instance_id"] for r in rows]

    cands = [bb.candidate(golden, r) for r in rows]
    assert [bb.ineligible(c, orig, set(ALL)) for c in cands] == [""] * 6
    assert {c["functions"][0] for c in cands} == {f"{MODULE[x]}:f{x}" for x in X}
    for c in cands:
        c["passed"] = set(ALL)
    drops = collections.Counter()
    bundles = bb.plan_bundles(golden, orig, cands, 2, 0, drops)
    assert [(len(t), rule) for t, rule, _ in bundles] == [(6, "6-fallback")], drops
    tasks = bundles[0][0]
    dest = bb.write_bundle(tmp_path / "ds", "toy.b00", "img", golden, tasks)

    # healthcheck in a copy of the image repo
    tree, statements = tmp_path / "testbed", tmp_path / "tasks"
    shutil.copytree(golden, tree)
    setup = tomllib.loads((dest / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    sh(tmp_path, setup.replace("/testbed", str(tree)).replace("/tasks", str(statements)))
    assert sh(tree, "git log --oneline").count("\n") == 1
    for path in ("pkg/m.py", "pkg/n.py"):
        assert (tree / path).read_text().count("'bug'") == 3
    assert not (tree / "tests/test_a.py").exists() and not (tree / "tests/test_shared.py").exists()
    assert "test_fc" not in (tree / "tests/test_c.py").read_text() and "test_c_other" in (tree / "tests/test_c.py").read_text()
    assert bb.leaks(tree, tasks) == []
    assert sorted(p.name for p in statements.iterdir()) == [f"{k}.md" for k in range(1, 7)] + ["INDEX.md"]
    assert (statements / "INDEX.md").read_text().count("\n") == 6

    graded = json.loads((dest / "tests/tasks.json").read_text())
    a = next(str(k) for k, c in enumerate(tasks, 1) if c["instance_id"].endswith("__a"))
    assert "tests/test_shared.py::test_fb2" not in graded[a]["p2p"]  # another task's F2P is never run
    assert graded[a]["p2p"] == ["tests/test_a.py::test_a_other", "tests/test_shared.py::test_shared_other"]

    assert '[agent]\ntimeout_sec = 3600.0' in (dest / "task.toml").read_text()
    assert '[verifier]\ntimeout_sec = 1800.0' in (dest / "task.toml").read_text()
    fresh = (dest / "instruction.md").read_text()
    assert "budget" not in fresh
    old = fresh.replace("hidden tests.", "hidden tests. You have a budget of 48000 tokens for the whole repository.")
    (dest / "instruction.md").write_text(old)
    manifest = tmp_path / "m.manifest.jsonl"
    manifest.write_text(json.dumps({"bundle": "toy.b00", "budget": 48000, "tasks": []}) + "\n")
    bb.drop_budget([dest, manifest])  # bundles built before the shared budget was dropped
    assert (dest / "instruction.md").read_text() == fresh
    assert json.loads(manifest.read_text()) == {"bundle": "toy.b00", "tasks": []}
    assert 100 < sum(bb.split_of(f"repo{i}") == "eval" for i in range(1000)) < 200

    # acceptance variants in the one set-up tree, twice: reset, fixes as solve.sh applies them, the bundle's test.sh
    tests = tmp_path / "tests_dir"
    shutil.copytree(dest / "tests", tests)
    logs = tmp_path / "logs"
    (tests / "check.py").write_text((tests / "check.py").read_text().replace("/logs/verifier", str(logs)))
    (tests / "test.sh").write_text(bb.TEST_SH.replace("/opt/miniconda3/envs/testbed/bin/python", sys.executable)
                                   .replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
    ks = range(1, 7)
    for fixes in [(), *((k,) for k in ks), tuple(ks)] * 2:
        cmd = (bb.variant_cmd(fixes).replace("/logs/verifier", str(logs)).replace("/solution/", f"{dest}/solution/")
               .replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
        tasks_json, _, grader, _ = sh(tmp_path, cmd).splitlines()
        assert json.loads(tasks_json) == {str(k): int(k in fixes) for k in ks}, fixes
        assert grader == "one run"

    # a fix that makes the code under test call pytest.skip solves nothing: task a scores 0 with every fix in
    a = int(a)
    skip = (bb.variant_cmd([str(k) for k in ks]).split(" && bash ")[0]
            + " && sed -i \"s/return 'value of a .*'/import pytest; pytest.skip('no')/\" pkg/m.py && bash /tests/test.sh")
    sh(tmp_path, skip.replace("/logs/verifier", str(logs)).replace("/solution/", f"{dest}/solution/")
       .replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
    assert json.loads((logs / "tasks.json").read_text()) == {str(k): int(k != a) for k in ks}
    assert json.loads((logs / "details.json").read_text())[str(a)] == {
        "f2p_skipped": 2, "not_passed": ["tests/test_a.py::test_fa", "tests/test_shared.py::test_fa2"]}

    # a hidden test file that does not collect: the one run is unusable, each task gets its own run under the same
    # rule, and only the task whose ids live in that file scores 0
    e = next(k for k, c in enumerate(tasks, 1) if c["instance_id"].endswith("__e"))
    (tests / "hidden_full/tests/test_e.py").write_text("def broken(:\n")
    cmd = (bb.variant_cmd([str(k) for k in ks]).replace("/logs/verifier", str(logs))
           .replace("/solution/", f"{dest}/solution/").replace("/tests/", f"{tests}/").replace("/testbed", str(tree)))
    tasks_json, _, grader, _ = sh(tmp_path, cmd).splitlines()
    assert json.loads(tasks_json) == {str(k): int(k != e) for k in ks} and grader.startswith("per task: ")


def test_diagnose_blames_the_wrong_cell():
    row = {"tasks": [{"k": k, "files": ["pkg/m.py"] if k < 3 else ["pkg/n.py"]} for k in (1, 2, 3)]}
    fixes = {"none": [], "1": ["1"], "2": ["2"], "3": ["3"], "all": ["1", "2", "3"], "s1": ["1"], "s2": ["2"]}
    want = lambda v: {str(k): int(str(k) in fixes[v]) for k in (1, 2, 3)}
    got = {v: want(v) for v in fixes}
    got["1"]["2"] = 1  # fixing 1 also solves 2: 2's failure comes from bug 1
    got["s1"]["2"] = 1
    got["none"]["3"] = 1  # 3 passes with every bug in
    rounds = [{v: {"tasks": t, "seconds": {"1": 1.0, "2": 1.0, "3": 70.0} if v == "all" else {}} for v, t in got.items()}] * 2
    blame, fatal = bb.diagnose(row, {"why": ["1: interaction", "none: nofix not 0"], "fixes": fixes, "rounds": rounds})
    assert fatal == [] and set(blame) == {"2", "3"}
    assert any("shared bug files: ['pkg/m.py']" in e for e in blame["2"])
    assert any(e.startswith("slow: 70.0") for e in blame["3"])


def test_cut_keeps_only_the_described_defect(tmp_path):
    """Task a's bug also changes a padding line its statement never mentions: reduce fixes that line in the bundle
    tree (the bake), a's fix restores only the return value, and grading still gives task k = 1 exactly when fix k is
    applied. A task whose statement describes none of its fix is dropped; a hidden source module makes a task
    ineligible."""
    from octopus.envs.swe.data.statement_coverage import units
    golden, orig, rows = toy_repo(tmp_path, extra="a")
    cands = [bb.candidate(golden, r) | {"passed": set(ALL)} for r in rows]
    tasks = bb.plan_bundles(golden, orig, cands, 1, 0, collections.Counter())[0][0]
    tree = tmp_path / "assembled"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(golden), str(tree)], check=True)
    assert bb.assemble(tree, orig, tasks)
    a = next(c for c in tasks if c["instance_id"].endswith("__a"))
    b = next(c for c in tasks if c["instance_id"].endswith("__b"))
    us = units(a["fix"])
    assert [u["plus"] for u in us] == [["    # padding"], [f"    return {GOOD['a']!r}"]]
    labels = {a["instance_id"]: {"units": us, "verdict": {"U1": {"label": "not described"}, "U2": {"label": "described"}}},
              b["instance_id"]: {"units": units(b["fix"]), "verdict": {"U1": {"label": "not described"}}}}
    bake, fixes, dropped, cut_ids = bb.cut(tree, tasks, labels)
    assert cut_ids == {a["instance_id"]} and list(dropped) == [b["instance_id"]]
    assert "+    # padding\n" in bake and "-    return 'bug'" not in bake
    assert "changed by the bug" not in fixes[a["instance_id"]] and f"+    return {GOOD['a']!r}" in fixes[a["instance_id"]]

    kept = [c for c in tasks if c is not b]
    assert bb.assemble(tree, orig, kept)  # b's bug leaves the tree; then as reduce_repo: cut, bake, the new fixes
    bake, fixes, dropped, cut_ids = bb.cut(tree, kept, labels)
    kept = [c | {"fix": fixes[c["instance_id"]]} for c in kept]
    subprocess.run(["git", "apply"], cwd=tree, input=bake, text=True, check=True)
    bb.sh(tree, bb.COMMIT)
    assert bb.leaks(tree, kept) == [] and bb.fixes_apply(tree, kept)
    dest = bb.write_bundle(tmp_path / "ds", "toy.b00", "img", golden, kept, bake)
    from octopus.envs.swe.data.statement_coverage import statement
    assert statement(dest, 1) == kept[0]["statement"] + "\n"  # the statements tar, not the bake patch before it
    cutree = healthcheck(tmp_path, golden, dest)
    m = (cutree / "pkg/m.py").read_text()
    fa = m.split("def fb")[0]
    assert "changed by the bug" not in m and "return 'bug'" in fa and GOOD["b"] in m  # a: only the return is the bug
    ks = range(1, len(kept) + 1)
    got = grading_matrix(tmp_path, dest, cutree, ks)
    assert got == {f: {str(k): int(k in f) for k in ks} for f in got}

    c = dict(a, deleted=a["deleted"] + ["pkg/core.py"])
    assert bb.ineligible(c, orig, set(ALL)) == "a hidden file is not a test file"


def test_reindent_puts_partial_code_at_the_units_indentation():
    assert bb.reindent(["if x:", "    y()"], ["        if not x:"]) == ["        if x:", "            y()"]
    assert bb.reindent(["        a", "        b"], ["        c"]) == ["        a", "        b"]
    assert bb.reindent(["  a", "b"], ["    c"]) == ["  a", "b"]  # not a uniform block: left as it is


def test_plan_bundles_draws_sizes_from_the_list(tmp_path):
    golden, orig, rows = toy_repo(tmp_path)
    cands = [bb.candidate(golden, r) | {"passed": set(ALL)} for r in rows]
    bundles = bb.plan_bundles(golden, orig, cands, 5, 0, collections.Counter(), sizes=[1, 2, 3])
    assert bundles and all(rule == "sizes" and len(t) == drawn for t, rule, drawn in bundles)
    ids = [c["instance_id"] for t, _, _ in bundles for c in t]
    assert len(ids) == len(set(ids)) <= 6


def test_plan_samefile_keeps_each_bundle_on_one_file(tmp_path):
    """--rule samefile: a bundle per module (its three bugs are <= NEAR lines apart); a far hunk only collides inside
    the same function or class where asked; a bug comment naming the change is caught; a bundle of <= 4 tasks is
    accepted on every pair of fixes."""
    golden, orig, rows = toy_repo(tmp_path)
    cands = [bb.candidate(golden, r) | {"passed": set(ALL)} for r in rows]
    bundles = bb.plan_samefile(golden, orig, cands, 5, 0, collections.Counter(), sizes=[3])
    assert sorted(sorted(c["instance_id"][-1] for c in t) for t, rule, drawn in bundles) == [list("abc"), list("def")]
    assert all(rule == "samefile" and drawn == 3 for _, rule, drawn in bundles)
    assert next(c for c in cands if c["instance_id"].endswith("__a"))["hunks"] == [[5, 5, ["fa@1"], []]]
    near, far, far_func, far_class = ({"hunks": [[s, s, f, c]]} for s, f, c in (
        (1 + bb.NEAR, [], []), (2 + bb.NEAR, [], []), (2 + bb.NEAR, ["g@2"], []), (2 + bb.NEAR, [], ["C@1"])))
    one = {"hunks": [[1, 1, ["g@2"], ["C@1"]]]}
    assert bb.collide(one, near) and not bb.collide(one, far) and not bb.collide(one, far_func)
    assert bb.collide(one, far_func, scope=True) and bb.collide(one, far_class, scope=True)
    fix = "-    x = a - 1  # Introduced an off-by-one error\n-    y = 2  # keep in sync with z\n+    x = a\n+    y = 2  # keep in sync with z\n"
    assert bb.coupled([dict(one, bug_files=["m.py"]), dict(near, bug_files=["m.py"]), dict(far_class, bug_files=["m.py"])])
    assert not bb.coupled([dict(one, bug_files=["m.py"]), dict(far_func, bug_files=["m.py"])])
    patch = "@@ -3,4 +3,4 @@ def g():\n a\n-b\n+B\n c\n-d\n+D\n"
    assert bb.hunks(patch, "", True) == [[4, 4, [], []], [6, 6, [], []]]
    replace = "@@ -10,3 +10,3 @@\n a\n-b\n-c\n+B\n+C\n"  # a replacement ends at its last line on that side (git -U0)
    assert bb.hunks(replace, "", False) == [[11, 12, [], []]] == bb.hunks(replace, "", True)
    assert bb.hunks("@@ -10,2 +10,3 @@\n a\n+x\n b\n", "", False) == [[10, 10, [], []]]  # insertion: the line before
    assert bb.hunks("@@ -10,0 +11 @@\n+x\n", "", False) == [[10, 10, [], []]]  # the same as git diff -U0 writes it
    assert "/opt/miniconda3/pkgs/*.conda" in bb.clean_command({"glom"}) and "*/_vendor/glom" in bb.clean_command({"glom"})
    assert "ensurepip" in bb.clean_command({"pygments"}) and "ensurepip" not in bb.clean_command({"glom"})
    assert bb.change_notes(fix) == ["Introduced an off-by-one error"] and bb.change_notes(fix.splitlines()[1] + "\n") == []
    assert list(bb.variants({"bundle": "x", "tasks": [{"k": k} for k in (1, 2, 3)]})) == \
        ["none", "1", "2", "3", "all", "p12", "p13", "p23"]
