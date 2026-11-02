import subprocess

from octopus.envs.swe.data.statement_coverage import edit, units, verdict

FIXED = """def wrap(func):
    if is_generator(func):
        wrapper = gen(func)
    else:
        wrapper = plain(func)
    return wrapper


def other():
    x = 1
    return x
"""
# two defects on paired lines (inverted test, None returned), one removed line in another function
BUGGY = FIXED.replace("if is_generator", "if not is_generator").replace("return wrapper", "return None").replace(
    "    x = 1\n", "")


def git_diff(tmp_path, old, new):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/m.py").write_text(old)
    (tmp_path / "b/m.py").write_text(new)
    out = subprocess.run(["git", "diff", "--no-index", "a/m.py", "b/m.py"], cwd=tmp_path, capture_output=True,
                         text=True).stdout
    return out.replace("a/a/m.py", "a/m.py").replace("b/b/m.py", "b/m.py")


def test_units_and_edit(tmp_path):
    fix = git_diff(tmp_path, BUGGY, FIXED)
    us = units(fix)
    assert [(u["minus"], u["plus"]) for u in us] == [
        (["    if not is_generator(func):"], ["    if is_generator(func):"]),
        (["    return None"], ["    return wrapper"]),
        ([], ["    x = 1"])]
    every = {u["id"]: u["plus"] for u in us}
    assert edit({"m.py": BUGGY}, fix, every)["m.py"] == FIXED
    # only the undescribed inversion fixed: the task's bug is the `return None` and the missing line left
    baked = edit({"m.py": BUGGY}, fix, {"U1": us[0]["plus"]})["m.py"]
    assert "if is_generator(func)" in baked and "return None" in baked and "x = 1" not in baked
    # a mixed unit takes Grok's partial code
    part = edit({"m.py": BUGGY}, fix, {"U2": ["    return wrapper or None"]})["m.py"]
    assert "return wrapper or None" in part and "if not is_generator" in part


def test_verdict_checks_every_unit():
    us = [{"id": "U1"}, {"id": "U2"}]
    v = verdict('noise {"U1": {"label": "described", "why": "x"}, "U2": {"label": "mixed", "partial": "y"}}', us)
    assert v["U2"]["partial"] == "y"
    for bad in ('{"U1": {"label": "described"}}', '{"U1": {"label": "x"}, "U2": {"label": "described"}}',
                '{"U1": {"label": "described"}, "U2": {"label": "mixed"}}'):
        try:
            verdict(bad, us)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_no_newline_at_end(tmp_path):
    fix = git_diff(tmp_path, "a = 1\nb = 2", "a = 1\nb = 3")
    assert "No newline at end of file" in fix
    us = units(fix)
    assert [(u["minus"], u["plus"]) for u in us] == [(["b = 2"], ["b = 3"])]
    assert edit({"m.py": "a = 1\nb = 2"}, fix, {"U1": ["b = 3"]})["m.py"] == "a = 1\nb = 3"


def test_region_units_for_rewrites(tmp_path):
    fix = git_diff(tmp_path, BUGGY, FIXED)
    us = units(fix, "lm_rewrite")  # one hunk: one unit from its first to its last changed line
    assert len(us) == 1 and us[0]["minus"][0] == "    if not is_generator(func):" and us[0]["plus"][-1] == "    x = 1"
    assert edit({"m.py": BUGGY}, fix, {"U1": us[0]["plus"]}, "lm_rewrite")["m.py"] == FIXED
    # Grok's partial: the original region with only the described defect put back
    partial = [l.replace("return wrapper", "return None") for l in us[0]["plus"]]
    assert edit({"m.py": BUGGY}, fix, {"U1": partial}, "lm_rewrite")["m.py"] == FIXED.replace("return wrapper", "return None")


def test_code_blocks_and_statement_overrides(tmp_path, monkeypatch):
    import json
    from octopus.envs.swe.data import statement_coverage as sc
    text = "Bug.\n```python\nimport m\n```\nthen\n```py\nm.f()\n```\nOutput:\n```\nx\n```\n"
    assert sc.code_blocks(text) == ["import m", "m.f()"]
    monkeypatch.setattr(sc, "statement", lambda d, k: text)
    (tmp_path / "b/solution").mkdir(parents=True)
    (tmp_path / "b/solution/fix_1.patch").write_text("+    return 'a value long enough to be a fix line'\n")
    rows = [{"bundle": "b", "k": 1, "instance_id": "i1", "new": ["import m, os", "m.f(); print(os.sep)"]},
            {"bundle": "b", "k": 1, "instance_id": "i2", "new": None},
            {"bundle": "b", "k": 1, "instance_id": "i3", "new": ["def g():\n    return 'a value long enough to be a fix line'", "y"]}]
    (tmp_path / "rep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sc.overrides(str(tmp_path), str(tmp_path / "rep.jsonl"), str(tmp_path / "o.json"))
    out = json.loads((tmp_path / "o.json").read_text())
    assert sc.code_blocks(out["i1"]) == ["import m, os", "m.f(); print(os.sep)"] and out["i1"].endswith("```\nx\n```\n")
    assert sc.code_blocks(out["i2"]) == [] and out["i2"].count("omitted") == 2
    assert sc.code_blocks(out["i3"]) == []  # a repair holding reference-fix code is dropped too
