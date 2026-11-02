"""Harbor task selection keeps holdouts, bottle, and the cold-start seven out."""

from octopus.envs.swe.data.emit_tasks import apply_oracle, emit
from octopus.envs.swe.data.oracles import FURL
from octopus.envs.swe.taskset import (
    HOLDOUT_IDS,
    VERIFIER_SHELL,
    is_excluded,
    render_task_files,
    select_task_specs,
    task_config_text,
    verifier_shell,
)

COLD_START = {
    "life4__textdistance.c3aca916.lm_rewrite__gdrh4f70",
    "python-hyper__h11.bed0dd4a.lm_rewrite__rtb2uk8p",
    "mozillazg__python-pinyin.e42dede5.lm_rewrite__wdpprvnh",
    "alecthomas__voluptuous.a7a55f83.lm_rewrite__gxejqjvw",
    "msiemens__tinydb.10644a0e.lm_rewrite__avoumrgh",
    "dbader__schedule.82a43db1.lm_rewrite__er3jyl8a",
    "hukkin__tomli.443a0c1b.lm_rewrite__oeqmfyq1",
}

IMAGE = "swebench/swesmith.x86_64.example_1776_pkg.abc:transport-deadbeef"


def _row(instance_id, issue="Fix the bug in pkg.", image=IMAGE, org_ok=True):
    return {"solver_task": {
        "instance_id": instance_id,
        "issue": issue,
        "image": image,
        "repo": "example/pkg",
        "base_commit": "abc",
    }}


def test_exclusions_and_instruction_split():
    rows = [
        _row("amueller__word_cloud.ec24191c.lm_rewrite__jhxlqo3m"),
        _row("prettytable__prettytable.ca90b055.lm_rewrite__guayblhb"),
        _row("bottlepy__bottle.a8dfef30.lm_rewrite__6wkup0zn"),
        *[_row(item) for item in COLD_START],
        _row("jaraco__inflect.c079a96a.lm_rewrite__keepme", issue="Remove the doubled decorator."),
        _row("missing__image.aaa.lm_rewrite__noimg", image="not-local:tag"),
    ]
    specs = select_task_specs(rows, {IMAGE}, COLD_START)
    ids = [spec["instance_id"] for spec in specs]
    assert ids == ["jaraco__inflect.c079a96a.lm_rewrite__keepme"]
    spec = specs[0]
    assert spec["instruction"] == "Remove the doubled decorator."
    files = render_task_files(spec)
    assert "pytest" not in files["instruction.md"]
    assert "/logs/verifier/reward.txt" not in files["instruction.md"]
    assert "/logs/verifier/reward.txt" in files["tests/test.sh"]
    assert "/opt/miniconda3/envs/testbed/bin/python /tests/check.py" in files["tests/test.sh"]
    assert "/opt/miniconda3/envs/testbed/bin/python" not in files["instruction.md"]
    config = files["task.toml"]
    assert spec["image"] in config
    assert "local-proxy" not in config
    assert "17931" not in config
    assert "OCTOPUS_API_KEY" not in config
    assert "sk-" not in config
    for holdout in HOLDOUT_IDS:
        assert is_excluded(holdout, set())
    assert is_excluded("bottlepy__bottle.a8dfef30.lm_rewrite__6wkup0zn", set())


def test_furl_instruction_matches_the_graded_check(tmp_path):
    image = "swebench/swesmith.x86_64.gruns_1776_furl.da386f68:transport-41d68498"
    rows = tmp_path / "rows"
    rows.mkdir()
    (rows / "row0.json").write_text(
        '{"solver_task": {"instance_id": "%s", "issue": "netloc becomes None", '
        '"image": "%s", "repo": "gruns/furl"}}' % (FURL, image),
    )
    written = emit(rows, {image}, set(), tmp_path / "tasks")
    assert written[0]["instruction"] != "netloc becomes None"
    task = tmp_path / "tasks" / FURL
    instruction = (task / "instruction.md").read_text()
    check = (task / "tests" / "check.py").read_text()
    assert 'furl("http://www.example.com:80/")' in instruction
    assert "443" in instruction
    assert "netloc" not in instruction
    assert "reward.txt" not in instruction
    assert "http://www.example.com:80/" in check
    assert "443" in check
    assert (task / "tests" / "test.sh").read_text() == verifier_shell(FURL)
    spec, extra = apply_oracle(written[0])
    assert extra["instruction"] in spec["instruction"]


def test_task_config_names_image_without_secrets():
    spec = select_task_specs(
        [_row("jaraco__inflect.c079a96a.lm_rewrite__keepme")],
        {IMAGE},
        set(),
    )[0]
    text = task_config_text(spec)
    assert IMAGE in text
    assert "Bearer" not in text
    assert "api.x.ai" not in text


def test_task_puts_testbed_on_the_instance_branch_without_history():
    import tomllib

    config = tomllib.loads(task_config_text({"instance_id": FURL, "image": "img:tag"}))
    setup = config["environment"]["healthcheck"]["command"]
    assert f"git checkout -q --force origin/{FURL}" in setup
    # main is the fix and the branch tip hides the tests: the agent gets one commit, no history
    assert setup.index(f"origin/{FURL}") < setup.index("rm -rf .git") < setup.index("git init")
    assert config["environment"]["healthcheck"]["timeout_sec"] >= 300


def test_verifier_restores_hidden_tests_from_task_files_then_grades_them():
    shell = verifier_shell(FURL)
    assert "origin/" not in shell  # history is gone by then; everything comes from /tests
    wholesale = "cp -a /tests/hidden_full/. /testbed/"
    restore = "git apply --3way --whitespace=nowarn /tests/hidden_tests.patch"
    grade = "/tests/check.py $(cat /tests/hidden_files.txt)"
    assert shell.index(wholesale) < shell.index(restore) < shell.index(grade)
    assert "echo 0 > /logs/verifier/reward.txt; exit 0" in shell  # unrestorable -> 0, never 1


def test_default_check_grades_hidden_files_and_skips_known_failures(tmp_path, monkeypatch):
    import subprocess
    import sys

    from octopus.envs.swe.taskset import DEFAULT_CHECK

    check = tmp_path / "check.py"
    check.write_text(DEFAULT_CHECK)
    (tmp_path / "known_failing.txt").write_text("tests/test_x.py::test_env\n")
    seen = {}

    def fake_run(argv, cwd, timeout):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(check), "tests/test_x.py"])
    try:
        exec(compile(DEFAULT_CHECK, str(check), "exec"), {"__file__": str(check), "__name__": "__main__"})
    except SystemExit as exit_:
        assert exit_.code == 0
    argv = seen["argv"]
    assert argv[-1] == "tests/test_x.py"
    assert argv[argv.index("--deselect") + 1] == "tests/test_x.py::test_env"
