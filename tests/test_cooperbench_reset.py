"""CooperBench check.py reset_tests: a staged new test file must not abort the restore."""

import os
import subprocess

from octopus.envs.swe.data.cooperbench import CHECK


def _reset_tests(repo):
    head = CHECK[CHECK.index("TEST_DIR"):CHECK.index("tasks =")]
    body = CHECK[CHECK.index("def git("):CHECK.index("\nsolved,")]
    src = (head + body).replace('cwd="/testbed"', "cwd=os.environ['TB']").replace(
        'os.path.join("/testbed", p)', "os.path.join(os.environ['TB'], p)")
    os.environ["TB"] = str(repo)
    ns = {"os": os, "re": __import__("re"), "subprocess": subprocess}
    exec(src, ns)
    ns["reset_tests"]()


def test_a_staged_new_test_file_is_removed_and_edited_tests_restored(tmp_path):
    repo = tmp_path / "testbed"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_old.py").write_text("def test_old():\n    assert False\n")
    (repo / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    (tests / "test_old.py").write_text("def test_old():\n    assert True\n")
    (tests / "test_feature1.py").write_text("def test_new():\n    assert True\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)

    _reset_tests(repo)

    assert not (tests / "test_feature1.py").exists()
    assert "assert False" in (tests / "test_old.py").read_text()
    assert (repo / "src.py").read_text() == "x = 1\n"
