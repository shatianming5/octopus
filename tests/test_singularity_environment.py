"""SingularityEnvironment: both task image names find one SIF; the worker binds each writable trial dir at its path, and
discarding a trial removes its RAM dir."""
import pytest

pytest.importorskip("harbor")

from octopus.envs.swe import singularity as se  # noqa: E402
from octopus.envs.swe.data import validate_tasks  # noqa: E402


def test_transport_and_hub_names_share_a_sif():
    assert (se.sif_name("swebench/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9:transport-fe4e5663")
            == se.sif_name("jyangballin/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9:latest")
            == "swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9.sif")


def test_command_binds_trial_dirs_and_discard_removes_them(tmp_path, monkeypatch):
    image = "jyangballin/swesmith.x86_64.a_1776_b.c:latest"
    monkeypatch.setattr(se, "OVERLAY", False)  # whatever this host exports
    (tmp_path / se.sif_name(image)).touch()
    monkeypatch.setenv("SINGULARITY_SIF_DIR", str(tmp_path))
    monkeypatch.setattr(validate_tasks, "HARBOR_ENV", "octopus.envs.swe.singularity:SingularityEnvironment")
    env = validate_tasks.make_env(image)
    assert isinstance(env, se.SingularityEnvironment)
    root = tmp_path / "trial"
    (root / "testbed").mkdir(parents=True)
    env._cid, env._root = str(root), root
    argv = env._command_argv("echo")
    assert argv[:4] == ["singularity", "exec", "--pwd=/", "--containall"] and argv[-3:] == ["/bin/bash", "-lc", "echo"]
    assert f"--home={root}/root:/root" in argv
    for name in ("testbed", "tmp", "logs", "tests", "solution", "tasks"):
        assert f"--bind={root}/{name}:/{name}" in argv
    assert not any(a.startswith("--overlay") for a in argv)
    monkeypatch.setattr(se, "OVERLAY", True)
    assert env._command_argv("echo") == ["singularity", "exec", "--pwd=/", "instance://trial", "/bin/bash", "-lc", "echo"]
    opts, up = env._options(), root / "overlay" / "upper"  # the instance's (start)
    assert f"--overlay={root}/overlay" in opts and "--no-mount=tmp" in opts and f"--home={up}/root:/root" in opts
    assert f"--bind={up}/logs:/logs" in opts and not any(a.endswith((":/testbed", ":/tmp")) for a in opts)
    monkeypatch.setattr(se.subprocess, "run", lambda *a, **k: None)  # _discard's instance stop
    env._discard()
    assert not root.exists() and env._cid == ""
