"""The scripts that edit SGLang internals run only on SGLang 0.5.19 (the target's sglang/_version.py); another version
exits non-zero naming the hooks to re-align. The 0.5.19 run copies the importable SGLang (skipped without it)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# script -> the SGLang it patches, as a path under a site-packages dir; in apply order (mount needs the cursor patch)
SCRIPTS = {
    "patches/apply_sglang_cursor.py": "sglang",
    "patches/apply_sglang_mount.py": "sglang",
    "patches/apply_sglang_lora_abort_release.py": "sglang",
    "patches/apply_sglang_lora_targets.py": "sglang",
    "patches/apply_sglang_top_k.py": "sglang",
}


def run(script, site, version):
    (site / "sglang/_version.py").write_text(f"__version__ = version = '{version}'\n")
    cmd = [sys.executable, REPO / script, site / SCRIPTS[script]]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("script", SCRIPTS)
def test_other_version_fails_loudly(tmp_path, script):
    (tmp_path / "sglang").mkdir()
    r = run(script, tmp_path, "0.5.22")
    assert r.returncode == 1
    assert "built for SGLang 0.5.19, found 0.5.22; re-align these hooks: " in r.stderr
    assert "srt/" in r.stderr  # the files it edits


def test_0_5_19_patches_as_before(tmp_path):
    import patches.apply_sglang_cursor as cursor
    import patches.apply_sglang_lora_abort_release as abort
    import patches.apply_sglang_mount as mount
    import patches.apply_sglang_top_k as top_k
    sglang = pytest.importorskip("sglang")
    if sglang.__version__ != "0.5.19":
        pytest.skip(f"the importable SGLang is {sglang.__version__}, not 0.5.19")
    src = Path(sglang.__file__).parent
    rels = {*cursor.EDITS, *mount.EDITS, "srt/managers/tokenizer_manager.py", "srt/lora/utils.py",
            "srt/lora/lora.py", "srt/models/granitemoehybrid.py", "srt/sampling/sampling_params.py"}
    done = [m.MARK for m in (cursor, mount, abort, top_k) if any(m.MARK in (src / rel).read_text() for rel in rels)]
    if done:  # install_env.sh patches the SGLang venv (lora_abort_release, top_k), the overlay adds cursor and mount
        pytest.skip(f"the importable SGLang already carries {done}: needs a pristine SGLang 0.5.19")
    for rel in rels:
        (tmp_path / "sglang" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, tmp_path / "sglang" / rel)
    for again in ("", "already "):
        for script in SCRIPTS:
            r = run(script, tmp_path, "0.5.19")
            assert r.returncode == 0, r.stderr
            assert r.stdout.startswith("patched" if script.endswith("lora_targets.py") else f"{again}patched"), r.stdout
