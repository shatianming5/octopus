"""patches/apply_sglang_granite_norm.py decides by the target's sglang/_version.py: 0.5.19 is patched (then "already
patched"), 0.5.22+ has the fix upstream and is left untouched, anything else exits non-zero. The target here is a fake
sglang dir holding the two 0.5.19 lines the patch edits.

Live parity (granite-4.0-h-350m, SGLang vs HF bf16 prompt logprobs) is in PR #174, not here: it needs a GPU server."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import patches.apply_sglang_granite_norm as granite  # noqa: E402

SCRIPT = Path(granite.__file__)
REL = "srt/models/granitemoehybrid.py"
SRC = "".join(old for old, _ in granite.EDITS)


def target(tmp_path, version):
    root = tmp_path / "sglang"
    (root / REL).parent.mkdir(parents=True)
    (root / REL).write_text(SRC)
    (root / "_version.py").write_text(f"__version__ = version = '{version}'\n")
    return root


def run(root):
    return subprocess.run([sys.executable, SCRIPT, root], capture_output=True, text=True, check=False)


def test_0_5_19_patches_once(tmp_path):
    root = target(tmp_path, "0.5.19")
    for out in ("patched", "already patched"):
        r = run(root)
        assert r.returncode == 0, r.stderr
        assert r.stdout.startswith(f"{out} "), r.stdout
    assert (root / REL).read_text() == "".join(new for _, new in granite.EDITS)


def test_0_5_22_upstream_left_alone(tmp_path):
    root = target(tmp_path, "0.5.22")
    r = run(root)
    assert r.returncode == 0, r.stderr
    assert "upstream since 0.5.22" in r.stdout
    assert (root / REL).read_text() == SRC


def test_0_5_21_fails_loudly(tmp_path):
    root = target(tmp_path, "0.5.21")
    r = run(root)
    assert r.returncode == 1
    assert "built for SGLang 0.5.19, found 0.5.21; re-align these hooks: srt/models/granitemoehybrid.py" in r.stderr
    assert (root / REL).read_text() == SRC
