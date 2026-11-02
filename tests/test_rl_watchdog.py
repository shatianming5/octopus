import re
import subprocess
from pathlib import Path

WATCHDOG = Path(__file__).resolve().parents[1] / "scripts" / "rl_env" / "rl_watchdog.sh"


def test_cut_mb_uses_value_from_sourced_env(tmp_path):
    """An env file that gets MB_TOKENS only through `source` is cut from the effective value, and the cut is appended."""
    cut_mb = re.search(r"^cut_mb\(\) \{.*\}$", WATCHDOG.read_text(), re.M).group(0)
    (tmp_path / "base.env").write_text("MB_TOKENS=196608\n")
    env = tmp_path / "run.env"
    env.write_text(f'source "{tmp_path}/base.env"\n')
    script = f'E={env}; log() {{ echo "$*"; }}\n{cut_mb}\ncut_mb "CUDA OOM #1"\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "MB_TOKENS 196608 -> 176947 (CUDA OOM #1)"
    assert env.read_text().splitlines()[-1].startswith("MB_TOKENS=176947  # watchdog cut ")
