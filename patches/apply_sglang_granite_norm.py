#!/usr/bin/env python3
"""Idempotent SGLang 0.5.19 patch: granitemoehybrid (granite-4.0-h) final norm without the residual.

Each decoder layer already adds its residual (hidden_states = residual + out * residual_multiplier), but the model's
forward passes it again to the final RMSNorm, which adds before normalising, so the last layer's input counts twice.
HF normalises hidden_states alone. Backport of upstream sgl-project/sglang 2c739b225b ("[Fix] Jet-Nemotron build and
Granite MoE hybrid final norm (#42306)", first released in v0.5.22). Decided by the target's sglang/_version.py:
0.5.19 is patched; 0.5.22 or newer already has the fix (restructured code) and is left untouched, exit 0; any other
version (0.5.20, 0.5.21: bug present, code may differ) exits non-zero.

  python3 patches/apply_sglang_granite_norm.py /path/to/site-packages/sglang
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from packaging.version import Version

EDITS = [
    ("                aux_hidden_states.append(hidden_states + residual)\n",
     "                aux_hidden_states.append(hidden_states)\n"),
    ("            hidden_states, _ = self.norm(hidden_states, residual)\n",
     "            hidden_states = self.norm(hidden_states)\n"),
]


def patch(root: Path) -> bool:
    path = root / "srt/models/granitemoehybrid.py"
    text = path.read_text()
    todo = [(old, new) for old, new in EDITS if new not in text]
    for old, new in todo:
        if text.count(old) != 1:
            raise SystemExit(f"needle not found exactly once in {path}: {old!r}")
        text = text.replace(old, new)
    if todo:
        path.write_text(text)
    return bool(todo)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    found = runpy.run_path(str(root / "_version.py"))["__version__"]
    if Version(found) >= Version("0.5.22"):
        print(f"SGLang {found} has the fix upstream since 0.5.22, nothing to do", root)
    elif found != "0.5.19":
        raise SystemExit(f"built for SGLang 0.5.19, found {found}; re-align these hooks: "
                         "srt/models/granitemoehybrid.py")
    else:
        print("patched" if patch(root) else "already patched", root)
