#!/usr/bin/env python3
"""Idempotent SGLang 0.5.19 patch: a top_k at or above the vocabulary size is the whole vocabulary (TOP_K_ALL).

SGLang reads "no top-k" only as -1 (TOP_K_ALL = 1 << 30). AReaL sends top_k=int(1e8) for it (GenerationHyperparameters
and its OpenAI client's default), so need_top_k_sampling was set for every request and the pytorch sampler ran the
top-k path, a full sort of the 248,320-token vocabulary per decode step (+1.0 ms at batch 16, +3.9 ms at 64 on a
4090D), to mask nothing. SamplingParams.verify, which knows the vocabulary size, now maps such a top_k to TOP_K_ALL:
the same distribution, the simple sampling path.

  python3 patches/apply_sglang_top_k.py /path/to/site-packages/sglang
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

MARK = "# octopus-top-k-all"
NEEDLE = """        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
"""
FIX = NEEDLE + f"""        if self.top_k >= vocab_size:  {MARK}: no truncation, as -1
            self.top_k = TOP_K_ALL
"""


def patch(root: Path) -> bool:
    path = root / "srt/sampling/sampling_params.py"
    text = path.read_text()
    if MARK in text:
        return False
    if text.count(NEEDLE) != 1:
        raise SystemExit(f"needle not found exactly once in {path}")
    path.write_text(text.replace(NEEDLE, FIX))
    return True


if __name__ == "__main__":
    root = Path(sys.argv[1])
    found = runpy.run_path(str(root / "_version.py"))["__version__"]
    if found != "0.5.19":
        raise SystemExit(f"built for SGLang 0.5.19, found {found}; re-align these hooks: "
                         "srt/sampling/sampling_params.py")
    print("patched" if patch(root) else "already patched", root)
