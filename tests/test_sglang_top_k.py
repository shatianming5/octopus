"""patches/apply_sglang_top_k.py on a copy of the importable SGLang 0.5.19's sampling_params.py: AReaL's
top_k=int(1e8) and any top_k >= the vocabulary become TOP_K_ALL (as -1), so the batch takes the simple sampling path
(need_top_k_sampling is `top_k != TOP_K_ALL`); a real top_k is kept. Needs the sglang venv (skipped without it)."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import patches.apply_sglang_top_k as top_k  # noqa: E402

VOCAB = 248320  # Qwen3.5


def test_top_k_at_or_above_vocab_is_all(tmp_path):
    src = Path(pytest.importorskip("sglang").__file__).parent / "srt/sampling/sampling_params.py"
    if top_k.MARK in src.read_text():
        pytest.skip("the importable SGLang is already patched")
    dst = tmp_path / "srt/sampling/sampling_params.py"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(src, dst)
    assert top_k.patch(tmp_path) and not top_k.patch(tmp_path)  # patched, then already patched
    spec = importlib.util.spec_from_file_location("patched_sampling_params", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, want in ((int(1e8), mod.TOP_K_ALL), (VOCAB, mod.TOP_K_ALL), (-1, mod.TOP_K_ALL), (VOCAB - 1, VOCAB - 1),
                    (20, 20)):
        p = mod.SamplingParams(top_k=k, temperature=1.0)
        p.verify(VOCAB)
        assert p.top_k == want, (k, p.top_k)
