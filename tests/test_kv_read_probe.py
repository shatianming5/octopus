"""CPU checks of the KV read probe (octopus/tools/kv_read_probe.py): fact selection and the K3 key re-rotation."""

from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb

import re

from octopus.tools.kv_read_probe import Probe, facts, slot


def test_facts_only_what_b_lacks():
    tool = ("[token budget left: 900 | calls left: 9]\n"
            "FAILED tests/test_api.py::test_parse_header - AssertionError\n"
            "  File \"/testbed/pkg/core.py\", line 88, in render_value\n"
            "  File \"/testbed/pkg/shared.py\", line 12, in helper_fn\n")
    mounted = "<|im_start|>user\n<tool_response>\n" + tool + "</tool_response><|im_end|>\n"
    known = "B already saw helper_fn in /testbed/pkg/shared.py"
    got = facts([tool], mounted, known)
    # FAILED line: cue "FAILED " too short; helper_fn: B has it; the core.py line is the one fact
    assert got == [{"cue": '  File "/testbed/pkg/core.py", line 88, in ', "answer": "render_value"}]
    assert all(mounted.count(f["cue"]) == 1 and f["cue"] + f["answer"] in mounted for f in got)


def test_k3_rotation_equals_rope_at_shifted_position():
    """Qwen3.5-4B's RoPE (partial 0.25 of head_dim 256, interleaved M-RoPE [11, 11, 10], all three ids equal for text):
    a key rotated at position p and then re-rotated by delta equals the key rotated at p + delta."""
    cfg = Qwen3_5TextConfig(head_dim=256, rope_parameters={"rope_type": "default", "rope_theta": 10_000_000,
                                                            "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                                            "mrope_interleaved": True})
    emb = Qwen3_5TextRotaryEmbedding(cfg)
    probe = Probe.__new__(Probe)
    probe.torch, probe.body = torch, SimpleNamespace(rotary_emb=emb)
    torch.manual_seed(0)
    k = torch.randn(1, 4, 6, 256)  # [batch, kv heads, tokens, head_dim]
    pos = torch.tensor([[5000, 5001, 5002, 5003, 5004, 5005]])
    for delta in (-4321, 0, 777):
        cos, sin = emb(k, pos)
        at_p = apply_rotary_pos_emb(k, k, cos, sin)[1]
        cos2, sin2 = emb(k, pos + delta)
        at_shift = apply_rotary_pos_emb(k, k, cos2, sin2)[1]
        assert torch.allclose(probe.rotate(at_p, delta), at_shift, atol=2e-3)
    assert torch.equal(probe.rotate(at_p, 0)[..., 64:], at_p[..., 64:])  # the 192 unrotated dims pass through


def test_slot_aligned_or_dropped():
    """A word tokenizer that glues leading whitespace to a word: the cue's trailing space moves into the answer slot."""
    enc = lambda s: re.findall(r"\s*[^\s]+|\s+", s)  # noqa: E731
    pre, ans = slot(enc, "def __init__(self, ", "pkg_srels")
    assert pre[-1] == " __init__(self," and ans == [" pkg_srels"]
    assert slot(enc, "path=/testbed/pk", "g/core.py")[1] is None  # the answer glues onto the cue's last token: dropped
