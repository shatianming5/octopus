"""Make transformers' Mamba-2 mixer (granite-4.0-h, ...) respect packed-sequence and prefix-tree boundaries.

Without the mamba_ssm / causal_conv1d packages transformers runs the mixer on pure-torch functions that take a packed
micro-batch for one sequence (state and convolution run across sequences) and have no per-sequence initial state.
patch_mamba2(modeling) runs a packed or tree micro-batch's chunked scan on flash-linear-attention's varlen simple-GLA
kernel, which takes per-sequence initial states (Mamba-2's SSD is simple GLA: per-head decay exp(dt*A), key B,
value dt*x, query C, plus the D skip), and its causal convolution on fla's varlen causal_conv1d
(prefix_tree.segment_conv); other calls (generation, a single unpacked sequence) keep transformers' functions
(installed through octopus/rl/model_hooks.py).
"""

from __future__ import annotations

import torch
from fla.ops.simple_gla import chunk_simple_gla

from octopus.rl.prefix_tree import tree_recurrence, varlen_conv


def patch_mamba2(modeling) -> None:
    """Route a transformers Mamba-2 modeling module's chunked scan and causal convolution through here (once)."""
    torch_scan = modeling.mamba2_chunk_scan
    if getattr(torch_scan, "octopus", False):
        return

    def mamba2_chunk_scan(x, dt, A, B, C, chunk_size, D=None, dt_bias=None, initial_states=None, dt_softplus=False,
                          dt_limit=(0.0, float("inf")), return_final_states=False, cu_seq_lens_q=None,
                          octopus_tree=None, **kwargs):
        """x: [1, T, H, P], dt: [1, T, H], A: [H], B/C: [1, T, G, N] -> [1, T, H, P] (fp32, as transformers')."""
        if octopus_tree is None and cu_seq_lens_q is None:
            return torch_scan(x, dt, A, B, C, chunk_size, D=D, dt_bias=dt_bias, initial_states=initial_states,
                              dt_softplus=dt_softplus, dt_limit=dt_limit, return_final_states=return_final_states,
                              **kwargs)
        if dt_bias is not None:
            dt = dt + dt_bias.to(dt.dtype)
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1]).float()
        q, k = (t.repeat_interleave(x.shape[2] // t.shape[2], dim=2) for t in (C, B))
        v, g = (x.float() * dt[..., None]).to(x.dtype), A.float() * dt
        if octopus_tree is not None:
            y, _ = tree_recurrence(chunk_simple_gla, octopus_tree, q, k, v, g, scale=1.0)
        else:
            y, _ = chunk_simple_gla(q, k, v, g, scale=1.0, cu_seqlens=cu_seq_lens_q.long())
        return y.float() + D[:, None] * x.float() if D is not None else y.float()

    mamba2_chunk_scan.octopus = True
    modeling.mamba2_chunk_scan = mamba2_chunk_scan
    modeling.causal_conv1d_fn = varlen_conv(modeling.causal_conv1d_fn)
