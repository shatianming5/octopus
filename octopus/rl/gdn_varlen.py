"""Make transformers' Gated DeltaNet (Qwen3.5, Qwen3-Next, ...) respect packed-sequence boundaries (cu_seq_lens_q).

Without the Hugging Face `kernels` hub packages, transformers falls back to pure-torch GDN functions
that accept cu_seqlens but ignore it, so the recurrent state and the short causal convolution run
across sequences packed into one micro-batch (measured: the 2nd packed sequence's logprobs off by up
to 16). patch_gdn(modeling) routes a GDN family's chunked delta rule to flash-linear-attention's varlen
kernel and its causal convolution to fla's varlen causal_conv1d (prefix_tree.segment_conv; installed through
octopus/rl/model_hooks.py).
"""

from __future__ import annotations

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

from octopus.rl.prefix_tree import tree_recurrence, varlen_conv


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                           output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                           cu_seqlens_cpu=None, octopus_tree=None, **_):
    if octopus_tree is not None:  # shared-prefix micro-batch (octopus/rl/prefix_tree.py)
        return tree_recurrence(fla_chunk_gated_delta_rule, octopus_tree, query, key, value, g, beta,
                               use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    return fla_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.long(), cu_seqlens_cpu=cu_seqlens_cpu,
    )


def patch_gdn(modeling) -> None:
    """Route a transformers GDN modeling module's torch fallbacks to the functions here (once)."""
    if modeling.torch_chunk_gated_delta_rule is chunk_gated_delta_rule:
        return
    modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling.causal_conv1d_fn = varlen_conv(modeling.causal_conv1d_fn)
