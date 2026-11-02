"""Sequence-tiled Qwen3.5 decoder layers, so one micro-batch can hold a 665k-token prefix tree on a 48 GB card.

Layer checkpointing keeps one tensor per layer, but recomputing a layer in backward holds all of that layer's
activations at once: ~35 GB for one full-attention layer at 262k tokens (fp32 RMSNorm copies, PEFT's fp32 input
casts for the LoRA, the 9216-wide MLP), and a GDN layer spreads ~78 KB per token (in_proj, the convolution, q and k
repeated to the value heads, the delta rule). patch_tiles(model) bounds what a layer holds at once:
  flat micro-batch  every token-wise part of a layer (norms, projections with their LoRA, output gates, MLP) runs TILE
                    tokens at a time, each tile checkpointed, so only the sequence mixers (FlashAttention, the GDN
                    convolution and delta rule) see full-length tensors; the GDN convolution is checkpointed on its own
  prefix tree       the layer streams over the tree's own kernel calls (octopus/rl/prefix_tree.py TreeMeta), each
                    checkpointed: a GDN layer runs one recurrence call (<= SEG tokens) at a time from in_proj to the
                    MLP, starting from its segments' parent final states and projecting their convolution context
                    again; an attention layer projects k, v over the whole tree (4 KB per token for Qwen3.5-4B), then
                    q, attention, o_proj and the MLP per FlashAttention call. Only the residual stream and k, v span
                    the tree, so the peak follows SEG, not the tree size.
Per token the arithmetic is the HF layer's; a flat micro-batch of at most TILE tokens runs the HF forward unchanged.
Chunk-major (octopus/rl/chunk_major.py) bounds a layer by its chunk instead and runs its backward under no_inner(): the
layer's own checkpoint stays, the tiles and kernel calls inside it run without one. Nested, each checkpoint level ran
most of the layer once more in that backward: up to 4 forwards instead of 2 (results/chunk-major/README.md).
"""

from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from octopus.rl import prefix_tree

TILE = 16384
_INNER = True  # tiles and a tree's kernel calls checkpointed (no_inner() turns this off)
_modeling = None  # the loaded model's transformers modeling module and its HF layer forward, set by patch_tiles
_hf_layer_forward = None
# torch.compile the element-wise blocks below (RMSNorms, the MLP's silu * up, the attention output gate): one fused
# kernel each instead of a chain of fp32 passes over the tile. GEMMs, FlashAttention and fla kernels stay as they are.
# OCTOPUS_COMPILE=0 runs them eagerly.
COMPILE = os.environ.get("OCTOPUS_COMPILE", "1") != "0"


def _fused(fn):
    """Compiled with dynamic sizes; callers pass 2-D (tokens, features) views, so a tile of a longer micro-batch
    (other strides) reuses the same graph instead of recompiling per micro-batch length."""
    return torch.compile(fn, dynamic=True) if COMPILE else fn


def _rows(x):
    return x.reshape(-1, x.shape[-1])


if COMPILE:  # one graph per (feature width, row stride, grad mode): RMSNorm alone sees 2560-wide rows and two
    # 256-wide q/k layouts, with and without grad, over the default limit of 8; micro-batch length adds none
    torch._dynamo.config.recompile_limit = 64



@_fused
def rms_norm(x, weight, eps: float):
    """Qwen3_5RMSNorm.forward: (x * rsqrt(mean(x^2) + eps)) * (1 + weight), in fp32."""
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).type_as(x)


@_fused
def rms_norm_gated(x, gate, weight, eps: float):
    """Qwen3_5RMSNormGated.forward: normalize in fp32, scale in x's dtype, gate by silu(gate) in fp32."""
    h = x.float()
    h = weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (h * F.silu(gate.float())).to(x.dtype)


@_fused
def silu_mul(gate, up):
    return F.silu(gate) * up


@_fused
def sigmoid_gate(o, gate):
    return o * torch.sigmoid(gate)


def _mlp_forward(self, x):
    gate = self.gate_proj(x)
    return self.down_proj(silu_mul(_rows(gate), _rows(self.up_proj(x))).view(gate.shape))



def _checkpoint(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False) if _INNER else fn(*args)


@contextlib.contextmanager
def no_inner():
    """Layers run without checkpoints inside them: their tiles and kernel calls keep their activations until the
    backward (each layer's own checkpoint, HF's, is unaffected)."""
    global _INNER
    _INNER = False
    try:
        yield
    finally:
        _INNER = True


def tiled(fn, *xs):
    """fn over [B, T, ...] inputs (None passes through), TILE tokens at a time, each tile recomputed in backward;
    its outputs (a tensor or a tuple) are concatenated along T. Tiles are split() views: the backward concatenates
    their gradients once (a slice's backward writes a zero tensor of the full input per tile, then adds them)."""
    parts = [None if x is None else x.split(TILE, dim=1) for x in xs]
    outs = [_checkpoint(fn, *(None if p is None else p[i] for p in parts)) for i in range(len(parts[0]))]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(parts, dim=1) for parts in zip(*outs))
    return torch.cat(outs, dim=1)


def _rotary(x, cos, sin):
    """[B, t, heads, D] rotated alone (the HF function rotates a q, k pair: the k here is empty)."""
    x = x.transpose(1, 2)
    return _modeling.apply_rotary_pos_emb(x, x[:, :0], cos, sin)[0].transpose(1, 2)


def _attention_q(attn, h, cos, sin):
    """Normed residual stream tile -> rotated q ([B, t, heads, D]) and the output gate."""
    q, gate = torch.chunk(attn.q_proj(h).view(*h.shape[:-1], -1, attn.head_dim * 2), 2, dim=-1)
    return _rotary(attn.q_norm(q), cos, sin), gate.reshape(*h.shape[:-1], -1)


def _attention_kv(attn, h, cos, sin):
    """Normed residual stream tile -> rotated k and v ([B, t, kv heads, D])."""
    shape = (*h.shape[:-1], -1, attn.head_dim)
    return _rotary(attn.k_norm(attn.k_proj(h).view(shape)), cos, sin), attn.v_proj(h).view(shape)


def _attention_in(layer, x, cos, sin):
    h = layer.input_layernorm(x)
    return *_attention_q(layer.self_attn, h, cos, sin), *_attention_kv(layer.self_attn, h, cos, sin)


def _attention_out(layer, x, o, gate):
    x = x + layer.self_attn.o_proj(sigmoid_gate(_rows(o.reshape(*o.shape[:2], -1)), _rows(gate)).view(gate.shape))
    return x + layer.mlp(layer.post_attention_layernorm(x))


def _gdn_in(layer, x, mask):
    lin = layer.linear_attn
    h = _modeling.apply_mask_to_padding_states(layer.input_layernorm(x), mask)
    return lin.in_proj_qkv(h), lin.in_proj_z(h), lin.in_proj_b(h).sigmoid(), lin.in_proj_a(h)


def _conv(lin, x, **kwargs):
    """The layer's causal convolution of [B, T, C] x (kwargs: packed-sequence bounds)."""
    return _modeling.causal_conv1d_fn(x.transpose(1, 2), lin.conv1d.weight.squeeze(1), lin.conv1d.bias,
                                      activation=lin.activation, **kwargs).transpose(1, 2)


def _gdn_conv(lin, qkv, conv):
    """Causal convolution (conv: [B, T, C] -> [B, T, C]), then q, k (repeated to the value heads) and v as their own
    tensors."""
    qkv = conv(qkv)
    b, t = qkv.shape[:2]
    q, k, v = torch.split(qkv, [lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
    q, k = (x.reshape(b, t, -1, lin.head_k_dim).repeat_interleave(lin.num_v_heads // lin.num_k_heads, dim=2)
            for x in (q, k))
    return q, k, v.reshape(b, t, -1, lin.head_v_dim).contiguous()


def _gdn_out(layer, x, core, z):
    lin = layer.linear_attn
    o = lin.norm(core.reshape(-1, lin.head_v_dim), z.reshape(-1, lin.head_v_dim)).reshape(*x.shape[:2], -1)
    x = x + lin.out_proj(o)
    return x + layer.mlp(layer.post_attention_layernorm(x))


def _gdn_delta_rule(lin, q, k, v, a, beta, **kwargs):
    g = -lin.A_log.float().exp() * F.softplus(a.float() + lin.dt_bias)
    return _modeling.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, **kwargs)


def _tree_gdn_call(layer, x, level, initial, ctx_rows):
    """One recurrence call (prefix_tree.Level, <= SEG tokens) of a GDN layer over the tree x: (the layer output of the
    call's rows, each segment's final state). ctx_rows: the rows level.ctx_index indexes (x, or x and more)."""
    lin = layer.linear_attn
    rows = prefix_tree.take(x, level.index)
    qkv, z, beta, a = tiled(lambda t: _gdn_in(layer, t, None), rows)
    ctx = (lin.in_proj_qkv(layer.input_layernorm(ctx_rows.index_select(1, level.ctx_index)))[0]
           if len(level.ctx_index) else None)
    q, k, v = _gdn_conv(lin, qkv, lambda y: prefix_tree.segment_conv(
        y, lin.conv1d.weight.squeeze(1), lin.conv1d.bias, lin.activation, level.cu, level.cu_cpu, ctx,
        level.ctx_dest))
    core, state = _gdn_delta_rule(lin, q, k, v, a, beta, initial_state=initial, output_final_state=True,
                                  cu_seqlens=level.cu, cu_seqlens_cpu=level.cu_cpu)
    return tiled(lambda t, c, z_: _gdn_out(layer, t, c, z_), rows, core, z), state


def _tree_attention_call(layer, x, cos, sin, k, v, rows, call):
    """One FlashAttention call (TreeMeta.attn_calls) of an attention layer over the tree x: the layer output of the
    call's rows (a slice), against k, v of the whole tree."""
    attn = layer.self_attn
    x, cos, sin = x[:, rows], cos[:, rows], sin[:, rows]
    q, gate = tiled(lambda t, c, s: _attention_q(attn, layer.input_layernorm(t), c, s), x, cos, sin)
    o = prefix_tree.attention_call(q[0], k[0], v[0], call, attn.scaling)[None]
    return tiled(lambda t, o_, g: _attention_out(layer, t, o_, g), x, o, gate)


def tree_layer_forward(layer, x, cos, sin, meta, io=None):
    """A decoder layer over a prefix tree micro-batch, one checkpointed kernel call at a time (module docstring).
    io: x is one chunk of the tree (octopus/rl/chunk_major.py); io.keys(k, v) appends the keys and values earlier chunks
    kept, io.rows(x) gives the rows the convolution contexts index (some of x's, then kept ones), io.finals holds the
    final states the chunk's segments start from."""
    out = torch.empty_like(x)
    if layer.block_type == "full_attention":
        attn = layer.self_attn
        k, v = tiled(lambda t, c, s: _attention_kv(attn, layer.input_layernorm(t), c, s), x, cos, sin)
        if io is not None:
            k, v = io.keys(k, v)
        start = 0
        for call in meta.calls(attn.layer_idx):
            rows = slice(start, start + call[0])
            out[:, rows] = _checkpoint(_tree_attention_call, layer, x, cos, sin, k, v, rows, call)
            start = rows.stop
        return out
    finals = {} if io is None else io.finals  # segment -> its final recurrent state
    ctx_rows = x if io is None else io.rows(x)
    for level in meta.levels:
        initial = torch.stack([finals[p] for p in level.parents]) if level.parents[0] >= 0 else None
        rows, state = _checkpoint(_tree_gdn_call, layer, x, level, initial, ctx_rows)
        if isinstance(level.index, slice):
            out[:, level.index] = rows
        else:
            out.index_copy_(1, level.index, rows)
        finals.update(zip(level.ids, state))
    return out


def tiled_layer_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                        past_key_values=None, **kwargs):
    if (meta := kwargs.get("octopus_tree")) is not None:
        return tree_layer_forward(self, hidden_states, *position_embeddings, meta, kwargs.get("octopus_io"))
    if hidden_states.shape[1] <= TILE:
        return _hf_layer_forward(self, hidden_states, position_embeddings, attention_mask, position_ids,
                                 past_key_values, **kwargs)
    if self.block_type == "full_attention":
        attn = self.self_attn
        q, gate, k, v = tiled(lambda x, c, s: _attention_in(self, x, c, s), hidden_states, *position_embeddings)
        interface = _modeling.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                                    _modeling.eager_attention_forward)
        o, _ = interface(attn, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attention_mask,
                         dropout=0.0, scaling=attn.scaling, **kwargs)
        return tiled(lambda x, o_, g: _attention_out(self, x, o_, g), hidden_states, o, gate)

    lin = self.linear_attn
    qkv, z, beta, a = tiled(lambda x, m: _gdn_in(self, x, m), hidden_states, attention_mask)
    conv_kwargs = dict(kwargs)  # the recompute in backward runs after cu_seq_lens_q is popped below
    # checkpointed, so neither the pre-activation nor the full conv output outlives the forward (v alone would keep
    # the latter)
    q, k, v = checkpoint(lambda x: _gdn_conv(lin, x, lambda y: _conv(lin, y, **conv_kwargs)), qkv,
                         use_reentrant=False)
    core, _ = _gdn_delta_rule(lin, q, k, v, a, beta, initial_state=None, output_final_state=False,
                              cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
    return tiled(lambda x, c, z_: _gdn_out(self, x, c, z_), hidden_states, core, z)


def patch_tiles(model) -> None:
    """Tile the loaded model's decoder layers (Qwen3.5 layout; model_hooks.py's qwen3_5 entry): its HF layer class
    (past FSDP2's FSDP<cls> subclass) and the RMSNorm, gated RMSNorm and MLP classes of its layers."""
    global _modeling, _hf_layer_forward
    layers = model.get_decoder().layers
    cls = next(c for c in type(layers[0]).__mro__ if c.__module__.startswith("transformers.models."))
    if cls.forward is tiled_layer_forward:
        return
    gdn = next(layer for layer in layers if layer.block_type == "linear_attention")
    _modeling, _hf_layer_forward = sys.modules[cls.__module__], cls.forward
    type(gdn.input_layernorm).forward = lambda self, x: rms_norm(_rows(x), self.weight, self.eps).view(x.shape)
    type(gdn.linear_attn.norm).forward = lambda self, x, gate=None: rms_norm_gated(
        _rows(x), _rows(gate), self.weight, self.variance_epsilon).view(x.shape)
    type(gdn.mlp).forward = _mlp_forward
    cls.forward = tiled_layer_forward
