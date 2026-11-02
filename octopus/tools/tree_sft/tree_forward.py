"""Shared-prefix ("tree-aware") forward and loss for Qwen3.5 text models.

Qwen3.5 interleaves Gated DeltaNet (linear attention, a recurrent state) with full attention, so
branches cannot share a prefix through a tree attention mask. Instead, per decoder layer, the root is
run once into a fresh cache and every child continues from that layer's state (K/V for full
attention; conv + recurrent state for linear attention), passed between checkpoint units as tensors.
Caches are rebuilt on recompute, so nothing is shared across the forward/backward boundary, and
gradients from every child flow back into the root through the state.

The loss (token cross-entropy on labelled child tokens) is computed in chunks on selected positions
only, which the flat baseline uses too (`labelled_ce`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

IGNORE = -100


def _update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
    # Same values as upstream, but rebinds instead of `copy_` into a static buffer: a child's update
    # must not overwrite the root state other children (and autograd) still read.
    if not self.has_previous_state:
        self.dtype, self.device = conv_states.dtype, conv_states.device
        self.max_batch_size, self.conv_kernel_size = conv_states.shape[0], conv_states.shape[-1]
        self.is_conv_states_initialized = True
        self.has_previous_state = True
        self.conv_states = conv_states
    else:
        n = conv_states.shape[-1]
        k = self.conv_kernel_size
        self.conv_states = conv_states[..., -k:] if n >= k else torch.cat([self.conv_states[..., n:], conv_states], -1)
    return self.conv_states


def _update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
    self.is_recurrent_states_initialized = True
    self.recurrent_states = recurrent_states
    return self.recurrent_states


def use_functional_linear_cache() -> None:
    LinearAttentionLayer.update_conv_state = _update_conv_state
    LinearAttentionLayer.update_recurrent_state = _update_recurrent_state


def labelled_ce(hidden: torch.Tensor, labels: torch.Tensor, lm_head, chunk: int = 1024) -> tuple[torch.Tensor, int]:
    """Summed CE of `lm_head(hidden[t])` against `labels[t]` over labelled positions, in checkpointed
    chunks so full-vocabulary logits never exist for the whole sequence. hidden/labels are aligned
    (hidden[t] predicts labels[t])."""
    keep = labels != IGNORE
    h, y = hidden[keep], labels[keep]
    total = hidden.new_zeros((), dtype=torch.float32)

    def piece(hc, yc):
        return F.cross_entropy(lm_head(hc).float(), yc, reduction="sum")

    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(piece, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return total, int(y.numel())


def _positions(start: int, length: int, device):
    pos = torch.arange(start, start + length, device=device).view(1, 1, -1).expand(4, 1, -1)
    return pos[0], pos[1:]


def _layer_state(cache: DynamicCache, layer_idx: int, full_attention: bool) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.layers[layer_idx]
    return (layer.keys, layer.values) if full_attention else (layer.conv_states, layer.recurrent_states)


def _cache_from_state(config, layer_idx: int, full_attention: bool, a: torch.Tensor, b: torch.Tensor) -> DynamicCache:
    """A cache whose `layer_idx` holds the root's state (K/V, or conv + recurrent state), so a child
    layer call continues right after the root."""
    cache = DynamicCache(config=config)
    layer = cache.layers[layer_idx]
    layer.dtype, layer.device = a.dtype, a.device
    if full_attention:
        layer.keys, layer.values = a, b
        layer.is_initialized = True
    else:
        layer.conv_states, layer.recurrent_states = a, b
        layer.max_batch_size, layer.conv_kernel_size = a.shape[0], a.shape[-1]
        layer.is_conv_states_initialized = layer.is_recurrent_states_initialized = True
        layer.has_previous_state = True
    return cache


def tree_loss(
    causal_lm, root_ids: torch.Tensor, children: list[tuple[torch.Tensor, torch.Tensor]], root_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Summed labelled CE over the root (if `root_labels`) and all children of one tree, with the root
    forwarded once.

    causal_lm: a Qwen3_5ForCausalLM (LoRA-wrapped modules are fine). root_ids: (Lr,) token ids.
    children: [(ids (Lc,), labels (Lc,)), ...] continuing right after the root.
    Returns (loss_sum, n_labelled_tokens).

    Per decoder layer the root is one checkpoint unit that also returns that layer's state; each
    child is its own checkpoint unit that rebuilds a cache from that state. Recompute memory is thus
    bounded by the longest segment, not the whole tree, and child gradients reach the root through
    the state tensors.
    """
    text = causal_lm.model
    config = text.config
    device = root_ids.device
    root_len = root_ids.shape[0]
    segments = [root_ids] + [ids for ids, _ in children]
    hidden = [text.embed_tokens(ids[None]) for ids in segments]
    geometry = []  # (text_position_ids, position_embeddings, full-attention mask)
    for i, h in enumerate(hidden):
        start = 0 if i == 0 else root_len
        text_pos, rope_pos = _positions(start, h.shape[1], device)
        # root: plain causal; child: every root key plus causal within the child (bottom-right aligned,
        # never materialised as a Lc x (Lr + Lc) matrix)
        mask = None if i == 0 else causal_lower_right(h.shape[1], root_len + h.shape[1])
        geometry.append((text_pos, text.rotary_emb(h, rope_pos), mask))

    def root_step(layer_idx: int, h: torch.Tensor):
        full_attention = config.layer_types[layer_idx] == "full_attention"
        cache = DynamicCache(config=config)
        text_pos, rope, _ = geometry[0]
        out = text.layers[layer_idx](
            h, position_embeddings=rope, attention_mask=None, position_ids=text_pos,
            past_key_values=cache, use_cache=True,
        )
        return (out, *_layer_state(cache, layer_idx, full_attention))

    def child_step(layer_idx: int, i: int, h: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        full_attention = config.layer_types[layer_idx] == "full_attention"
        text_pos, rope, mask = geometry[i]
        return text.layers[layer_idx](
            h, position_embeddings=rope, attention_mask=mask if full_attention else None, position_ids=text_pos,
            past_key_values=_cache_from_state(config, layer_idx, full_attention, a, b), use_cache=True,
        )

    for idx in range(config.num_hidden_layers):
        root_out, a, b = checkpoint(root_step, idx, hidden[0], use_reentrant=False)
        hidden = [root_out] + [
            checkpoint(child_step, idx, i, h, a, b, use_reentrant=False) for i, h in enumerate(hidden[1:], start=1)
        ]

    root_last = text.norm(hidden[0][:, -1:])
    loss, count = hidden[0].new_zeros((), dtype=torch.float32), 0
    if root_labels is not None and bool((root_labels != IGNORE).any()):
        loss, count = labelled_ce(text.norm(hidden[0][0, :-1]), root_labels[1:], causal_lm.lm_head)
    for (ids, labels), h in zip(children, hidden[1:]):
        # the first child token is predicted by the root's last position
        pred = torch.cat([root_last, text.norm(h[:, :-1])], dim=1)[0]
        part, n = labelled_ce(pred, labels, causal_lm.lm_head)
        loss, count = loss + part, count + n
    return loss, count


def flat_loss(causal_lm, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The same labelled CE for one ordinary sequence (method A), via the stock forward."""
    out = causal_lm.model(input_ids=input_ids[None], use_cache=False)
    return labelled_ce(out.last_hidden_state[0, :-1], labels[1:], causal_lm.lm_head)
