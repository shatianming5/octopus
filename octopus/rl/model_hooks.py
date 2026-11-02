"""Training-side hooks per transformers model family: what packed micro-batches and the prefix tree need from a model's
sequence-state layers (octopus/rl/prefix_tree.py, gdn_varlen.py, seq_tiles.py).

A family is the transformers modeling module (transformers.models.<family>), found from the config's class. Its entry
gives the left context of its causal convolution (from the family's own kernel attribute), the patch that makes its
recurrent kernels respect packed-sequence and tree boundaries, and optionally the sequence-tiled layer forward.
A model of an unregistered family is pure attention (no hooks, no convolution context) unless anything marks
sequence state: a non-attention layer type, a conv attribute in its config, or a stateful model class
(transformers' `_is_stateful`); then patch_model raises instead of running with the state crossing sequences.
Adding a family is one FAMILIES entry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from octopus.rl import prefix_tree


def _deferred(module: str, name: str) -> Callable:
    """module.name(arg), imported at the call (the varlen kernels need fla: CUDA hosts only)."""
    return lambda arg: getattr(importlib.import_module(module), name)(arg)


_gdn_kernels = _deferred("octopus.rl.gdn_varlen", "patch_gdn")
_mamba2_kernels = _deferred("octopus.rl.mamba2_varlen", "patch_mamba2")
_qwen3_5_tiles = _deferred("octopus.rl.seq_tiles", "patch_tiles")


@dataclass(frozen=True)
class Family:
    conv_context: Callable  # text config -> tokens of left context its causal convolution reads (kernel - 1)
    patch: Callable  # modeling module -> None: varlen / tree kernels for its recurrent layers
    tiles: Callable | None = None  # loaded model -> None: sequence-tiled layer forward (its layer layout)


_GDN = dict(conv_context=lambda c: c.linear_conv_kernel_dim - 1, patch=_gdn_kernels)
FAMILIES = {
    "qwen3_5": Family(**_GDN, tiles=_qwen3_5_tiles),
    "qwen3_5_moe": Family(**_GDN),  # sparse MoE MLP: no tiled forward
    "qwen3_next": Family(**_GDN),  # fused in_proj_qkvz: no tiled forward
    "granitemoehybrid": Family(conv_context=lambda c: c.mamba_d_conv - 1, patch=_mamba2_kernels),  # Mamba-2 + attention
}
ATTENTION_LAYERS = {"full_attention", "sliding_attention", "chunked_attention"}


def modeling_module(config):
    """transformers' modeling_<family> beside the config's configuration_<family>."""
    return importlib.import_module(type(config).__module__.replace(".configuration_", ".modeling_"))


def family(config) -> Family | None:
    """The config's registered family, None for pure attention; raises for unregistered sequence state."""
    modeling = modeling_module(config)
    name = modeling.__name__.split(".")[-2]
    if name in FAMILIES:
        return FAMILIES[name]
    text = config.get_text_config()
    stateful = [c.__name__ for c in vars(modeling).values()
                if isinstance(c, type) and c.__module__ == modeling.__name__ and getattr(c, "_is_stateful", False)]
    state = (sorted(set(getattr(text, "layer_types", None) or ()) - ATTENTION_LAYERS)
             + [a for a in vars(text) if "conv" in a] + stateful)
    if state:
        raise NotImplementedError(f"{name}: sequence state ({', '.join(state)}) without training hooks; "
                                  f"register the family in octopus/rl/model_hooks.py FAMILIES")
    return None


def conv_context(config) -> int:
    f = family(config)
    return f.conv_context(config.get_text_config()) if f else 0


def patch_model(config) -> None:
    """Install the family's kernels and set the prefix tree's convolution context (0 for pure attention)."""
    prefix_tree.CONV_CONTEXT = conv_context(config)
    if f := family(config):
        f.patch(modeling_module(config))


def tile_model(model) -> None:
    """The family's sequence-tiled layer forward; a family without one keeps the HF forward."""
    f = family(model.config)
    if f and f.tiles:
        f.tiles(model)
