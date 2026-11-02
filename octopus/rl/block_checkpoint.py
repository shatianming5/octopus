"""Block checkpointing: a micro-batch's forward keeps the input of every BLOCK-th decoder layer for backward, not the
input of every layer.

HF layer checkpointing (gradient_checkpointing_enable) keeps each decoder layer's input: 5 KB per token per layer for
Qwen3.5-4B (bf16, hidden 2560), which lm_head.host_activations holds in pinned host memory: 32 layers x 5 KB = 160 KB
per token, 110-140 GB per rank for the 665k-994k-token micro-batches of search-v9-rest (node-138, 503 GB: a rank was
OOM-killed with 4 ranks). checkpoint_blocks(model) makes each run of BLOCK consecutive layers one more checkpoint around
their own: the forward keeps only the block's input; the backward of the block's last layer recomputes its first
BLOCK-1 layers (their inputs then wait on the GPU, BLOCK x 5 KB per token), and each layer recomputes and differentiates
itself as before. The same kernels run on the same inputs, so the gradients are the same; the cost is one more forward
of BLOCK-1 of each BLOCK layers (and their FSDP all-gather). FSDP2 records those recomputed forwards in its post-forward
order, so at each block boundary its backward prefetch unshards a mistargeted layer, whose buffer stays pending until
the end of the backward: up to (blocks - 1) extra unshard buffers accumulate (7 for 32 layers at BLOCK 4; not measured
on their own, they are in the peaks below). Measured with both (results/block-checkpoint, 2 FSDP2 ranks, BLOCK 4): host
170 -> 50 KiB per tree token, forward + backward +11%, GPU peak 16.1 -> 19.5 GiB at 199k tree tokens; the 994k-tree-token
unit of search-5271 ran out of a 48 GB card in the block recompute with BLOCK 4 (45.0 GiB allocated) and with BLOCK 2
(44.6 GiB), while a 640k-tree-token unit (search-6296) peaked at 37.8 GiB with BLOCK 4, 9.6 GiB under the card. So the
block size is picked per micro-batch: BLOCK up to BLOCK_TOKENS tokens (650k: that measurement, 2 ranks of a 48 GB
card; more ranks shard more), per layer above; the size must be identical on every FSDP rank, so it follows the
largest micro-batch of the ranks' current ones (checkpoint_blocks). A rank's
pinned host memory follows its largest micro-batch: 50 KiB per token of one run in blocks, 170 of one run per layer
(every rank's micro-batch beside a unit over BLOCK_TOKENS on any rank).

OCTOPUS_BLOCK_CHECKPOINT (a host setting, octopus/rl/config.py HOST_ENV) sets BLOCK; 1 is HF per-layer checkpointing
for every micro-batch.
"""

from __future__ import annotations

import functools
import inspect
import os

from torch.utils.checkpoint import _checkpoint_without_reentrant_generator, noop_context_fn

BLOCK = int(os.environ.get("OCTOPUS_BLOCK_CHECKPOINT", "4"))
BLOCK_TOKENS = 650_000  # the longest micro-batch run in blocks (module docstring)
# torch.utils.checkpoint.checkpoint's own (private) generator, called positionally below
_PARAMS = ["fn", "preserve_rng_state", "context_fn", "determinism_check", "debug", "early_stop"]


def checkpoint_blocks(model, k: int = BLOCK, max_tokens: int = BLOCK_TOKENS):
    """Group the decoder layers into blocks of k (the last one may be shorter) in each micro-batch of at most
    max_tokens tokens; a longer one, and k = 1, keep HF's per-layer checkpoints. Needs HF gradient checkpointing
    enabled: each layer's own checkpoint function (HF's) stays the per-layer checkpoint inside its block.

    The block size must be the same on every FSDP rank: a rank in blocks all-gathers k-1 layers more in its backward,
    and ranks running different block sizes issue their collectives out of order and hang. Returns the setter of the
    next forward's token count (call it on every rank with the max over the ranks, actor.py); unset, a forward uses its
    own count. None for k <= 1."""
    if k <= 1:
        return None
    if list(inspect.signature(_checkpoint_without_reentrant_generator).parameters)[:6] != _PARAMS:
        raise RuntimeError(f"torch.utils.checkpoint._checkpoint_without_reentrant_generator is not "
                           f"({', '.join(_PARAMS)}, *args): check octopus/rl/block_checkpoint.py against this torch")
    layers = model.get_decoder().layers
    block = {}  # the next forward's token count, its block size; the open block: checkpoint, calls, last output
    for i, layer in enumerate(layers):
        layer._gradient_checkpointing_func = functools.partial(
            _block_layer, block, layer._gradient_checkpointing_func, i, len(layers), k, max_tokens)
    return functools.partial(block.__setitem__, "tokens")


def _block_layer(block: dict, layer_checkpoint, i: int, n: int, k: int, max_tokens: int, call, *args):
    """Layer i's checkpoint function (GradientCheckpointingLayer.__call__: call is its forward with the keyword
    arguments bound, args its hidden states). Layer 0 picks the forward's block size from its token count (the one set
for it, else its own). The first
    layer of a block opens the block's checkpoint, which saves the block input (torch.utils.checkpoint.checkpoint,
    split around the block's layers); the last one closes it. In the backward the block recomputes from its input,
    layer after layer, each in its own checkpoint as in the forward: so each layer's input must be the previous
    layer's output."""
    if i == 0:
        block["k"] = k if block.pop("tokens", args[0].shape[:-1].numel()) <= max_tokens else 1
    k = block["k"]
    if k == 1:
        return layer_checkpoint(call, *args)
    if i % k == 0:
        calls = block["calls"] = []

        def recompute(*xs):
            for c in calls:
                xs = (layer_checkpoint(c, *xs),)

        block["frame"] = _checkpoint_without_reentrant_generator(recompute, True, noop_context_fn, "default", False,
                                                                 True, *args)
        next(block["frame"])
    elif len(args) != 1 or args[0] is not block["out"]:
        raise RuntimeError("block checkpointing: a decoder layer's input is not the previous layer's output, which "
                           "its block's recompute would feed it (OCTOPUS_BLOCK_CHECKPOINT=1 checkpoints per layer)")
    block["calls"].append(call)
    out = block["out"] = layer_checkpoint(call, *args)
    if i % k == k - 1 or i == n - 1:
        next(block.pop("frame"), None)
        del block["calls"], block["out"]  # the calls hold the layers' kwargs (rotary cos/sin, the prefix tree)
    return out
