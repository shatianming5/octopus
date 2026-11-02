"""Shared-prefix training (built for Qwen3.5: 8 full-attention + 24 Gated DeltaNet layers; model_hooks.py per family).

The episodes of a GRPO group share their start (root + task message, ~22k tokens) and each adds ~7k of
its own. A micro-batch holding several of them is run ONCE on its prefix tree (every shared run of
tokens stored once, pre-order) and the per-token (logprob, entropy) are expanded back to the flat
per-sequence layout AReaL computes its loss on; gradients flow back through that expansion.

Exactness per layer type, so the result equals running each sequence alone:
  full attention  each tree node's queries attend to its ancestors' keys + its own (causal), one
                  FlashAttention varlen call with the ancestors' K/V gathered per node
  recurrent       nodes run level by level (Gated DeltaNet, Mamba-2); a node's recurrence starts from its parent's
                  final state (fla's varlen kernels take per-sequence initial states and return their gradients),
                  and its causal convolution sees the last (kernel-1) tokens of its ancestor path
  LM head         logprobs for each (tree position, next token) pair a sequence needs, so a branch point
                  scores every child's first token; only the pairs of positions the loss reads (`scored`: AReaL's
                  loss_mask, octopus/rl/actor.py scored_positions), the others' output is 0
AReaL's own tree mode is not used: it only masks softmax attention (GDN would run straight across
branches) and builds a dense T x T mask, impossible at ~90k tokens.

Mounts (cross-branch read, docs/cross-branch-read.md) make the tree a DAG for attention only: from a mount point on, a
sequence's full-attention queries also see token ranges of other sequences (keys as those sequences computed them, at
their own positions, with the segments a source itself mounted inside its range), its recurrent layers and convolution
still follow its own path, and its positions continue at max(own next position, every mounted segment's last position
+ 1) (rule K2 of docs/kv-read-probe.md). Which full-attention layers read the mounted keys is the server's setting,
OCTOPUS_MOUNT_LAYERS (octopus/serving/kv_mount.py mount_layers; unset: all): the others attend to the sequence's own path.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from octopus.serving.kv_mount import mount_layers

MOUNT_LAYERS = mount_layers()  # decoder layer ids whose attention reads the mounted keys, None for all
CONV_CONTEXT = 0  # tokens of left context of the model's causal convolution: set by model_hooks.patch_model
SEG = 32768  # per kernel call: recurrent-layer tokens (a multiple of fla's 64-token chunk), FlashAttention gathered keys
Mounts = dict[int, list[tuple[int, int, int, int]]]  # sequence -> [(at, src, src_start, src_end)], see build_tree


@dataclass
class Node:
    start: int  # [start, end) in tree order
    end: int
    parent: int  # index into nodes, -1 for a root
    depth: int
    path: np.ndarray  # tree indices of the ancestors' tokens and this node's, in order
    mounted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # tree indices of the segments
    # mounted at this node's start or an ancestor's: attention keys, gathered before the path


@dataclass
class Tree:
    tokens: np.ndarray  # (T_tree,)
    nodes: list[Node]  # pre-order: node ranges tile [0, T_tree)
    position_ids: np.ndarray  # (T_tree,) position of each token on its own path
    pair_pos: np.ndarray  # (P,) tree position whose hidden state scores a label
    pair_label: np.ndarray  # (P,) the label (next token)
    flat_pair: np.ndarray  # (T_flat,) pair index for each flat position, -1 when nothing follows it or it is not scored


def build_tree(flat: np.ndarray, cu: list[int], mounts: Mounts | None = None, scored: np.ndarray | None = None) -> Tree:
    """Prefix tree of the sequences flat[cu[i]:cu[i+1]] (a padding tail is just another sequence).

    scored: (T_flat,) bool, the flat positions whose head output is used (None: all). A pair is scored when any of its
    flat positions is; the other positions get no pair (head output 0), as a sequence's last position.

    mounts: {sequence i: [(at, src, src_start, src_end), ...]}: tokens at >= `at` of sequence i attend (full
    attention only) to tokens [src_start, src_end) of sequence src as well, and to what src mounted at a point strictly
    inside that range (transitively). A mount point starts a node, and sequences share a node only when their tokens and
    their mounts so far are the same. Mounts form a DAG: no sequence reads itself through a chain of reads."""
    seqs = [flat[a:b] for a, b in zip(cu[:-1], cu[1:])]
    at: list[dict[int, tuple]] = [{} for _ in seqs]  # sequence -> {mount point: ((src, src_start, src_end), ...)}
    for i, ms in (mounts or {}).items():
        for a, src, s0, s1 in sorted(ms, key=lambda m: m[0]):  # segments at one point keep their order
            at[i][a] = at[i].get(a, ()) + ((src, s0, s1),)
    tokens: list[np.ndarray] = []
    nodes: list[Node] = []
    node_mounts: list[tuple] = []
    terminal = [-1] * len(seqs)
    n_tree = 0

    def add(ids: list[int], offset: int, parent: int):  # ids share the token and the mounts at offset
        nonlocal n_tree
        ref = seqs[ids[0]]
        length = len(ref) - offset
        for i in ids:
            length = min([length] + [a - offset for a in at[i] if a > offset])
        for i in ids[1:]:
            s = seqs[i]
            m = min(len(ref), len(s))
            diff = np.flatnonzero(ref[offset:m] != s[offset:m])
            length = min(length, int(diff[0]) if diff.size else m - offset)
        path_parent = nodes[parent].path if parent >= 0 else np.empty(0, dtype=np.int64)
        nodes.append(Node(n_tree, n_tree + length, parent, nodes[parent].depth + 1 if parent >= 0 else 0,
                          np.concatenate([path_parent, np.arange(n_tree, n_tree + length)])))
        node_mounts.append(at[ids[0]].get(offset, ()))
        tokens.append(ref[offset:offset + length])
        n_tree += length
        children(ids, offset + length, len(nodes) - 1)

    def children(ids: list[int], end: int, here: int):
        rest: dict[tuple, list[int]] = {}
        for i in ids:
            if len(seqs[i]) == end:
                terminal[i] = here
            else:
                rest.setdefault((int(seqs[i][end]), at[i].get(end, ())), []).append(i)
        for group in rest.values():
            add(group, end, here)

    children(list(range(len(seqs))), 0, -1)
    own_keys: dict[int, np.ndarray] = {}

    def keys_at(k: int) -> np.ndarray:  # the keys node k's own mounts add (a source can come later in pre-order)
        if k not in own_keys:
            own_keys[k] = np.concatenate([np.empty(0, dtype=np.int64)] + [segment(*m) for m in node_mounts[k]])
        return own_keys[k]

    def segment(src: int, s0: int, s1: int) -> np.ndarray:
        """Tokens [s0, s1) of src and, from a read of src's own inside the range (s0 < its mount point < s1), the keys
        that read mounted: on the server those rows sit between src's tokens, inside the mounted range."""
        inner, j = [], terminal[src]
        while j >= 0:
            if node_mounts[j] and s0 < len(nodes[j].path) - (nodes[j].end - nodes[j].start) < s1:
                inner.append(keys_at(j))
            j = nodes[j].parent
        return np.concatenate([nodes[terminal[src]].path[s0:s1]] + inner)

    for k, node in enumerate(nodes):  # pre-order: a parent's mounted keys are known
        node.mounted = np.concatenate([nodes[node.parent].mounted if node.parent >= 0 else node.mounted, keys_at(k)])
    tree_tokens = np.concatenate(tokens) if tokens else np.empty(0, dtype=flat.dtype)
    node_of = np.repeat(np.arange(len(nodes)), [n.end - n.start for n in nodes])
    first: dict[int, int] = {}

    def first_pos(k: int) -> int:  # the parent's next position; after a mount, K2: max(that, mounted end)
        if k not in first:
            n = nodes[k]
            p = first_pos(n.parent) + nodes[n.parent].end - nodes[n.parent].start if n.parent >= 0 else 0
            for src, _, s1 in node_mounts[k]:
                last = nodes[terminal[src]].path[s1 - 1]
                j = node_of[last]
                p = max(p, first_pos(j) + int(last) - nodes[j].start + 1)
            first[k] = p
        return first[k]

    position_ids = np.empty(n_tree, dtype=np.int64)
    for k, node in enumerate(nodes):
        base = first_pos(k)
        position_ids[node.start:node.end] = np.arange(base, base + node.end - node.start)
    flat_to_tree = np.concatenate([nodes[t].path if t >= 0 else np.empty(0, dtype=np.int64) for t in terminal])
    # the label of flat position p is flat token p+1, except at each sequence's last position
    has_next = np.ones(len(flat), dtype=bool) if scored is None else scored.astype(bool)
    has_next[np.asarray(cu[1:]) - 1] = False
    pos = flat_to_tree[has_next]
    label = flat[1:][has_next[:-1]].astype(np.int64)
    bits = int(label.max(initial=0)).bit_length()  # (pos, label) packed into one int64, sorted as the pairs
    pairs, inverse = np.unique(pos.astype(np.int64) << bits | label, return_inverse=True)
    flat_pair = np.full(len(flat), -1, dtype=np.int64)
    flat_pair[has_next] = inverse
    return Tree(tree_tokens, nodes, position_ids, pairs >> bits, pairs & ((1 << bits) - 1), flat_pair)


class Level(NamedTuple):
    """One recurrent kernel call: segments of one depth, at most SEG tokens."""
    ids: list[int]  # segment ids
    parents: list[int]  # each one's parent segment, -1 at the roots
    index: slice | torch.Tensor  # their tree tokens, in order (a slice when contiguous)
    cu: torch.Tensor  # cu_seqlens of the segments
    cu_cpu: torch.Tensor  # the same on the host: fla builds its chunk indices from it without a device sync
    lens: list[int]  # segment lengths
    ctx_index: torch.Tensor  # tree indices of each segment's convolution context, segment by segment
    ctx_lens: list[int]
    ctx_dest: torch.Tensor  # their rows in the segments' convolution initial states (conv_dest)


class TreeMeta:
    """The tree as device tensors for the patched layers (passed to the model as the `octopus_tree` kwarg)."""

    def __init__(self, tree: Tree, device):
        nodes = tree.nodes
        self.n_tokens = len(tree.tokens)
        # attention: a node's keys are its mounted segments, then its path: its own tokens last, so the causal mask
        # holds; the layers outside MOUNT_LAYERS read the path alone (own_calls)
        self.attn_calls, self.own_calls = layer_calls([n.end - n.start for n in nodes], [n.mounted for n in nodes],
                                                      [n.path for n in nodes], device)
        segs, ctx = segments(nodes)
        self.conv_keep = [b - a for a, b, _, _ in segs]
        self.conv_ctx = [len(c) for c in ctx]
        self.conv_ctx_index = _index(ctx, device)
        self.conv_cu_cpu = torch.tensor(np.cumsum([0] + self.conv_keep), dtype=torch.long)
        self.conv_cu = self.conv_cu_cpu.to(device)
        self.conv_dest = conv_dest(self.conv_ctx, device)
        self.levels = levels(dict(enumerate(segs)), ctx, device)
        self.pair_pos = torch.from_numpy(tree.pair_pos).to(device)
        self.pair_label = torch.from_numpy(tree.pair_label).to(device)
        # flat position -> pair row; -1 (no next token) -> an appended zero row
        flat_pair = tree.flat_pair.copy()
        flat_pair[flat_pair < 0] = len(tree.pair_pos)
        self.flat_pair = torch.from_numpy(flat_pair).to(device)

    def calls(self, layer_idx: int) -> list[tuple]:
        """The FlashAttention calls of decoder layer layer_idx (MOUNT_LAYERS: which layers read the mounted keys)."""
        return self.attn_calls if MOUNT_LAYERS is None or layer_idx in MOUNT_LAYERS else self.own_calls


def segments(nodes: list[Node]) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Recurrence segments of at most SEG tokens, (start, end, parent segment, depth) in tree order (a node is a chain
    of segments, each starting from the previous one's final state; at most SEG tokens per call: fla's per-call buffers
    stay small at 262k), and each one's convolution context: up to CONV_CONTEXT tokens of its ancestor path."""
    segs: list[tuple[int, int, int, int]] = []
    ctx: list[np.ndarray] = []  # per segment: tree indices of its convolution context
    last: dict[int, int] = {}  # node -> its last segment
    for i, n in enumerate(nodes):  # pre-order: parents first
        parent = last[n.parent] if n.parent >= 0 else -1
        for a in range(n.start, n.end, SEG):
            segs.append((a, min(a + SEG, n.end), parent, segs[parent][3] + 1 if parent >= 0 else 0))
            before = len(n.path) - (n.end - a)  # path tokens before the segment
            ctx.append(n.path[max(0, before - CONV_CONTEXT):before])
            parent = len(segs) - 1
        last[i] = parent
    return segs, ctx


def _index(parts, device) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([np.empty(0, dtype=np.int64), *parts])).to(device)


def conv_dest(ctx_lens: list[int], device) -> torch.Tensor:
    """Where each segment's convolution context goes in the [segments x W, C] rows of their initial states (W =
    CONV_CONTEXT + 1, fla's causal_conv1d layout [segments, C, W]): right-aligned, slot W-1 the token just before the
    segment. fla reads slots 1..W-1; the rest stay zero, a sequence start's zero padding."""
    w = CONV_CONTEXT + 1
    return _index([i * w + w - n + np.arange(n) for i, n in enumerate(ctx_lens)], device)


def levels(segs: dict[int, tuple[int, int, int, int]], ctx, device) -> list[Level]:
    """The recurrence level by level: segments {id: (start, end, parent id, depth)} (start, end: rows of the tokens they
    run on) of one depth, at most SEG tokens per call; ctx[id]: the rows of each one's convolution context."""
    out = []
    for depth in range(max(sg[3] for sg in segs.values()) + 1):
        calls, size = [], SEG
        for i, sg in segs.items():
            if sg[3] == depth:
                if size + sg[1] - sg[0] > SEG:
                    calls.append([])
                    size = 0
                calls[-1].append(i)
                size += sg[1] - sg[0]
        for ids in calls:
            spans = [segs[i][:2] for i in ids]
            index = (slice(spans[0][0], spans[-1][1]) if all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
                     else _index([np.arange(a, b) for a, b in spans], device))
            lens = [b - a for a, b in spans]
            cu = torch.tensor(np.cumsum([0] + lens), dtype=torch.long)
            ctx_lens = [len(ctx[i]) for i in ids]
            out.append(Level(ids, [segs[i][2] for i in ids], index, cu.to(device), cu, lens,
                             _index([ctx[i] for i in ids], device), ctx_lens, conv_dest(ctx_lens, device)))
    return out


def attention_calls(q_lens: list[int], keys_of: list[np.ndarray], device) -> list[tuple]:
    """(query rows, key index, cu_q, cu_k, max_q, max_k) per FlashAttention call, the calls' rows tiling the tree:
    consecutive nodes (q_lens: their query rows), at most SEG gathered keys per call (a node with more keys alone): its
    backward expands a call's keys to every query head (GQA), 16 KB per key for Qwen3.5-4B."""
    calls, groups, keys = [], [], SEG
    for i in range(len(q_lens)):
        if keys + len(keys_of[i]) > SEG:
            groups.append([])
            keys = 0
        groups[-1].append(i)
        keys += len(keys_of[i])
    for g in groups:
        q, k = [q_lens[i] for i in g], [len(keys_of[i]) for i in g]
        calls.append((
            sum(q),
            torch.from_numpy(np.concatenate([keys_of[i] for i in g])).to(device),
            torch.tensor(np.cumsum([0] + q), dtype=torch.int32, device=device),
            torch.tensor(np.cumsum([0] + k), dtype=torch.int32, device=device), max(q), max(k)))
    return calls


def layer_calls(q_lens: list[int], mounted: list[np.ndarray], paths: list[np.ndarray], device) -> tuple[list, list]:
    """attention_calls of nodes whose keys are their mounted segments, then their path (own tokens last: the causal mask
    holds), and of the same nodes reading their path alone (the layers outside MOUNT_LAYERS; the same calls when every
    layer reads the mounts or nothing is mounted)."""
    calls = attention_calls(q_lens, [np.concatenate([m, p]) for m, p in zip(mounted, paths)], device)
    return calls, (calls if MOUNT_LAYERS is None or not any(len(m) for m in mounted)
                   else attention_calls(q_lens, paths, device))


def take(x: torch.Tensor, index) -> torch.Tensor:
    """Tokens `index` (a slice or index tensor) of [B, T, ...] x: by index_select, not x[:, index], so the backward is
    one index_add_ (no index repeats) where advanced indexing sorts the indices."""
    return x[:, index] if isinstance(index, slice) else x.index_select(1, index)


def segment_conv(x: torch.Tensor, weight: torch.Tensor, bias, activation, cu: torch.Tensor,
                 cu_cpu: torch.Tensor | None, ctx: torch.Tensor | None = None,
                 dest: torch.Tensor | None = None) -> torch.Tensor:
    """Causal convolution of each segment of x [1, T, C] (cu_seqlens cu, on the host cu_cpu: no device sync) by
    weight [C, W], one call of fla's Triton kernel; ctx [n, C]: the segments' left contexts, at rows dest of their
    initial states (conv_dest). The kernel takes the gradients of x and ctx (accumulating in fp32, rounding once); it
    reads the output's gradient as contiguous, so a strided or expanded one is made contiguous first."""
    from fla.modules.conv import causal_conv1d

    init = None
    if ctx is not None:
        w = weight.shape[-1]
        init = x.new_zeros((len(cu) - 1) * w, x.shape[-1]).index_copy(0, dest, ctx).view(-1, w, x.shape[-1])
        init = init.transpose(1, 2)
    y = causal_conv1d(x, weight, bias, initial_state=init, activation=activation, cu_seqlens=cu,
                      cu_seqlens_cpu=cu_cpu)[0]
    if y.requires_grad:  # fla's backward reads dy as contiguous: a strided or stride-0 one gives garbage or a fault
        y.register_hook(torch.Tensor.contiguous)
    return y


def warm_conv(channels: int, width: int, bias: bool, dtype, device) -> None:
    """segment_conv forward and backward once per autotune key of fla's kernels, on a still empty card: mid-step,
    autotuning benchmarks its configs with extra buffers (actor._check_weights_loaded: an 88k-token micro-batch ran out
    of memory that way). The key is channels, width, cdiv(T, 1024) (a training call has T <= SEG tokens) and the dtypes
    of the tensor arguments, so with and without contexts (a level of roots has none). Every length bucket compiles anew:
    25 min on an empty Triton cache at 8192 channels, then 3.2 s for a later process on the host (FLA_CACHE_RESULTS,
    TRITON_CACHE_DIR keep them; results/chunk-major/README.md)."""
    weight = torch.zeros(channels, width, dtype=dtype, device=device)
    b = torch.zeros(channels, dtype=dtype, device=device) if bias else None
    ctx = torch.zeros(width - 1, channels, dtype=dtype, device=device, requires_grad=True)
    slots = torch.arange(1, width, device=device)  # one segment, its context in slots 1..W-1
    for t in range(1024, SEG + 1, 1024):
        cu = torch.tensor([0, t])
        for c in (ctx, None):
            x = torch.zeros(1, t, channels, dtype=dtype, device=device, requires_grad=True)
            y = segment_conv(x, weight, b, "silu", cu.to(device), cu, c, slots)
            y.backward(torch.ones_like(y))  # a dense gradient, as training's


def tree_conv(x: torch.Tensor, weight, bias, activation, meta: TreeMeta) -> torch.Tensor:
    """x: [1, C, T_tree] -> the same, each segment convolved after the last tokens of its ancestor path."""
    y = x.transpose(1, 2)
    ctx = y[0].index_select(0, meta.conv_ctx_index) if len(meta.conv_ctx_index) else None  # a single root: none
    y = segment_conv(y, weight, bias, activation, meta.conv_cu, meta.conv_cu_cpu, ctx, meta.conv_dest)
    return y.transpose(1, 2)


def varlen_conv(conv):
    """A family's torch causal_conv1d_fn(x [batch, channels, time], weight, bias, activation=...) made to convolve each
    packed sequence (cu_seq_lens_q) or tree segment (octopus_tree) on its own (segment_conv); other calls (generation,
    one unpacked sequence) keep `conv`."""
    def causal_conv1d_fn(x, weight, bias=None, activation=None, cu_seq_lens_q=None, octopus_tree=None, **kwargs):
        if octopus_tree is not None:
            return tree_conv(x, weight, bias, activation, octopus_tree)
        if cu_seq_lens_q is None:
            return conv(x, weight, bias, activation=activation, **kwargs)
        return segment_conv(x.transpose(1, 2), weight, bias, activation, cu_seq_lens_q.long(), None).transpose(1, 2)
    return causal_conv1d_fn


def tree_recurrence(chunk, meta: TreeMeta, *xs, **kw):
    """A chunked linear recurrence over the tree, level by level; `chunk` is an fla varlen kernel (chunk_gated_delta_rule,
    chunk_simple_gla) and xs its leading [1, T_tree, ...] inputs in its order (q, k, v, g[, beta]); each call gets the
    level's cu_seqlens on the device and the host. Returns (output [1, T_tree, H, Dv], None)."""
    outputs = {}
    finals: dict[int, torch.Tensor] = {}
    for level in meta.levels:
        initial = None
        if level.parents[0] >= 0:  # every segment of a level below the roots has a parent
            initial = torch.stack([finals[p] for p in level.parents])
        out, state = chunk(*(take(x, level.index) for x in xs), initial_state=initial, output_final_state=True,
                           cu_seqlens=level.cu, cu_seqlens_cpu=level.cu_cpu, **kw)
        for i, o, s in zip(level.ids, out.split(level.lens, dim=1), state):
            outputs[i], finals[i] = o, s
    return torch.cat([outputs[i] for i in range(len(outputs))], dim=1), None


def attention_call(q, k, v, call: tuple, softmax_scale=None) -> torch.Tensor:
    """One of TreeMeta's FlashAttention calls: q [n, H, D] its query rows, k/v [T_tree, Hkv, D] -> [n, H, D]."""
    from flash_attn import flash_attn_varlen_func

    _, keys, cu_q, cu_k, max_q, max_k = call
    return flash_attn_varlen_func(q, k.index_select(0, keys), v.index_select(0, keys), cu_q, cu_k, max_q, max_k,
                                  softmax_scale=softmax_scale, causal=True)  # causal aligned bottom-right


def tree_attention(q, k, v, meta: TreeMeta, softmax_scale=None, layer_idx=None):
    """q: [1, T_tree, H, D], k/v: [1, T_tree, Hkv, D] -> [1, T_tree, H, D] for decoder layer layer_idx."""
    calls = meta.calls(layer_idx)
    out = [attention_call(q_, k[0], v[0], call, softmax_scale)
           for q_, call in zip(q[0].split([c[0] for c in calls]), calls)]
    return (torch.cat(out) if len(out) > 1 else out[0]).unsqueeze(0)


def patch_attention():
    """Route HF flash attention to tree_attention when the call carries `octopus_tree`."""
    from transformers.integrations import flash_attention

    original = flash_attention._flash_attention_forward
    if getattr(original, "octopus_tree_patch", False):
        return

    def forward(query_states, key_states, value_states, attention_mask, *args, octopus_tree=None, **kwargs):
        if octopus_tree is None:
            return original(query_states, key_states, value_states, attention_mask, *args, **kwargs)
        return tree_attention(query_states, key_states, value_states, octopus_tree, kwargs.get("softmax_scale"),
                              kwargs["layer_idx"])

    forward.octopus_tree_patch = True
    flash_attention._flash_attention_forward = forward


def tree_head_output(chunk_fn, hidden: torch.Tensor, meta: TreeMeta) -> torch.Tensor:
    """[1, T_tree, H] hidden -> [1, T_flat, C] head columns per flat position via the label pairs;
    chunk_fn(hidden_rows, labels) -> [n, C] is the chunked head (positions with no next token get 0)."""
    pairs = chunk_fn(hidden[0].index_select(0, meta.pair_pos), meta.pair_label)
    pairs = torch.cat([pairs, pairs.new_zeros(1, pairs.shape[1])])
    return pairs.index_select(0, meta.flat_pair).unsqueeze(0)


def tree_inputs(input_ids: torch.Tensor, cu_seq_lens: torch.Tensor, mounts: Mounts | None = None,
                scored: torch.Tensor | None = None, with_tree: bool = False) -> tuple[TreeMeta, dict] | None:
    """Tree kwargs for a flat packed micro-batch, or None when its sequences share nothing, mount nothing and it is
    short. mounts, scored: see build_tree (sequence indices in cu order; scored: flat positions). with_tree: the
    TreeMeta keeps the host-side Tree as .tree (chunk-major plans its chunks from it, octopus/rl/chunk_major.py
    ChunkPlan), and every micro-batch gets one: FSDP ranks must issue the same collectives in the same order, and
    a flat micro-batch's backward gathers and reduce-scatters layer by layer where chunk-major's gathers every layer
    first and reduce-scatters them all at the end (one rank flat beside one chunk-major: NCCL paired them wrongly,
    NaN gradients; results/chunk-major/README.md "Ranks on different paths")."""
    flat = input_ids.reshape(-1).cpu().numpy()
    cu = cu_seq_lens.tolist()
    if cu[-1] < len(flat):  # padding tail
        cu.append(len(flat))
    tree = build_tree(flat, cu, mounts, None if scored is None else scored.cpu().numpy())
    # longer: the tree path bounds each recurrent kernel call
    if not with_tree and not mounts and len(tree.tokens) == len(flat) and len(flat) <= SEG:
        return None
    meta = TreeMeta(tree, input_ids.device)
    if with_tree:
        meta.tree = tree
    return meta, {"input_ids": torch.from_numpy(tree.tokens.astype(np.int64)).to(input_ids.device)[None],
                  "position_ids": torch.from_numpy(tree.position_ids).to(input_ids.device)[None],
                  "octopus_tree": meta, "use_cache": False}


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    m = min(len(a), len(b))
    diff = np.flatnonzero(a[:m] != b[:m])
    return int(diff[0]) if diff.size else m


def tree_tokens(seqs: list[np.ndarray]) -> int:
    """Prefix-tree size of the sequences: in lexicographic order each adds len - LCP(previous)."""
    seqs = sorted(seqs, key=lambda s: s.tolist())
    return sum(len(s) - (_lcp(p, s) if p is not None else 0) for p, s in zip([None] + seqs[:-1], seqs))


# Seconds of attention per query-key pair over seconds of everything else per prefix-tree token, measured for
# Qwen3.5-4B on a 4090D (results/dp-balance/README.md: ppo_update chunk-major on mb-256190, FlashAttention timed by
# bench_attn.py); another model or card needs its own (OCTOPUS_KAPPA, a host setting).
KAPPA = float(os.environ.get("OCTOPUS_KAPPA", "1.15e-5"))


def cost(seqs: list[np.ndarray], mounts: Mounts | None = None) -> float:
    """A micro-batch's training cost in tree tokens: its prefix-tree tokens + KAPPA x the causal attention query-key
    pairs (each node's rows against its mounted keys and path): at 256k tree tokens attention is a quarter of a training
    micro-batch, and a long path costs it quadratically."""
    tree = build_tree(np.concatenate(seqs), np.cumsum([0] + [len(s) for s in seqs]).tolist(), mounts or None)
    pairs = 0
    for n in tree.nodes:
        q, keys = n.end - n.start, len(n.mounted) + len(n.path)
        pairs += q * (keys - q) + q * (q + 1) // 2
    return len(tree.tokens) + KAPPA * pairs


def subset(seqs: list[np.ndarray], mounts: Mounts, rows: list[int]) -> tuple[list[np.ndarray], Mounts]:
    """The sequences at rows (each reader's sources among them) and their mounts, renumbered."""
    at = {r: i for i, r in enumerate(rows)}
    return [seqs[r] for r in rows], {at[r]: [(p, at[src], a, b) for p, src, a, b in mounts[r]]
                                     for r in rows if r in mounts}


def who() -> str:
    """This process in a log line: its global rank (the data-parallel rank under pure FSDP), or its pid off the trainer
    (a rollout worker, a script)."""
    return f"rank {torch.distributed.get_rank()}" if torch.distributed.is_initialized() else f"pid {os.getpid()}"


def allocate_prefix_groups(seqs: list[np.ndarray], cap: int, min_groups: int = 1, divisor: int = 1,
                           mounts: Mounts | None = None) -> list[list[int]]:
    """Micro-batches (lists of sequence indices) sized by prefix-tree tokens <= cap. In lexicographic order a
    sequence adds exactly len - LCP(previous) tokens to the tree, so walking that order keeps sequences that
    share a start together and sizes each chunk exactly; chunks are then packed first-fit-decreasing (bins add
    chunk sizes, an upper bound). At least min_groups micro-batches (a multiple of divisor) when there are enough
    sequences. (Keying on a fixed-length start fails: every task shares the long system prompt.)

    mounts (build_tree's format): a reader and its sources are one unit, never split (build_tree needs a mount's source
    in the reader's micro-batch). A unit larger than cap gets a micro-batch of its own over the cap, as a single sequence
    longer than cap does, and is counted in a log line; it is never cut.

    Returned largest first: with OCTOPUS_CHUNK_MAJOR off, FSDP runs micro-batch j of every data-parallel rank in
    lockstep (per-layer all-gathers), so a step costs sum over j of the largest j-th micro-batch; with every rank in
    the same order the full ones run together and the partial ones share the last slot (bundle-e8f steps 31-38: -7%).
    Chunk-major syncs once per call instead (chunk_major.across_micro_batches)."""
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].tolist())
    rank = {i: r for r, i in enumerate(order)}
    root = list(range(len(seqs)))  # union-find over mount edges

    def find(i: int) -> int:
        while root[i] != i:
            i = root[i]
        return i
    for i, ms in (mounts or {}).items():
        for _, src, _, _ in ms:
            root[find(src)] = find(i)
    units: dict[int, list[int]] = {}  # in lexicographic order of each unit's first sequence
    for i in order:
        units.setdefault(find(i), []).append(i)

    def added(ranks: list[int], i: int) -> int:  # tokens sequence i adds to the tree of the sequences at these ranks
        k = bisect.bisect(ranks, rank[i])
        near = [order[r] for r in ranks[max(k - 1, 0):k + 1]]  # its lexicographic neighbours share the most
        return len(seqs[i]) - max([_lcp(seqs[j], seqs[i]) for j in near] + [0])

    chunks: list[tuple[int, list[int]]] = []  # (tree tokens, indices)
    size, cur, ranks = 0, [], []
    for unit in units.values():
        add, grown = 0, list(ranks)
        for i in unit:
            add += added(grown, i)
            bisect.insort(grown, rank[i])
        if cur and size + add > cap:
            chunks.append((size, cur))
            size, cur, ranks = 0, [], []
            add, grown = tree_tokens([seqs[i] for i in unit]), sorted(rank[i] for i in unit)
        if len(unit) > 1 and add > cap and not cur:
            print(f"[octopus] {who()}: mount unit over the micro-batch cap: {add} > {cap} tree tokens ({len(unit)} "
                  "sequences)", flush=True)
        size, cur, ranks = size + add, cur + unit, grown
    if cur:
        chunks.append((size, cur))
    bins: list[tuple[int, list[int]]] = []
    for size, idx in sorted(chunks, key=lambda c: -c[0]):
        for b, (used, members) in enumerate(bins):
            if used + size <= cap:
                bins[b] = (used + size, members + idx)
                break
        else:
            bins.append((size, list(idx)))
    out = [sorted(m) for _, m in bins]
    target = -(-max(min_groups, len(out)) // divisor) * divisor

    def parts(m: list[int]) -> list[list[int]]:  # a micro-batch's units, by first index
        by: dict[int, list[int]] = {}
        for i in m:
            by.setdefault(find(i), []).append(i)
        return list(by.values())
    while len(out) < target and any(len(parts(m)) > 1 for m in out):  # split the fullest splittable micro-batch
        k = max((j for j in range(len(out)) if len(parts(out[j])) > 1), key=lambda j: len(out[j]))
        p = parts(out.pop(k))
        out += [sorted(sum(p[: len(p) // 2], [])), sorted(sum(p[len(p) // 2:], []))]
    return sorted((sorted(m) for m in out), key=lambda m: -tree_tokens([seqs[i] for i in m]))


# Reads through AReaL's export (docs/cross-branch-read.md "Export"). AReaL packs and splits only [B, L] tensors, so a
# leaf carries its reads per token, and sequences are matched by their leaf interaction ids, not by batch rows.
MOUNT_KEYS = ("octopus_node", "octopus_mount", "octopus_mount_start", "octopus_mount_end")


def node_id(interaction_id: str) -> int:
    """A leaf's id in the long tensors: 60 random bits of its completion id (chatcmpl-<29 hex>)."""
    return int(interaction_id[-15:], 16)


def record_reads(leaves: dict) -> None:
    """The proxy's concat export (patches/areal-cursor.patch, SessionData.export_interactions): a leaf whose chain holds
    reads (the proxy's metadata["mount"] = {source, start, end, at} on a reader interaction) gets metadata["octopus_reads"]
    = [[at, source leaf, start, end], ...]. The leaf's tokens are its chain's at the same positions, so `at` holds in
    the leaf and every later token of it is computed with the read. The source leaf is the first leaf below the source
    interaction: each has its tokens, reads and positions up to it. A session without reads is left as it is."""
    holder: dict[str, str] = {}
    for lid, leaf in leaves.items():
        x = leaf
        while x is not None:
            holder.setdefault(x.interaction_id, lid)
            x = x.parent
    for leaf in leaves.values():
        reads, x = [], leaf
        while x is not None:
            if m := x.metadata.get("mount"):
                reads.append([m["at"], holder[m["source"]], m["start"], m["end"]])
            x = x.parent
        if reads:
            leaf.metadata["octopus_reads"] = reads


def read_tensors(leaf_id: str, reads, shape) -> dict[str, torch.Tensor]:
    """A leaf's record_reads as [1, L] long tensors: octopus_node = the leaf's id at its first token; at each read's
    `at`, octopus_mount = the source leaf's id and octopus_mount_start/end = the range [start, end) of its tokens."""
    out = {k: torch.zeros(shape, dtype=torch.long) for k in MOUNT_KEYS}
    out["octopus_node"].view(-1)[0] = node_id(leaf_id)
    for at, src, start, end in reads:
        for k, v in zip(MOUNT_KEYS[1:], (node_id(src), start, end)):
            out[k].view(-1)[at] = v
    return out


def seq_mounts(cu: list[int], node: np.ndarray, src: np.ndarray, start: np.ndarray, end: np.ndarray) -> Mounts:
    """build_tree's mounts from packed read_tensors (sequence i = positions cu[i]:cu[i+1]). A source leaf that is not
    among these sequences raises: a reader never runs without its read."""
    holder = {int(node[c]): j for j, c in enumerate(cu[:-1])}
    mounts: Mounts = {}
    for p in np.flatnonzero(src[:cu[-1]]):
        i = int(np.searchsorted(cu, p, "right")) - 1
        if (j := holder.get(int(src[p]))) is None:
            raise ValueError(f"sequence {i} reads leaf {int(src[p]):x} at token {p - cu[i]}, which is not in its "
                             "micro-batch")
        mounts.setdefault(i, []).append((int(p - cu[i]), j, int(start[p]), int(end[p])))
    return mounts
