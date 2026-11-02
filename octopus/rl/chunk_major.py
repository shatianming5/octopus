"""Chunk-major training of one prefix-tree micro-batch (octopus/rl/prefix_tree.py): the tree runs through the whole
model one chunk of at most `tokens` tree tokens at a time, so activations span a chunk, not the tree.

Chunks are runs of the tree's recurrence segments (prefix_tree.segments) in a topological order: a node after its parent
and after the nodes holding the keys it mounts, otherwise in pre-order. A chunk's tokens are its rows, in that order.
What a chunk needs from earlier chunks, they keep (per decoder layer):
  attention     k, v of each token a later chunk's queries attend to (an ancestor's, a mounted one, or an earlier part
                of the query's own node); the chunk's FlashAttention calls index them after its own rows
  recurrence    the final state of each segment whose child segment runs in a later chunk
  convolution   the layer input of each token in a later segment's convolution context, indexed after the chunk's rows
run_forward runs the chunks in order without autograd and keeps those. run_backward runs them in reverse, each again
with autograd, and backpropagates its share of the output gradient together with the gradients later chunks sent to the
tensors it kept (summed in ACC_DTYPE): what a chunk reads of earlier chunks is loaded by an op (_Kept) whose backward
adds the gradient to those sums, so no copy of it waits for the chunk's backward beyond the layer inputs that hold it.
Parameter gradients add up to those of the whole tree. On a GPU the kept tensors and their gradient sums wait in pinned
host memory (HOST, octopus/rl/pinned_slab.py): their slots are ordered by owning chunk, so a chunk writes its own as one
block and reads an earlier chunk's as a few runs (whole node parts).
chunk_major_output is the two as one autograd op, called inside the model's forward.

FSDP2 (fully_shard, torch 2.9; AReaL wraps each decoder layer, the input embedding and the root) expects one forward
and one backward per module and micro-batch. Around a chunk loop (_gathered) the wrapped modules' parameters are
gathered once and stay gathered, and their gradients add up unreduced. After each chunk's backward the modules are set
back to idle, so the next chunk's forward runs FSDP's hooks again: its input cast (mixed precision) and its
post-backward, which moves the chunk's gradient into the reduce-dtype (fp32) sum. Left in pre-backward, the later
chunks would skip both, their gradients adding up in the parameter dtype (bf16); left in post-backward at the end,
FSDP's final callback would skip the modules and never reduce their gradients. The gradients are reduce-scattered once,
by that final callback, which FSDP queued on the outer backward when the root's pre-backward hook saw the chunk-major
output's gradient. No collective runs per chunk, so ranks may run different numbers of chunks.

Across the micro-batches of an actor call (across_micro_batches, octopus/rl/actor.py forward_backward_batch) the same
holds per call: every module, the root's too, stays gathered from the call's first forward to its end and adds its
gradients up unreduced until the last micro-batch's backward, whose final callback reduce-scatters them once and
reshards. Otherwise each micro-batch gathers the parameters again and reduce-scatters its gradients, and every rank
waits there for the slowest one's micro-batch: a call takes the sum over micro-batches of the slowest rank's, instead of
the slowest rank's sum. The parameters stay gathered no longer than over one micro-batch already; the unsharded
gradients (LoRA's in fp32: 0.45 GiB for r64 on Qwen3.5-4B) wait over the call.
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import os

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from octopus.rl import prefix_tree, seq_tiles
from octopus.rl.pinned_slab import PinnedSlab
from octopus.rl.prefix_tree import Tree, TreeMeta

# host settings (octopus/rl/config.py HOST_ENV): the actor trains each prefix-tree micro-batch chunk-major
# (octopus/rl/actor.py install_forward), in chunks of at most TOKENS tree tokens (32k: the 994k-token unit peaked at
# 31.9 GiB on one card; before one checkpoint per layer in run_backward, 64k ran out of a 40 GiB cap there and took
# 35.4 GiB on each of 2 ranks; results/chunk-major)
ENABLED = os.environ.get("OCTOPUS_CHUNK_MAJOR", "0") == "1"
TOKENS = int(os.environ.get("OCTOPUS_CHUNK_TOKENS", "32768"))
KINDS = {"k": "kv", "v": "kv", "x": "x", "s": "s"}  # kept tensor -> what indexes it: tokens (kv, x) or segments (s)
HOST = True  # on a GPU: kept tensors and their gradient sums in pinned host memory
ACC_DTYPE = torch.float32  # the gradient sums of the kept tensors: at least this precise


class Chunk:
    """One chunk as seq_tiles.tree_layer_forward's meta (attn_calls, own_calls, levels in chunk rows), its tokens
    (ids, pos; tree_index, for diagnostics such as results/chunk-major/layerdiff.py), its label pairs (pair_rows of
    the tree's, at rows pair_at) and, per kind, the slots of the kept tensors it reads (ext: runs [a, b)) and the block
    of those it keeps (own: its rows or segments, slots lo, hi)."""
    calls = TreeMeta.calls


class ChunkPlan:
    """The chunks of a prefix tree (build_tree) of at most `tokens` tree tokens each (whole segments: a chunk of fewer
    than SEG tokens still holds one), in run order, and how many tensors of each kind they keep."""

    def __init__(self, tree: Tree, tokens: int, device):
        nodes = tree.nodes
        segs, ctx = prefix_tree.segments(nodes)
        node_of = np.repeat(np.arange(len(nodes)), [n.end - n.start for n in nodes])
        seg_node = node_of[[sg[0] for sg in segs]]
        first = np.searchsorted(seg_node, np.arange(len(nodes) + 1))  # node k: segments first[k]..first[k+1]
        chunk_segs, size = [], tokens
        for k in _topological(nodes, node_of):
            for s in range(first[k], first[k + 1]):
                if size + segs[s][1] - segs[s][0] > tokens:
                    chunk_segs.append([])
                    size = 0
                chunk_segs[-1].append(s)
                size += segs[s][1] - segs[s][0]
        seg_chunk = np.empty(len(segs), dtype=np.int64)
        tok_chunk, row = np.empty(len(tree.tokens), dtype=np.int64), np.empty(len(tree.tokens), dtype=np.int64)
        seg_row = np.empty(len(segs), dtype=np.int64)
        toks = [np.concatenate([np.arange(*segs[s][:2]) for s in ids]) for ids in chunk_segs]
        for c, (ids, tok) in enumerate(zip(chunk_segs, toks)):
            seg_chunk[ids], tok_chunk[tok], row[tok], seg_row[ids] = c, c, np.arange(len(tok)), np.arange(len(ids))
        # slot order of the kept tensors: by owning chunk, then row (tokens) or position (segments)
        order = {"kv": tok_chunk * len(row) + row, "s": seg_chunk * len(segs) + seg_row}
        order["x"] = order["kv"]
        as_tensor = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)  # noqa: E731

        self.chunks: list[Chunk] = []
        exts: list[dict[str, np.ndarray]] = []  # per chunk: tree indices (kv, x) or segments (s) read from earlier ones
        for c, (ids, tok) in enumerate(zip(chunk_segs, toks)):
            n_rows = len(tok)

            def external(parts: list[np.ndarray], here: bool = False) -> np.ndarray:
                """The tree indices among parts in earlier chunks, in slot order (here: this chunk's rows among them)"""
                t = np.concatenate([np.empty(0, dtype=np.int64), *parts])
                if here:
                    return np.unique(row[t[tok_chunk[t] == c]])
                t = np.unique(t[tok_chunk[t] < c])
                return t[np.argsort(order["kv"][t])]

            def rows(t: np.ndarray, ext: np.ndarray, own: np.ndarray | None = None) -> np.ndarray:
                """t's rows: in this chunk (their positions among its rows `own`, default all), then ext's after"""
                here = row[t] if own is None else np.searchsorted(own, row[t])
                after = (n_rows if own is None else len(own)) + np.searchsorted(order["kv"][ext], order["kv"][t])
                return np.where(tok_chunk[t] == c, here, after)

            spans: dict[int, tuple[int, int]] = {}  # node -> its tokens in this chunk (consecutive segments)
            for s in ids:
                spans[seg_node[s]] = (spans.get(seg_node[s], segs[s])[0], segs[s][1])
            # a node's part [a, b) attends to the node's keys up to b: mounted segments, then its path
            paths = [nodes[k].path[:len(nodes[k].path) - (nodes[k].end - b)] for k, (_, b) in spans.items()]
            mounted = [nodes[k].mounted for k in spans]
            ext_kv, ext_x = external(mounted + paths), external([ctx[s] for s in ids])
            ch = Chunk()
            ch.attn_calls, ch.own_calls = prefix_tree.layer_calls(
                [b - a for a, b in spans.values()], [rows(m, ext_kv) for m in mounted],
                [rows(p, ext_kv) for p in paths], device)
            starts = {s: int(row[segs[s][0]]) for s in ids}
            ctx_own = external([ctx[s] for s in ids], here=True)  # convolution contexts: these rows, then ext_x
            ch.levels = prefix_tree.levels(
                {s: (starts[s], starts[s] + segs[s][1] - segs[s][0], *segs[s][2:]) for s in ids},
                {s: rows(ctx[s], ext_x, ctx_own) for s in ids}, device)
            ch.ctx_rows = as_tensor(ctx_own)
            ch.tree_index = tok
            ch.ids = as_tensor(tree.tokens[tok])[None]
            ch.pos = as_tensor(tree.position_ids[tok])[None]
            pairs = np.flatnonzero(tok_chunk[tree.pair_pos] == c)
            ch.pair_rows, ch.pair_at = as_tensor(pairs), as_tensor(row[tree.pair_pos[pairs]])
            ch.labels = as_tensor(tree.pair_label[pairs])
            parents = np.unique(np.array([p for p in (segs[s][2] for s in ids) if p >= 0 and seg_chunk[p] < c],
                                         dtype=np.int64))
            exts.append({"kv": ext_kv, "x": ext_x, "s": parents[np.argsort(order["s"][parents])]})
            self.chunks.append(ch)

        # kept: what any chunk reads from an earlier one, one slot each, in slot order
        kept = {}
        for kind in ("kv", "x", "s"):
            items = np.unique(np.concatenate([np.empty(0, dtype=np.int64)] + [e[kind] for e in exts]))
            kept[kind] = items[np.argsort(order[kind][items])]
        owner = {"kv": tok_chunk[kept["kv"]], "x": tok_chunk[kept["x"]], "s": seg_chunk[kept["s"]]}
        for c, (ch, ext) in enumerate(zip(self.chunks, exts)):
            ch.ext_segs = ext["s"].tolist()
            ch.ext = {kind: _runs(np.searchsorted(order[kind][kept[kind]], order[kind][e])) for kind, e in ext.items()}
            ch.own = {}
            for kind in kept:
                lo, hi = (int(i) for i in np.searchsorted(owner[kind], [c, c + 1]))
                mine = kept[kind][lo:hi]
                ch.own[kind] = (mine.tolist() if kind == "s" else as_tensor(row[mine]), lo, hi)
        self.n_kept = {key: len(kept[kind]) for key, kind in KINDS.items()}
        self.n_pairs = len(tree.pair_pos)
        flat_pair = tree.flat_pair.copy()  # flat position -> pair row; -1 (no next token) -> an appended zero row
        flat_pair[flat_pair < 0] = self.n_pairs
        self.flat_pair = torch.from_numpy(flat_pair).to(device)


def _runs(slots: np.ndarray) -> list[tuple[int, int]]:
    """Ascending slots as runs [a, b) of consecutive ones."""
    cut = np.flatnonzero(np.diff(slots) != 1) + 1
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(slots, cut)] if len(slots) else []


def _load(store: torch.Tensor, runs: list, device) -> torch.Tensor:
    """The rows of `store` (kept slots, in host or device memory) in runs, back to back on the device."""
    out = torch.empty((sum(b - a for a, b in runs), *store.shape[1:]), dtype=store.dtype, device=device)
    i = 0
    for a, b in runs:
        out[i:i + b - a].copy_(store[a:b], non_blocking=True)
        i += b - a
    return out


def _save(store: torch.Tensor, runs: list, t: torch.Tensor) -> None:
    i = 0
    for a, b in runs:
        store[a:b].copy_(t[i:i + b - a], non_blocking=True)
        i += b - a


_host = PinnedSlab()  # the stores of a micro-batch on a GPU (HOST)
# a micro-batch's kept tensors wait in the slab for its backward (chunk_major_output with grad). A backward that raises
# or never runs leaves it set, and every later forward fails loudly, until release_host (the actor's call ends)
_pending = False


def _store(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Slots for kept tensors (or their sums) computed on `device`: on a GPU (HOST) in the pinned slab, else on the
    device."""
    if HOST and device.type == "cuda":
        return _host.take(shape, dtype)
    return torch.empty(shape, dtype=dtype, device=device)


def _start() -> None:
    """A micro-batch's stores are carved from the slab's start, so no other micro-batch's may wait there still."""
    assert not _pending, ("chunk-major: a micro-batch's forward ran before the previous one's backward, which reads "
                          "the kept tensors this one would overwrite in the slab")
    if HOST and torch.cuda.is_available():
        _host.start()


def release_host(slab: bool = True) -> None:
    """End of an actor call (compute_logp, ppo_update), also one that raised: no micro-batch waits for its backward any
    more; with `slab` the slab goes back to the OS (ppo_update, as lm_head.release_host_activations). compute_logp keeps
    it for the ppo_update that follows, whose micro-batches carve more of it: unpinning and pinning it again in between
    took 4.5-6.4 s per step at 8.7-12.2 GiB slabs, ~0.48 s per GiB (results/chunk-major/README.md "Use in training")."""
    global _pending
    _pending = False
    if slab:
        _host.release()


def _topological(nodes: list, node_of: np.ndarray) -> list[int]:
    """Node order: each after its parent and the nodes holding its mounted keys, the lowest pre-order index first (a
    tree without mounts: pre-order)."""
    deps = [set(node_of[n.mounted].tolist()) | ({n.parent} if n.parent >= 0 else set()) for n in nodes]
    users: list[list[int]] = [[] for _ in nodes]
    for k, d in enumerate(deps):
        for j in d:
            users[j].append(k)
    waiting = [len(d) for d in deps]
    ready = [k for k, w in enumerate(waiting) if not w]  # ascending: a heap
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        for u in users[k]:
            waiting[u] -= 1
            if not waiting[u]:
                heapq.heappush(ready, u)
    assert len(order) == len(nodes), "mounts form a cycle"
    return order


_ANCHOR = torch.zeros((), requires_grad=True)  # an input that needs a gradient: an op's output gets one (never set)


class _Kept(torch.autograd.Function):
    """Kept tensors of earlier chunks, loaded onto the device by load(); the backward hands their gradient to add()."""

    @staticmethod
    def forward(ctx, load, add, _anchor):
        ctx.add = add
        return load()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        ctx.add(grad)
        return None, None, None


class _IO:
    """A chunk's link to earlier chunks in one decoder layer (seq_tiles.tree_layer_forward's io): the kept k, v
    (attention) or layer-input rows (convolution) of earlier chunks, loaded by the functions in `ext` (each call loads
    them again: the layer's recompute in backward calls them too), after the chunk's own k, v or the chunk rows its
    convolution contexts read (ctx); and their final states its segments start from (finals, by segment)."""

    def __init__(self, ext=(), finals=None, ctx=None):
        self.ext, self.finals, self.ctx, self.own = ext, finals or {}, ctx, None

    def keys(self, k, v):
        self.own = k, v
        return tuple(torch.cat([t, e()[None]], 1) for t, e in zip((k, v), self.ext)) if self.ext else (k, v)

    def rows(self, x):
        rows = x.index_select(1, self.ctx)
        return torch.cat([rows, self.ext[0]()[None]], 1) if self.ext else rows


def _pass(model, head, ch: Chunk, kept: dict, sums: dict | None = None):
    """Chunk ch through the model: (head output of its label pairs, [(key, (lo, hi), tensor)] of what it keeps for later
    chunks, its slots lo:hi). It reads the earlier chunks' tensors from kept[(layer, kind)]; with `sums` (backward)
    their gradients are added to sums[(layer, kind)]."""
    lm = model.get_decoder()
    x = lm.embed_tokens(ch.ids)

    def ext(key, runs):  # the function that loads the kept rows of key in runs, through _Kept
        def load():
            return _load(kept[key], runs, x.device)

        def add(grad):
            acc = _load(sums[key], runs, grad.device)
            acc += grad
            _save(sums[key], runs, acc)
        return lambda: _Kept.apply(load, add, _ANCHOR)

    pe = lm.rotary_emb(x, ch.pos)
    own = []  # (key, (lo, hi), tensor)
    for i, layer in enumerate(lm.layers):
        if layer.block_type == "full_attention":
            io = _IO(tuple(ext((i, kind), ch.ext["kv"]) for kind in "kv") if ch.ext["kv"] else ())
        else:
            states = ext((i, "s"), ch.ext["s"])() if ch.ext["s"] else None
            io = _IO((ext((i, "x"), ch.ext["x"]),) if ch.ext["x"] else (),
                     None if states is None else dict(zip(ch.ext_segs, states)), ch.ctx_rows)
        y = layer(x, position_embeddings=pe, octopus_tree=ch, octopus_io=io)
        if layer.block_type == "full_attention":
            rows, lo, hi = ch.own["kv"]
            own += [((i, kind), (lo, hi), t[0].index_select(0, rows)) for kind, t in zip("kv", io.own)]
        else:
            rows, lo, hi = ch.own["x"]
            own.append(((i, "x"), (lo, hi), x[0].index_select(0, rows)))
            segs, lo, hi = ch.own["s"]
            if segs:
                own.append(((i, "s"), (lo, hi), torch.stack([io.finals[s] for s in segs])))
        x = y
    out = head(lm.norm(x)[0].index_select(0, ch.pair_at), ch.labels)
    return out, [o for o in own if o[1][1] > o[1][0]]


@torch.no_grad()
def run_forward(plan: ChunkPlan, model, head) -> tuple[torch.Tensor, dict]:
    """head(hidden rows [n, H], labels [n]) -> [n, C] per label pair. Returns the [1, T_flat, C] head output per flat
    position (as prefix_tree.tree_head_output; 0 where no token follows) and the kept tensors run_backward reads."""
    kept, pairs = {}, None
    _start()
    with _gathered(model):
        for ch in plan.chunks:
            out, own = _pass(model, head, ch, kept)
            for key, (lo, hi), t in own:
                if key not in kept:
                    kept[key] = _store((plan.n_kept[key[1]], *t.shape[1:]), t.dtype, t.device)
                kept[key][lo:hi].copy_(t, non_blocking=True)
            if pairs is None:
                pairs = out.new_zeros(plan.n_pairs + 1, out.shape[1])
            pairs[ch.pair_rows] = out
    return pairs.index_select(0, plan.flat_pair)[None], kept


def run_backward(plan: ChunkPlan, model, head, kept: dict, grad: torch.Tensor) -> None:
    """Add to the parameters' .grad the gradient of the run_forward output's dot product with grad [1, T_flat, C]."""
    grad_pairs = grad.new_zeros(plan.n_pairs + 1, grad.shape[-1]).index_add_(0, plan.flat_pair, grad[0])
    if grad.is_cuda:
        torch.cuda.current_stream().synchronize()  # nothing queued still reads the slab the sums are zeroed in
    # key -> gradient of kept[key] from the chunks run so far (the later ones), at least ACC_DTYPE
    sums = {key: _store(t.shape, torch.promote_types(t.dtype, ACC_DTYPE), grad.device).zero_()
            for key, t in kept.items()}
    # only each layer's own checkpoint stays (seq_tiles.no_inner), also for its recompute inside autograd.backward: a
    # layer's backward holds its activations over the chunk and the keys and values each of the chunk's attention calls
    # gathered, 4 KB per key (their sum over the calls follows the tree's paths, not TOKENS: 1.7-3.9 GiB at most in a
    # 32k chunk of the 256k-994k units; results/chunk-major/README.md)
    with _gathered(model, backward=True) as idle, seq_tiles.no_inner():
        for ch in reversed(plan.chunks):
            with torch.enable_grad():
                out, own = _pass(model, head, ch, kept, sums)
                own = [o for o in own if o[2].requires_grad]  # a frozen embedding: layer 0's input has no gradient
                torch.autograd.backward([out] + [t for _, _, t in own], [grad_pairs.index_select(0, ch.pair_rows)] + [
                    sums[key][lo:hi].to(t.device, non_blocking=True).to(t.dtype) for key, (lo, hi), t in own])
            idle()


class _ChunkMajor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, model, head, _anchor):
        global _pending
        out, ctx.kept = run_forward(plan, model, head)
        ctx.plan, ctx.model, ctx.head = plan, model, head
        _pending = True
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        global _pending
        run_backward(ctx.plan, ctx.model, ctx.head, ctx.kept, grad)
        _pending = False
        del ctx.kept
        return None, None, None, None




def chunk_major_output(plan: ChunkPlan, model, head) -> torch.Tensor:
    """run_forward's [1, T_flat, C] output, with run_backward as its backward when grad is enabled. Call it inside the
    forward of `model`, the FSDP root when sharded (module docstring)."""
    if not torch.is_grad_enabled():
        return run_forward(plan, model, head)[0]
    if not (model.training and getattr(model, "is_gradient_checkpointing", False)):  # as lm_head.check_checkpointing
        raise RuntimeError("chunk-major's backward needs HF layer checkpointing (run_backward): without it every layer "
                           "would keep its activations over a whole chunk")
    return _ChunkMajor.apply(plan, model, head, _ANCHOR)


_STATE_ATTRS = ("_training_state", "_fsdp_param_group", "_state_ctx")
_GROUP_ATTRS = ("_training_state", "_post_forward_indices", "post_forward_mesh_info", "reshard_after_backward",
                "reduce_grads", "fsdp_params")
_PARAM_ATTRS = ("unsharded_accumulated_grad",)


def _fsdp_states(model) -> list:
    """The FSDP2 states of `model` (the root, first) and of the fully_shard modules below it; none when unsharded."""
    from torch.distributed.fsdp import FSDPModule
    sharded = [m for m in model.modules() if isinstance(m, FSDPModule)]
    if not sharded:
        return []
    states = [m._get_fsdp_state() for m in sharded]
    assert sharded[0] is model and states[0]._is_root, "chunk-major on a sharded model: pass its FSDP root"
    for s in states:
        if not all(hasattr(s, a) for a in _STATE_ATTRS) or not (
                s._fsdp_param_group is None or (all(hasattr(s._fsdp_param_group, a) for a in _GROUP_ATTRS) and all(
                    hasattr(p, a) for p in s._fsdp_param_group.fsdp_params for a in _PARAM_ATTRS))):
            raise RuntimeError(f"torch {torch.__version__}: FSDP2 internals differ from torch 2.9's; check "
                               "octopus/rl/chunk_major.py _gathered and across_micro_batches against this torch")
    return states


@contextlib.contextmanager
def _gathered(model, backward: bool = False):
    """FSDP2 around a chunk loop (module docstring; nothing without FSDP): the modules below the root keep their
    parameters gathered after forward and backward and, with `backward`, do not reduce their gradients; yields the
    function that sets them idle after a chunk's backward. Afterwards their settings are restored and, after a forward
    loop, their parameters resharded (after the backward, FSDP's final callback reshards and reduces): as
    across_micro_batches set them, within an actor call not before its end."""
    states = _fsdp_states(model)
    if backward and states:
        assert states[0]._state_ctx.post_backward_final_callback_queued, \
            "chunk-major backward without FSDP's final callback queued: call chunk_major_output in the root's forward"
    groups = [s._fsdp_param_group for s in states[1:] if s._fsdp_param_group is not None]
    saved = [(g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads) for g in groups]
    for g in groups:
        g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads = None, False, False
    idle = functools.partial(_idle, states[1:])
    try:
        yield idle
    finally:
        idle()
        for g, (mesh, after_backward, reduce) in zip(groups, saved):
            g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads = mesh, after_backward, reduce
            if not backward and mesh is not None:
                g.reshard()


@contextlib.contextmanager
def across_micro_batches(model, n: int):
    """FSDP2 across an actor call's n micro-batches (module docstring; nothing without FSDP), each one forward of
    `model`, the FSDP root, and in training one backward: the modules keep their parameters gathered and, but in the
    n-th micro-batch's backward, do not reduce or reshard. Afterwards their settings are restored and the parameters
    resharded; a call that raised also drops the gradients it left unreduced, which the next call's reduce-scatter
    would add to its own."""
    groups, saved, count = [], [], 0

    def pre_forward(module, args):
        nonlocal count
        if not count:  # FSDP's root pre-forward hook, which runs first, has initialized the states
            groups[:] = [s._fsdp_param_group for s in _fsdp_states(model) if s._fsdp_param_group is not None]
            saved[:] = [(g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads) for g in groups]
            for g in groups:
                g.post_forward_mesh_info = None
        count += 1
        assert count <= n, f"micro-batch forward {count} of {n}"
        for g, (_, after_backward, reduce) in zip(groups, saved):
            g.reshard_after_backward, g.reduce_grads = (after_backward, reduce) if count == n else (False, False)

    handle = model.register_forward_pre_hook(pre_forward)
    try:
        yield
        assert count == n, f"{count} micro-batch forwards of {n}: the gradients of the last were never reduced"
    except BaseException:
        for p in (p for g in groups for p in g.fsdp_params):
            p.unsharded_accumulated_grad = None
            if hasattr(p, "_unsharded_param"):
                p._unsharded_param.grad = None
        raise
    finally:
        handle.remove()
        for g, (mesh, after_backward, reduce) in zip(groups, saved):
            g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads = mesh, after_backward, reduce
            g.reshard()


def _idle(states: list) -> None:
    """FSDP2 modules back to idle: not in pre-backward (their next forward runs its hooks), no backward prefetch."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    for s in states:
        s._training_state = TrainingState.IDLE
        if (g := s._fsdp_param_group) is not None:
            g._training_state = TrainingState.IDLE
            g._post_forward_indices.clear()
