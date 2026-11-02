import numpy as np
import pytest
import torch

from octopus.rl import prefix_tree
from octopus.rl.prefix_tree import TreeMeta, build_tree, tree_conv, tree_head_output


def _flat(seqs):
    return np.concatenate(seqs), [0, *np.cumsum([len(s) for s in seqs]).tolist()]


def torch_segment_conv(x, weight, bias, activation, cu, cu_cpu, ctx=None, dest=None):
    """prefix_tree.segment_conv in plain torch (fla's Triton kernel runs on CUDA only): each segment of x [1, T, C]
    convolved after the last W-1 slots of its initial state (ctx at rows dest; zero elsewhere)."""
    w, c = weight.shape[-1], x.shape[-1]
    bounds = (cu if cu_cpu is None else cu_cpu).tolist()
    init = x.new_zeros((len(bounds) - 1) * w, c)
    if ctx is not None:
        init = init.index_copy(0, dest, ctx)
    init = init.view(-1, w, c)
    y = torch.cat([torch.nn.functional.conv1d(torch.cat([init[i, 1:], x[0, a:b]]).t()[None], weight[:, None], bias,
                                              groups=c)[0].t()
                   for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))])
    return (torch.nn.functional.silu(y) if activation in ("silu", "swish") else y)[None]


def test_tree_dedupes_shared_prefixes_and_keeps_every_sequence_exact():
    rng = np.random.default_rng(0)
    prefix = rng.integers(0, 1000, 50)
    seqs = [np.concatenate([prefix, rng.integers(0, 1000, n)]) for n in (7, 3, 12)]
    seqs.append(np.concatenate([prefix[:20], rng.integers(0, 1000, 5)]))  # branches inside the prefix
    seqs.append(seqs[1].copy())  # a duplicate
    seqs.append(rng.integers(1000, 2000, 9))  # shares nothing
    flat, cu = _flat(seqs)
    tree = build_tree(flat, cu)
    assert len(tree.tokens) == 50 + 7 + 3 + 12 + 5 + 9
    for node in tree.nodes:  # pre-order ranges tile the tree; a node's path ends with its own tokens
        assert (node.path[-(node.end - node.start):] == np.arange(node.start, node.end)).all()
    starts = [n.start for n in tree.nodes]
    assert starts == sorted(starts) and tree.nodes[-1].end == len(tree.tokens)
    for s, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        for p in range(a, b):
            row = tree.flat_pair[p]
            if p == b - 1:
                assert row == -1  # nothing follows a sequence's last token
                continue
            pos, label = tree.pair_pos[row], tree.pair_label[row]
            assert tree.tokens[pos] == flat[p] and label == flat[p + 1]
            assert tree.position_ids[pos] == p - a  # its position on its own path


@pytest.mark.parametrize("seg", [32768, 3])  # 3: nodes split into segments, each with its own context
def test_tree_conv_sees_its_ancestor_tail(monkeypatch, seg):
    monkeypatch.setattr(prefix_tree, "CONV_CONTEXT", 3)  # the kernel 4 below
    monkeypatch.setattr(prefix_tree, "SEG", seg)
    monkeypatch.setattr(prefix_tree, "segment_conv", torch_segment_conv)  # the layout of contexts, on the CPU
    flat, cu = _flat([np.arange(10), np.r_[np.arange(6), 50, 51]])  # branch after 6 shared tokens
    tree = build_tree(flat, cu)
    meta = TreeMeta(tree, "cpu")
    x = torch.from_numpy(tree.tokens.astype(np.float32))[None, None]  # one channel = the token ids

    def window_sum(seg):  # causal conv with kernel 4: sum of the last 4 inputs
        return torch.nn.functional.conv1d(torch.nn.functional.pad(seg, (3, 0)), torch.ones(1, 1, 4))

    out = tree_conv(x, torch.ones(1, 4), None, None, meta)[0, 0]
    for a, b in zip(cu[:-1], cu[1:]):  # every sequence's own causal conv, read back at its tree positions
        own = window_sum(torch.from_numpy(flat[a:b].astype(np.float32))[None, None])[0, 0]
        positions = [tree.pair_pos[tree.flat_pair[p]] for p in range(a, b - 1)]
        assert torch.equal(out[positions], own[:-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla's Triton kernel needs CUDA")
def test_segment_conv_on_fla_is_the_torch_convolution_forward_and_backward(monkeypatch):
    """Every level of a tree whose nodes split into segments, some with fewer context tokens than the kernel's (near the
    roots), and tree_conv over the whole tree: fla against torch in fp64 on the same bf16-valued inputs (output, d x,
    d ctx), within 1e-5 in fp32, and in bf16 no further from it than torch's bf16 convolution."""
    fla_conv = prefix_tree.segment_conv
    monkeypatch.setattr(prefix_tree, "CONV_CONTEXT", 3)
    monkeypatch.setattr(prefix_tree, "SEG", 64)
    rng = np.random.default_rng(5)
    trunk = rng.integers(0, 1000, 300)
    seqs = [np.r_[trunk[:cut], rng.integers(1000, 2000, n)]
            for cut, n in ((300, 90), (1, 200), (2, 70), (150, 5), (150, 260), (0, 40))]
    meta = TreeMeta(build_tree(*_flat(seqs)), "cuda")
    g = torch.Generator("cuda").manual_seed(0)
    c = 256

    def bf16_valued(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16().double()

    weight = bf16_valued(c, 4, scale=0.5)
    worst = {}
    for level in [*meta.levels, None]:  # None: tree_conv over the whole tree, its contexts gathered from x
        t, n = (sum(level.lens), len(level.ctx_index)) if level else (meta.n_tokens, 0)
        x0, ctx0, dy = bf16_valued(1, t, c), bf16_valued(n, c), bf16_valued(1, t, c)
        runs = {}
        for name, conv, dtype in (("ref", torch_segment_conv, torch.float64),
                                  ("torch", torch_segment_conv, torch.bfloat16),
                                  ("fla", fla_conv, torch.bfloat16), ("fla32", fla_conv, torch.float32)):
            x, ctx = (v.to(dtype, copy=True).requires_grad_() for v in (x0, ctx0))
            if level is None:
                monkeypatch.setattr(prefix_tree, "segment_conv", conv)
                y = tree_conv(x.transpose(1, 2), weight.to(dtype), None, "silu", meta).transpose(1, 2)
            else:
                y = conv(x, weight.to(dtype), None, "silu", level.cu, level.cu_cpu, ctx if n else None, level.ctx_dest)
            y.backward(dy.to(dtype))
            runs[name] = (y.detach(), x.grad, ctx.grad if n else None)
        for what, i in (("y", 0), ("dx", 1), ("dctx", 2)):
            for name in ("torch", "fla", "fla32") if runs["ref"][i] is not None else ():
                err = (runs[name][i].double() - runs["ref"][i]).abs().max().item()
                worst[name, what] = max(worst.get((name, what), 0.0), err)
    print("max |diff| against fp64 torch:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert all(worst["fla32", w] <= 1e-5 for w in ("y", "dx", "dctx")), worst
    assert all(worst["fla", w] <= worst["torch", w] for w in ("y", "dx", "dctx")), worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla's Triton kernel needs CUDA")
@pytest.mark.parametrize("bias", [False, True])  # GDN's convolution has none, Mamba-2's has one
def test_warm_conv_leaves_no_autotuning_for_training_calls(monkeypatch, bias):
    """After warm_conv, the calls training makes (a level with contexts, a level of roots or a flat micro-batch: no
    initial states; any length up to SEG) find every autotune key (Triton's: D, W, NB and each tensor argument's dtype)
    already tuned."""
    from fla.modules.conv.triton import kernels

    monkeypatch.setattr(prefix_tree, "SEG", 4096)
    monkeypatch.setattr(prefix_tree, "CONV_CONTEXT", 3)
    c, dtype = 96 + 32 * bias, torch.bfloat16  # widths no other test tuned
    prefix_tree.warm_conv(c, 4, bias, dtype, "cuda")
    tuners = [kernels.causal_conv1d_fwd_kernel.fn, kernels.causal_conv1d_bwd_kernel.fn]
    tuned = [set(t.cache) for t in tuners]
    weight = torch.randn(c, 4, dtype=dtype, device="cuda")
    b = torch.randn(c, dtype=dtype, device="cuda") if bias else None
    for lens in ([1500], [700, 2100, 900], [4096]):
        cu = torch.tensor(np.cumsum([0] + lens))
        x = torch.randn(1, sum(lens), c, dtype=dtype, device="cuda", requires_grad=True)
        ctx = torch.randn(3 * len(lens), c, dtype=dtype, device="cuda", requires_grad=True)
        for context in ((ctx, prefix_tree.conv_dest([3] * len(lens), "cuda")), (None, None)):
            y = prefix_tree.segment_conv(x, weight, b, "silu", cu.cuda(), cu, *context)
            y.backward(torch.randn_like(y))
    assert [set(t.cache) for t in tuners] == tuned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla's Triton kernel needs CUDA")
def test_segment_conv_takes_strided_and_expanded_output_gradients():
    """fla's backward reads the output's gradient as contiguous: an expanded (sum()) or transposed one gives the
    gradients of a dense copy."""
    g = torch.Generator("cuda").manual_seed(1)
    c, cu = 64, torch.tensor([0, 300, 700])
    weight = torch.randn(c, 4, device="cuda", generator=g)
    x0 = torch.randn(1, 700, c, device="cuda", generator=g)
    ctx0 = torch.randn(6, c, device="cuda", generator=g)
    dest = torch.tensor([1, 2, 3, 5, 6, 7], device="cuda")
    dense = torch.randn(1, c, 700, device="cuda", generator=g)
    for dy in (torch.ones(1, 1, 1, device="cuda").expand(1, 700, c), dense.transpose(1, 2)):
        grads = []
        for conv in (prefix_tree.segment_conv, torch_segment_conv):
            x, ctx = (v.clone().requires_grad_() for v in (x0, ctx0))
            conv(x, weight, None, "silu", cu.cuda(), cu, ctx, dest).backward(dy)
            grads.append((x.grad, ctx.grad))
        for a, b in zip(*grads):
            assert (a - b).abs().max() <= 1e-5


def test_tree_head_output_expands_pairs_to_flat_positions():
    flat, cu = _flat([np.array([1, 2, 3, 4]), np.array([1, 2, 9])])
    meta = TreeMeta(build_tree(flat, cu), "cpu")
    hidden = torch.arange(len(meta.pair_pos) * 0 + 5, dtype=torch.float32)[None, :, None]  # tree tokens 1 2 3 4 9

    def head(rows, labels):  # logprob := hidden value * 10 + label, entropy := label
        return torch.stack([rows[:, 0] * 10 + labels, labels.float()], -1)

    out = tree_head_output(head, hidden, meta)[0]
    # flat position 1 ("2" in both sequences) has a different next token per sequence: 3 vs 9
    assert out[1, 1] == 3 and out[5, 1] == 9 and out[1, 0] == 1 * 10 + 3 and out[5, 0] == 1 * 10 + 9
    assert (out[[3, 6]] == 0).all()  # last positions


def test_tree_scores_only_the_pairs_of_scored_positions():
    flat, cu = _flat([np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5]), np.array([1, 2, 9])])
    scored = np.zeros(len(flat), dtype=bool)
    scored[[1, 9]] = True  # "2" -> 3 in the first sequence (the second's position 5 too); "2" -> 9 in the third
    tree, full = build_tree(flat, cu, scored=scored), build_tree(flat, cu)
    assert len(tree.pair_pos) == 2 and (tree.flat_pair[~scored] == -1).all()  # position 5 too: its output is 0
    for p in (1, 9):  # the same (tree position, label) as without scoring
        assert (tree.pair_pos[tree.flat_pair[p]], tree.pair_label[tree.flat_pair[p]]) == (
            full.pair_pos[full.flat_pair[p]], full.pair_label[full.flat_pair[p]])
    assert full.flat_pair[1] == full.flat_pair[5]


def test_allocation_keeps_groups_together_and_counts_tree_tokens():
    from octopus.rl.prefix_tree import allocate_prefix_groups, tree_tokens
    rng = np.random.default_rng(1)
    groups = []
    system = rng.integers(0, 1000, 300)  # every task starts with the same system prompt
    for g in range(3):  # 3 GRPO groups of 8: a 2000-token shared start + 200-500 own tokens each
        start = np.concatenate([system, rng.integers(0, 1000, 1700)])
        groups.append([np.concatenate([start, rng.integers(0, 1000, rng.integers(200, 500))]) for _ in range(8)])
    seqs = [s for g in groups for s in g]
    mbs = allocate_prefix_groups(seqs, cap=6000)
    assert sorted(i for m in mbs for i in m) == list(range(24))
    for m in mbs:  # every micro-batch fits the cap in tree tokens, though not in flat tokens
        tree = build_tree(np.concatenate([seqs[i] for i in m]), [0, *np.cumsum([len(seqs[i]) for i in m]).tolist()])
        assert len(tree.tokens) <= 6000
    assert len(mbs) == 3 and all(len({i // 8 for i in m}) == 1 for m in mbs)  # one whole group each
    sizes = [tree_tokens([seqs[i] for i in m]) for m in allocate_prefix_groups(seqs, cap=2600)]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]  # largest first: ranks' full ones in lockstep
    assert len(allocate_prefix_groups(seqs, cap=2600)) > 3  # a group too big for one micro-batch is split
    assert len(allocate_prefix_groups(seqs, cap=6000, min_groups=5)) == 5


def test_mount_splits_nodes_gathers_mounted_keys_and_continues_at_k2_positions():
    ctx, cont = np.arange(10), np.arange(100, 106)
    a = np.r_[np.arange(4), np.arange(200, 209)]  # sibling A: shares 4 tokens with B's context, 13 tokens
    seqs = [np.r_[ctx, cont], np.r_[ctx, cont], a, np.r_[ctx, cont, 300, 301]]  # B, C (= B, no mount), A, D (B + 2)
    mount = (10, 2, 8, 13)  # from B's token 10 on, attend to A's tokens 8..12 (positions 8..12, A ahead of B)
    flat, cu = _flat(seqs)
    tree = build_tree(flat, cu, {0: [mount], 3: [mount]})
    assert len(tree.tokens) == 4 + 6 + 6 + 2 + 6 + 9  # B and D share the mounted node, C does not

    def node_at(seq, p):  # the node holding flat position p of sequence seq
        t = tree.pair_pos[tree.flat_pair[cu[seq] + p]]
        return next(k for k, n in enumerate(tree.nodes) if n.start <= t < n.end)

    b, c, d = node_at(0, 10), node_at(1, 10), node_at(3, 16)
    a_tree = tree.nodes[node_at(2, 11)].path  # A's tokens in tree order
    assert tree.nodes[b].start == tree.pair_pos[tree.flat_pair[cu[0] + 10]]  # the mount point starts a node
    assert tree.nodes[b].parent == tree.nodes[c].parent == node_at(0, 9)  # GDN/conv: same parent chain as C
    assert (tree.nodes[b].mounted == a_tree[8:13]).all() and (tree.tokens[tree.nodes[b].mounted] == a[8:13]).all()
    assert (tree.nodes[d].mounted == a_tree[8:13]).all() and tree.nodes[d].parent == b  # inherited by descendants
    assert all(not len(n.mounted) for k, n in enumerate(tree.nodes) if k not in (b, d))

    # attention keys per node: mounted segment first, then the path (own tokens last)
    meta = TreeMeta(tree, "cpu")
    (_, keys, cu_q, cu_k, _, _), = meta.attn_calls
    per_node = np.split(keys.numpy(), cu_k[1:-1].numpy())
    for k, n in enumerate(tree.nodes):
        assert (per_node[k] == np.r_[n.mounted, n.path]).all()
    assert (cu_q.diff().numpy() == [n.end - n.start for n in tree.nodes]).all()
    assert meta.own_calls is meta.attn_calls  # OCTOPUS_MOUNT_LAYERS unset: every attention layer reads the mount
    prefix_tree.MOUNT_LAYERS = frozenset({3})
    try:  # a subset: the other layers' keys are each node's path alone
        (_, keys, _, cu_k, _, _), = TreeMeta(tree, "cpu").own_calls
    finally:
        prefix_tree.MOUNT_LAYERS = None
    per_node = np.split(keys.numpy(), cu_k[1:-1].numpy())
    assert all((per_node[k] == n.path).all() for k, n in enumerate(tree.nodes))

    # positions: K2 continues B at max(10, 12 + 1) = 13; mounted tokens keep A's positions; C is unshifted
    pos = lambda seq, p: tree.position_ids[tree.pair_pos[tree.flat_pair[cu[seq] + p]]]  # noqa: E731
    assert [pos(0, p) for p in range(15)] == list(range(10)) + list(range(13, 18))
    assert [pos(3, p) for p in range(17)] == list(range(10)) + list(range(13, 20))
    assert [pos(1, p) for p in range(15)] == list(range(15))
    assert (tree.position_ids[a_tree] == np.arange(13)).all()
    # A behind B: the mount keeps B's own positions
    behind = build_tree(flat, cu, {0: [(10, 2, 4, 8)]})
    b_pos = [behind.position_ids[behind.pair_pos[behind.flat_pair[cu[0] + p]]] for p in range(15)]
    assert b_pos == list(range(15))


def test_two_segments_and_a_transitive_mount():
    """B mounts two segments at one point: both sets of keys, and K2 after the later of them. C mounts a range of A
    that holds A's own read of X: its keys are A's tokens in the range and X's segment that A mounted there (on the
    server, A's placeholders inside the range), and C continues after A's last mounted token. A read point at the
    range's start or end is outside it (the server's placeholders sit before the token at that index)."""
    x, a = np.arange(500, 520), np.arange(200, 230)
    b, c = np.r_[np.arange(10), np.arange(100, 106)], np.r_[np.arange(10), np.arange(300, 306)]
    flat, cu = _flat([x, a, b, c])
    mounts = {1: [(12, 0, 5, 15)], 2: [(10, 1, 2, 8), (10, 0, 10, 18)], 3: [(10, 1, 8, 20)]}
    tree = build_tree(flat, cu, mounts)

    def node_of(t, seq, p):
        return next(n for n in t.nodes if n.start <= t.pair_pos[t.flat_pair[cu[seq] + p]] < n.end)
    x_tree, a_tree = node_of(tree, 0, 18).path, node_of(tree, 1, 28).path
    assert (node_of(tree, 1, 12).mounted == x_tree[5:15]).all()
    assert (node_of(tree, 2, 10).mounted == np.r_[a_tree[2:8], x_tree[10:18]]).all()
    assert (node_of(tree, 3, 10).mounted == np.r_[a_tree[8:20], x_tree[5:15]]).all()
    assert (tree.tokens[node_of(tree, 3, 15).mounted] == np.r_[a[8:20], x[5:15]]).all()
    pos = lambda t, seq, p: int(t.position_ids[t.pair_pos[t.flat_pair[cu[seq] + p]]])  # noqa: E731
    assert [pos(tree, 1, p) for p in (11, 12, 28)] == [11, 15, 31]  # A after X's last mounted key 14
    assert pos(tree, 2, 10) == 18  # max(10, A's 7 + 1, X's 17 + 1)
    assert pos(tree, 3, 10) == pos(tree, 1, 19) + 1 == 23  # after A's token 19 (at 22), not after X's
    for rng in ((12, 20), (2, 12)):  # A's read at the range's start or end: A's tokens only
        t = build_tree(flat, cu, {1: mounts[1], 3: [(10, 1, *rng)]})
        assert (node_of(t, 3, 10).mounted == node_of(t, 1, 28).path[slice(*rng)]).all()


# Reads through the export (docs/cross-branch-read.md "Export") ---------------------------------------------------------
def _episode():
    """Concat interactions as the proxy keeps them: r0 forks A and B (C, E are more threads from r0). A's a2 is the source
    of two reads and has two leaves below it (a3, a3b); B's b2 reads A's a1..a2 and b3 continues after it (the reader
    leaf); C's c1 reads the same range; E's e1 reads B's b1..b3, which holds B's own read (a source that is itself a
    reader, read transitively)."""
    from types import SimpleNamespace

    rng, n = np.random.default_rng(3), iter(range(1, 100))

    def x(parent, delta, out, mount=None):
        inp = (parent.model_response.input_tokens + parent.model_response.output_tokens if parent else []) + delta
        return SimpleNamespace(interaction_id=f"chatcmpl-{next(n):029x}", parent=parent, metadata={"mount": mount} if mount
                               else {}, model_response=SimpleNamespace(input_tokens=inp, output_tokens=out))
    tok = lambda k: rng.integers(10, 1000, k).tolist()  # noqa: E731
    r0 = x(None, tok(30), tok(5))
    a1 = x(r0, tok(6), tok(7))
    a2 = x(a1, tok(4), tok(5))
    a3, a3b = x(a2, tok(3), tok(4)), x(a2, tok(5), tok(2))
    end = lambda r: len(r.model_response.input_tokens) + len(r.model_response.output_tokens)  # noqa: E731
    rng_a = {"source": a2.interaction_id, "start": end(r0), "end": len(a2.model_response.input_tokens)}
    b1 = x(r0, tok(3), tok(4))
    b2 = x(b1, tok(2), tok(6), {**rng_a, "at": end(b1)})
    b3 = x(b2, tok(5), tok(3))
    c1 = x(r0, tok(9), tok(6), {**rng_a, "at": end(r0)})
    e1 = x(r0, tok(4), tok(5), {"source": b3.interaction_id, "start": end(r0), "end": len(b3.model_response.input_tokens),
                                "at": end(r0)})
    return {v.interaction_id: v for v in (a3, a3b, b3, c1, e1)}, (r0, a2, b2, e1)


def _batch(leaves):
    """The batch the actor splits: each leaf's tokens with its read_tensors, padded to [B, L]."""
    import torch

    from octopus.rl.prefix_tree import MOUNT_KEYS, read_tensors
    rows = []
    for lid, leaf in leaves.items():
        ids = leaf.model_response.input_tokens + leaf.model_response.output_tokens
        rows.append({"input_ids": torch.tensor([ids]),
                     **read_tensors(lid, leaf.metadata.get("octopus_reads", ()), (1, len(ids)))})
    width = max(r["input_ids"].shape[1] for r in rows)
    pad = lambda t: torch.nn.functional.pad(t, (0, width - t.shape[1]))  # noqa: E731
    batch = {k: torch.cat([pad(r[k]) for r in rows]) for k in ("input_ids", *MOUNT_KEYS)}
    batch["attention_mask"] = torch.cat([pad(torch.ones_like(r["input_ids"])) for r in rows])
    return batch


def _mounts(batch, rows=None):
    """actor.packed_mounts on the rows (all by default), as _prepare_mb_list and the packed forward see them."""
    from octopus.rl.prefix_tree import MOUNT_KEYS, seq_mounts
    rows = list(range(len(batch["input_ids"]))) if rows is None else rows
    mask = batch["attention_mask"][rows].bool()
    flat = {k: batch[k][rows][mask].numpy() for k in ("input_ids", *MOUNT_KEYS)}
    cu = np.cumsum([0] + mask.sum(1).tolist()).tolist()
    return flat["input_ids"], cu, seq_mounts(cu, *(flat[k] for k in MOUNT_KEYS))


def test_export_reads_become_tree_mounts_on_the_right_sequences():
    from octopus.rl.prefix_tree import record_reads
    leaves, (r0, a2, b2, e1) = _episode()
    record_reads(leaves)
    lid = list(leaves)  # a3, a3b, b3, c1, e1
    m = lambda leaf: leaf.metadata["mount"]  # noqa: E731
    b_read, e_read = [b2.metadata["mount"][k] for k in ("at", "start", "end")], [m(e1)[k] for k in ("at", "start", "end")]
    assert leaves[lid[2]].metadata["octopus_reads"] == [[b_read[0], lid[0], *b_read[1:]]]  # the reader leaf b3 below b2
    assert leaves[lid[3]].metadata["octopus_reads"] == [[m(leaves[lid[3]])["at"], lid[0], *b_read[1:]]]  # a3: first leaf
    assert leaves[lid[4]].metadata["octopus_reads"] == [[e_read[0], lid[2], *e_read[1:]]]
    assert "octopus_reads" not in leaves[lid[0]].metadata and "octopus_reads" not in leaves[lid[1]].metadata
    flat, cu, mounts = _mounts(_batch(leaves))
    assert mounts == {2: [(b_read[0], 0, *b_read[1:])], 3: [(m(leaves[lid[3]])["at"], 0, *b_read[1:])],
                      4: [(e_read[0], 2, *e_read[1:])]}
    tree = build_tree(flat, cu, mounts)
    node_of = lambda seq, p: next(n for n in tree.nodes  # noqa: E731
                                  if n.start <= tree.pair_pos[tree.flat_pair[cu[seq] + p]] < n.end)
    a_tree = node_of(0, cu[1] - cu[0] - 2).path  # a3's tokens in tree order (a3b shares them up to a2's input)
    a_in = len(a2.model_response.input_tokens)
    assert (node_of(1, a_in - 1).path[:a_in] == a_tree[:a_in]).all()
    for seq in (2, 3):  # from the read on, every later token of the reader leaf sees A's range, as A computed it
        at, _, s, e = mounts[seq][0]
        for p in (at, cu[seq + 1] - cu[seq] - 2):
            assert (node_of(seq, p).mounted == a_tree[s:e]).all()
        assert (tree.tokens[a_tree[s:e]] == flat[cu[0] + s:cu[0] + e]).all()
        assert not len(node_of(seq, at - 1).mounted)
    at, _, s, e = mounts[4][0]
    b_tree = node_of(2, cu[3] - cu[2] - 2).path
    b_at, _, a_s, a_e = mounts[2][0]  # B's read of A, inside E's range: E sees it too (on the server, B's placeholders)
    assert s < b_at < e and (node_of(4, at).mounted == np.r_[b_tree[s:e], a_tree[a_s:a_e]]).all()
    pos = lambda seq, p: int(tree.position_ids[tree.pair_pos[tree.flat_pair[cu[seq] + p]]])  # noqa: E731
    assert pos(4, at) == max(at, pos(2, e - 1) + 1)  # K2 off B's positions, which its own read shifted
    b_at, _, _, a_end = mounts[2][0]
    assert pos(2, b_at) == max(b_at, a_end)  # B continues at max(own next position, A's last mounted + 1)


def test_cost_is_tree_tokens_and_causal_attention_pairs():
    from octopus.rl.prefix_tree import KAPPA, cost
    a, b = np.arange(300), np.r_[np.arange(100), np.arange(1000, 1150)]  # 100 shared tokens, then 200 and 150 own
    pairs = 100 * 101 // 2 + 200 * 100 + 200 * 201 // 2 + 150 * 100 + 150 * 151 // 2  # root, then each child
    assert cost([a, b]) == pytest.approx(450 + KAPPA * pairs)
    assert cost([a]) == pytest.approx(300 + KAPPA * (300 * 301 // 2))
    # a reader's queries also see the 40 keys it mounts: 150 own rows x 40 more pairs
    assert cost([a, b], {1: [(100, 0, 10, 50)]}) == pytest.approx(450 + KAPPA * (pairs + 150 * 40))


def test_allocation_keeps_a_read_with_its_source_and_fails_loudly_without_it(capsys):
    from octopus.rl.prefix_tree import allocate_prefix_groups, record_reads, tree_tokens
    leaves, _ = _episode()
    record_reads(leaves)
    flat, cu, mounts = _mounts(batch := _batch(leaves))
    rng = np.random.default_rng(4)
    seqs = [flat[a:b] for a, b in zip(cu[:-1], cu[1:])]
    seqs += [np.r_[5, rng.integers(10, 1000, 60)] for _ in range(4)]  # other episodes, lexicographically first
    cap = 80
    plain = allocate_prefix_groups(seqs, cap)
    assert any(2 in m and 0 not in m for m in plain)  # by tokens alone, B's leaf lands away from its source a3
    for kw in ({}, {"min_groups": len(seqs)}):  # the walk and the split for min_groups never part a reader and source
        mbs = allocate_prefix_groups(seqs, cap, mounts=mounts, **kw)
        assert sorted(i for m in mbs for i in m) == list(range(len(seqs)))
        assert all({0, 2, 3, 4} <= set(m) or not {0, 2, 3, 4} & set(m) for m in mbs)
    assert allocate_prefix_groups(seqs, cap, mounts={}) == plain
    unit = tree_tokens([seqs[i] for i in (0, 2, 3, 4)])
    capsys.readouterr()
    mbs = allocate_prefix_groups(seqs, unit - 1, mounts=mounts)  # over the cap: a micro-batch of its own, counted
    assert [0, 2, 3, 4] in mbs and f"mount unit over the micro-batch cap: {unit} > {unit - 1}" in capsys.readouterr().out
    with pytest.raises(ValueError, match="reads leaf .* which is not in its micro-batch"):
        _mounts(batch, rows=[1, 2])  # B's leaf without a3: never run without its read
    assert _mounts(batch, rows=[0, 1])[2] == {}


def test_conv_context_and_label_packing_follow_the_model():
    from transformers import GraniteMoeHybridConfig, Lfm2Config, LlamaConfig, MambaConfig, Qwen3_5Config, \
        Qwen3_5TextConfig, Qwen3NextConfig, RwkvConfig

    from octopus.rl.model_hooks import _mamba2_kernels, conv_context, family

    assert conv_context(Qwen3_5TextConfig()) == conv_context(Qwen3_5Config()) == 3  # kernel 4; text or VL config
    assert conv_context(Qwen3NextConfig(linear_conv_kernel_dim=6)) == 5
    assert conv_context(LlamaConfig()) == 0  # pure attention
    assert conv_context(GraniteMoeHybridConfig(mamba_d_conv=6)) == 5  # Mamba-2 hybrid
    assert family(GraniteMoeHybridConfig()).patch is _mamba2_kernels
    for config in (MambaConfig(), RwkvConfig(), Lfm2Config()):  # sequence state without registered hooks
        with pytest.raises(NotImplementedError):
            conv_context(config)
    flat, cu = _flat([np.array([5, (1 << 21) + 3, 7]), np.array([5, (1 << 21) + 3, (1 << 20) + 9])])  # vocab >= 2^20
    tree = build_tree(flat, cu)
    for p in (0, 1, 3, 4):
        row = tree.flat_pair[p]
        assert tree.tokens[tree.pair_pos[row]] == flat[p] and tree.pair_label[row] == flat[p + 1]
