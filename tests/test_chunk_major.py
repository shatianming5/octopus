"""Chunk-major (octopus/rl/chunk_major.py) against the whole-tree streamed layers (seq_tiles.tree_layer_forward), on
CPU in fp64: FlashAttention and fla (recurrence, convolution) replaced by plain torch, seq_tiles' fp32 blocks run in
fp64, so the two differ by summation order alone. Alone, and as AReaL trains it: PEFT LoRA, fully_shard on 2 gloo
ranks."""

import contextlib
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from octopus.rl import chunk_major, prefix_tree, seq_tiles  # noqa: E402
from octopus.rl.prefix_tree import TreeMeta, build_tree, tree_head_output  # noqa: E402
from test_prefix_tree import torch_segment_conv  # noqa: E402  (fla's convolution runs on CUDA only)

VOCAB = 1024


def _attention_call(q, k, v, call, softmax_scale=None):
    """prefix_tree.attention_call (FlashAttention varlen, causal aligned bottom-right) node by node with SDPA."""
    _, keys, cu_q, cu_k, _, _ = call
    out = []
    for i in range(len(cu_q) - 1):
        q_ = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)
        k_, v_ = (t.index_select(0, keys[cu_k[i]:cu_k[i + 1]]).repeat_interleave(q.shape[1] // k.shape[1], 1)
                  .transpose(0, 1) for t in (k, v))
        mask = torch.ones(q_.shape[1], k_.shape[1], dtype=torch.bool).tril(k_.shape[1] - q_.shape[1])
        out.append(F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, scale=softmax_scale).transpose(0, 1))
    return torch.cat(out)


def _delta_rule(q, k, v, g, beta, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None, **_):
    """fla's varlen chunk_gated_delta_rule (q, k l2-normed in the kernel) token by token in the input dtype:
    (output, each segment's final state). The call also passes cu_seqlens on the host (prefix_tree.Level.cu_cpu)."""
    assert cu_seqlens_cpu.device.type == "cpu" and torch.equal(cu_seqlens_cpu, cu_seqlens.cpu())
    q, k = (t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) for t in (q, k))
    q = q * q.shape[-1] ** -0.5
    out, finals = [], []
    for j, (a, b) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        s = initial_state[j] if initial_state is not None else q.new_zeros(q.shape[2], q.shape[3], v.shape[3])
        for t in range(a, b):
            kt = k[0, t, :, :, None]
            s = s * g[0, t].exp()[:, None, None]
            s = s + kt * ((v[0, t] - (s * kt).sum(1)) * beta[0, t, :, None])[:, None]
            out.append((s * q[0, t, :, :, None]).sum(1))
        finals.append(s)
    return torch.stack(out)[None], torch.stack(finals)


def _tiny_model(set_):
    """The tiny fp64 Qwen3.5 (2 GDN + 2 attention layers) with the CPU kernels and SEG 64, installed by
    set_(module, name, value): monkeypatch.setattr, or setattr in a worker process."""
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                            layer_types=["linear_attention", "full_attention"] * 2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=128, linear_value_head_dim=128, initializer_range=0.1,
                            tie_word_embeddings=False)
    model = Qwen3_5ForCausalLM(cfg).to(torch.float64)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    seq_tiles.patch_tiles(model)
    rms = lambda x, eps: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # noqa: E731
    for name, fn in dict(rms_norm=lambda x, w, eps: rms(x, eps) * (1.0 + w),
                         rms_norm_gated=lambda x, gate, w, eps: w * rms(x, eps) * F.silu(gate),
                         silu_mul=lambda gate, up: F.silu(gate) * up,
                         sigmoid_gate=lambda o, gate: o * torch.sigmoid(gate),
                         _gdn_delta_rule=lambda lin, q, k, v, a, beta, **kw: _delta_rule(
                             q, k, v, -lin.A_log.exp() * F.softplus(a + lin.dt_bias), beta, **kw)).items():
        set_(seq_tiles, name, fn)
    set_(prefix_tree, "attention_call", _attention_call)
    set_(prefix_tree, "segment_conv", torch_segment_conv)
    set_(prefix_tree, "CONV_CONTEXT", cfg.linear_conv_kernel_dim - 1)
    set_(prefix_tree, "SEG", 64)
    set_(seq_tiles, "TILE", 48)
    return model


@pytest.fixture
def model(monkeypatch):
    return _tiny_model(monkeypatch.setattr)


def _head(model):
    return lambda h, y: torch.log_softmax(model.get_output_embeddings()(h), -1).gather(-1, y[:, None])


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def _whole_tree(model, tree, w):
    """The tree through the whole-tree streamed layers: the head output; (out * w).sum() backpropagated into .grad."""
    meta = TreeMeta(tree, "cpu")
    hidden = model.get_decoder()(input_ids=torch.from_numpy(tree.tokens)[None],
                                 position_ids=torch.from_numpy(tree.position_ids)[None], octopus_tree=meta,
                                 use_cache=False).last_hidden_state
    out = tree_head_output(_head(model), hidden, meta)
    (out * w).sum().backward()
    return out.detach()


def _chunk_major(model, tree, w, tokens):
    model.zero_grad(set_to_none=True)
    plan = chunk_major.ChunkPlan(tree, tokens, "cpu")
    out, kept = chunk_major.run_forward(plan, model, _head(model))
    chunk_major.run_backward(plan, model, _head(model), kept, w)
    return out, _grads(model), len(plan.chunks)


def _flat(seqs):
    return np.concatenate(seqs).astype(np.int64), np.cumsum([0] + [len(s) for s in seqs]).tolist()


def _tok(rng, n):
    return rng.integers(3, VOCAB, n)


def _mount():
    """test_seq_tiles' mounted tree: A shares a prefix with B and C; B reads A's own tokens from its token 210 on."""
    rng = np.random.default_rng(1)
    prefix = _tok(rng, 150)
    seqs = [np.r_[prefix, _tok(rng, 200)], np.r_[prefix, _tok(rng, 200)], np.r_[prefix, _tok(rng, 90)]]
    return seqs, {1: [(210, 0, 150, 330)]}


def _transitive():
    """test_prefix_tree's two segments and transitive mount, 8x longer: B mounts two segments at one point, C mounts a
    range of A that holds A's own read of X."""
    rng = np.random.default_rng(2)
    prefix = _tok(rng, 80)
    seqs = [_tok(rng, 160), _tok(rng, 240), np.r_[prefix, _tok(rng, 48)], np.r_[prefix, _tok(rng, 48)]]
    return seqs, {1: [(96, 0, 40, 120)], 2: [(80, 1, 16, 64), (80, 0, 80, 144)], 3: [(80, 1, 64, 160)]}


def _source_later():
    """The reader comes first in pre-order, its source after it: the chunk order moves the source ahead."""
    rng = np.random.default_rng(3)
    return [np.r_[1, _tok(rng, 199)], np.r_[2, _tok(rng, 249)]], {0: [(120, 1, 30, 200)]}


def _random_dag(seed):
    """Sequences branching off earlier ones; each reads ranges of the sequences before it in a random order."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(rng.integers(3, 7)):
        if seqs and rng.random() < 0.7:
            base = seqs[rng.integers(len(seqs))]
            seqs.append(np.r_[base[:rng.integers(1, len(base))], _tok(rng, rng.integers(1, 120))])
        else:
            seqs.append(_tok(rng, rng.integers(20, 160)))
    order = rng.permutation(len(seqs))
    mounts = {}
    for r, i in enumerate(order[1:], 1):
        for _ in range(rng.integers(0, 3)):
            src = int(order[rng.integers(r)])
            s0 = int(rng.integers(0, len(seqs[src]) - 1))
            mounts.setdefault(int(i), []).append((int(rng.integers(1, len(seqs[i]))), src, s0,
                                                  int(rng.integers(s0 + 1, len(seqs[src]) + 1))))
    return seqs, mounts


def _long_node():
    """A 600-token node (10 segments) read from its middle by a sibling several chunks long, itself read by a third."""
    rng = np.random.default_rng(7)
    prefix = _tok(rng, 30)
    return ([np.r_[prefix, _tok(rng, 600)], np.r_[prefix, _tok(rng, 400)], np.r_[prefix, _tok(rng, 50)]],
            {1: [(250, 0, 300, 330)], 2: [(60, 1, 100, 380)]})


CASES = {"mount": _mount, "transitive": _transitive, "source_later": _source_later, "long_node": _long_node,
         **{f"random{s}": (lambda s=s: _random_dag(s)) for s in range(4)}}


def _case(name):
    """The case's tree and a fixed random weight per flat position (the loss is (out * w).sum())."""
    tree = build_tree(*_flat(CASES[name]()[0]), CASES[name]()[1])
    return tree, torch.randn(1, len(tree.flat_pair), 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("case", CASES)
def test_chunk_major_matches_the_whole_tree(model, case):
    """logp and every parameter gradient equal the whole tree's (1e-9, fp64) in 1, 2 and many chunks, also chunks
    below SEG (a chunk then holds one segment, or several shorter ones)."""
    tree, w = _case(case)
    model.zero_grad(set_to_none=True)
    ref_out = _whole_tree(model, tree, w)
    ref = _grads(model)
    assert min(g.abs().max() for g in ref.values()) > 1e-6  # every parameter has a gradient to compare
    counts = []
    for tokens in (10**9, len(tree.tokens) // 2 + 64, 64, 32):  # a segment has at most SEG = 64 tokens; 32 < SEG
        out, grads, count = _chunk_major(model, tree, w, tokens)
        counts.append(count)
        diff = max((grads[name] - g).abs().max().item() for name, g in ref.items())
        print(f"{case}: {count} chunks, logp max diff {(out - ref_out).abs().max().item():.1e}, grads {diff:.1e}")
        assert (out - ref_out).abs().max() <= 1e-9
        for name, g in ref.items():
            assert (grads[name] - g).abs().max() <= 1e-9, (tokens, name)
    assert counts[:2] == [1, 2] and counts[2] > 2


def test_chunk_major_with_mount_layers(model, monkeypatch):
    """OCTOPUS_MOUNT_LAYERS: only layer 3 reads the mounted keys, layer 1 its own path."""
    monkeypatch.setattr(prefix_tree, "MOUNT_LAYERS", frozenset({3}))
    tree, w = _case("mount")
    ref_out = _whole_tree(model, tree, w)
    ref = _grads(model)
    out, grads, _ = _chunk_major(model, tree, w, 64)
    assert (out - ref_out).abs().max() <= 1e-9
    assert all((grads[k] - g).abs().max() <= 1e-9 for k, g in ref.items())


def test_backward_runs_each_layer_twice(model, monkeypatch):
    """run_backward runs each layer of a chunk twice, its forward with autograd and the recompute of its own checkpoint,
    with no checkpoint nested inside (seq_tiles.no_inner); the gradients equal, bit for bit, those with the tiles and
    kernel calls checkpointed too, which ran each kernel call up to 4 times (FlashAttention 2.5 times). The inner
    checkpoints are back on afterwards, also after a backward that raised."""
    tree, w = _case("long_node")
    layers = model.get_decoder().layers
    rows, calls = {}, [0]  # MLP rows per layer type, FlashAttention calls
    for layer in layers:
        layer.mlp.gate_proj.register_forward_hook(
            lambda m, args, out, kind=layer.block_type: rows.update({kind: rows.get(kind, 0) + args[0].shape[1]}))
    attention_call = prefix_tree.attention_call

    def counted_call(*args, **kwargs):
        calls[0] += 1
        return attention_call(*args, **kwargs)

    monkeypatch.setattr(prefix_tree, "attention_call", counted_call)
    runs, no_inner = [], seq_tiles.no_inner
    for inner in (no_inner, contextlib.nullcontext):  # the second: every checkpoint nested, as before
        monkeypatch.setattr(seq_tiles, "no_inner", inner)
        model.zero_grad(set_to_none=True)
        plan = chunk_major.ChunkPlan(tree, 256, "cpu")
        calls[0] = 0
        _, kept = chunk_major.run_forward(plan, model, _head(model))
        forward_calls, calls[0] = calls[0], 0
        rows.clear()
        chunk_major.run_backward(plan, model, _head(model), kept, w)
        runs.append((_grads(model), dict(rows), calls[0]))
        assert seq_tiles._INNER
    (grads, rows, n_calls), (nested, nested_rows, nested_calls) = runs
    assert len(plan.chunks) > 2 and n_calls == 2 * forward_calls and nested_calls > n_calls
    per_kind = {kind: sum(layer.block_type == kind for layer in layers) for kind in rows}
    assert rows == {kind: 2 * n * len(tree.tokens) for kind, n in per_kind.items()}
    assert all(nested_rows[kind] > rows[kind] for kind in rows)
    assert all(torch.equal(grads[k], g) for k, g in nested.items())

    def failing_head(h, y):
        assert not seq_tiles._INNER
        raise RuntimeError("head")

    monkeypatch.setattr(seq_tiles, "no_inner", no_inner)
    _, kept = chunk_major.run_forward(plan, model, _head(model))
    with pytest.raises(RuntimeError, match="head"):
        chunk_major.run_backward(plan, model, failing_head, kept, w)
    assert seq_tiles._INNER


def test_training_needs_layer_checkpointing(model):
    """Without HF layer checkpointing the chunk-major backward would hold every layer's activations over a chunk
    (seq_tiles.no_inner): a forward with grad refuses before running."""
    tree, _ = _case("mount")
    model.gradient_checkpointing_disable()
    with pytest.raises(RuntimeError, match="layer checkpointing"):
        chunk_major.chunk_major_output(chunk_major.ChunkPlan(tree, 64, "cpu"), model, _head(model))


def test_a_forward_before_the_previous_backward_fails_loudly(model):
    """A micro-batch's kept tensors wait in the (GPU path's) pinned slab for its backward: the next forward, which
    would carve its own from the slab's start, raises until then."""
    tree, w = _case("mount")
    plan = chunk_major.ChunkPlan(tree, 64, "cpu")
    out = chunk_major.chunk_major_output(plan, model, _head(model))
    with pytest.raises(AssertionError, match="before the previous one's backward"):
        chunk_major.chunk_major_output(plan, model, _head(model))
    with torch.no_grad(), pytest.raises(AssertionError, match="before the previous one's backward"):
        chunk_major.chunk_major_output(plan, model, _head(model))  # compute_logp's forward alike
    (out * w).sum().backward()
    with torch.no_grad():
        chunk_major.chunk_major_output(plan, model, _head(model))


@pytest.mark.parametrize("case", ["mount", "random0"])
def test_scored_tree_matches_the_full_tree_at_its_positions(model, case):
    """A tree scoring 5% of the flat positions (build_tree scored: AReaL's loss_mask), whole and chunk-major (some chunks
    score nothing): its logp equals the full tree's there and is 0 elsewhere; the gradients of a loss over those
    positions equal the full tree's (1e-9, fp64)."""
    seqs, mounts = CASES[case]()
    flat, cu = _flat(seqs)
    scored = np.random.default_rng(5).random(len(flat)) < 0.05
    tree, (full, w) = build_tree(flat, cu, mounts, scored), _case(case)
    w = w * torch.from_numpy(scored)[None, :, None]
    model.zero_grad(set_to_none=True)
    ref_out = _whole_tree(model, full, w)
    ref = _grads(model)
    model.zero_grad(set_to_none=True)
    whole_out = _whole_tree(model, tree, w)
    whole = _grads(model)
    out, grads, _ = _chunk_major(model, tree, w, 64)
    assert any(not len(ch.pair_rows) for ch in chunk_major.ChunkPlan(tree, 64, "cpu").chunks)
    for o, g in ((whole_out, whole), (out, grads)):
        assert (o[0, scored] - ref_out[0, scored]).abs().max() <= 1e-9 and (o[0, ~scored] == 0).all()
        assert all((g[k] - r).abs().max() <= 1e-9 for k, r in ref.items())


# per rank: its micro-batches (case, chunk tokens, loss scale); rank 1's second is AReaL's dummy (its first, loss x 0)
RANK_MBS = [[("mount", 64, 1.0), ("random0", 128, 1.0)], [("transitive", 10**9, 1.0), ("transitive", 10**9, 0.0)]]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                "in_proj_b", "in_proj_a", "out_proj"]


def _fsdp_setup(rank: int, path: str, mixed: bool):
    """One of 2 gloo ranks: the LoRA model sharded as AReaL's apply_fsdp2 does, its forward chunk_major_output (as
    actor.py's); returns it, an unsharded copy and the head. mixed: FSDP computes in fp32 and reduces in fp64 (as AReaL
    in bf16 and fp32), the model kept in fp64."""
    import datetime
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))  # a per-chunk collective hangs: fail fast
    model = get_peft_model(_tiny_model(setattr), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                            target_modules=LORA_TARGETS))
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:  # PEFT starts B at 0: A would get no gradient
            torch.nn.init.normal_(p, std=0.1)
    reference = copy.deepcopy(model)
    shard = dict(mesh=init_device_mesh("cpu", (2,)), mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float64) if mixed else MixedPrecisionPolicy())
    for m in model.modules():  # AReaL's apply_fsdp2: each decoder layer, the input embedding (untied), the root
        if type(m).__name__ == "Qwen3_5DecoderLayer" or isinstance(m, torch.nn.Embedding):
            fully_shard(m, **shard)
    fully_shard(model, **shard)
    head = _head(model)
    model.forward = lambda plan: chunk_major.chunk_major_output(plan, model, head)  # as actor.py's forward
    return model, reference, head


def _fsdp_worker(rank: int, path: str, mixed: bool) -> None:
    """RANK_MBS[rank] trained chunk-major on a sharded LoRA model (_fsdp_setup; rank 0 in 13 and 6 chunks, rank 1 in
    1), against an unsharded copy trained on every rank's micro-batches whole."""
    import torch.distributed as dist
    model, reference, head = _fsdp_setup(rank, path, mixed)
    posts = [0] * len(model.get_decoder().layers)  # each decoder layer's FSDP post-backward calls

    def counted(i, post_backward):
        def count(*args):
            posts[i] += 1
            return post_backward(*args)
        return count
    for i, layer in enumerate(model.get_decoder().layers):
        group = layer._get_fsdp_state()._fsdp_param_group
        group.post_backward = counted(i, group.post_backward)

    def whole(ref_model):  # every rank's micro-batches through the whole tree: head outputs, LoRA grads (FSDP: mean)
        outs = {}
        for mbs in RANK_MBS:
            for case, _, scale in mbs:
                tree, w = _case(case)
                outs[case] = _whole_tree(ref_model, tree, w.to(ref_model.dtype) * scale / 2)
        return outs, _grads(ref_model)

    def max_diff(a, b):
        return max((a[k].double() - b[k].double()).abs().max().item() for k in b)

    # fp64: rounding alone (1e-9). Mixed: the whole tree in fp32 sets the scale of fp32 rounding; chunk-major in fp32
    # (fp64 reduce) stays within 3x of it, where a wiring slip is O(1)
    tol_out = tol_grad = 1e-9
    if mixed:
        out32, grads32 = whole(copy.deepcopy(reference).float())
    ref_out, ref = whole(reference)
    if mixed:
        tol_out, tol_grad = 3 * max_diff(out32, ref_out), 3 * max_diff(grads32, ref)
    for case, tokens, scale in RANK_MBS[rank]:
        tree, w = _case(case)
        plan = chunk_major.ChunkPlan(tree, tokens, "cpu")
        ((model(plan) * w).sum() * scale).backward()
        # every chunk's backward ran each layer's post-backward (gradients summed in the reduce dtype), and the final
        # callback once more (the one reduce-scatter)
        assert posts == [len(plan.chunks) + 1] * len(posts), (case, posts)
        posts[:] = [0] * len(posts)
    grads = {n: p.grad.full_tensor() for n, p in model.named_parameters() if p.requires_grad}
    assert grads.keys() == ref.keys() and all("lora_" in n for n in ref)
    diff = max_diff(grads, ref)
    print(f"rank {rank} mixed={mixed}: {len(ref)} LoRA grads, max diff {diff:.1e} (bound {tol_grad:.1e}), largest "
          f"{max(g.abs().max().item() for g in ref.values()):.1e}")
    assert diff <= tol_grad
    states = chunk_major._fsdp_states(model)
    assert all(s._fsdp_param_group.is_sharded for s in states[1:])  # nothing stays gathered after the backward

    with torch.no_grad():  # compute_logp
        case, tokens, _ = RANK_MBS[rank][0]
        out = model(chunk_major.ChunkPlan(_case(case)[0], tokens, "cpu"))
    assert (out.double() - ref_out[case]).abs().max() <= tol_out
    assert all(s._fsdp_param_group.is_sharded for s in states[1:])

    model.unshard()  # misuse: called outside the root's forward, FSDP queues no final callback for the backward
    out = chunk_major.chunk_major_output(chunk_major.ChunkPlan(_case(case)[0], tokens, "cpu"), model, head)
    with pytest.raises(AssertionError, match="final callback"):
        out.sum().backward()
    dist.destroy_process_group()


@pytest.mark.parametrize("mixed", [False, True])
def test_fsdp_lora_two_ranks_match_the_whole_tree(tmp_path, mixed):
    """Merged LoRA gradients of 2 FSDP2 ranks with different chunk counts and micro-batch counts (a dummy), 2
    micro-batches accumulated, equal the unsharded whole tree's (fp64: 1e-9); the no-grad forward too; a backward
    outside the root's forward fails loudly."""
    pytest.importorskip("peft")
    torch.multiprocessing.spawn(_fsdp_worker, args=(str(tmp_path / "pg"), mixed), nprocs=2)


# per rank: 3 micro-batches (case, chunk tokens, loss scale) of different sizes; rank 1's third is AReaL's dummy
ACROSS_MBS = [[("mount", 64, 1.0), ("random0", 128, 1.0), ("long_node", 256, 1.0)],
              [("transitive", 10**9, 1.0), ("source_later", 64, 1.0), ("source_later", 64, 0.0)]]


def _across_worker(rank: int, path: str, mixed: bool) -> None:
    """ACROSS_MBS[rank] as compute_logp and ppo_update run them (_fsdp_setup's model), per micro-batch and in
    chunk_major.across_micro_batches, counting FSDP's collectives."""
    import torch.distributed as dist
    model, _, _ = _fsdp_setup(rank, path, mixed)
    calls = {"all_gather_into_tensor": [], "reduce_scatter_tensor": []}  # per collective: the micro-batch it ran in
    mb = [0]
    for name in calls:  # FSDP2's collectives (torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py)
        def counted(*args, _f=getattr(dist, name), _name=name, **kwargs):
            calls[_name].append(mb[0])
            return _f(*args, **kwargs)
        setattr(dist, name, counted)
    mbs = [(chunk_major.ChunkPlan(_case(case)[0], tokens, "cpu"), _case(case)[1] * scale)
           for case, tokens, scale in ACROSS_MBS[rank]]

    def run(across, train, n=len(mbs)):
        for c in calls.values():
            c.clear()
        model.zero_grad(set_to_none=True)
        outs = []
        with (chunk_major.across_micro_batches(model, len(mbs)) if across else contextlib.nullcontext()), \
                torch.set_grad_enabled(train):
            for mb[0], (plan, w) in enumerate(mbs[:n]):
                out = model(plan)
                if train:
                    (out * w).sum().backward()
                outs.append(out.detach())
        grads = {k: p.grad.full_tensor() for k, p in model.named_parameters() if p.grad is not None} if train else {}
        return outs, grads, {name: sorted(set(c)) for name, c in calls.items()}

    def settings():
        return [(g.post_forward_mesh_info, g.reshard_after_backward, g.reduce_grads, g.is_sharded) for g in groups]

    logp, _, per_mb = run(False, False)  # the per-micro-batch path: every micro-batch gathers again
    assert per_mb["all_gather_into_tensor"] == [0, 1, 2], per_mb
    states = chunk_major._fsdp_states(model)
    groups = [s._fsdp_param_group for s in states]
    model.reshard()
    before = settings()
    assert all(sharded for *_, sharded in before)
    outs, _, across = run(True, False)  # compute_logp: gathered once, resharded at the end
    assert across == {"all_gather_into_tensor": [0], "reduce_scatter_tensor": []}, across
    assert all(torch.equal(a, b) for a, b in zip(outs, logp)) and settings() == before

    train, ref, per_mb = run(False, True)
    assert per_mb["reduce_scatter_tensor"] == [0, 1, 2], per_mb
    for _ in range(2):  # the second after a call that raised with 2 micro-batches' gradients unreduced
        outs, grads, across = run(True, True)
        assert across == {"all_gather_into_tensor": [0], "reduce_scatter_tensor": [2]}, across
        assert all(torch.equal(a, b) for a, b in zip(outs, train)) and settings() == before
        diff = max((grads[k] - g).abs().max().item() for k, g in ref.items())
        print(f"rank {rank} mixed={mixed}: across micro-batches, LoRA grads max diff {diff:.1e} (largest "
              f"{max(g.abs().max().item() for g in ref.values()):.1e})")
        assert grads.keys() == ref.keys() and diff <= 1e-12  # one reduce-scatter of the sum: fp64 summation order
        with pytest.raises(AssertionError, match="never reduced"):
            run(True, True, n=2)
        assert settings() == before
        assert all(p.unsharded_accumulated_grad is None and p._unsharded_param.grad is None
                   for g in groups for p in g.fsdp_params)
    dist.destroy_process_group()


@pytest.mark.parametrize("mixed", [False, True])
def test_fsdp_across_micro_batches(tmp_path, mixed):
    """chunk_major.across_micro_batches on 2 FSDP2 ranks, 3 micro-batches each (rank 1's last a dummy): compute_logp
    gathers once and returns the per-micro-batch path's logits bit for bit; ppo_update also reduce-scatters once, in
    the last micro-batch's backward, its gradients those of the per-micro-batch path up to fp64 summation order (1e-12);
    both leave the settings as they were and the parameters sharded. A call that raises (here: it ran 2 of its 3
    micro-batches) restores them too and drops its unreduced gradients: the next call's equal the reference."""
    pytest.importorskip("peft")
    torch.multiprocessing.spawn(_across_worker, args=(str(tmp_path / "pg"), mixed), nprocs=2)
