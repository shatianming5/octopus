import functools

import pytest
import torch

pytest.importorskip("fla")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("fla and FlashAttention kernels need CUDA", allow_module_level=True)

from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from octopus.rl.model_hooks import patch_model, tile_model  # noqa: E402
from octopus.rl import prefix_tree, seq_tiles  # noqa: E402


def _model(**overrides):
    cfg = Qwen3_5TextConfig(**{**dict(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                                      layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
                                      num_key_value_heads=1, head_dim=256, linear_num_key_heads=2,
                                      linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128),
                               **overrides})
    cfg._attn_implementation = "flash_attention_2"
    model = Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    patch_model(cfg)
    tile_model(model)
    prefix_tree.patch_attention()
    return model


def _logp_and_grads(model, kwargs, labels, rows=slice(None)):
    """logp of labels at hidden rows, then the weight-matrix grads of its mean (what the LoRA trains); per-head scalars
    such as dt_bias sum cancelling terms over every token, so their bf16 grads differ by ~20% between any two summation
    orders (fp32: within 2%)."""
    model.zero_grad(set_to_none=True)
    logits = model(**kwargs).logits[0, rows].float()
    logp = torch.log_softmax(logits, -1).gather(-1, labels[:, None])[:, 0]
    logp.mean().backward()
    return logp.detach(), {n: p.grad.float() for n, p in model.named_parameters() if p.ndim == 2}


def _assert_close(a, b):
    (logp_a, grads_a), (logp_b, grads_b) = a, b
    assert (logp_a - logp_b).abs().max() < 2e-2
    assert grads_a.keys() == grads_b.keys()
    for name, g in grads_b.items():  # ~1% apart; a wiring slip is O(1)
        assert (grads_a[name] - g).norm() <= 5e-2 * g.norm(), name


def test_flat_tiles_and_tree_streaming_match_the_hf_layers_forward_and_backward(monkeypatch):
    """Sequence tiles (TILE) of a flat micro-batch, and the same micro-batch as a prefix tree streamed in SEG-token
    kernel calls, shrunk to 64 / 128 tokens, against the plain HF layers on the same two packed sequences."""
    torch.manual_seed(0)
    model = _model()
    lens = [300, 213]  # two packed sequences; 513 tokens = 4 tiles of 128 and a ragged one
    ids = torch.randint(0, 1024, (1, sum(lens)), device="cuda")
    ids[0, 0], ids[0, lens[0]] = 1, 2  # nothing shared: the tree is the flat layout, segmented
    cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(n) for n in lens])[None].cuda()

    def run(tile, seg):
        monkeypatch.setattr(seq_tiles, "TILE", tile)
        monkeypatch.setattr(prefix_tree, "SEG", seg)
        kwargs = dict(input_ids=ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max(lens),
                      max_length_k=max(lens), use_cache=False)
        tree = prefix_tree.tree_inputs(ids, cu)
        assert (tree is not None) == (seg < sum(lens))
        if tree is not None:
            assert len(tree[0].levels) == 5  # 300 = 128+128+44 and 213 = 128+85: five calls of <= 128 tokens
            kwargs = tree[1]
        return _logp_and_grads(model, kwargs, ids[0, 1:], slice(None, -1))

    plain = run(10**9, 10**9)  # one tile, no tree: the HF layer forward
    _assert_close(run(128, 10**9), plain)  # flat, tiled
    _assert_close(run(64, 128), plain)  # tree, streamed


def _mounted_tree(n_prefix=150):
    """A shares a prefix with B and C; B reads A's own tokens from its token 210 on and continues after the read."""
    g = torch.Generator().manual_seed(1)
    tok = lambda n: torch.randint(3, 1024, (n,), generator=g)  # noqa: E731
    prefix = tok(n_prefix)
    seqs = [torch.cat([prefix, tok(200)]), torch.cat([prefix, tok(60), tok(140)]), torch.cat([prefix, tok(90)])]
    flat = torch.cat(seqs)[None].cuda()
    cu = torch.tensor([0, *torch.tensor([len(s) for s in seqs]).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return flat, cu, {1: [(n_prefix + 60, 0, n_prefix, n_prefix + 180)]}


def test_tree_streaming_matches_the_hf_layers_tree_path_with_mounts(monkeypatch):
    """Shared prefixes, a mount and nodes of several segments (SEG 64): the streamed layers against the HF layer
    forward on the same tree (the tree kernels of octopus/rl/prefix_tree.py inside the HF layers)."""
    torch.manual_seed(0)
    model = _model()
    monkeypatch.setattr(seq_tiles, "TILE", 48)
    monkeypatch.setattr(prefix_tree, "SEG", 64)
    flat, cu, mounts = _mounted_tree()
    meta, kwargs = prefix_tree.tree_inputs(flat, cu, mounts)
    assert any(len(lv.ids) > 1 for lv in meta.levels) and any(len(lv.ctx_index) for lv in meta.levels)
    assert any(isinstance(lv.index, torch.Tensor) for lv in meta.levels)  # a call of non-adjacent segments
    streamed = _logp_and_grads(model, kwargs, meta.pair_label, meta.pair_pos)
    cls = next(c for c in type(model.model.layers[0]).__mro__ if c.__module__.startswith("transformers.models."))
    monkeypatch.setattr(cls, "forward", seq_tiles._hf_layer_forward)
    _assert_close(streamed, _logp_and_grads(model, kwargs, meta.pair_label, meta.pair_pos))


@pytest.mark.parametrize("train", [False, True])
def test_tree_streaming_peak_follows_seg_not_the_tree(monkeypatch, train):
    """No GDN-wide tensor spans the tree: with the GDN value heads 32x wider than the residual stream, the peak grows by
    less than one value row (8 KB) per added tree token; the HF layers on the same tree grow by more."""
    torch.manual_seed(0)
    model = _model(hidden_size=128, intermediate_size=256, head_dim=128, linear_num_key_heads=4,
                   linear_num_value_heads=32)
    monkeypatch.setattr(seq_tiles, "TILE", 512)
    monkeypatch.setattr(prefix_tree, "SEG", 1024)
    row = 32 * 128 * 2  # bytes

    def growth(added=8192):  # peak bytes per tree token, from a 8k-token to a 16k-token shared prefix
        peaks = []
        for n_prefix in (8192, 8192 + added):
            _, kwargs = prefix_tree.tree_inputs(*_mounted_tree(n_prefix))

            def run():
                with torch.set_grad_enabled(train):
                    out = model.model(**kwargs).last_hidden_state
                    if train:
                        out.float().square().mean().backward()
            run()  # compiles, allocates the grads
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            run()
            peaks.append(torch.cuda.max_memory_allocated() - base)
        return (peaks[1] - peaks[0]) / added

    streamed = growth()
    cls = next(c for c in type(model.model.layers[0]).__mro__ if c.__module__.startswith("transformers.models."))
    monkeypatch.setattr(cls, "forward", seq_tiles._hf_layer_forward)
    hf = growth()
    print(f"peak bytes per tree token: streamed {streamed:.0f}, HF layers {hf:.0f}")
    assert streamed < row < hf


@pytest.mark.skipif(not seq_tiles.COMPILE, reason="OCTOPUS_COMPILE=0")
def test_compiled_elementwise_blocks_match_eager():
    from octopus.rl import lm_head

    torch.manual_seed(0)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x, w, z = torch.randn(300, 2560, **bf), torch.randn(2560, **bf) * 0.1, torch.randn(300, 2560, **bf)
    logits, labels = torch.randn(64, 1000, **bf), torch.randint(0, 1000, (64,), device="cuda")
    cases = [(seq_tiles.rms_norm, (x, w, 1e-6)), (seq_tiles.rms_norm_gated, (x, z, w, 1e-6)), (seq_tiles.silu_mul, (x, z)),
             (seq_tiles.sigmoid_gate, (x, z)), (lm_head._head_columns, (logits, 1.0, labels))]
    for fn, args in cases:
        outs, grads = [], []
        for f in (fn, fn._torchdynamo_orig_callable):  # compiled, eager
            leaves = [a.detach().clone().requires_grad_() if torch.is_tensor(a) and a.is_floating_point() else a for a in args]
            out = f(*leaves)
            out.float().pow(2).sum().backward()
            outs.append(out.float())
            grads.append([a.grad.float() for a in leaves if torch.is_tensor(a) and a.grad is not None])
        assert (outs[0] - outs[1]).abs().max() <= 2e-2 * outs[1].abs().max() + 1e-3, fn
        for g0, g1 in zip(*grads):
            assert (g0 - g1).norm() <= 2e-2 * g1.norm() + 1e-6, fn


@pytest.mark.parametrize("kernel", ["gated_delta_rule", "simple_gla"])
def test_host_cu_seqlens_leave_fla_bit_identical(kernel):
    """prefix_tree.Level.cu_cpu: fla's varlen kernels give the same output, final states and gradients as from the
    device cu_seqlens alone (from which fla read the lengths back to the host: a device sync per call)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.simple_gla import chunk_simple_gla

    torch.manual_seed(0)
    cu = torch.tensor([0, 100, 164, 300, 301])  # segments of 100, 64, 136 and 1 tokens
    n, h, d = int(cu[-1]), 4, 64
    q, k, v = (torch.randn(1, n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    g = torch.nn.functional.logsigmoid(torch.randn(1, n, h, device="cuda"))
    h0 = torch.randn(len(cu) - 1, h, d, d, device="cuda")
    if kernel == "gated_delta_rule":
        fn = functools.partial(chunk_gated_delta_rule, use_qk_l2norm_in_kernel=True)
        xs = (q, k, v, g, torch.rand(1, n, h, device="cuda", dtype=torch.bfloat16))
    else:
        fn, xs = chunk_simple_gla, (q, k, v, g)
    runs = []
    for host in (None, cu):
        leaves = [x.clone().requires_grad_() for x in (*xs, h0)]
        out, state = fn(*leaves[:-1], initial_state=leaves[-1], output_final_state=True, cu_seqlens=cu.cuda(),
                        cu_seqlens_cpu=host)
        (out.float().square().sum() + state.square().sum()).backward()
        runs.append([out, state] + [x.grad for x in leaves])
    assert all(torch.equal(a, b) for a, b in zip(*runs))
