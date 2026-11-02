"""block_checkpoint.checkpoint_blocks: blocks of k layers save one input per block instead of one per layer, and the
logprobs and gradients stay bitwise those of per-layer checkpointing (CPU, fp64): the HF layers on a flat micro-batch,
and the streamed prefix-tree layers (octopus/rl/seq_tiles.py) on a tree with shared prefixes, a mount and several
kernel calls per layer, with torch stand-ins for the CUDA kernels (FlashAttention, fla)."""

import datetime
import functools

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, modeling_qwen3_5

from octopus.rl import prefix_tree, seq_tiles
from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.model_hooks import conv_context, tile_model
from test_prefix_tree import torch_segment_conv

LAYERS = ["linear_attention", "linear_attention", "full_attention", "linear_attention", "full_attention"]
HIDDEN = 64
# HF's torch GDN functions, as at collection: a GPU test run before these (test_seq_tiles' patch_model) leaves them
# replaced by fla's Triton kernels for the rest of the session, which cannot take these CPU tensors
_HF_GDN = {name: getattr(modeling_qwen3_5, name) for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")}


@pytest.fixture(autouse=True)
def _hf_gdn(monkeypatch):
    for name, fn in _HF_GDN.items():
        monkeypatch.setattr(modeling_qwen3_5, name, fn)


def _model():
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=len(LAYERS),
                            layer_types=LAYERS, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                            linear_value_head_dim=16)
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).double()
    with torch.no_grad():  # a trained-looking LoRA-free model: every weight gets a gradient worth comparing
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


def _run(model, kwargs, rows, labels, k):
    """logp of labels at the logit rows, every parameter's gradient of its mean, and the bytes the forward saved for
    backward (what lm_head.host_activations would pin)."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # per-layer again
    if k:
        checkpoint_blocks(model, k)
    model.zero_grad(set_to_none=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel() * t.element_size()) or t,
                                                  lambda t: t):
        logits = model(**kwargs).logits[0, rows]
    logp = torch.log_softmax(logits, -1).gather(-1, labels[:, None])[:, 0]
    logp.mean().backward()
    return logp.detach(), {n: p.grad for n, p in model.named_parameters()}, sum(saved)


def _assert_identical_and_smaller(model, kwargs, rows, labels, tokens):
    logp, grads, saved = _run(model, kwargs, rows, labels, None)
    for k in (2, 3):  # blocks [0 1] [2 3] [4] and [0 1 2] [3 4]
        logp_k, grads_k, saved_k = _run(model, kwargs, rows, labels, k)
        assert torch.equal(logp_k, logp)
        assert grads_k.keys() == grads.keys() and all(torch.equal(grads_k[n], g) for n, g in grads.items()), k
        blocks = -(-len(LAYERS) // k)
        assert saved - saved_k == (len(LAYERS) - blocks) * tokens * HIDDEN * 8  # one fp64 layer input per layer less


def test_block_size_per_micro_batch():
    """One step of two micro-batches, one under max_tokens (blocks of 3) and one over it (per layer): the summed
    gradients are bitwise per-layer checkpointing's, and only the short micro-batch saves less."""
    model = _model()
    short, long = torch.randint(0, 512, (1, 64)), torch.randint(0, 512, (1, 160))

    def step(k):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if k:
            checkpoint_blocks(model, k, max_tokens=100)
        model.zero_grad(set_to_none=True)
        saved = []
        for ids in (short, long):
            n = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: n.append(t.numel() * t.element_size()) or t,
                                                          lambda t: t):
                logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
            torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).mean().backward()
            saved.append(sum(n))
        return {n: p.grad for n, p in model.named_parameters()}, saved

    grads, saved = step(None)
    grads_k, saved_k = step(3)
    assert all(torch.equal(grads_k[n], g) for n, g in grads.items())
    assert saved[0] - saved_k[0] == (len(LAYERS) - 2) * 64 * HIDDEN * 8 and saved_k[1] == saved[1]


class _Layer(torch.nn.Module):
    """HF GradientCheckpointingLayer's call shape around a residual tanh layer."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)
        self._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))

    def __call__(self, x):
        return self._gradient_checkpointing_func(functools.partial(super().__call__), x)


class _Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(6))

    def get_decoder(self):
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _fsdp_rank(rank, lens, port, out):
    """One gloo rank of FSDP2 (reshard_after_forward, as AReaL) over its own micro-batch length: per layer, then blocks
    of 3 with max_tokens 100, the block size from the ranks' largest micro-batch (actor.py)."""
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    grads = []
    for k in (None, 3):
        torch.manual_seed(0)
        model = _Stack().double()
        for layer in model.layers:
            fully_shard(layer, mesh=init_device_mesh("cpu", (2,)), reshard_after_forward=True)
        fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
        torch.manual_seed(10 + rank)
        x = torch.randn(1, lens[rank], 16, dtype=torch.double)
        if k:
            tokens = torch.tensor(x.shape[1])
            dist.all_reduce(tokens, op=dist.ReduceOp.MAX)
            checkpoint_blocks(model, k, max_tokens=100)(int(tokens))
        model(x).square().mean().backward()
        grads.append({n: p.grad.full_tensor() for n, p in model.named_parameters()})
    out.put((rank, all(torch.equal(grads[1][n], g) for n, g in grads[0].items())))
    dist.destroy_process_group()


@pytest.mark.parametrize("lens", [(64, 80), (64, 160)])
def test_fsdp_ranks_agree_on_the_block_size(lens):
    """Ranks under and over max_tokens in the same step: one block size on both (else the FSDP all-gathers of the
    block recompute pair up out of order and the ranks hang), gradients bitwise per-layer."""
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = 29700 + lens[1]
    procs = [ctx.Process(target=_fsdp_rank, args=(r, lens, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(out.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=30)
    assert results == {0: True, 1: True}


def test_flat_micro_batch():
    model = _model()
    ids = torch.randint(0, 512, (1, 96))
    _assert_identical_and_smaller(model, dict(input_ids=ids, use_cache=False), slice(None, -1), ids[0, 1:], 96)


def _attention_call(q, k, v, call, softmax_scale=None):
    """prefix_tree.attention_call with torch's attention: FlashAttention varlen, causal aligned bottom-right."""
    _, keys, cu_q, cu_k, _, _ = call
    k, v = (x.index_select(0, keys).repeat_interleave(q.shape[1] // k.shape[1], dim=1) for x in (k, v))
    out = []
    for qs, qe, ks, ke in zip(cu_q.tolist(), cu_q.tolist()[1:], cu_k.tolist(), cu_k.tolist()[1:]):
        mask = torch.ones(qe - qs, ke - ks, dtype=torch.bool).tril(ke - ks - (qe - qs))
        out.append(F.scaled_dot_product_attention(q[qs:qe].transpose(0, 1), k[ks:ke].transpose(0, 1),
                                                  v[ks:ke].transpose(0, 1), attn_mask=mask,
                                                  scale=softmax_scale).transpose(0, 1))
    return torch.cat(out)


def _segmented(rule):
    """HF's torch delta rule made fla's varlen call: each cu_seqlens segment from its own initial state."""
    def chunk(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None, **kw):
        cu = cu_seqlens.tolist()
        outs = [rule(*(x[:, s:e] for x in (q, k, v, g, beta)), output_final_state=True,
                     initial_state=None if initial_state is None else initial_state[j:j + 1], **kw)
                for j, (s, e) in enumerate(zip(cu, cu[1:]))]
        return torch.cat([o for o, _ in outs], dim=1), torch.cat([s for _, s in outs])
    return chunk


def test_streamed_tree_micro_batch(monkeypatch):
    model = _model()
    layer_cls = type(model.model.layers[0])
    for cls, name in ((layer_cls, "forward"), (type(model.model.norm), "forward"),
                      (type(model.model.layers[0].linear_attn.norm), "forward"), (type(model.model.layers[0].mlp), "forward"),
                      (seq_tiles, "_modeling"), (seq_tiles, "_hf_layer_forward")):
        monkeypatch.setattr(cls, name, getattr(cls, name))  # tile_model patches these classes; restored after the test
    tile_model(model)
    monkeypatch.setattr(prefix_tree, "attention_call", _attention_call)
    monkeypatch.setattr(prefix_tree, "segment_conv", torch_segment_conv)  # fla's runs on CUDA only
    monkeypatch.setattr(modeling_qwen3_5, "torch_chunk_gated_delta_rule",
                        _segmented(modeling_qwen3_5.torch_chunk_gated_delta_rule))
    monkeypatch.setattr(seq_tiles, "TILE", 24)
    monkeypatch.setattr(prefix_tree, "SEG", 64)
    monkeypatch.setattr(prefix_tree, "CONV_CONTEXT", conv_context(model.config))  # what patch_model sets (it needs fla)
    g = torch.Generator().manual_seed(1)
    tok = lambda n: torch.randint(3, 512, (n,), generator=g)  # noqa: E731
    prefix = tok(150)
    seqs = [torch.cat([prefix, tok(200)]), torch.cat([prefix, tok(60), tok(140)]), torch.cat([prefix, tok(90)])]
    cu = torch.tensor([0, *torch.tensor([len(s) for s in seqs]).cumsum(0).tolist()], dtype=torch.int32)
    # as test_seq_tiles' tree: the second sequence reads the first one's own tokens from its token 210 on, then continues
    meta, kwargs = prefix_tree.tree_inputs(torch.cat(seqs)[None], cu, {1: [(210, 0, 150, 330)]})
    assert any(len(lv.ids) > 1 for lv in meta.levels) and any(len(lv.ctx_index) for lv in meta.levels)
    _assert_identical_and_smaller(model, kwargs, meta.pair_pos, meta.pair_label, kwargs["input_ids"].shape[1])
