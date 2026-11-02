"""Chunked LM head: each position's (logprob of its next token, entropy) without materialising the [tokens x vocab]
logits (80k tokens x 248k vocab would need ~40 GB). Bound to the model's lm_head module as its forward; the labels and
scored positions (flat) and the prefix tree (tree path) are set on the module per call; bind_head installs it.
Only scored positions run the head (actor.scored_positions: those AReaL's loss_mask trains, not the prompt or tool
results); the others' output is 0."""

from __future__ import annotations

import types

import torch
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import MoeModelOutputWithPast

from octopus.rl.pinned_slab import PinnedSlab
from octopus.rl.prefix_tree import tree_head_output
from octopus.rl.seq_tiles import _fused

CHUNK = 2048


_LOGITS = object()  # stands for the logits in a recorded op


class _Tail(torch.Tensor):
    """Logits as a model's forward handles them after its lm_head. While probing (chain = {ops, last}) every op on them
    is recorded, each on the previous op's result; on the bound head's output (skip = the number recorded) those ops
    leave that output unchanged."""
    chain = None
    skip = 0
    out = None

    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        tail = next(a for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, _Tail))
        with torch._C.DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)
        chain = tail.chain
        if chain is not None and (result is None or tail is not chain["last"]):
            raise NotImplementedError(f"post-head logits: {getattr(func, '__name__', func)} "
                                      f"{'in place' if result is None else 'on an earlier result'} cannot be replayed")
        if not isinstance(result, torch.Tensor):  # shape, dtype, ...
            return result
        if chain is None:
            return _skipping(tail.out, tail.skip - 1)
        chain["ops"].append(pytree.tree_map(lambda a: _LOGITS if isinstance(a, _Tail) else a, (func, args, kwargs)))
        result = result.as_subclass(_Tail)
        result.chain = chain
        chain["last"] = result
        return result


def _skipping(out: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return out
    tail = out.as_subclass(_Tail)
    tail.out, tail.skip = out, n
    return tail


def _post_head(ops: list, logits: torch.Tensor) -> torch.Tensor:
    for func, args, kwargs in ops:
        args, kwargs = pytree.tree_map(lambda a: logits if a is _LOGITS else a, (args, kwargs))
        logits = func(*args, **kwargs)
    return logits


def _post_head_ops(model, head) -> list:
    """The ops the model's own forward applies to its lm_head output (granite's / logits_scaling, Falcon-H1's
    * lm_head_multiplier, Gemma's final_logit_softcapping, ...): its forward run once with its other submodules
    stubbed (zero hidden states) and the head returning a recording tensor. Raises unless the returned logits are the
    end of one chain of ops on the head's output."""
    lm = next(m for m in model.modules() if any(c is head for c in m.children()))
    chain: dict = {"ops": [], "last": None}

    def logits(*_, **__):
        tail = torch.zeros(1, 1, head.weight.shape[0], dtype=head.weight.dtype).as_subclass(_Tail)
        tail.chain = chain
        chain["last"] = tail
        return tail
    hidden = MoeModelOutputWithPast(last_hidden_state=torch.zeros(1, 1, head.weight.shape[1], dtype=head.weight.dtype))
    stubs = {c: (lambda *_, **__: hidden) for c in lm.children() if c is not head} | {head: logits}
    saved = {c: vars(c).get("forward") for c in stubs}
    try:
        for c, f in stubs.items():
            c.forward = f
        out = type(lm).forward(lm, input_ids=torch.zeros(1, 1, dtype=torch.long), use_cache=False)
    finally:
        for c, f in saved.items():
            if f is None:
                del c.forward
            else:
                c.forward = f
    if chain["last"] is None or out.logits is not chain["last"]:
        raise NotImplementedError(f"{type(lm).__name__}: its forward's logits are not its lm_head output after one chain "
                                  "of ops; the chunked head cannot reproduce them")
    return chain["ops"]


def _head_chunk(weight: torch.Tensor, temperature: float, post: list, hidden: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    # bf16 matmul (cuBLAS), the model's post-head ops, fused fp32 softmax
    return _head_columns(_post_head(post, hidden @ weight.T), temperature, labels)


@_fused
def _head_columns(logits: torch.Tensor, temperature: float, labels: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)  # fp32 softmax (as SGLang)
    entropy = -(logp.exp() * logp).sum(-1).detach()
    return torch.stack([logp.gather(-1, labels[:, None])[:, 0], entropy], dim=-1)


def _chunked_rows(self, rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """[n, H] hidden rows, [n] labels -> [n, 2] (logprob, entropy), CHUNK rows at a time (recomputed in backward).
    self.weight is read when each chunk runs (forward and backward recompute), after FSDP unshards it."""
    return torch.cat([checkpoint(lambda h, y: _head_chunk(self.weight, self.octopus_temperature, self.octopus_post,
                                                          h, y),
                                 h, y, use_reentrant=False)  # split(): one gradient concat, no per-chunk zero tensor
                      for h, y in zip(rows.split(CHUNK), labels.split(CHUNK))])


def _scored_rows(self, rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """_chunked_rows at the positions self.octopus_scored marks ([n] bool, None: all), 0 at the others."""
    if self.octopus_scored is None:
        return _chunked_rows(self, rows, labels)
    at = self.octopus_scored.nonzero()[:, 0]
    out = _chunked_rows(self, rows.index_select(0, at), labels.index_select(0, at))
    return out.new_zeros(len(rows), out.shape[1]).index_copy(0, at, out)


def _chunked_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
    """[1, T, H] hidden states -> [1, T_flat, 2] (logprob of each flat position's next token, entropy). The model's
    forward then applies its post-head ops to this output: they leave it as it is (they ran on each chunk's logits).
    The tree path scores the pairs its tree kept (prefix_tree.build_tree scored)."""
    if self.octopus_tree is not None:  # hidden is over the prefix tree; expand to the flat layout
        out = tree_head_output(lambda rows, y: _chunked_rows(self, rows, y), hidden, self.octopus_tree)
    else:
        out = _scored_rows(self, hidden.reshape(-1, hidden.shape[-1]), self.octopus_labels).reshape(
            *hidden.shape[:-1], 2)
    return _skipping(out, len(self.octopus_post))


def bind_head(model, temperature: float):
    """Install the chunked head as the model's lm_head forward at this temperature, with the post-head ops of the
    model's own forward applied to each chunk's logits before the softmax. Returns the head module."""
    head = model.get_output_embeddings()
    head.octopus_post = _post_head_ops(model, head)
    head.octopus_temperature = temperature
    head.octopus_scored = None
    head.forward = types.MethodType(_chunked_head_forward, head)
    return head


def check_checkpointing(model) -> None:
    """Activations go to pinned host memory (save_on_cpu), which cannot be swapped: without layer checkpointing a
    90k-token micro-batch pins every intermediate tensor and starves the whole machine (node-94, 2026-10-01).
    HF checkpoints layers only when gradient checkpointing is enabled and the model is in train() mode."""
    if not (model.training and getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("save_on_cpu without layer checkpointing would pin every activation in host memory")


_host = PinnedSlab()  # the page-locked buffer the saved activations of each micro-batch reuse


def host_activations(model, gpu_bytes: float = 0):
    """The saved-tensor hooks one micro-batch's forward runs in: what save_on_cpu does, but into one page-locked
    buffer kept across micro-batches (octopus/rl/pinned_slab.py; release_host_activations frees it between train
    calls). Each micro-batch's backward has run (and its host-to-device copies were queued on the same stream) before
    the next forward overwrites the buffer. The first saved tensors that fit in gpu_bytes (actor.py ACT_GPU_GB) stay
    on the GPU instead, as without the hooks: spare card memory replaces host memory, gradients unchanged."""
    check_checkpointing(model)
    _host.start()
    kept = 0

    def pack(t):
        nonlocal kept
        if not t.is_cuda:
            return t.device, t
        n = t.numel() * t.element_size()
        if kept + n <= gpu_bytes:
            kept += n
            return t.device, t
        host = _host.take(t.shape, t.dtype)
        host.copy_(t)
        return t.device, host

    def unpack(packed):
        device, host = packed
        return host.to(device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def release_host_activations() -> None:
    """End of a train call (actor.py ppo_update): the buffer and torch's pinned cache go back to the OS, so the pinned
    memory is not held between steps (PinnedSlab.release; 484 of 503 GB host RAM in use on node-138 without)."""
    _host.release()
