"""lm_head.host_activations: the GPU share of the saved activations changes where they wait, not the gradients."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from octopus.rl import actor, lm_head


def test_budget_shrinks_as_micro_batches_grow(monkeypatch):
    monkeypatch.setattr(actor, "ACT_GPU_GB", 18.0)
    monkeypatch.setattr(actor, "ACT_GPU_TARGET_GB", 40.0)
    assert actor.act_gpu_budget(65536) == 18.0  # small: the cap
    assert 12 < actor.act_gpu_budget(229376) < 14  # bundle-e9's OOM at a fixed 18
    assert 4.5 < actor.act_gpu_budget(319604) < 5.5
    assert actor.act_gpu_budget(400000) == 0.0
    monkeypatch.setattr(actor, "ACT_GPU_GB", 0.0)  # off
    assert actor.act_gpu_budget(65536) == 0.0


class Stack(torch.nn.Module):
    is_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(256, 256) for _ in range(8))

    def forward(self, x):
        for layer in self.layers:  # each checkpoint saves its input: 8 saved tensors of x's size
            x = checkpoint(lambda h, f=layer: torch.tanh(f(h)), x, use_reentrant=False)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_share_keeps_gradients_and_leaves_nothing_behind():
    torch.manual_seed(0)
    model = Stack().cuda().train()
    x = torch.randn(4096, 256, device="cuda")
    size = x.numel() * x.element_size()
    model(x).sum().backward()  # cuBLAS workspace and the gradients' first allocation
    grads = {}
    for budget, offloaded in ((0, 8), (3 * size, 5), (1e12, 0), (0, 8)):
        model.zero_grad(set_to_none=True)
        before = torch.cuda.memory_allocated()
        with lm_head.host_activations(model, budget):
            y = model(x)
        assert lm_head._host.need == offloaded * size
        # y, and the kept inputs past x itself (already allocated)
        assert torch.cuda.memory_allocated() - before == (1 + max(0, 7 - offloaded)) * size
        y.square().sum().backward()
        del y
        grads[budget] = [p.grad.cpu() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        assert torch.cuda.memory_allocated() == before
    for budget in (3 * size, 1e12):
        assert all(torch.equal(a, b) for a, b in zip(grads[0], grads[budget]))


class _Cudart:
    def __init__(self):
        self.registered, self.log = {}, []

    def cudaHostRegister(self, ptr, size, flags):
        self.registered[ptr] = size

    def cudaHostUnregister(self, ptr):
        self.log.append("unregister")
        del self.registered[ptr]


def test_release_frees_the_buffer_between_steps(monkeypatch):
    """CPU: the end of a train call unpins and frees the buffer, after a synchronize (as the grow path does); the next
    call's first micro-batch pins it again at its size, or at the last micro-batch's if that is larger."""
    cudart = _Cudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: cudart.log.append("sync"))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: None)
    monkeypatch.setattr(lm_head, "_host", lm_head.PinnedSlab())
    lm_head._host.need = 1 << 20  # a step's micro-batch saved 1 MiB
    model = Stack().train()

    def micro_batch(saved):  # host_activations as a micro-batch enters it; returns the pinned size
        with lm_head.host_activations(model):
            pass
        assert cudart.registered == {lm_head._host.buf.data_ptr(): lm_head._host.buf.numel()}
        lm_head._host.need = saved
        return lm_head._host.buf.numel()

    assert micro_batch(1 << 21) == 1 << 20
    assert micro_batch(1 << 10) == 1 << 21  # grown: the 1 MiB buffer unpinned after a synchronize
    assert cudart.log == ["sync", "unregister"]
    lm_head.release_host_activations()  # end of ppo_update; the last micro-batch (a dummy) saved 1 KiB
    assert lm_head._host.buf.numel() == 0 and cudart.registered == {} and cudart.log[2:] == ["sync", "unregister"]
    lm_head.release_host_activations()  # nothing pinned: a no-op
    assert len(cudart.log) == 4
    assert micro_batch(1 << 22) == 1 << 21  # the next step's first micro-batch: the buffer's size, not the dummy's
    lm_head.release_host_activations()
    assert micro_batch(0) == 1 << 22  # the last micro-batch outgrew the buffer: its size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_release_between_steps_keeps_gradients():
    """Two steps with the buffer released between them give the same gradients as two steps with it kept. p.grad.cpu()
    synchronizes before each release, so this does not show that release's own synchronize is needed (that rests on
    freeing not being ordered on the stream)."""
    torch.manual_seed(0)
    model = Stack().cuda().train()
    x = torch.randn(4096, 256, device="cuda")
    grads = []
    for release in (False, True):
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            with lm_head.host_activations(model):
                y = model(x)
            y.square().sum().backward()
            grads.append([p.grad.cpu() for p in model.parameters()])
            if release:
                lm_head.release_host_activations()
                assert lm_head._host.buf.numel() == 0
    lm_head.release_host_activations()
    assert all(torch.equal(a, b) for step in grads[1:] for a, b in zip(grads[0], step))
