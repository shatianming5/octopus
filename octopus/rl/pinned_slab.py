"""Page-locked host memory reused across micro-batches (lm_head.host_activations' saved activations, chunk_major's kept
tensors). Pinning costs ~0.5 s/GB, ~20 s per 255k-token micro-batch when each one pinned afresh, and torch's pinned
cache rounds every block up to a power of two and only reuses a block for the same size (first 3-rank bundle step:
77-109 GB per rank). A slab is pinned at exactly the size the largest micro-batch so far carved (cudaHostRegister, not
rounded); a micro-batch that outgrows it takes the rest from torch's pinned cache once, and the slab grows to its size
when the next one starts."""

from __future__ import annotations

import math

import torch


class PinnedSlab:
    def __init__(self):
        self.buf = torch.empty(0, dtype=torch.uint8)
        self.need = 0  # bytes the current (or last) micro-batch carved

    def start(self) -> None:
        """A micro-batch starts carving at the slab's start, the slab first grown to the last micro-batch's need. The
        previous micro-batch's copies out of it were queued on the same stream, before this one's."""
        torch._C._host_emptyCache()  # blocks an outgrown micro-batch took from torch's pinned cache
        if self.need > self.buf.numel():
            self._unpin()
            self.buf = torch.empty(self.need, dtype=torch.uint8)
            _check(torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), self.need, 0), "cudaHostRegister")
        self.need = 0

    def take(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Pinned room for a [shape] dtype tensor: the slab's next bytes, past its end torch's pinned cache."""
        n = math.prod(shape) * dtype.itemsize
        start, self.need = self.need, self.need + (n + 511) // 512 * 512
        if self.need > self.buf.numel():
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        return self.buf[start:start + n].view(dtype).view(shape)

    def release(self) -> None:
        """End of a train call: hand the slab and torch's pinned cache back to the OS (search-v9-rest on node-138:
        ~100 GB per rank held after one 665k-token micro-batch). The next call's first micro-batch pins it again at the
        size it had (or the last micro-batch's, if larger): sized to the last micro-batch alone, often a small padding
        dummy, the larger ones would take torch's power-of-two pinned blocks every step."""
        self.need = max(self.need, self.buf.numel())
        self._unpin()
        torch._C._host_emptyCache()

    def _unpin(self) -> None:
        """Unpin and free once the copies queued out of the slab ran: freeing is not ordered on the stream."""
        if self.buf.numel():
            torch.cuda.synchronize()
            _check(torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr()), "cudaHostUnregister")
            self.buf = torch.empty(0, dtype=torch.uint8)


def _check(error, call: str) -> None:
    if error is not None and int(getattr(error, "value", error)) != 0:
        raise RuntimeError(f"{call} failed: {error}")
