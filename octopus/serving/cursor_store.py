"""SGLang's cursor store (docs/cursor-protocol.md): the token ids of finished requests by response id.

SGLang imports this file as sglang/srt/octopus_cursor_store.py (a shim, patches/apply_sglang_cursor.py). Each cursor holds
its whole sequence as one array('i') (4 bytes a token): a request from it is one dict lookup and one C-level copy, and
eviction frees exactly the bytes it counts. Parent-linked chunks would store a thread in O(L) instead of O(n·L), but an
evicted parent would stay alive under its children (the budget would no longer be the memory) and a lookup would walk
the chain. Full arrays are small enough: the recorded bundle-e11 episode (121 requests, 21k tokens at most) stores
6 MiB, so the default 2 GiB holds ~340 such episodes. Least recently used cursors go first; a thread continues from
its newest one, so what is evicted is old rewind points and finished threads.

A cursor that continues a mounted read (patches/apply_sglang_mount.py, docs/cursor-protocol.md "Mount") also keeps
its position breakpoints in `pos`: ((index, offset, mount), ...), token i at position i + the offset of the last
breakpoint at or before i; mount (the source's tokens up to the mount's end, ...) lets the server rebuild the mounted
rows. A thread's later cursors share these objects; the budget does not count them.

A chat cursor also keeps `ctx`, the stub render_delta builds for templates that read the history (delta_render.py
context_stub: the thread's system message, shared by its cursors, and the cursor's reply); the budget does not count it.
And `prompt`, its prompt's length: where read_thread's mount finds the reply's turn (kv_mount.prepare_chat), in the
prompt only, as the AReaL proxy does in its input tokens.
"""

import os
from array import array
from collections import OrderedDict


class CursorStore:
    def __init__(self, budget_bytes: int):
        self.budget, self.used, self.seqs, self.pos, self.ctx, self.prompt = budget_bytes, 0, OrderedDict(), {}, {}, {}

    def get(self, cursor: str) -> array | None:
        seq = self.seqs.get(cursor)
        if seq is not None:
            self.seqs.move_to_end(cursor)
        return seq

    def put(self, cursor: str, tokens: list[int], pos: tuple = (), ctx: list[dict] | None = None,
            prompt: int | None = None) -> None:
        old = self.seqs.pop(cursor, None)
        if old is not None:
            self.used -= nbytes(old)
        self.seqs[cursor] = seq = array("i", tokens)
        self.used += nbytes(seq)
        if pos:
            self.pos[cursor] = pos
        else:
            self.pos.pop(cursor, None)
        if ctx:
            self.ctx[cursor] = ctx
        else:
            self.ctx.pop(cursor, None)
        if prompt is not None:
            self.prompt[cursor] = prompt
        else:
            self.prompt.pop(cursor, None)
        while self.used > self.budget:
            gone, seq = self.seqs.popitem(last=False)
            self.used -= nbytes(seq)
            self.pos.pop(gone, None)
            self.ctx.pop(gone, None)
            self.prompt.pop(gone, None)

    def put_reply(self, out: dict, eos: int, pad: int | None, end: list[int], ctx: list[dict] | None = None,
                  pos: tuple = ()) -> None:
        """A chat reply (a finished TokenizerManager output with prompt_token_ids): AReaL concat's parent tokens, the
        prompt + the output without its trailing stop tokens (eos, pad, the stop it matched, e.g. GLM's <|observation|>;
        kept when cut by length or aborted) + end, the reply's terminator (delta_render.terminator); ctx its stub, pos
        its position breakpoints (a mount, kv_mount.prepare)."""
        ids, n, finish = out["output_ids"], len(out["output_ids"]), out["meta_info"]["finish_reason"]
        if finish["type"] not in ("length", "abort"):
            while n and ids[n - 1] in (eos, pad, finish.get("matched")):
                n -= 1
        prompt = out["prompt_token_ids"]
        self.put(out["meta_info"]["id"], prompt + ids[:n] + end, pos, ctx, len(prompt))


def nbytes(seq: array) -> int:
    return len(seq) * seq.itemsize


CURSORS = CursorStore(int(os.environ.get("OCTOPUS_CURSOR_BYTES", 2 << 30)))
