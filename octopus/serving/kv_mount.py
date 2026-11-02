"""Attention-only mount of other cursors' tokens on SGLang /generate and /v1/chat/completions (docs/cursor-protocol.md
"Mount").

SGLang imports this file as sglang/srt/octopus_mount.py (a shim from patches/apply_sglang_mount.py). A cursor request
`{"cursor": B, "input_ids": delta, "mount": [{"cursor": A1, "start": s1, "end": e1}, ...]}` (on chat: delta =
render_delta of the messages) runs as the sequence

    tokens(B) + M1 placeholders + M2 placeholders + ... + delta,  Mi = ei - si

whose prefix rows are B's own (radix cache, through B's last token, with B's GDN state there), then each segment's
pages for Ai's tokens [si, ei) in request order (radix cache, FULL pages locked for the request), so the 8
full-attention layers (Qwen3.5) see B + A1[s1:e1] + A2[s2:e2] + ... + delta in prefill and every decode step. The GDN
layers continue from B's state and never see the mount: the mounted rows are a prefix the request does not run. A model
with only full-attention layers sees the mount in every layer. A segment's source rows may hold a mount of their own
(Ai continues a mount whose placeholders lie inside [si, ei)): the segment brings those rows too, as Ai's own rows saw
them. When B's KV ends short of its last token (always after a generated reply: that token has no KV yet) or a
segment's pages are gone, the scheduler aborts the request before it runs and the HTTP side prefills what is missing
(prefill-only requests: the same tokens, positions and mounts; numerics within bf16 prefill-split precision) and
retries. A prefill-only request snapshots its recurrent state at its
end (finish), so on a GDN/mamba model a rebuilt B has its state at B's last token under either mamba strategy
(no_buffer caches one at every sequence end anyway, extra_buffer otherwise only at state-chunk and 256-token decode
boundaries). Overlap scheduling may be on: SGLang launches every batch after the scheduler stream's work
(forward_stream.wait_stream) and fences result processing behind the batch in flight (_apply_war_barrier), which
orders the end-of-request bookkeeping; a prefill-only request stays out of the next decode batch (prefill_only), which
would step its state past its end. Sliding-window attention is refused (400).
Which attention layers see a mount is one setting, OCTOPUS_MOUNT_LAYERS (mount_layers): unset, every attention layer
(above); a list of decoder layer ids, only those, and every other attention layer attends to the request's own rows
alone (its row without the placeholder rows, layer_view), positions unchanged. The server reads it at launch and
reports it (stats, /get_server_info); the trainer's prefix tree reads the same variable and refuses servers that report
another value (octopus/rl/train.py), so inference and training run one token DAG.
Positions (K2, docs/kv-read-probe.md): mounted keys keep their source's positions; delta and output continue at
max(next position of B, last mounted position + 1 over all segments), a constant offset from the token index that the
cursor store keeps for every later cursor of B (`pos` breakpoints).

Zero-copy: the request's row holds the segments' own pages, and B's continuation (B + placeholders + delta + output) is
cached as an ordinary radix path whose placeholder entries are those pages, shared (lend): a shared page counts how
many more radix nodes hold it, and an evicted node frees only the pages no other node holds (owned), so a page is
freed once, by whichever of its holders leaves the tree last, in any order and across any radix namespace (LoRA
versions). Nothing stays locked once the request ends. Later requests from B's cursors reuse that path and keep seeing
the mount. A placeholder is a negative id unique to its segment, so no other request can match that path. Each breakpoint keeps the mount that
made it (where it sits, each segment's start, source tokens up to its end and source breakpoints), so whatever of a
mounted sequence is not resident (the cache kept it only in part, the path was evicted later) is rebuilt: the
request goes again carrying its last mount, and a sequence it needs that is missing (the tokens before that mount, or a
source's) is prefilled first, the same way (same tokens, positions and mounts; numerics within bf16 prefill-split
precision). The HTTP side and the scheduler side share this module.
"""

import asyncio
import contextlib
import copy
import itertools
import os
import uuid
from array import array

NO_CURSOR_KV = "octopus-mount: the cursor's KV is not resident through its last token"
NO_MOUNT_KV = "octopus-mount: the mounted pages are not resident"
EVICTED = "octopus-mount: the KV of a mounted read is evicted"
UNSUPPORTED_PREFIX = "octopus-mount: this server cannot mount:"
_PLACEHOLDERS = itertools.count(1)
UNSUPPORTED = None  # scheduler process: why it cannot mount (serve), None when it can
_REBUILDING, _ABORTED = {}, set()  # id -> GenerateReqInput of each mount request in generate; ids aborted meanwhile
_REF, _REF_ROOT = {}, None  # scheduler process: page slot -> how many more radix nodes hold it (lend, owned)


def mount_layers() -> frozenset | None:
    """OCTOPUS_MOUNT_LAYERS: comma-separated decoder layer ids whose attention reads mounted rows; None (unset or empty)
    for every attention layer."""
    value = os.environ.get("OCTOPUS_MOUNT_LAYERS", "")
    return frozenset(int(x) for x in value.split(",")) if value.strip() else None


MOUNT_LAYERS = mount_layers()


def offset_at(pos: tuple, i: int) -> int:
    """Position offset of token i under breakpoints ((index, offset, mount), ...)."""
    off = 0
    for j, o, _ in pos:
        if j <= i:
            off = o
    return off


# ---------------------------------------------------------------- HTTP process

def prepare(obj, cursors, tm) -> None:
    """A /generate or /v1/chat/completions request (obj: its GenerateReqInput; with obj.cursor, input_ids are the tokens
    after the cursor's): input_ids become the whole sequence and obj.octopus_pos the new cursor's position breakpoints,
    () without a cursor or a mount behind it. A breakpoint is (index, offset, mount): tokens from index on run at
    index + offset, and mount = (at, ((s, A's tokens up to e, A's breakpoints), ...)) is the mount whose placeholders
    end there, one entry per segment. Validation errors raise HTTPException (400; 404 for an unknown cursor)."""
    from fastapi import HTTPException

    obj.octopus_pos = ()
    if obj.cursor is None:
        if obj.mount is not None:
            raise HTTPException(status_code=400, detail="mount needs a cursor")
        return
    parent = cursors.get(obj.cursor)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"cursor {obj.cursor} is unknown or evicted")
    pos = cursors.pos.get(obj.cursor, ())
    n = len(parent)
    if obj.mount is None:
        if pos:  # B descends from a mount: its tokens hold placeholders and its positions an offset
            _params(obj)["octopus_pos"] = pos[-1][1]
        obj.input_ids, obj.octopus_pos = parent.tolist() + obj.input_ids, pos
        return
    if not isinstance(obj.mount, list) or not obj.mount or not obj.input_ids:
        raise HTTPException(status_code=400, detail="mount: a list of {cursor, start, end}, input_ids not empty")
    args = tm.server_args
    if args.disable_radix_cache or args.page_size not in (None, 1):
        raise HTTPException(status_code=400, detail="mount needs the radix cache and page size 1")
    ids, segs, nxt = parent.tolist(), [], n + offset_at(pos, n)
    for m in obj.mount:
        try:
            cursor, s, e = m["cursor"], int(m["start"]), int(m["end"])
            a = cursors.get(cursor)
        except (KeyError, TypeError, ValueError) as err:
            raise HTTPException(status_code=400, detail=f"mount entry {m!r}: {err!r}") from err
        if a is None:
            raise HTTPException(status_code=404, detail=f"mount cursor {cursor} is unknown or evicted")
        if not 0 <= s < e <= len(a):
            raise HTTPException(status_code=400, detail=f"mount range [{s}, {e}) is not inside cursor {cursor}")
        if e < len(a) and a[e] < 0 and a[e - 1] < 0:  # it would split a placeholder run, or cut between two
            raise HTTPException(status_code=400, detail=f"mount range [{s}, {e}) ends inside a mount of cursor {cursor}")
        a_pos = tuple(p for p in cursors.pos.get(cursor, ()) if p[0] < e)
        segs.append((s, a[:e], a_pos))
        ids += [-next(_PLACEHOLDERS)] * (e - s)
        nxt = max(nxt, e + offset_at(a_pos, e - 1))  # K2: after B and after every segment's last mounted key
    lp_start = obj.logprob_start_len
    if isinstance(lp_start, int) and 0 <= lp_start < len(ids):  # B's and the mounted rows are not run: no logits there
        raise HTTPException(status_code=400, detail=f"logprob_start_len {lp_start} is inside the cursor or the mount "
                                                    f"(the request's own tokens start at {len(ids)})")
    pos += ((len(ids), nxt - len(ids), (n, tuple(segs))),)
    _params(obj).update(_mount_params(pos[-1]))
    obj.input_ids, obj.octopus_pos = ids + obj.input_ids, pos


def _mount_params(breakpoint) -> dict:
    _, off, (at, segs) = breakpoint
    return {"octopus_pos": off, "octopus_mount_at": at, "octopus_mounts": [[s, a.tolist()] for s, a, _ in segs]}


def _params(obj) -> dict:
    obj.sampling_params = sp = dict(obj.sampling_params or {})
    sp["custom_params"] = cp = dict(sp.get("custom_params") or {})
    return cp


async def generate(tm, obj, request):
    """The outputs of a request prepared by prepare, as TokenizerManager.generate_request yields them (streaming or
    not): those of the attempt that runs. When the scheduler aborts an attempt because rows it needs are not resident,
    this rebuilds them and sends it again: B's KV short of B's last token (always, when B ended on a reply: that token
    has no KV yet) or holding placeholders that are gone: the tokens before its last mount are rebuilt; a segment's
    pages gone: its source's tokens are; a request from a mounted cursor whose rows past the mount are gone (the cache
    kept them only in part, or the path was evicted later) goes again carrying its last mount. B's and the sources'
    tokens are rebuilt together. A rebuild is prefill-only with the same tokens, positions and mounts as the first
    computation, its numerics within bf16 prefill-split precision of it (another split than the resident rows). The
    scheduler aborts an attempt before it runs anything, so all of this comes before the request's first output: a
    stream shows only the attempt that runs. Rebuilds stop once the client is gone (request: the HTTP request) or the
    request is aborted (abort: by rid or all), which then gets SGLang's abort answer."""
    if not obj.octopus_pos:
        async for out in tm.generate_request(obj, request):
            yield out
        return
    _REBUILDING[id(obj)] = obj
    try:
        first, rest = await _attempt(tm, obj, request, obj.octopus_pos, request, obj)
    finally:
        del _REBUILDING[id(obj)]
        _ABORTED.discard(id(obj))
    yield first
    if rest is not None:
        async for out in rest:
            yield out


async def _attempt(tm, obj, request, pos, client, root):
    """(first output, the rest) of the attempt of obj (breakpoints pos) that runs, after rebuilding what it misses;
    (SGLang's abort answer, None) once root (the request the rebuilds are for) is aborted. The scheduler's abort for
    missing rows is a stream's first output, and raised for a non-streaming request."""
    if not pos:
        outs = tm.generate_request(obj, request)
        return await outs.__anext__(), outs
    at, segs = pos[-1][2]
    cp, attempt, rid = obj.sampling_params["custom_params"], obj, obj.rid
    # ponytail: no retry cap. A retry follows rebuilds whose paths are the most recently used, so it fails again only
    # if one is evicted before the retry is matched. Near a full pool that repeats: slice 6 at --mem-fraction-static
    # 0.7 saw requests wait 17 min (docs/cross-branch-read.md); locking the rebuilt paths until the retry was admitted
    # made waits longer at the rollout config. A cap would turn the wait into a failed request.
    while id(root) not in _ABORTED:
        # a rid of its own (SGLang keeps per-rid state past an attempt's end), but a client's rid is kept: the response
        # id is the cursor key. obj carries it at once, so create_abort_task (a stream whose client is gone) and
        # abort() reach the attempt that runs.
        attempt = copy.copy(obj)
        attempt.rid = rid or uuid.uuid4().hex
        obj.rid, obj.is_single = attempt.rid, True
        outs = tm.generate_request(attempt, request)
        try:
            out = await outs.__anext__()
            reason = out["meta_info"].get("finish_reason") or {}
            missing = str(reason.get("message")) if reason.get("type") == "abort" else ""
        except ValueError as err:
            out, missing = err, str(err)
        if not any(m in missing for m in (NO_CURSOR_KV, NO_MOUNT_KV, EVICTED)):
            if isinstance(out, ValueError):
                raise out
            return out, outs
        # a stream's aborted attempt is a suspended generator: closed now, its cleanup (SGLang drops its rid's state)
        # runs before the retry, not when it is collected, by then possibly under the retry's rid (a client's)
        await outs.aclose()
        if client is not None and await client.is_disconnected():  # an agent timeout or abort: nobody waits
            raise ValueError(f"{missing} (the client disconnected: no rebuild)")
        if EVICTED in missing and "octopus_mount_at" not in cp:
            cp.update(_mount_params(pos[-1]))
        else:
            # all at once (a resident one costs a one-token prefill): rebuilt one after the other, the first was
            # often evicted while the second ran (slice 6: a request waited 20 min)
            await _rebuild(tm, obj.lora_path, [(obj.input_ids[:at], pos[:-1])] + [(a, a_pos) for _, a, a_pos in segs],
                           client, root)
    # what TokenizerManager._handle_abort_req answers for a request aborted before it ran
    meta = {"id": attempt.rid, "finish_reason": {"type": "abort", "message": "Abort in waiting queue"},
            "weight_version": tm.config_value("weight_version"), "completion_tokens": 0}
    if obj.return_logprob:
        meta.update(input_token_logprobs=[], output_token_logprobs=[], output_token_logprobs_length=0)
    return {"text": "", "output_ids": [], "meta_info": meta}, None


def prepare_chat(obj, request, cursors, tm) -> None:
    """prepare for /v1/chat/completions (evaluation sends here; RL reaches /generate through the AReaL proxy): obj, the
    request's GenerateReqInput (input_ids: render_delta of the messages after the cursor), takes the request's cursor and
    read_thread's mount {"from", "to", "thread"} as /generate's mount. The mounted tokens are the ones the AReaL proxy
    takes (its _read_range): delta_render.read_range in "to"'s tokens."""
    from fastapi import HTTPException

    from octopus.serving.delta_render import read_range

    obj.cursor, obj.mount = request.cursor, request.mount
    if request.mount is not None and request.cursor is not None:
        a, b = cursors.get(request.mount.get("from")), cursors.get(request.mount.get("to"))
        # a reply's turn is found in its prompt alone, as the proxy does in its input tokens: chat replies keep its
        # length (cursor_store.put_reply), /generate ones do not
        n, m = cursors.prompt.get(request.mount.get("from")), cursors.prompt.get(request.mount.get("to"))
        if a is None or b is None or n is None or m is None:
            raise HTTPException(status_code=404, detail=f"mount {request.mount} does not name stored chat replies")
        if len(a) > len(b) or b[:len(a)] != a:
            raise HTTPException(status_code=400, detail=f"mount {request.mount}: \"from\" is not \"to\" or before it")
        start, end = read_range(tm.tokenizer, request.mount, a[:n], b[:m], len(b))
        obj.mount = [{"cursor": request.mount["to"], "start": start, "end": end}]
    prepare(obj, cursors, tm)


async def chat(tm, obj, raw):
    """generate for /v1/chat/completions, streaming or not: as through the AReaL proxy, prompt_tokens leaves out the
    placeholders (the reader's own tokens)."""
    hidden = sum(i - at for i, _, (at, *_) in obj.octopus_pos)
    async for out in generate(tm, obj, raw):
        if hidden and "prompt_tokens" in out["meta_info"]:  # not on an abort answer
            out["meta_info"]["prompt_tokens"] -= hidden
        yield out


def abort(rid: str, abort_all: bool) -> None:
    """Start of TokenizerManager.abort_request: SGLang drops an abort for a rid it does not track, and a mount request
    has none while its rows are rebuilt (between its attempts); mark it, so its rebuild stops (abort_all: AReaL's
    pause_generation at a weight update)."""
    for key, obj in _REBUILDING.items():
        if abort_all or rid and isinstance(obj.rid, str) and obj.rid.startswith(rid):  # SGLang matches rid prefixes
            _ABORTED.add(key)


async def _rebuild(tm, lora_path, seqs, client, root):
    """Make the KV of each sequence (ids, breakpoints pos) of seqs resident by prefill-only requests: the ones without a
    mount as one batch request, so the scheduler receives them together (one message) and splits them into prefill
    batches by their lengths and the free pool (chunked-prefill and max-prefill budgets), no longer by arrival timing:
    on an idle server the same rebuild gets the same batches (sent one by one, with overlap off it ran in one batch or
    two by arrival: other prefill numerics, so another greedy reply for the same request on either endpoint). Under
    load other requests share those batches; with dp attention SGLang sends a batch's items one by one. One holding a
    mount carries its last mount (which rebuilds what that needs first) and still goes on its own, its batch by
    arrival. Nothing once the request they are for is aborted."""
    if id(root) in _ABORTED:  # e.g. an abort_all that came after the scheduler aborted the attempt for missing rows
        return
    from sglang.srt.managers.io_struct import GenerateReqInput

    plain = [list(ids) for ids, pos in seqs if not pos]
    todo = [anext(tm.generate_request(GenerateReqInput(input_ids=plain, lora_path=lora_path, sampling_params=[
        {"max_new_tokens": 0} for _ in plain]), None))] if plain else []
    # ponytail: a sequence holding a mount goes on its own (its attempt may be aborted for rows it misses, then
    # rebuilds and retries), so its batch composition still follows arrival; batch it too if that case must not
    # depend on arrival
    todo += [_attempt(tm, GenerateReqInput(input_ids=list(ids), lora_path=lora_path, sampling_params={
        "max_new_tokens": 0, "custom_params": _mount_params(pos[-1])}), None, pos, client, root)
        for ids, pos in seqs if pos]
    await asyncio.gather(*todo)


# ---------------------------------------------------------------- scheduler process

def serve(tree_cache, runner) -> None:
    """Scheduler init: a mount needs UnifiedRadixCache (its hooks; LMCache, FlexKV, the C++ tree and custom backends are
    other classes) and attention without a window: neither the model runner's (any model that has one: hybrid or pure
    SWA, chunked attention) nor an attention layer's own (models that set it only per layer); the window would hide
    mounted rows and the cache frees SWA pages outside it. Shared pages are only checked against the paths this module
    hooks: hierarchical cache (host copies, load-back), streaming sessions and the session radix cache (their own
    insert and lock paths), speculative decoding (draft pools, EAGLE's bigram keys) and PD disaggregation (KV transfer)
    are refused. A layer subset (MOUNT_LAYERS) must name attention layers of the model (else the server does not start)
    and needs the triton attention backend without deterministic inference (its unified extend kernel does not
    take layer_kv) or decode context parallelism (per-rank KV indices: layer_view's row offsets do not hold)."""
    global UNSUPPORTED
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if MOUNT_LAYERS is not None:
        attention = {m.layer_id for m in runner.model.modules() if isinstance(m, RadixAttention)}
        if not MOUNT_LAYERS <= attention:
            raise ValueError(f"OCTOPUS_MOUNT_LAYERS {sorted(MOUNT_LAYERS - attention)} are not attention layers of this "
                             f"model ({sorted(attention)})")
    window = runner.sliding_window_size
    if window is None:
        window = max((m.sliding_window_size for m in runner.model.modules() if isinstance(m, RadixAttention)), default=-1)
        window = None if window < 0 else window
    if window is not None:
        UNSUPPORTED = f"{UNSUPPORTED_PREFIX} sliding-window attention (window {window})"
    elif not isinstance(tree_cache, UnifiedRadixCache):
        UNSUPPORTED = f"{UNSUPPORTED_PREFIX} the radix cache {type(tree_cache).__name__} (needs UnifiedRadixCache)"
    else:
        args = runner.server_args
        off = [f for f, on in (("--enable-hierarchical-cache", args.enable_hierarchical_cache),
                               ("--enable-streaming-session", args.enable_streaming_session),
                               ("--enable-session-radix-cache", args.enable_session_radix_cache),
                               ("--speculative-algorithm", args.speculative_algorithm is not None),
                               ("--enable-unified-memory", args.enable_unified_memory),
                               ("--enable-hisparse", args.enable_hisparse),
                               ("--disaggregation-mode", args.disaggregation_mode != "null")) if on]
        if MOUNT_LAYERS is not None:
            backends = {runner.prefill_attention_backend_str, runner.decode_attention_backend_str}
            off += [f"OCTOPUS_MOUNT_LAYERS with {f}" for f, on in (
                (f"attention backends {sorted(backends)} (needs triton)", backends != {"triton"}),
                ("--enable-deterministic-inference", args.enable_deterministic_inference),
                ("--dcp-size > 1", args.dcp_size > 1)) if on]
        if off:
            UNSUPPORTED = f"{UNSUPPORTED_PREFIX} {', '.join(off)}"


def resolve(req, tree_cache) -> bool:
    """After Req.init_next_round_input matched the prefix: mount every segment's pages after B's prefix (all or none),
    set the position offset, and abort (returning True) a request whose prefill would contain placeholders. The tree
    owns the mounted rows, so the request's own KV starts after them (cache_protected_len): it never frees them."""
    cp = req.sampling_params.custom_params
    if not cp or "octopus_pos" not in cp or req.to_finish is not None:
        return False
    req.octopus_pos_offset, req.octopus_mount_hit = cp["octopus_pos"], None  # a queued request is re-matched each round
    if MOUNT_LAYERS is not None and not hasattr(req, "octopus_hidden"):  # its placeholder rows, for layer_view
        import numpy as np

        req.octopus_hidden = np.flatnonzero(np.asarray(req.origin_input_ids) < 0)
    if UNSUPPORTED is not None and "octopus_mount_at" in cp:  # the HTTP side answers 400 with this reason
        return _abort(req, UNSUPPORTED)
    n = len(req.prefix_indices)
    fill = req.full_untruncated_fill_ids
    if cp.get("octopus_mount_at") == n:
        import torch
        from sglang.srt.mem_cache.radix_cache import RadixKey

        rows, hits, at = [], [], n
        for s, toks in cp["octopus_mounts"]:
            idx, node = match_kv(tree_cache, RadixKey(array("q", toks), req.extra_key, cache_salt=req.cache_salt))
            if len(idx) < len(toks):
                break
            rows.append(idx[s:].to(req.prefix_indices.dtype))
            hits.append((node, len(toks) - s, rows[-1], len(toks)))  # source node, M, its pages, e
            at += len(toks) - s
        else:
            req.prefix_indices = torch.cat([req.prefix_indices, *rows])
            req.octopus_mount_hit, req.octopus_mount_at = hits, n
            n = req.kv.cache_protected_len = len(req.prefix_indices)
    if n < len(fill) and min(fill[n:]) < 0:
        at = cp.get("octopus_mount_at")
        return _abort(req, EVICTED if at is None or n < at and min(fill[n:at], default=0) < 0
                      else NO_CURSOR_KV if n < at else NO_MOUNT_KV)
    return False


def _abort(req, message: str) -> bool:
    req.octopus_pos_offset, logprob = 0, req.return_logprob  # the abort's one-token stand-in runs at position 0
    req.set_finish_with_abort(message)
    req.return_logprob = logprob  # the tokenizer manager still expects the logprob fields it asked for
    return True


def match_kv(cache, key):
    """UnifiedRadixCache: FULL KV pages of the longest device-resident prefix of key (no mamba state needed: the mount
    reads K/V only) and the id of its node, refreshed as recently used."""
    import torch
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    core, vals, actions = cache.tree_core, [], []
    node = core.root_node
    while len(key) and key.child_key(core.page_size) in node.children:
        child = node.children[key.child_key(core.page_size)]
        if child.evicted:
            break
        n = child.key.match(key, page_size=core.page_size)
        split = n < len(child.key)
        if split:
            child, action = core._split_node(child.key, child, n)
            actions += [action] if action is not None else []
        vals.append(child.component_data[ComponentType.FULL].value)
        node, key = child, key[n:]
        if split:
            break
    cache._apply_cache_actions(actions)
    walk = node
    while walk is not None:
        core._touch_node(walk)
        walk = walk.parent
    return (torch.cat(vals) if vals else torch.empty(0, dtype=torch.int64)), node.id


def _lock(tree, node):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    return tree.inc_lock_ref(node, skip_lock_components=(ComponentType.MAMBA,))  # FULL only: the mount reads K/V


def _unlock(tree, node, skip_ids) -> None:
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

    tree.dec_lock_ref(node, DecLockRefParams(skip_lock_node_ids=skip_ids))


@contextlib.contextmanager
def check_lock(adder, req):
    """Around PrefillAdder.add_one_req's budget check, which runs with the request's prefix locked: the segments' paths
    locked as well, so the check does not count them as evictable. Locked only after the check (lock), a mount request
    could over-commit the pool by those paths; then the batch's allocation fails (Out of memory) and the scheduler
    exits."""
    hits = getattr(req, "octopus_mount_hit", None)
    if not hits or getattr(req, "octopus_mount_node", None) is not None:
        yield
        return
    held = [(h[0], _lock(adder.tree_cache, h[0])) for h in hits]
    try:
        yield
    finally:
        for node, h in held:
            _unlock(adder.tree_cache, node, h.skip_lock_node_ids)


def lock(adder, req) -> None:
    """When the scheduler takes the request (PrefillAdder._req_inc_lock_ref): lock each segment's pages (FULL only, the
    node's mamba state is not read) until the request ends (lend)."""
    hits = getattr(req, "octopus_mount_hit", None)
    if hits and getattr(req, "octopus_mount_node", None) is None:
        req.octopus_mount_node = [(h, _lock(adder.tree_cache, h[0])) for h in hits]


def finish(cache, req, is_insert: bool, kv_len: int) -> None:
    """Start of UnifiedRadixCache.cache_finished_req (finish, abort and retraction all pass here): under the extra_buffer
    mamba strategy a prefill-only request (a rebuild) snapshots its recurrent state here."""
    if is_insert and cache.enable_mamba_extra_buffer and prefill_only([req]):
        snapshot(cache.req_to_token_pool, req, kv_len)


def lend(cache, req, result) -> None:
    """End of UnifiedRadixCache.cache_finished_req, after the insert (result: None without one): the rows the insert
    added of each segment are entries of the request's new path (their key: the segment's placeholders) holding the
    source's pages, which its nodes hold too. Each such page gains a holder, and the source nodes holding those rows
    are marked shared (owned checks them). The tree counts a shared page in every node holding it: it is taken off the
    evictable count once per added holder, and owned counts only the pages it frees, so available + evictable +
    protected stays the pool. Every segment is unlocked: nothing stays locked once the request ends."""
    held = getattr(req, "octopus_mount_node", None)
    if held is None:
        return
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    req.octopus_mount_node = req.octopus_mount_hit = None
    core, kept, old = cache.tree_core, 0, 0
    if result is not None and result.last_device_node is not None:
        old = result.prefix_len  # the path existed up to here; the insert added [old, kept)
        node = core.node_by_id(result.last_device_node)  # the end of the inserted path: its length is what was kept
        while node is not core.root_node:
            kept += len(node.key)
            node = node.parent
    refs, a = _refs(core), req.octopus_mount_at
    for (src, M, rows, e), lk in held:
        lo, hi = max(old, a) - a, min(kept, a + M) - a  # the segment's rows the insert added
        a += M
        if hi > lo:
            core.component_evictable_size_[ComponentType.FULL] -= hi - lo
            for slot in rows[lo:hi].tolist():
                refs[slot] = refs.get(slot, 0) + 1
            node, end = core.node_by_id(src), e  # the source path's nodes holding its tokens [e - M + lo, e - M + hi)
            while node is not core.root_node and end > e - M + lo:
                if end - len(node.key) < e - M + hi:
                    node.octopus_shared = True
                end -= len(node.key)
                node = node.parent
        _unlock(cache, src, lk.skip_lock_node_ids)


def split(new_parent, child) -> None:
    """FullComponent.redistribute_on_node_split: a shared node's front part may hold its shared pages."""
    if getattr(child, "octopus_shared", False):
        new_parent.octopus_shared = True


def owned(cache, node, value):
    """FullComponent.evict_component: the pages an evicted node frees and counts. A node holding shared pages (a
    source, marked by lend, or a continuation: placeholder keys) frees only those no other node holds; the rest lose a
    holder."""
    refs = _refs(cache.tree_core)
    if not refs or not (getattr(node, "octopus_shared", False) or min(node.key.raw_token_ids(), default=0) < 0):
        return value
    own = []
    # ponytail: one device-to-host read per evicted node that holds shared pages; a CPU mirror of node values if
    # eviction-heavy runs show it
    for slot in value.tolist():
        r = refs.get(slot, 0)
        if r == 0:
            own.append(slot)
        elif r == 1:
            del refs[slot]
        else:
            refs[slot] = r - 1
    return value if len(own) == len(value) else value.new_tensor(own)


def _refs(core) -> dict:
    """The shared pages of this tree; a flush builds a new root (and empties the pool): they go with it."""
    global _REF_ROOT
    if _REF_ROOT is not core.root_node:
        _REF.clear()
        _REF_ROOT = core.root_node
    return _REF


def stats(cache) -> dict:
    """Scheduler.get_internal_state: the layers that read a mount (MOUNT_LAYERS, None for all: the trainer checks it)
    and, for leak and page checks, locked tokens (0 when idle), free slots and the extra holders of shared pages."""
    out = {"octopus_mount_layers": None if MOUNT_LAYERS is None else sorted(MOUNT_LAYERS)}
    if not hasattr(cache, "full_protected_size"):
        return out
    out.update(octopus_full_protected=cache.full_protected_size(),
               octopus_available=cache.token_to_kv_pool_allocator.available_size())
    if hasattr(cache, "tree_core"):
        out["octopus_shared_holds"] = sum(_refs(cache.tree_core).values())
    return out


def snapshot(pool, req, kv_len: int) -> None:
    """extra_buffer keeps a finished request's KV only up to its last tracked recurrent state (a chunk or interval
    boundary) and caches that state: track the live state at kv_len instead (copied into the request's next track
    slot, the slot SGLang itself tracks into), so the whole sequence is cached with its state at its end. The live
    state is the state at kv_len: the request is prefill-only and stayed out of the next decode batch
    (prefill_only), and result processing runs after its batch."""
    i = req.kv.mamba_next_track_idx
    t = pool.translate_mamba_indices
    pool.mamba_pool.copy_from(t(req.kv.mamba_pool_idx[None]), t(req.kv.mamba_ping_pong_track_buffer[i:i + 1]))
    req.kv.mamba_last_track_idx, req.kv.mamba_last_track_seqlen = i, kv_len


def prefill_only(reqs) -> list:
    """Scheduler.get_next_batch_to_run, before the last (prefill) batch merges into the running batch: its prefill-only
    requests stay out. They finish after their prefill, but under overlap that is known only once its results are
    processed, after the next decode batch is launched, which would step their live state past their end."""
    return [r for r in reqs if r.sampling_params.max_new_tokens == 0]


def unfinished(cache, req) -> bool:
    """Start of UnifiedRadixCache.cache_unfinished_req (after a prefill or a chunk): while the row holds the segments'
    pages under the request's lock nothing of it is inserted; the next chunk's prefix is the row, as on the cache's
    disabled path."""
    if getattr(req, "octopus_mount_node", None) is None:
        return False
    import torch

    n = len(req.get_fill_ids())
    req.prefix_indices = cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n].to(dtype=torch.int64, copy=True)
    return True


def hide_mounts(ret, batch) -> None:
    """ForwardBatch.init_new, under a layer subset: each request's placeholder rows (resolve) when any request of the
    batch holds some (ForwardBatch.octopus_hidden, a field: the runners pass dataclasses.replace copies), for layer_view;
    such a batch runs without CUDA graphs (ModelRunner.forward)."""
    hidden = [getattr(r, "octopus_hidden", None) for r in batch.reqs] if MOUNT_LAYERS is not None else ()
    if any(h is not None and len(h) for h in hidden):
        ret.octopus_hidden = hidden


def layer_view(backend, forward_batch) -> None:
    """End of TritonAttnBackend.init_forward_metadata: a second view of the batch's KV rows (prefix rows in extend, all
    rows in decode) without the placeholder rows, which the attention layers outside MOUNT_LAYERS read (layer_kv): the
    request's own rows, at their own positions. Batches with no mounted rows (hide_mounts) keep one view."""
    hidden = getattr(forward_batch, "octopus_hidden", None)
    meta = backend.forward_metadata
    if hidden is None:
        return
    import numpy as np
    import torch

    # ponytail: one host read of the indptr and a boolean gather per forward of such a batch (eager); CUDA-graph
    # buffers for the second view if subset throughput matters
    ptr = meta.kv_indptr.tolist()
    hide = np.concatenate([ptr[i] + h[h < ptr[i + 1] - ptr[i]] for i, h in enumerate(hidden) if h is not None])
    keep = torch.ones(ptr[-1], dtype=torch.bool, device=meta.kv_indices.device)
    keep[torch.from_numpy(hide).to(keep.device)] = False
    kept = torch.cat([keep.new_zeros(1, dtype=torch.int64), keep.cumsum(0)])
    meta.octopus_kv = kept[meta.kv_indptr.long()].to(meta.kv_indptr.dtype), meta.kv_indices[:ptr[-1]][keep]


def layer_kv(meta, layer, kv_indptr, kv_indices):
    """TritonAttnBackend.forward_extend / forward_decode: the KV view of this attention layer (layer_view)."""
    view = getattr(meta, "octopus_kv", None)
    return view if view is not None and layer.layer_id not in MOUNT_LAYERS else (kv_indptr, kv_indices)


def offset_positions(ret, batch) -> None:
    """ForwardBatch.init_new: add each request's position offset to its tokens' positions (and M-RoPE positions)."""
    offs = [getattr(r, "octopus_pos_offset", 0) for r in batch.reqs]
    if not any(offs):
        return
    import torch

    o = torch.tensor(offs, dtype=ret.positions.dtype, device=ret.positions.device)
    if not ret.forward_mode.is_decode():
        o = torch.repeat_interleave(o, ret.extend_seq_lens, output_size=ret.extend_num_tokens)
    ret.positions = ret.positions + o
    if ret.mrope_positions is not None:
        ret.mrope_positions = ret.mrope_positions + o.to(ret.mrope_positions.dtype)
