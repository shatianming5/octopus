"""Attention-only mount on SGLang /generate (patches/apply_sglang_mount.py, octopus/serving/kv_mount.py).

CPU: position math anywhere (fastapi installed); the patch test needs an unpatched SGLang 0.5.19 importable, the HTTP
test the patched one (PYTHONPATH=<mount overlay>:<this repo>). Live: a patched server (OCTOPUS_SGLANG_URL; either mamba
strategy, overlap on or off) serving Qwen3.5-4B-text with the mount points of the #129 probe (OCTOPUS_MOUNT_DATA,
written by `python tests/test_sglang_mount.py prep ...` on node-94, see main()), a dense model (OCTOPUS_DENSE_MODEL) or
a hybrid one (OCTOPUS_HYBRID_MODEL), each against its transformers reference.
"""

import asyncio
import copy
import gzip
import json
import os
import random
import shutil
import statistics
import sys
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from octopus.serving.cursor_store import CursorStore  # noqa: E402
from octopus.serving.kv_mount import EVICTED, NO_CURSOR_KV, NO_MOUNT_KV, offset_at  # noqa: E402
from test_delta_render import tok  # noqa: E402,F401  (fixture)

ARGS = SimpleNamespace(disable_radix_cache=False, page_size=None)
TM = SimpleNamespace(server_args=ARGS)


def run_prepare(obj, store):
    """kv_mount.prepare: the request's whole input ids and its breakpoints."""
    from octopus.serving.kv_mount import prepare
    prepare(obj, store, TM)
    return obj.input_ids, obj.octopus_pos


def test_offset_at():
    assert offset_at((), 5) == 0
    pos = ((10, -3, None), (20, 4, None))
    assert [offset_at(pos, i) for i in (9, 10, 19, 20, 99)] == [0, -3, -3, 4, 4]


def test_prepare_k2_positions_and_placeholders():
    """B continues at max(B's next position, last mounted position + 1); the new cursor keeps that offset; a later
    request from it carries the offset; a mount of a cursor that has its own offset uses A's positions; several
    segments in one request each get their own placeholder id, and B continues after the latest of them."""
    pytest.importorskip("fastapi")
    from octopus.serving.kv_mount import prepare
    store = CursorStore(1 << 20)
    store.put("B", list(range(100)))
    store.put("A", list(range(1000, 1300)))
    req = lambda **kw: SimpleNamespace(**{"sampling_params": None, "lora_path": None,  # noqa: E731
                                          "logprob_start_len": -1, **kw})
    for (s, e), nxt in (((50, 80), 100), ((250, 300), 300)):  # peer behind: B's position; ahead: after the mount
        obj = req(cursor="B", input_ids=[7, 8], mount=[{"cursor": "A", "start": s, "end": e}], logprob_start_len=100 + e - s)
        ids, pos = run_prepare(obj, store)
        M, ph = e - s, ids[100]
        assert ids == list(range(100)) + [ph] * M + [7, 8] and ph < 0
        # delta token i (index >= 100 + M) at i + offset: first at nxt; the breakpoint keeps its mount (rebuilds)
        assert pos == ((100 + M, nxt - 100 - M, (100, ((s, array("i", range(1000, 1000 + e)), ()),))),)
        cp = obj.sampling_params["custom_params"]
        assert cp == {"octopus_pos": nxt - 100 - M, "octopus_mount_at": 100,
                      "octopus_mounts": [[s, list(range(1000, 1000 + e))]]}
    store.put("B2", ids + [9], pos)  # a later request from B's new cursor: same offset, no mount
    obj = req(cursor="B2", input_ids=[5], mount=None)
    ids2, pos2 = run_prepare(obj, store)
    assert ids2 == ids + [9, 5] and pos2 == pos
    assert obj.sampling_params["custom_params"] == {"octopus_pos": pos[-1][1]}
    obj = req(cursor="B", input_ids=[1], mount=[{"cursor": "B2", "start": 150, "end": 153}])  # A = B2: offset
    _, pos3 = run_prepare(obj, store)
    assert pos3[0][:2] == (103, 153 + pos[-1][1] - 103)  # B2's last mounted key sits at 152 + its offset
    assert pos3[0][2][1][0][2] == pos  # B2's breakpoints before the mount's end: its own mount, to rebuild B2's rows
    from fastapi import HTTPException
    for end in (120, 149):  # ends inside B2's placeholders: would split that run in the tree; 100 and 150 are its edges
        with pytest.raises(HTTPException) as err:
            prepare(req(cursor="B", input_ids=[1], mount=[{"cursor": "B2", "start": 0, "end": end}]), store, TM)
        assert err.value.status_code == 400 and "inside a mount" in err.value.detail
    for end in (100, 150):
        prepare(req(cursor="B", input_ids=[1], mount=[{"cursor": "B2", "start": 0, "end": end}]), store, TM)
    obj = req(cursor=None, input_ids=[1], mount=None)  # no cursor: today's request
    prepare(obj, store, TM)
    assert obj.input_ids == [1] and obj.octopus_pos == () and obj.sampling_params is None
    obj = req(cursor="B", input_ids=[1], mount=[{"cursor": "A", "start": 10, "end": 20},
                                                {"cursor": "B2", "start": 150, "end": 153}])  # two segments
    ids4, pos4 = run_prepare(obj, store)
    assert ids4[100:110] == [ids4[100]] * 10 and ids4[110:113] == [ids4[110]] * 3 and ids4[100] != ids4[110] < 0
    assert ids4[113:] == [1] and pos4[0][:2] == (113, 153 + pos[-1][1] - 113)  # B2's segment ends at 302: the latest
    assert pos4[0][2] == (100, ((10, array("i", range(1000, 1020)), ()), (150, store.get("B2")[:153], pos)))
    assert obj.sampling_params["custom_params"]["octopus_mounts"] == [[10, list(range(1000, 1020))],
                                                                      [150, store.get("B2")[:153].tolist()]]


@pytest.mark.parametrize("bad, code", [
    ({"mount": [{"cursor": "A", "start": 0, "end": 301}]}, 400), ({"mount": [{"cursor": "A", "start": 5, "end": 5}]}, 400),
    ({"mount": [{"cursor": "nope", "start": 0, "end": 1}]}, 404), ({"input_ids": []}, 400),
    ({"cursor": "nope"}, 404), ({"cursor": None}, 400), ({"mount": []}, 400), ({"mount": [{"cursor": "A", "start": 0, "end": 1}, {"cursor": "nope", "start": 0, "end": 1}]},
                            404), ({"args": {"disable_radix_cache": True}}, 400),
    ({"args": {"page_size": 16}}, 400),
    ({"mount": [{"cursor": "A", "end": 1}]}, 400), ({"mount": [{"cursor": "A", "start": "x", "end": 1}]}, 400),
    ({"mount": ["A"]}, 400), ({"mount": [{"cursor": ["A"], "start": 0, "end": 1}]}, 400),
    ({"logprob_start_len": 12}, 400),  # inside B (3) + the mount (10): those rows have no logits
])
def test_prepare_rejects(bad, code):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from octopus.serving.kv_mount import prepare
    store = CursorStore(1 << 20)
    store.put("B", [1, 2, 3])
    store.put("A", list(range(300)))
    body = {"cursor": "B", "input_ids": [4], "mount": [{"cursor": "A", "start": 0, "end": 10}],
            "sampling_params": None, "lora_path": None, "logprob_start_len": -1, **bad}
    args = SimpleNamespace(**{**vars(ARGS), **body.pop("args", {})})
    with pytest.raises(HTTPException) as e:
        prepare(SimpleNamespace(**body), store, SimpleNamespace(server_args=args))
    assert e.value.status_code == code


def test_failed_rematch_leaves_no_stale_mount_hit(monkeypatch):
    """A queued mount request is re-matched every round: its segments' pages go after B's prefix (all of them, in
    request order, with the tree owning them: cache_protected_len); when one segment's pages are gone in a later round
    nothing is mounted, the request aborts, and its one-token stand-in must not lock the node of the earlier round's
    match."""
    pytest.importorskip("sglang.srt.mem_cache.radix_cache")
    import torch

    import octopus.serving.kv_mount as km
    cp = {"octopus_pos": -2, "octopus_mount_at": 3, "octopus_mounts": [[1, [10, 11, 12]], [0, [20, 21]]]}

    class Req(SimpleNamespace):
        def set_finish_with_abort(self, msg):
            self.to_finish, self.aborted = msg, msg
    req = Req(sampling_params=SimpleNamespace(custom_params=cp), to_finish=None, kv=SimpleNamespace(cache_protected_len=3),
              full_untruncated_fill_ids=array("q", [1, 2, 3, -5, -5, -6, -6, 4]), extra_key=None, cache_salt=None,
              return_logprob=False)
    pages = {10: (torch.tensor([7, 8, 9]), 42), 20: (torch.tensor([5, 6]), 44)}  # round 1: both sources resident
    monkeypatch.setattr(km, "match_kv", lambda cache, key: pages[key.token_ids[0]])
    req.prefix_indices = torch.tensor([0, 1, 2])
    assert not km.resolve(req, None) and [h[:2] for h in req.octopus_mount_hit] == [(42, 2), (44, 2)]
    assert req.prefix_indices.tolist() == [0, 1, 2, 8, 9, 5, 6] and req.kv.cache_protected_len == 7
    pages[20] = (torch.tensor([5]), 45)  # round 2: the second source lost a page, the request aborts
    req.prefix_indices, req.kv.cache_protected_len = torch.tensor([0, 1, 2]), 3
    assert km.resolve(req, None) and req.aborted == NO_MOUNT_KV and req.octopus_mount_hit is None
    assert req.prefix_indices.tolist() == [0, 1, 2] and req.kv.cache_protected_len == 3
    adder = SimpleNamespace(tree_cache=MagicMock())
    km.lock(adder, req)
    adder.tree_cache.inc_lock_ref.assert_not_called()
    assert getattr(req, "octopus_mount_node", None) is None


def test_unsupported_scheduler_refuses_a_mount(monkeypatch):
    """serve (Scheduler init) records why a scheduler cannot mount: the model runner's sliding window (any model with
    one) or an attention layer's own, a radix cache that is not UnifiedRadixCache, or a feature whose own cache paths
    the shared pages are not checked against (hierarchical cache, sessions, speculative decoding, PD). Then a mount
    request aborts with that reason before any match (the HTTP side answers 400 with it); a request from a mounted
    cursor without a new mount is not refused."""
    pytest.importorskip("sglang.srt.mem_cache.unified_radix_cache")
    import torch

    import octopus.serving.kv_mount as km
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    unified = UnifiedRadixCache.__new__(UnifiedRadixCache)

    def runner(window, layer_windows=(), **on):
        layers = []
        for i, w in enumerate(layer_windows):
            layers.append(RadixAttention.__new__(RadixAttention))
            torch.nn.Module.__init__(layers[-1])
            layers[-1].sliding_window_size, layers[-1].layer_id = w, i
        backends = {k: on.pop(k) for k in ("prefill_attention_backend_str", "decode_attention_backend_str") if k in on}
        args = SimpleNamespace(**{"enable_hierarchical_cache": False, "enable_streaming_session": False,
                                  "enable_session_radix_cache": False, "speculative_algorithm": None,
                                  "enable_unified_memory": False, "enable_hisparse": False,
                                  "disaggregation_mode": "null", "enable_deterministic_inference": False,
                                  "dcp_size": 1, **on})
        return SimpleNamespace(sliding_window_size=window, model=torch.nn.ModuleList(layers), server_args=args,
                               **backends)

    class Req(SimpleNamespace):
        def set_finish_with_abort(self, msg):
            self.to_finish = msg
    for cache, run, why in ((unified, runner(4096), "sliding-window attention (window 4096)"),
                            (unified, runner(None, (-1, 1024)), "sliding-window attention (window 1024)"),  # per layer
                            (object(), runner(None), "the radix cache object (needs UnifiedRadixCache)"),
                            (unified, runner(None, enable_hierarchical_cache=True, speculative_algorithm="EAGLE"),
                             "--enable-hierarchical-cache, --speculative-algorithm")):
        monkeypatch.setattr(km, "UNSUPPORTED", None)
        km.serve(cache, run)
        cp = {"octopus_pos": 5, "octopus_mount_at": 3, "octopus_mounts": [[0, [1]]]}
        req = Req(sampling_params=SimpleNamespace(custom_params=cp), to_finish=None, return_logprob=True)
        monkeypatch.setattr(km, "match_kv", None)  # never reached
        assert km.resolve(req, cache) and req.to_finish == f"{km.UNSUPPORTED_PREFIX} {why}" and req.return_logprob
    monkeypatch.setattr(km, "UNSUPPORTED", None)
    km.serve(unified, runner(None, (-1, -1)))
    assert km.UNSUPPORTED is None
    # a layer subset: attention layers of the model only (the server does not start), on the triton backend only
    monkeypatch.setattr(km, "MOUNT_LAYERS", frozenset({1}))
    km.serve(unified, runner(None, (-1, -1), prefill_attention_backend_str="triton",
                             decode_attention_backend_str="triton"))
    assert km.UNSUPPORTED is None
    km.serve(unified, runner(None, (-1, -1), prefill_attention_backend_str="triton",
                             decode_attention_backend_str="flashinfer"))
    assert km.UNSUPPORTED == (f"{km.UNSUPPORTED_PREFIX} OCTOPUS_MOUNT_LAYERS with attention backends "
                              "['flashinfer', 'triton'] (needs triton)")
    km.serve(unified, runner(None, (-1, -1), prefill_attention_backend_str="triton", decode_attention_backend_str="triton",
                             enable_deterministic_inference=True, dcp_size=2))
    assert km.UNSUPPORTED == (f"{km.UNSUPPORTED_PREFIX} OCTOPUS_MOUNT_LAYERS with --enable-deterministic-inference, "
                              "OCTOPUS_MOUNT_LAYERS with --dcp-size > 1")
    monkeypatch.setattr(km, "MOUNT_LAYERS", frozenset({1, 2}))
    with pytest.raises(ValueError, match=r"\[2\] are not attention layers of this model \(\[0, 1\]\)"):
        km.serve(unified, runner(None, (-1, -1)))


def test_lock_locks_each_segment_once():
    """No end-of-request copy: nothing is charged to the round's budget; each segment's source is locked once."""
    pytest.importorskip("sglang.srt.mem_cache.unified_cache.component_type")
    import octopus.serving.kv_mount as km
    req = SimpleNamespace(octopus_mount_hit=[(42, 7, None, 9), (43, 2, None, 2)])
    adder = SimpleNamespace(tree_cache=MagicMock(), rem_total_token_offset=5)
    km.lock(adder, req)
    km.lock(adder, req)  # a chunked request passes again: locked once
    assert [c[0][0] for c in adder.tree_cache.inc_lock_ref.call_args_list] == [42, 43]
    assert [h[:2] for h, _ in req.octopus_mount_node] == [(42, 7), (43, 2)]
    assert adder.rem_total_token_offset == 5


def test_check_lock_holds_the_mounted_path_for_the_budget_check():
    """add_one_req checks the budget with the prefix locked: A's path is locked for that check too (unlocked, it
    counted as evictable and a mount request could over-commit the pool: Out of memory in alloc_for_extend)."""
    pytest.importorskip("sglang.srt.mem_cache.unified_cache.component_type")
    import octopus.serving.kv_mount as km
    tree = MagicMock()
    tree.inc_lock_ref.return_value = SimpleNamespace(skip_lock_node_ids={3: 1})
    adder, req = SimpleNamespace(tree_cache=tree), SimpleNamespace(octopus_mount_hit=[(42, 3, -1)])
    with km.check_lock(adder, req):
        assert tree.inc_lock_ref.call_args[0][0] == 42 and not tree.dec_lock_ref.called
    assert tree.dec_lock_ref.call_args[0][0] == 42 and tree.dec_lock_ref.call_args[0][1].skip_lock_node_ids == {3: 1}
    tree.reset_mock()
    req.octopus_mount_node = 42  # a later chunk: A is locked for the request already
    for r in (req, SimpleNamespace()):
        with km.check_lock(adder, r):
            pass
    assert not tree.inc_lock_ref.called


class _Pool:
    """The real UnifiedRadixCache (FULL only, CPU allocator) with the mount patch, a mount request's end as
    cache_finished_req runs it (match, lock, insert with the tree owning the mounted rows, lend), and the checks: every
    freed slot, the pool balance, no slot freed while a node still holds it."""

    def __init__(self, total=64):
        urc = pytest.importorskip("sglang.srt.mem_cache.unified_radix_cache")
        from sglang.srt.mem_cache.unified_cache.components import full_component
        if "octopus-mount" not in Path(full_component.__file__).read_text():
            pytest.skip("SGLang is not mount-patched: PYTHONPATH=<mount overlay>")
        import torch
        from sglang.srt import octopus_mount as km  # the module the patched hooks import
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        self.torch, self.km, self.total, self.FULL = torch, km, total, ComponentType.FULL
        self.alloc = TokenToKVPoolAllocator(total, torch.bfloat16, "cpu", SimpleNamespace(), need_sort=False)
        self.cache = urc.UnifiedRadixCache(CacheInitParams(
            disable=False, req_to_token_pool=None, token_to_kv_pool_allocator=self.alloc, page_size=1,
            tree_components=(ComponentType.FULL,)))
        self.freed, free = [], self.alloc.free
        self.alloc.free = lambda idx: (self.freed.extend(idx.tolist()), free(idx))

    def key(self, ids, ns=None):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        return RadixKey(array("q", ids), ns)

    def put(self, ids, ns=None):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        rows = self.alloc.alloc(len(ids))
        self.cache.insert(InsertParams(key=self.key(ids, ns), value=rows))
        return rows

    def mount(self, prefix, prefix_rows, segs, own_ids, ns=None, cached=True, phs=None):
        """A finished mount request from prefix: segs = [(source tokens up to e, s)], then its own tokens."""
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        req, rows, ids, hits = SimpleNamespace(), [prefix_rows], list(prefix), []
        for k, (toks, s) in enumerate(segs):
            idx, node = self.km.match_kv(self.cache, self.key(toks, ns))
            rows.append(idx[s:])
            hits.append((node, len(toks) - s, rows[-1], len(toks)))
            ids += [phs[k] if phs else -next(self.km._PLACEHOLDERS)] * (len(toks) - s)
        req.octopus_mount_hit, req.octopus_mount_at = hits, len(prefix)
        self.km.lock(SimpleNamespace(tree_cache=self.cache), req)
        own = self.alloc.alloc(len(own_ids))
        row = self.torch.cat(rows + [own])
        result = None
        if cached:
            result = self.cache.insert(InsertParams(key=self.key(ids + own_ids, ns), value=row, prev_prefix_len=len(ids)))
        else:
            self.alloc.free(own)  # the request frees its own KV, never the mounted rows (cache_protected_len)
        self.km.lend(self.cache, req, result)
        return ids + own_ids, row, own

    def held(self) -> set:
        return {v for n in self.cache.tree_core._node_arena.values() if n is not self.cache.tree_core.root_node
                and (v_ := n.component_data[self.FULL].value) is not None for v in v_.tolist()}

    def balanced(self) -> bool:
        a = self.alloc.available_size() + self.cache.full_evictable_size() + self.cache.full_protected_size()
        return a == self.total

    def drain(self):
        """Evict one node's worth at a time: the pool balances and no slot a node still holds is free."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        while self.alloc.available_size() < self.total:
            before = len(self.freed)
            self.cache.evict(EvictParams(num_tokens=1))
            assert self.balanced() and not self.held() & set(self.freed) and len(self.freed) > before


def test_zero_copy_shares_mounted_pages_and_frees_each_once():
    """B's continuation mounts two segments (A's and X's pages) without a copy; C mounts a range of B's continuation
    that holds B's placeholders (transitive), so C's row holds A's, X's and B's pages too. No slot is allocated for
    mounted rows; nothing stays locked once a request ends (idle: available + evictable = pool); a rebuild re-mounting
    into its old path adds no holder; eviction, one node at a time in LRU order, never frees a slot a node still holds
    and frees each of the 33 slots exactly once."""
    p = _Pool()
    km, b = p.km, list(range(101, 106))
    a, x = list(range(1, 11)), list(range(21, 29))
    a_rows, x_rows, b_rows = p.put(a), p.put(x), p.put(b)
    p.mount(b, b_rows, [(a[:8], 2)], [9], cached=False)  # aborted: nothing cached
    assert p.cache.full_protected_size() == 0 and km.stats(p.cache)["octopus_shared_holds"] == 0 and p.balanced()
    free_before = p.alloc.available_size()
    b_ids, b_row, b_own = p.mount(b, b_rows, [(a[:8], 2), (x[:4], 0)], [201, 202, 203])
    assert free_before - p.alloc.available_size() == 3  # B's own tokens only: no slot for the 10 mounted rows
    assert km.stats(p.cache) == {"octopus_mount_layers": None, "octopus_full_protected": 0,
                                 "octopus_available": p.alloc.available_size(), "octopus_shared_holds": 10}
    assert p.balanced() and b_row[5:15].tolist() == a_rows[2:8].tolist() + x_rows[:4].tolist()
    st = km.stats(p.cache)
    p.mount(b, b_rows, [(a[:8], 2), (x[:4], 0)], [201, 202, 203], phs=[b_ids[5], b_ids[11]])  # a rebuild
    assert km.stats(p.cache) == st and p.balanced()  # its own 3 slots duplicated the path's: the insert frees them
    c = [301, 302]
    c_ids, c_row, c_own = p.mount(c, p.put(c), [(b_ids[:16], 5)], [401])  # B's placeholders + its token 201
    assert c_row[2:13].tolist() == b_row[5:16].tolist()  # A's, X's and B's own slots, none copied
    assert km.stats(p.cache)["octopus_shared_holds"] == 21 and p.cache.full_protected_size() == 0
    assert p.alloc.available_size() + p.cache.full_evictable_size() == p.total  # idle: the whole pool is admissible
    p.drain()
    assert not km._refs(p.cache.tree_core) and sorted(p.freed) == list(range(1, 34))  # each slot once


def test_zero_copy_across_lora_versions():
    """Review of #185: SGLang keys the radix tree by LoRA version (extra_key). After a weight update a mounted cursor's
    continuation is rebuilt in the new version's namespace with the same placeholders: its rows there hold the new
    version's source pages, and those stay allocated while that continuation is cached, whatever is evicted first."""
    p = _Pool()
    a, b, ph = list(range(1, 11)), list(range(101, 106)), -next(p.km._PLACEHOLDERS)
    p.put(a, "v1")
    p.mount(b, p.put(b, "v1"), [(a[:8], 2)], [201], ns="v1", phs=[ph])
    p.put(list(range(901, 921)), "v0")
    p.put(a, "v2")
    _, row2, _ = p.mount(b, p.put(b, "v2"), [(a[:8], 2)], [201], ns="v2", phs=[ph])
    assert p.km.stats(p.cache)["octopus_shared_holds"] == 12 and p.cache.full_protected_size() == 0 and p.balanced()
    p.drain()  # never frees row2's mounted slots while the v2 continuation holds them
    assert sorted(p.freed) == list(range(1, 53)) and not p.km._refs(p.cache.tree_core)


def test_zero_copy_idle_pool_is_admissible_and_evictable():
    """Review of #185: with a cached continuation reading a 40-token source, nothing is locked at idle, so admission
    sees the whole pool, and one eviction for the pool frees it all (source and continuation in either order)."""
    p = _Pool()
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    big, b = list(range(1, 41)), list(range(101, 106))
    p.put(big)
    p.mount(b, p.put(b), [(big, 0)], [201])
    assert p.cache.full_protected_size() == 0
    assert p.alloc.available_size() + p.cache.full_evictable_size() == p.total == 64
    p.cache.evict(EvictParams(num_tokens=p.total))
    assert p.alloc.available_size() == p.total and p.balanced() and sorted(p.freed) == list(range(1, 47))


@pytest.mark.parametrize("seed", [0, 4, 5])  # each fails without owned, the split hook or the source mark
def test_zero_copy_fuzz(seed):
    """Review of #185 (its fuzz): random sequences in two LoRA namespaces, mounts (one or two segments, transitive,
    aborted, kept in part, rebuilt into their old path with the same placeholders), plain continuations, running
    requests holding locks while the pool evicts, random evictions. After every step: no slot a node or a running row
    holds is free, nothing is freed twice, available + evictable + protected (+ the running mount's own) = pool, and at
    idle nothing is locked and evictable = the distinct slots in the tree; at the end the pool empties and no holder
    count is left. It fails without owned, without the split hook or without the source mark."""
    p = _Pool(1500)
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, EvictParams, InsertParams
    km, cache, core, rng, tok = p.km, p.cache, p.cache.tree_core, random.Random(seed), iter(range(1000, 10**9))
    free = set(range(1, p.total + 1))
    _alloc, _free = p.alloc.alloc, p.alloc.free

    def alloc(n):
        r = _alloc(n)
        for x in [] if r is None else r.tolist():
            assert x in free
            free.discard(x)
        return r

    def release(idx):
        for x in idx.tolist():
            assert x not in free, f"double free {x}"
            free.add(x)
        _free(idx)
    p.alloc.alloc, p.alloc.free = alloc, release
    seqs, running, own_running = [], [], [0]

    def check(step):
        held = p.held()
        assert not held & free and not {x for r in running for x in r} & free, step
        assert (p.alloc.available_size() + cache.full_evictable_size() + cache.full_protected_size() + own_running[0]
                == p.total and p.alloc.available_size() == len(free)), step
        if not running:
            assert cache.full_protected_size() == 0 and cache.full_evictable_size() == len(held), step

    def room(n):
        if p.alloc.available_size() < n:
            cache.evict(EvictParams(num_tokens=n - p.alloc.available_size()))
        return alloc(n) if p.alloc.available_size() >= n else None

    def resident(ns, toks):
        idx, node = km.match_kv(cache, p.key(toks, ns))
        return (idx, node) if len(idx) == len(toks) else None

    def unlock(node, lk):
        cache.dec_lock_ref(node, DecLockRefParams(skip_lock_node_ids=lk.skip_lock_node_ids))

    def new_base():
        ns = rng.choice(["v1", "v2"])
        parent = rng.choice([x for x in seqs if x[0] == ns] or [None]) if rng.random() < 0.6 else None
        toks = (parent[1][:rng.randint(1, len(parent[1]))] if parent else []) + [next(tok) for _ in range(rng.randint(3, 25))]
        pre = next((i for i in range(len(toks), 0, -1) if parent and resident(ns, toks[:i])), 0)
        idx, node = resident(ns, toks[:pre]) if pre else (p.torch.empty(0, dtype=p.torch.int64), core.root_node.id)
        lk = cache.inc_lock_ref(node)
        own = room(len(toks) - pre)
        if own is not None:
            cache.insert(InsertParams(key=p.key(toks, ns), value=p.torch.cat([idx, own]), prev_prefix_len=pre))
            seqs.append((ns, toks))
        unlock(node, lk)

    def mount(again=None):
        if again:
            ns, b, segs, own_ids = again
        else:
            ns = rng.choice(["v1", "v2"])
            cand = [x[1] for x in seqs[-40:] if x[0] == ns]
            if len(cand) < 2:
                return
            b, segs, own_ids = rng.choice(cand), [], [next(tok) for _ in range(rng.randint(1, 6))]
            for _ in range(rng.randint(1, 2)):
                a = rng.choice(cand)
                s = rng.randrange(len(a))
                e = rng.randint(s + 1, len(a))
                if not (e < len(a) and a[e] < 0 and a[e - 1] < 0):  # else prepare's 400
                    segs.append((a[:e], s, -next(km._PLACEHOLDERS)))
            if not segs:
                return
        hb = resident(ns, b)
        if hb is None:
            return
        blk, hits, rows, ids = cache.inc_lock_ref(hb[1]), [], [hb[0]], list(b)
        for toks, s, ph in segs:
            h = resident(ns, toks)
            if h is None:
                return unlock(hb[1], blk)
            hits.append((h[1], len(toks) - s, h[0][s:], len(toks)))
            rows.append(h[0][s:])
            ids += [ph] * (len(toks) - s)
        req, adder = SimpleNamespace(octopus_mount_hit=hits, octopus_mount_at=len(b)), SimpleNamespace(tree_cache=cache)
        with km.check_lock(adder, req):
            km.lock(adder, req)
        own = room(len(own_ids))
        if own is None:
            km.lend(cache, req, None)
            return unlock(hb[1], blk)
        row = p.torch.cat(rows + [own])
        running.append(row.tolist())
        own_running[0] = len(own)
        check("mount running")
        if rng.random() < 0.3:
            cache.evict(EvictParams(num_tokens=rng.randint(1, 60)))
            check("evict while a mount runs")
        running.pop()
        own_running[0], full, mode, result = 0, ids + own_ids, rng.random(), None
        if mode < 0.15:  # aborted: the request frees its own KV
            release(row[len(ids):])
        else:
            keep = len(full) if mode < 0.8 else rng.randint(len(b), len(full))  # the cache keeps a prefix
            result = cache.insert(InsertParams(key=p.key(full[:keep], ns), value=row[:keep].clone(),
                                               prev_prefix_len=len(ids)))
            if max(keep, len(ids)) < len(full):
                release(row[max(keep, len(ids)):])
        unlock(hb[1], blk)
        km.lend(cache, req, result)
        if result is not None and keep == len(full):
            seqs.append((ns, full))
            if rng.random() < 0.3:  # rebuilt into the old path with the same placeholders
                mount((ns, b, segs, own_ids))

    def continuation():
        if seqs:
            ns, t = rng.choice(seqs[-40:])
            h = resident(ns, t)
            if h:
                lk = cache.inc_lock_ref(h[1])
                own_ids = [next(tok) for _ in range(rng.randint(1, 8))]
                own = room(len(own_ids))
                if own is not None:
                    cache.insert(InsertParams(key=p.key(t + own_ids, ns), value=p.torch.cat([h[0], own]),
                                              prev_prefix_len=len(t)))
                    seqs.append((ns, t + own_ids))
                unlock(h[1], lk)

    for step in range(3000):
        r = rng.random()
        if r < 0.25:
            new_base()
        elif r < 0.6:
            mount()
        elif r < 0.75:
            continuation()
        elif r < 0.85 and seqs:  # a running request holds a cached path while the pool evicts
            h = resident(*rng.choice(seqs))
            if h:
                lk = cache.inc_lock_ref(h[1])
                running.append(h[0].tolist())
                cache.evict(EvictParams(num_tokens=rng.randint(1, 100)))
                check(f"{step}: evict under a lock")
                running.pop()
                unlock(h[1], lk)
        else:
            cache.evict(EvictParams(num_tokens=rng.randint(1, 40)))
        check(step)
    cache.evict(EvictParams(num_tokens=p.total))
    check("end")
    assert p.alloc.available_size() == p.total and not km._refs(core)


def test_prefill_only_request_snapshots_its_recurrent_state_under_extra_buffer():
    """extra_buffer caches a finished request only up to its last tracked state: a prefill-only one (a rebuild) copies
    its live state into its next track slot and tracks it at kv_len, so the cache keeps the whole sequence with its
    state at the end; requests that generate, retractions and no_buffer are untouched. The scheduler keeps
    prefill-only requests out of the next decode batch (prefill_only)."""
    pytest.importorskip("torch")
    import torch

    import octopus.serving.kv_mount as km
    copies = []
    mamba_pool = SimpleNamespace(copy_from=lambda src, dst: copies.append((src.tolist(), dst.tolist())))
    pool = SimpleNamespace(mamba_pool=mamba_pool, translate_mamba_indices=lambda ix: ix + 100)
    req = lambda n: SimpleNamespace(sampling_params=SimpleNamespace(max_new_tokens=n), kv=SimpleNamespace(  # noqa: E731
        mamba_pool_idx=torch.tensor(5), mamba_ping_pong_track_buffer=torch.tensor([7, 8]), mamba_next_track_idx=1,
        mamba_last_track_idx=0, mamba_last_track_seqlen=64))
    for extra, n, insert, snapped in ((True, 0, True, True), (True, 0, False, False),  # retraction: nothing cached
                                      (False, 0, True, False), (True, 16, True, False)):  # no_buffer caches it anyway
        copies.clear()
        r = req(n)
        cache = SimpleNamespace(enable_mamba_extra_buffer=extra, req_to_token_pool=pool)
        assert km.finish(cache, r, insert, 90) is None
        assert copies == ([([105], [108])] if snapped else [])
        assert (r.kv.mamba_last_track_idx, r.kv.mamba_last_track_seqlen) == ((1, 90) if snapped else (0, 64))
    reqs = [req(0), req(16), req(None)]
    assert km.prefill_only(reqs) == reqs[:1]


def test_layer_subset_hides_the_mounted_rows_from_the_other_layers(monkeypatch):
    """OCTOPUS_MOUNT_LAYERS: unset is every layer (no second view, nothing recorded); a subset records each request's
    placeholder rows (resolve, ForwardBatch.init_new) and the attention layers outside it read the batch's KV rows
    without them (each request's own rows in order, rows past a request's KV length ignored); the others the full view."""
    import numpy as np
    import torch

    import octopus.serving.kv_mount as km
    for value, layers in (("", None), (" 3,7, 11", frozenset({3, 7, 11}))):
        monkeypatch.setenv("OCTOPUS_MOUNT_LAYERS", value)
        assert km.mount_layers() == layers
    assert km.stats(None) == {"octopus_mount_layers": None}
    reqs = [SimpleNamespace(octopus_hidden=np.array([2, 3])), SimpleNamespace(), SimpleNamespace(octopus_hidden=np.array([1, 9]))]
    fb = SimpleNamespace()
    km.hide_mounts(fb, SimpleNamespace(reqs=reqs))
    assert not hasattr(fb, "octopus_hidden")  # every layer reads the mount: one view
    monkeypatch.setattr(km, "MOUNT_LAYERS", frozenset({1}))
    assert km.stats(None) == {"octopus_mount_layers": [1]}
    req = SimpleNamespace(sampling_params=SimpleNamespace(custom_params={"octopus_pos": 0}), to_finish=None,
                          origin_input_ids=[4, 5, -8, -8, -9, 6], prefix_indices=torch.arange(6),
                          full_untruncated_fill_ids=[4, 5, -8, -8, -9, 6, 7])
    assert not km.resolve(req, None) and req.octopus_hidden.tolist() == [2, 3, 4]
    km.hide_mounts(fb, SimpleNamespace(reqs=[SimpleNamespace(), SimpleNamespace(octopus_hidden=np.array([], int))]))
    assert not hasattr(fb, "octopus_hidden")  # no request holds mounted rows: one view, CUDA graphs as usual
    km.hide_mounts(fb, SimpleNamespace(reqs=reqs))
    assert fb.octopus_hidden[1] is None
    meta = SimpleNamespace(kv_indptr=torch.tensor([0, 5, 8, 12], dtype=torch.int32),
                           kv_indices=torch.arange(100, 115))  # over-allocated past the last request
    km.layer_view(SimpleNamespace(forward_metadata=meta), fb)
    indptr, indices = meta.octopus_kv
    assert indptr.dtype == torch.int32 and indptr.tolist() == [0, 3, 6, 9]
    assert indices.tolist() == [100, 101, 104, 105, 106, 107, 108, 110, 111]  # row 9 of the third is past its 4 rows
    layer = lambda i: SimpleNamespace(layer_id=i)  # noqa: E731
    assert km.layer_kv(meta, layer(0), "ptr", "idx") == (indptr, indices)
    assert km.layer_kv(meta, layer(1), "ptr", "idx") == ("ptr", "idx")
    plain = SimpleNamespace(kv_indptr=None, kv_indices=None)
    km.layer_view(SimpleNamespace(forward_metadata=plain), SimpleNamespace())
    assert km.layer_kv(plain, layer(0), "ptr", "idx") == ("ptr", "idx")


def test_store_drops_positions_with_the_cursor():
    s = CursorStore(budget_bytes=40)
    s.put("a", [1, 2, 3], ((2, 5),))
    s.put("b", [1, 2, 3, 4, 5, 6, 7, 8])  # over budget: a goes, and its breakpoints with it
    assert s.get("a") is None and s.pos == {}
    assert s.used == 32
    s.put("b", [1], ((0, 1),))
    s.put("b", [1])
    assert s.pos == {}


def sglang_root() -> Path:
    return Path(pytest.importorskip("sglang").__file__).parent


def test_patch_applies_once_with_the_cursor_and_lora_patches(tmp_path):
    """On a copy of the importable SGLang: the mount patch needs the cursor patch; in every order with the lora patch
    each second run is a no-op, every edit lands once and the result compiles."""
    import py_compile

    import patches.apply_sglang_cursor as cursor
    import patches.apply_sglang_lora_abort_release as lora
    import patches.apply_sglang_mount as mount
    src = sglang_root()
    if cursor.MARK in (src / "srt/entrypoints/http_server.py").read_text():
        pytest.skip("the importable SGLang is already cursor-patched: run without the overlay")
    rels = {*cursor.EDITS, *mount.EDITS, "srt/managers/tokenizer_manager.py"}
    for order in ((cursor, mount, lora), (lora, cursor, mount), (cursor, lora, mount)):
        root = tmp_path / "-".join(m.__name__.rsplit("_", 1)[-1] for m in order) / "sglang"
        for rel in rels:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, root / rel)
        with pytest.raises(SystemExit):
            mount.patch(root)  # before the cursor patch
        assert [m.patch(root) for m in order][order.index(mount)]
        assert [m.patch(root) for m in order] == [False, False, False]
        for rel, edits in mount.EDITS.items():
            text = (root / rel).read_text()
            assert all(text.count(new) == 1 for _, new in edits), rel
        for path in [*map(root.joinpath, rels), root / "srt/octopus_mount.py", root / "srt/octopus_cursor_store.py"]:
            py_compile.compile(str(path), doraise=True)
    # the shims are the code dir's modules themselves: one copy, the trainer's (docs/cursor-protocol.md "Deployment")
    import importlib
    import octopus.serving.cursor_store
    import octopus.serving.delta_render
    import octopus.serving.kv_mount
    sys.path.insert(0, str(root / "srt"))
    try:
        for shim, real in (("octopus_mount", octopus.serving.kv_mount), ("octopus_cursor_store", octopus.serving.cursor_store),
                           ("octopus_delta_render", octopus.serving.delta_render)):
            assert importlib.import_module(shim) is real
            sys.modules.pop(shim)
    finally:
        sys.path.remove(str(root / "srt"))


def test_generate_mount_http(monkeypatch):
    """The patched /generate handler on a stub tokenizer manager: a mount request goes out as B + placeholders + delta
    with the mount in custom_params; a scheduler abort for B's missing KV prefills B and A together (prefill-only) and
    retries; the new cursor keeps its breakpoints. Rows that are gone are rebuilt, never a 404: a continuation of the
    mount reply goes again carrying the mount; a mount from it whose prefix lost its mounted rows rebuilds that prefix
    with its own mount first. A client that is gone gets no rebuild."""
    hs = pytest.importorskip("sglang.srt.entrypoints.http_server")
    if not hasattr(hs, "octopus_mount"):
        pytest.skip("SGLang is not mount-patched: PYTHONPATH=<mount overlay>")
    import copy

    from fastapi import HTTPException
    from sglang.srt.managers.io_struct import GenerateReqInput
    seen, outs = [], []

    async def generate(obj, raw):
        batch = isinstance(obj.input_ids[0], list)  # the rebuilds without a mount: one message
        for ids, sp in zip(obj.input_ids, obj.sampling_params) if batch else ((obj.input_ids, obj.sampling_params),):
            seen.append(SimpleNamespace(input_ids=list(ids), sampling_params=copy.deepcopy(sp), batch=batch))
        if batch:
            yield [outs.pop(0) for _ in obj.input_ids]
            return
        out = outs.pop(0)
        if isinstance(out, Exception):
            raise out
        for o in out if isinstance(out, list) else [out]:  # a list: a stream's chunks
            yield {"meta_info": {}, **o}
    tm = SimpleNamespace(generate_request=generate, server_args=ARGS, create_abort_task=lambda obj: None,
                         incremental_streaming_output=False)
    raw = lambda gone=False: SimpleNamespace(is_disconnected=AsyncMock(return_value=gone))  # noqa: E731  the client
    monkeypatch.setattr(hs, "_global_state", SimpleNamespace(tokenizer_manager=tm))
    hs.CURSORS.put("B", [1, 2, 3])
    hs.CURSORS.put("A", [10, 11, 12, 13])
    outs[:] = [ValueError(NO_CURSOR_KV), {}, {}, {"output_ids": [5, 6], "meta_info": {"id": "B1"}}]
    asyncio.run(hs.generate_request(GenerateReqInput(cursor="B", input_ids=[4], mount=[
        {"cursor": "A", "start": 1, "end": 3}], sampling_params={"temperature": 0}), raw()))
    ph = seen[0].input_ids[3]
    b1 = [1, 2, 3, ph, ph, 4, 5, 6]
    assert [o.input_ids for o in seen] == [b1[:6], [1, 2, 3], [10, 11, 12], b1[:6]]  # B and A rebuilt together,
    assert [o.batch for o in seen] == [False, True, True, False]  # in one message
    assert [o.sampling_params.get("max_new_tokens") for o in seen] == [None, 0, 0, None]
    assert seen[1].sampling_params == {"max_new_tokens": 0}
    mount = {"octopus_pos": -2, "octopus_mount_at": 3, "octopus_mounts": [[1, [10, 11, 12]]]}
    assert seen[0].sampling_params == {"temperature": 0, "custom_params": mount}
    assert hs.CURSORS.get("B1").tolist() == b1 and hs.CURSORS.pos["B1"][0][:2] == (5, -2)
    seen.clear()  # the radix cache lost everything: a continuation of the mount reply
    outs[:] = [ValueError(EVICTED), ValueError(NO_MOUNT_KV), {}, {}, {"output_ids": [9], "meta_info": {"id": "B2"}}]
    asyncio.run(hs.generate_request(GenerateReqInput(cursor="B1", input_ids=[7]), raw()))
    assert [o.input_ids for o in seen] == [b1 + [7], b1 + [7], [1, 2, 3], [10, 11, 12], b1 + [7]]
    assert seen[0].sampling_params["custom_params"] == {"octopus_pos": -2}  # first as a plain continuation
    assert all(seen[i].sampling_params["custom_params"] == mount for i in (1, 4))  # then carrying its mount
    assert hs.CURSORS.get("B2").tolist() == b1 + [7, 9] and hs.CURSORS.pos["B2"] is hs.CURSORS.pos["B1"]
    seen.clear()  # a mount from B1 whose prefix lost its mounted rows: B1 is rebuilt with its own mount first
    outs[:] = [ValueError(EVICTED), {}, {}, {"output_ids": [1], "meta_info": {"id": "B3"}}]
    asyncio.run(hs.generate_request(GenerateReqInput(cursor="B1", input_ids=[8], mount=[
        {"cursor": "A", "start": 2, "end": 4}]), raw()))
    ph2 = seen[0].input_ids[8]
    assert ph2 < 0 and ph2 != ph
    assert [o.input_ids for o in seen] == [b1 + [ph2, ph2, 8], [10, 11, 12, 13], b1, b1 + [ph2, ph2, 8]]
    assert seen[2].sampling_params == {"max_new_tokens": 0, "custom_params": mount}
    assert seen[0].sampling_params["custom_params"]["octopus_mount_at"] == 8
    assert [p[:2] for p in hs.CURSORS.pos["B3"]] == [(5, -2), (10, -4)]
    seen.clear()  # two segments, one of them B1's own rows after its placeholders: every source is rebuilt
    outs[:] = [ValueError(NO_MOUNT_KV), {}, {}, {}, {"output_ids": [2], "meta_info": {"id": "B4"}}]
    asyncio.run(hs.generate_request(GenerateReqInput(cursor="B", input_ids=[3], mount=[
        {"cursor": "A", "start": 0, "end": 2}, {"cursor": "B1", "start": 3, "end": 7}]), raw()))
    p1, p2 = seen[0].input_ids[3], seen[0].input_ids[5]
    assert seen[0].input_ids == [1, 2, 3, p1, p1, p2, p2, p2, p2, 3] and p1 != p2 < 0
    assert [o.input_ids for o in seen[1:]] == [[1, 2, 3], [10, 11], b1[:7], seen[0].input_ids]
    assert seen[3].sampling_params == {"max_new_tokens": 0, "custom_params": mount}  # B1's rows with its own mount
    assert seen[0].sampling_params["custom_params"]["octopus_mounts"] == [[0, [10, 11]], [3, b1[:7]]]
    assert hs.CURSORS.pos["B4"][0][:2] == (9, 6 - 2 - 9 + 1)  # after B1's token 6 (at 6 - 2): 5 - 9
    seen.clear()  # streaming: the abort for missing rows is the attempt's first output; the stream shows only the retry
    gone = {"meta_info": {"finish_reason": {"type": "abort", "message": NO_CURSOR_KV, "status_code": 400}}}
    chunks = [{"output_ids": [5], "meta_info": {"id": "S", "finish_reason": None}},
              {"output_ids": [5, 6], "meta_info": {"id": "S", "finish_reason": {"type": "stop"}}}]
    outs[:] = [[gone], {}, {}, chunks]

    async def stream(obj):
        return [c async for c in (await hs.generate_request(obj, raw())).body_iterator]
    body = asyncio.run(stream(GenerateReqInput(cursor="B", input_ids=[4], mount=[
        {"cursor": "A", "start": 1, "end": 3}], stream=True)))
    assert [o.input_ids[:3] for o in seen] == [[1, 2, 3], [1, 2, 3], [10, 11, 12], [1, 2, 3]]
    assert [json.loads(c[len(b"data: "):]) for c in body[:-1]] == chunks and body[-1] == b"data: [DONE]\n\n"
    assert hs.CURSORS.get("S").tolist() == seen[3].input_ids + [5, 6]  # the last chunk's whole output: a cursor
    assert hs.CURSORS.pos["S"][0][:2] == (5, -2)
    seen.clear()  # the client is gone (an agent timeout or abort): no rebuild for it
    outs[:] = [ValueError(NO_CURSOR_KV)]
    try:
        asyncio.run(hs.generate_request(GenerateReqInput(cursor="B", input_ids=[4], mount=[
            {"cursor": "A", "start": 1, "end": 3}]), raw(gone=True)))
    except ValueError as err:  # or the handler's error response
        assert "disconnected" in str(err)
    assert len(seen) == 1 and not outs
    with pytest.raises(HTTPException) as e:
        asyncio.run(hs.generate_request(GenerateReqInput(input_ids=[4], mount=[{"cursor": "A", "start": 1, "end": 3}]),
                                        raw()))
    assert e.value.status_code == 400
    outs[:] = [{"output_ids": [8], "meta_info": {"id": "P"}}]  # no cursor, no mount: today's request
    asyncio.run(hs.generate_request(GenerateReqInput(input_ids=[1, 2]), raw()))
    assert seen[-1].sampling_params is None and hs.CURSORS.pos.get("P") is None


def test_abort_by_rid_stops_a_rebuild():
    """An abort of a mount request while its rows are rebuilt (its rid is in no rid_to_state, so SGLang alone drops it)
    stops the rebuild: no retry, and the request gets SGLang's abort answer; an abort of another rid does not. An
    abort_all that crosses the attempt's own abort for missing rows sends no rebuild at all."""
    pytest.importorskip("sglang.srt.managers.io_struct")
    from octopus.serving import kv_mount
    from sglang.srt.managers.io_struct import GenerateReqInput
    seen = []

    async def generate(obj, raw):
        seen.append(list(obj.input_ids))
        if len(seen) == 1:
            raise ValueError(NO_CURSOR_KV)
        kv_mount.abort("other", False)
        kv_mount.abort("r1", False)  # /abort_request {"rid": "r1"} while B and A are rebuilt
        yield {}
    tm = SimpleNamespace(generate_request=generate, config_value={"weight_version": "v3"}.get)
    pos = ((5, -2, (3, ((1, array("i", [10, 11, 12]), ()),))),)

    def first(obj):
        async def run():
            return await anext(kv_mount.generate(tm, obj, None))
        obj.octopus_pos = pos
        return asyncio.run(run())
    ret = first(GenerateReqInput(rid="r1", input_ids=[1, 2, 3, -7, -7, 4], return_logprob=True,
                                 sampling_params={"custom_params": {}}))
    assert seen == [[1, 2, 3, -7, -7, 4], [[1, 2, 3], [10, 11, 12]]]  # the attempt, B's and A's rebuild; no retry
    assert ret == {"text": "", "output_ids": [], "meta_info": {
        "id": "r1", "finish_reason": {"type": "abort", "message": "Abort in waiting queue"}, "weight_version": "v3",
        "completion_tokens": 0, "input_token_logprobs": [], "output_token_logprobs": [], "output_token_logprobs_length": 0}}
    assert not kv_mount._REBUILDING and not kv_mount._ABORTED
    seen.clear()

    async def crossing(obj, raw):  # pause_generation's abort_all lands, then the attempt's NO_CURSOR_KV abort arrives
        seen.append(list(obj.input_ids))
        kv_mount.abort("", True)
        raise ValueError(NO_CURSOR_KV)
        yield
    tm.generate_request = crossing
    ret = first(GenerateReqInput(rid="r2", input_ids=[1, 2, 3, -7, -7, 4], sampling_params={"custom_params": {}}))
    assert seen == [[1, 2, 3, -7, -7, 4]] and ret["meta_info"]["finish_reason"]["type"] == "abort"  # no rebuild
    assert ret["meta_info"]["id"] == "r2" and not kv_mount._REBUILDING and not kv_mount._ABORTED


def test_chat_mount(tok):
    """/v1/chat/completions (evaluation's endpoint) takes read_thread's mount as the AReaL proxy does: the request runs
    as B + placeholders + delta with the range from "from"'s turn to "to"'s turn of "to"'s tokens in custom_params,
    prompt_tokens leaves out the placeholders, metadata.octopus_mount confirms the mount, and the reply is stored with
    its breakpoints, so the next chat request from it keeps the offset. Streamed, both run the same sequence through the
    same generate. A mount without a cursor, or with "from" not before "to", is a 400."""
    from test_sglang_cursor import KWARGS, TOOLS, request, serving

    from octopus.serving.delta_render import render_delta
    chat, s = serving(tok)
    if "octopus-mount" not in Path(chat.__file__).read_text():
        pytest.skip("SGLang is not mount-patched: PYTHONPATH=<mount overlay>")
    s.tokenizer_manager.server_args.disable_radix_cache, s.tokenizer_manager.server_args.page_size = False, None
    eos, im, seen = tok.eos_token_id, tok.convert_tokens_to_ids("<|im_start|>"), []

    async def generate(obj, raw):
        seen.append(copy.deepcopy(obj))
        yield {"text": "Hi", "output_ids": [13, 14, eos], "prompt_token_ids": list(obj.input_ids), "meta_info": {
            "id": f"mr{len(seen)}", "finish_reason": {"type": "stop", "matched": eos}, "prompt_tokens": len(obj.input_ids),
            "completion_tokens": 3, "cached_tokens": 0, "weight_version": "default"}}
    s.tokenizer_manager.generate_request = generate
    a1 = [7, im, 20, 21, eos]  # the source thread: "from"'s turn starts at 1, "to"'s at 6
    a2 = a1 + [30, im, 40, 41, eos]
    chat.CURSORS.put("src1", a1, prompt=2)  # chat replies: their prompt length is kept (put_reply)
    chat.CURSORS.put("src2", a2, prompt=7)
    chat.CURSORS.put("rd", [7, 8, 9])
    delta = [{"role": "tool", "content": "Mounted."}]
    read = {"from": "src1", "to": "src2", "thread": "x"}
    r = asyncio.run(s.handle_request(request(messages=delta, cursor="rd", mount=read, tools=TOOLS,
                                             chat_template_kwargs=KWARGS), None))
    tail = render_delta(tok, delta, TOOLS, KWARGS)
    ph = seen[0].input_ids[3]
    assert ph < 0 and seen[0].input_ids == [7, 8, 9] + [ph] * 5 + tail
    assert seen[0].sampling_params["custom_params"] == {"octopus_pos": -2, "octopus_mount_at": 3,
                                                        "octopus_mounts": [[1, a2[:6]]]}
    assert r.usage.prompt_tokens == 3 + len(tail)  # the reader's own tokens
    assert r.metadata["octopus_mount"] == 5  # what the client checks (tree)
    stored = [7, 8, 9] + [ph] * 5 + tail + [13, 14, eos]
    assert chat.CURSORS.get("mr1").tolist() == stored  # K2: the delta runs from A's next position, 6
    assert chat.CURSORS.pos["mr1"][0][:2] == (8, -2)
    more = [{"role": "user", "content": "Go on."}]
    r = asyncio.run(s.handle_request(request(messages=more, cursor="mr1", tools=TOOLS, chat_template_kwargs=KWARGS),
                                     None))  # continues the mount
    assert seen[1].input_ids == stored + render_delta(tok, more, TOOLS, KWARGS)
    assert seen[1].sampling_params["custom_params"] == {"octopus_pos": -2}
    assert r.usage.prompt_tokens == len(seen[1].input_ids) - 5 and chat.CURSORS.pos["mr2"] is chat.CURSORS.pos["mr1"]
    assert "octopus_mount" not in r.metadata  # no mount of its own
    s.tokenizer_manager.server_args.incremental_streaming_output = False
    s.tokenizer_manager.server_args.stream_response_default_include_usage = False

    async def streamed(**body):  # the streamed request: the same sequence through the same generate
        r = await s.handle_request(request(tools=TOOLS, chat_template_kwargs=KWARGS, stream=True, **body), None)
        return [c async for c in r.body_iterator]
    assert "Hi" in "".join(asyncio.run(streamed(messages=more, cursor="mr1")))
    assert seen[2].input_ids == seen[1].input_ids and seen[2].sampling_params == seen[1].sampling_params
    assert chat.CURSORS.get("mr3").tolist() == seen[2].input_ids + [13, 14, eos]  # a streamed reply is a cursor too
    assert chat.CURSORS.pos["mr3"] is chat.CURSORS.pos["mr1"]
    asyncio.run(streamed(messages=delta, cursor="rd", mount=read))
    assert seen[3].input_ids[:3] + seen[3].input_ids[8:] == [7, 8, 9] + tail and min(seen[3].input_ids[3:8]) < 0
    assert chat.CURSORS.pos["mr4"][0][:2] == (8, -2)  # the streamed read's reply keeps its breakpoints
    for bad in ({"messages": delta, "mount": {"from": "src1", "to": "src2"}},
                {"messages": delta, "cursor": "rd", "mount": {"from": "src2", "to": "src1"}}):
        assert asyncio.run(s.handle_request(request(**bad), None)).status_code == 400
    assert len(seen) == 4


# Live server -------------------------------------------------------------------------------------------------------
URL, DATA = os.environ.get("OCTOPUS_SGLANG_URL"), os.environ.get("OCTOPUS_MOUNT_DATA")
live = pytest.mark.skipif(not (URL and DATA), reason="needs OCTOPUS_SGLANG_URL and OCTOPUS_MOUNT_DATA")
# Against the offline K2 (transformers, flash-attn 2 + fla, bf16): mean |d logprob| <= 0.03 and argmax agreement >= 0.98
# (bar revised in review of PR #132 from the 0.02 example: 0.020 measured with 4096-token chunks, 0.014 with 128).
MEAN_ABS, ARGMAX, EVICTION = 0.03, 0.98, 1e-3


def post(path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()  # no body: GET
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            body = r.read()
            return r.status, json.loads(body) if body[:1] in (b"{", b"[") else body.decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def gen(body: dict) -> dict:
    code, r = post("/generate", body)
    assert code == 200, r
    return r


def cursor_of(ids: list[int]) -> str:
    return gen({"input_ids": ids, "sampling_params": {"max_new_tokens": 0}})["meta_info"]["id"]


def scored(cursor: str, ids: list[int], start: int, mount=None):
    """Teacher-forced log-probs and argmax of the input tokens from index `start` of the whole sequence (SGLang's
    input logprobs start with a None entry for the first token of the range: the range starts one token earlier)."""
    body = {"cursor": cursor, "input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": start - 1, "top_logprobs_num": 1}
    if mount:
        body["mount"] = mount
    m = gen(body)["meta_info"]
    lps, tops = m["input_token_logprobs"][1:], m["input_top_logprobs"][1:]
    assert [x[1] for x in lps] == ids[len(ids) - len(lps):], "input logprobs are not aligned with the scored tokens"
    return [x[0] for x in lps], [x[0][1] for x in tops]


def kv() -> dict:
    """The scheduler's KV accounting (kv_mount.stats)."""
    code, info = post("/get_server_info")
    return info["internal_states"][0]


def leaked() -> int:
    """Locked tokens: 0 when idle (shared pages hold no lock)."""
    return kv()["octopus_full_protected"]


@live
def test_live_mount_equals_offline_k2():
    """Per #129 mount point: B and A as prefill-only cursors, then B's probe continuation with the mount. Teacher-forced
    answer log-probs vs the offline K2 run (and the no-mount request vs offline none, the kernel baseline), argmax at
    every step of the offline greedy continuation, and free greedy decoding. Then the eviction path (flush the radix
    cache: B's KV and A's pages are recomputed by prefill) gives the same log-probs, and nothing stays locked."""
    rows = [r for r in json.loads(gzip.open(DATA).read()) if r["probes"]]  # every probe of a few was not token-aligned
    rows = rows[:int(os.environ.get("OCTOPUS_MOUNT_POINTS", len(rows)))]
    d = {"k2": [], "none": []}
    agree = {"k2": [], "none": []}
    exact, per_point, evicted = [], [], []
    for n, r in enumerate(rows):
        b, a, a0, M = r["b"], r["a"], r["a0"], r["M"]
        cb, ca = cursor_of(b), cursor_of(a)
        mount = [{"cursor": ca, "start": a0, "end": a0 + M}]
        pt = []
        for p in r["probes"]:
            q, ans = p["q"], p["ans"]
            lp, _ = scored(cb, q + ans, len(b) + M + len(q), mount)
            dd = [abs(x - y) for x, y in zip(lp, p["k2_lp"])]
            d["k2"] += dd
            pt += dd
            _, am = scored(cb, q + p["k2_gen"], len(b) + M + len(q), mount)
            agree["k2"] += [x == y for x, y in zip(am, p["k2_gen"])]
            out = gen({"cursor": cb, "input_ids": q, "mount": mount,
                       "sampling_params": {"max_new_tokens": 16, "temperature": 0}})["output_ids"]
            exact.append(out == p["k2_gen"][:len(out)])
            lp0, _ = scored(cb, q + ans, len(b) + len(q))
            d["none"] += [abs(x - y) for x, y in zip(lp0, p["none_lp"])]
            _, am0 = scored(cb, q + p["none_gen"], len(b) + len(q))
            agree["none"] += [x == y for x, y in zip(am0, p["none_gen"])]
        per_point.append(statistics.mean(pt))
        if n < 5:  # eviction path: same request after the whole radix cache is flushed
            assert post("/flush_cache", {})[0] == 200
            p = r["probes"][0]
            lp2, _ = scored(cb, p["q"] + p["ans"], len(b) + M + len(p["q"]), mount)
            lp1, _ = scored(cb, p["q"] + p["ans"], len(b) + M + len(p["q"]), mount)
            evicted.append(max(abs(x - y) for x, y in zip(lp2, lp1)))
        print(f"{n + 1}/{len(rows)} p_b {len(b)} M {M} ahead_by {a0 + M - len(b)}: mean |d| {per_point[-1]:.4f}",
              flush=True)
    stats = {k: {"tokens": len(d[k]), "mean_abs": statistics.mean(d[k]), "max_abs": max(d[k]),
                 "argmax_agree": statistics.mean(agree[k]), "argmax_steps": len(agree[k])} for k in d}
    stats.update(mount_points=len(rows), probes=len(exact), greedy16_exact=statistics.mean(exact),
                 worst_point_mean_abs=max(per_point), eviction_max_abs=evicted, leaked_after=leaked())
    print(json.dumps(stats, indent=1))
    assert stats["leaked_after"] == 0
    assert stats["k2"]["mean_abs"] <= MEAN_ABS and stats["k2"]["argmax_agree"] >= ARGMAX
    assert max(evicted) <= EVICTION


@live
def test_live_mount_continuation_keeps_offset_and_frees():
    """A mount reply's cursor continues with the mount still visible and B's position offset: its next tokens' log-probs
    equal one mounted request over reply + next tokens up to prefill-split numerics (SGLang without any mount moves
    such log-probs by 0.01-0.4 nats when the same tokens are prefilled in other chunks); many mount requests leave
    nothing locked. After /flush_cache the rows are rebuilt: the mount reply's continuation, a second mount from it
    (a prefix holding a mount) and a mount of its own rows (a source holding a mount) give the resident log-probs up
    to the same prefill-split numerics. Locked tokens are only those of the sources lending rows to a cached
    continuation (leaked: 0)."""
    rows = json.loads(gzip.open(DATA).read())[:3]
    params = {"sampling_params": {"max_new_tokens": 8, "temperature": 0}}
    d = []
    for r in rows:
        b, a, a0, M = r["b"], r["a"], r["a0"], r["M"]
        cb, ca = cursor_of(b), cursor_of(a)
        mount = [{"cursor": ca, "start": a0, "end": a0 + M}]
        q, ans = r["probes"][0]["q"], r["probes"][0]["ans"]
        first = gen({"cursor": cb, "input_ids": q, "mount": mount, **params})
        out = first["output_ids"]
        lp_cont, am_cont = scored(first["meta_info"]["id"], ans, len(b) + M + len(q) + len(out))  # reply's cursor
        lp_once, am_once = scored(cb, q + out + ans, len(b) + M + len(q) + len(out), mount)  # one mounted request
        assert am_cont == am_once
        d += [abs(x - y) for x, y in zip(lp_cont, lp_once)]
    print(f"continuation vs one mounted request: {len(d)} tokens, mean |d| {statistics.mean(d):.4f}, max {max(d):.4f}")
    assert statistics.mean(d) <= 0.05
    for _ in range(30):
        gen({"cursor": cb, "input_ids": q, "mount": mount, **params})
    assert leaked() == 0
    r1, n1 = first["meta_info"]["id"], len(b) + M + len(q) + len(out)
    second = gen({"cursor": r1, "input_ids": ans, "mount": mount, **params})
    r2, n2 = second["meta_info"]["id"], n1 + len(ans) + M + len(second["output_ids"])
    own = [{"cursor": r1, "start": len(b) + M, "end": len(b) + M + len(q)}]  # r1's q rows, after its placeholders
    probes = [lambda: scored(r1, q + ans, n1), lambda: scored(r2, q + ans, n2),  # q + ans: enough tokens to average
              lambda: scored(cb, q + q + ans, len(b) + len(q) + len(q), own)]
    resident = [f() for f in probes]
    assert post("/flush_cache", {})[0] == 200
    for f, (lp, am) in zip(probes, resident):
        lp2, am2 = f()
        print(f"rebuilt vs resident: {len(lp)} tokens, mean |d| {statistics.mean(abs(x - y) for x, y in zip(lp, lp2)):.4f}")
        assert am2 == am and statistics.mean(abs(x - y) for x, y in zip(lp, lp2)) <= 0.05
    assert leaked() == 0


live_url = pytest.mark.skipif(not URL, reason="needs OCTOPUS_SGLANG_URL")
CHAT = {"model": "m", "temperature": 0, "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": False}}
NOTES = gzip.decompress((Path(__file__).parent / "fixtures/mount_text.txt.gz").read_bytes()).decode()[:4000]


def chat(body: dict) -> dict:
    code, r = post("/v1/chat/completions", {**CHAT, "logprobs": True, "return_prompt_token_ids": True,
                                            "return_token_ids": True, **body})
    assert code == 200, r
    return r


def stream(path: str, body: dict) -> list[dict]:
    """The events of a streaming request (SSE `data:` lines but [DONE])."""
    req = urllib.request.Request(URL + path, json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return [json.loads(line[6:]) for line in map(bytes.decode, r) if line.startswith("data: {")]


def stored(r: dict) -> int:
    """A chat reply's cursor length: prompt + output without its stop + the terminator (eos), + 1 when cut by length."""
    c = r["choices"][0]
    return len(c["prompt_token_ids"]) + len(c["response_token_ids"]) + (c["finish_reason"] == "length")


@live_url
def test_live_chat_mount_equals_generate_and_streams():
    """(a) A chat request with read_thread's mount (branch B reads branch A's turns) and one continuing its reply's
    cursor give /generate's tokens and log-probs on the same sequence (/generate's mount: the segment of A's tokens the
    chat mount names; its input ids: the chat prompt's after B's tokens and the placeholders), resident and after
    /flush_cache (rebuilt). (b) Streaming chat and /generate give the non-streaming text, tokens and log-probs, also
    when the request starts with a rebuild. Nothing stays locked."""
    user = lambda text: {"messages": [{"role": "user", "content": text}]}  # noqa: E731
    r0 = chat(user(f"Notes:\n{NOTES}\nName two topics in these notes."))
    a = chat({"cursor": r0["id"], **user("Branch A: summarize the first paragraph.")})
    a2 = chat({"cursor": a["id"], **user("Branch A: and the second one?")})
    b = chat({"cursor": r0["id"], **user("Branch B: list the people named.")})
    im = r0["choices"][0]["prompt_token_ids"][0]  # <|im_start|>: where a turn starts
    turn = lambda r: len(p := r["choices"][0]["prompt_token_ids"]) - 1 - p[::-1].index(im)  # noqa: E731
    mount = [{"cursor": a2["id"], "start": turn(a), "end": turn(a2)}]  # "from" a's turn to a2's, in a2's tokens
    read = {"cursor": b["id"], "mount": {"from": a["id"], "to": a2["id"], "thread": "A"},
            **user("Branch A's work is above. In one sentence, what did branch A find?")}
    params = {"sampling_params": {"temperature": 0, "max_new_tokens": 48, "skip_special_tokens": False},
              "return_logprob": True}

    def same(c, g, what, exact=True):
        """exact: both ran on the same cache state; else the second matched the first's cached path deeper and
        prefilled from another split (mean |d| <= 0.05, the resident-vs-rebuilt bar above)."""
        lc = [x["logprob"] for x in c["choices"][0]["logprobs"]["content"]]
        lg = [x[0] for x in g["meta_info"]["output_token_logprobs"]]
        d = [abs(x - y) for x, y in zip(lc, lg)]
        print(f"{what}: {len(lg)} tokens, |d logprob| chat vs /generate max {max(d):.2e} mean {statistics.mean(d):.2e}")
        assert c["choices"][0]["response_token_ids"] == g["output_ids"] and len(lc) == len(lg)
        assert max(d) <= EVICTION if exact else statistics.mean(d) <= 0.05

    def pair(body, n, flush, **gmount):
        """The chat request and the /generate request on the same sequence: the chat prompt's ids after the n tokens
        of the cursor (and the mount's placeholders). After a flush each, both rebuild."""
        if flush:
            assert post("/flush_cache", {})[0] == 200
        c = chat(body)
        g = {"cursor": body["cursor"], "input_ids": c["choices"][0]["prompt_token_ids"][n:], **params, **gmount}
        if flush:
            assert post("/flush_cache", {})[0] == 200
        return c, gen(g), g

    M = mount[0]["end"] - mount[0]["start"]
    for flush in (False, True):
        c, g, gbody = pair(read, stored(b) + M, flush, mount=mount)
        assert c["metadata"]["octopus_mount"] == M and min(c["choices"][0]["prompt_token_ids"]) < 0
        same(c, g, f"mount read{' rebuilt' if flush else ''}")
        cont = {"cursor": c["id"], "messages": [{"role": "user", "content": "And what should branch B do next?"}]}
        c2, g2, gbody2 = pair(cont, stored(c), flush)
        # resident, the continuation's second request matches the first's cached path deeper (another prefill split)
        same(c2, g2, f"continuation of the mount reply{' rebuilt' if flush else ''}", exact=flush)
        # (b) on the same cache state as the non-streamed reply: resident, the mount read (new placeholders: only B is
        # matched, as for its reference); after a flush, both
        cases = (("mount read", read, c, gbody, g),) + ((("continuation", cont, c2, gbody2, g2),) if flush else ())
        for name, body, ref, _, _ in cases:  # (b) chat streams
            if flush:
                assert post("/flush_cache", {})[0] == 200
            chat_stream(name + (" rebuilt" if flush else ""), body, ref)
        for name, _, _, body, ref in cases:  # (b) /generate streams
            if flush:
                assert post("/flush_cache", {})[0] == 200
            last = stream("/generate", body)[-1]
            assert last["output_ids"] == ref["output_ids"], name
            assert last["meta_info"]["output_token_logprobs"] == ref["meta_info"]["output_token_logprobs"], name
            gen({"cursor": last["meta_info"]["id"], "input_ids": [im], "sampling_params": {"max_new_tokens": 1}})
            print(f"/generate stream {name}{' rebuilt' if flush else ''}: output ids and log-probs equal")
    plain = {"messages": [{"role": "user", "content": "Name one planet."}]}  # SGLang's own stream, the same relation
    assert post("/flush_cache", {})[0] == 200
    ref = chat(plain)
    assert post("/flush_cache", {})[0] == 200
    chat_stream("no cursor", plain, ref)
    assert leaked() == 0


def chat_stream(name: str, body: dict, ref: dict) -> None:
    """A streamed chat request gives the non-streamed reply ref's text and log-probs."""
    ev = stream("/v1/chat/completions", {**CHAT, "logprobs": True, **body})
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev if e["choices"])
    lps = [x["logprob"] for e in ev if e["choices"] and e["choices"][0].get("logprobs")
           for x in e["choices"][0]["logprobs"]["content"]]
    want = [x["logprob"] for x in ref["choices"][0]["logprobs"]["content"]]
    assert text == ref["choices"][0]["message"]["content"], (name, text)
    # SGLang's chat stream sends no chunk for the stop token (it has no text), so not its log-prob either
    assert lps == want[:len(want) - (ref["choices"][0]["finish_reason"] == "stop")], name
    print(f"chat stream {name}: {len(ev)} events, text and {len(lps)} log-probs equal")


@live_url
def test_live_openai_sdk_fork_and_read():
    """(c) An agent on the openai package, chat only: a main turn, a fork of two branches from its cursor (extra_body
    cursor), two more steps on branch A, branch B reading A's turns by read_thread's mount (extra_body mount, reply ids;
    the server confirms the mounted token count in metadata), then a streamed continuation of the mount reply that
    equals the non-streamed one (each from an empty cache: the same request on a warm cache prefills from another split
    and may decode other greedy tokens late in a reply)."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=URL + "/v1", api_key="none")

    def ask(messages, stream=False, **extra):
        return client.chat.completions.create(
            model="m", messages=messages, temperature=0, max_tokens=48, stream=stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, **extra})

    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    main = ask([{"role": "system", "content": "You coordinate two branches."}] + user(f"Notes:\n{NOTES[:2000]}\n"
                                                                                       "Plan the work."))
    fa = ask(user("You are branch A: find the main claim."), cursor=main.id)
    fb = ask(user("You are branch B: wait for branch A."), cursor=main.id)
    fa2 = ask(user("Branch A: state the claim in one line."), cursor=fa.id)
    fa3 = ask(user("Branch A: done."), cursor=fa2.id)
    read = ask(user("Branch A's work is above. Report it."), cursor=fb.id,
               mount={"from": fa.id, "to": fa3.id, "thread": "A"})
    mounted = read.to_dict()["metadata"]["octopus_mount"]
    assert read.choices[0].message.content and mounted > fa2.usage.completion_tokens
    assert post("/flush_cache", {})[0] == 200  # both from an empty cache: rebuilt the same way, the same prefill split
    done = ask(user("Finish: one sentence."), cursor=read.id)
    assert post("/flush_cache", {})[0] == 200
    text = "".join(ch.choices[0].delta.content or "" for ch in ask(user("Finish: one sentence."), stream=True,
                                                                   cursor=read.id) if ch.choices)
    print(f"sdk episode: read mounted {mounted} tokens of branch A, {read.usage.prompt_tokens} own prompt tokens; "
          f"streamed {text!r}")
    assert text == done.choices[0].message.content and leaked() == 0


@live_url
def test_live_openai_sdk_streaming_only():
    """A streaming-only agent on the openai package: every request streamed, each reply's chunk id its cursor. It forks,
    continues streamed replies, and branch B mounts the turns of streamed branch A (the server resolves "from"/"to"
    against the streamed replies: the same request unstreamed reports the mounted tokens), then continues the read."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=URL + "/v1", api_key="none")

    def ask(messages, **extra):  # (cursor, text) of a streamed reply
        chunks = list(client.chat.completions.create(
            model="m", messages=messages, temperature=0, max_tokens=48, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, **extra}))
        return chunks[0].id, "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)

    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    main = ask(user(f"Notes:\n{NOTES[:2000]}\nPlan the work."))
    fa = ask(user("You are branch A: find the main claim."), cursor=main[0])
    fb = ask(user("You are branch B: wait for branch A."), cursor=main[0])
    fa2 = ask(user("Branch A: state the claim in one line."), cursor=fa[0])
    mount = {"from": fa[0], "to": fa2[0], "thread": "A"}
    read = ask(user("Branch A's work is above. Report it."), cursor=fb[0], mount=mount)
    done = ask(user("Finish: one sentence."), cursor=read[0])
    check = chat({"cursor": fb[0], "mount": mount, "messages": user("Branch A's work is above. Report it.")})
    print(f"streaming-only episode: read {read[1]!r}; then {done[1]!r}; mounted {check['metadata']['octopus_mount']}")
    assert all(t for _, t in (main, fa, fb, fa2, read, done)) and check["metadata"]["octopus_mount"] > 0
    assert leaked() == 0


DENSE =os.environ.get("OCTOPUS_DENSE_MODEL")  # the served model's HF path: the exact reference runs in transformers
DENSE_CASES = ((1500, (8000, 3000), (2000, 2600)), (3000, (8000, 1500), (400, 1000)))  # p_B, (A's start, length), [s, e)


def dense_text(tok) -> list[int]:
    return tok(gzip.decompress((Path(__file__).parent / "fixtures/mount_text.txt.gz").read_bytes()).decode())["input_ids"]


def dense_logits(model, b, a, s, e, x, nxt, layers=None, embeds=None):
    """transformers (sdpa) on the exact mounted sequence b + a[:e] + x: B and A[:e] see only their own prefix, x sees
    its own predecessors, B, and A[s:e] in the attention layers `layers` (None: every layer; the others see B only),
    at positions [0, len(b)), [0, e), nxt, nxt + 1, ...; the logits of x's rows (embeds: the inputs' embeddings, to
    take gradients through)."""
    import torch

    nb, L = len(b), len(b) + e + len(x)
    mask = torch.ones(L, L, dtype=torch.bool, device="cuda").tril()
    mask[nb:nb + e, :nb] = False
    mask[nb + e:, nb:nb + s] = False
    own = mask.clone()
    own[nb + e:, nb:nb + e] = False  # the layers outside the subset: B and x's own predecessors

    def swap(module, args, kwargs):
        kwargs["attention_mask"] = own[None, None]
        return args, kwargs
    hooks = [m.self_attn.register_forward_pre_hook(swap, with_kwargs=True)
             for i, m in enumerate(model.model.layers) if layers is not None and i not in layers]
    pos = torch.tensor([[*range(nb), *range(e), *range(nxt, nxt + len(x))]], device="cuda")
    ids = None if embeds is not None else torch.tensor([b + a[:e] + x], device="cuda")
    try:
        return model(input_ids=ids, inputs_embeds=embeds, attention_mask=mask[None, None], position_ids=pos,
                     logits_to_keep=len(x)).logits[0]
    finally:
        for h in hooks:
            h.remove()


@pytest.mark.skipif(not (URL and DENSE), reason="needs OCTOPUS_SGLANG_URL serving OCTOPUS_DENSE_MODEL")
def test_live_dense_mount_equals_hf():
    """A model with only full-attention layers (no mamba flag, overlap off): peer ahead and peer behind, the mount
    request's teacher-forced log-probs (prefill) and its greedy reply (decode: CUDA graphs without a layer subset, eager
    under one, the batch holding mounted rows) vs transformers on the exact
    sequence: B's KV, A's KV rows [s, e) computed in A's own context, the request's tokens at the K2 positions (one
    forward, a mask per segment), A's rows in the layers the server mounts into (OCTOPUS_MOUNT_LAYERS, as the server
    reports it). Bar: the kernel's own error, the same server without a mount vs transformers on
    B + q + ans (bf16 SGLang vs bf16 transformers already differ by more than the Qwen3.5 bar of 0.03 on a 36-layer
    model), plus 0.01 nats and 0.02 argmax agreement. The same reference without the offset (and, under a layer subset,
    with the mount in every layer) is printed for scale. After
    /flush_cache the rows are rebuilt (B and A prefilled together: other prefill splits than the resident run, so not
    bit-identical to it) and meet the same bar against the reference; nothing stays locked."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DENSE)
    model = AutoModelForCausalLM.from_pretrained(DENSE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    text = dense_text(tok)
    layers = kv()["octopus_mount_layers"]

    def ref(b, a, s, e, x, nxt, layers=layers):
        """Log-probs of x[1:] and the argmax before each (dense_logits; no a: x right after B)."""
        with torch.no_grad():
            out = dense_logits(model, b, a, s, e, x, nxt, layers)[:-1]
        out = out.float().log_softmax(-1)[len(q) - 1:]  # the rows after q
        return out.gather(1, torch.tensor(x[len(q):], device="cuda")[:, None])[:, 0].tolist(), out.argmax(-1).tolist()

    def diff(lp, am, ref_lp, ref_am):
        return [abs(x - y) for x, y in zip(lp, ref_lp)], [x == y for x, y in zip(am, ref_am)]

    q = tok("\n\nContinue the mounted passage:\n")["input_ids"]
    st = {k: ([], []) for k in ("mount", "no_mount", "decode", "no_offset", "rebuilt")
          + (("every_layer",) if layers is not None else ())}
    for nb, (a0, na), (s, e) in DENSE_CASES:
        b, a = text[:nb], text[a0:a0 + na]
        M, nxt, ans = e - s, max(nb, e), a[e:e + 64]  # K2: after B and after the last mounted key
        assert post("/flush_cache", {})[0] == 200  # B's and A's KV from these prefills alone, not an earlier split
        cb, ca = cursor_of(b), cursor_of(a)
        mount = [{"cursor": ca, "start": s, "end": e}]
        lp = scored(cb, q + ans, nb + M + len(q), mount)
        ref_k2 = ref(b, a, s, e, q + ans, nxt)
        r = gen({"cursor": cb, "input_ids": q, "mount": mount, "return_logprob": True,
                 "sampling_params": {"max_new_tokens": 32, "temperature": 0}})
        add = {"mount": diff(*lp, *ref_k2), "no_offset": diff(*lp, *ref(b, a, s, e, q + ans, nb + M)),
               "no_mount": diff(*scored(cb, q + ans, nb + len(q)), *ref(b, [], 0, 0, q + ans, nb)),
               "decode": diff([x[0] for x in r["meta_info"]["output_token_logprobs"]], r["output_ids"],
                              *ref(b, a, s, e, q + r["output_ids"], nxt))}
        if layers is not None:
            add["every_layer"] = diff(*lp, *ref(b, a, s, e, q + ans, nxt, None))
        assert post("/flush_cache", {})[0] == 200  # eviction: B's and A's rows are rebuilt by prefill
        add["rebuilt"] = diff(*scored(cb, q + ans, nb + M + len(q), mount), *ref_k2)
        for k, (dd, ag) in add.items():
            st[k][0].extend(dd)
            st[k][1].extend(ag)
        print(f"p_b {nb} M {M} offset {nxt - nb - M}: " + ", ".join(
            f"{k} mean |d| {statistics.mean(dd):.4f} argmax {statistics.mean(ag):.3f}" for k, (dd, ag) in add.items()),
            flush=True)
    stats = {k: {"tokens": len(dd), "mean_abs": statistics.mean(dd), "max_abs": max(dd),
                 "argmax_agree": statistics.mean(ag)} for k, (dd, ag) in st.items()}
    stats["leaked_after"] = leaked()
    print(json.dumps({"mount_layers": layers, **stats}, indent=1))
    base = stats["no_mount"]
    assert stats["leaked_after"] == 0
    for k in ("mount", "decode", "rebuilt"):
        assert stats[k]["mean_abs"] <= base["mean_abs"] + 0.01
        assert stats[k]["argmax_agree"] >= base["argmax_agree"] - 0.02


@pytest.mark.skipif(not (DENSE and os.environ.get("OCTOPUS_MOUNT_LAYERS")),
                    reason="needs OCTOPUS_DENSE_MODEL (a CUDA device, flash-attn) and a layer subset OCTOPUS_MOUNT_LAYERS")
def test_tree_dense_subset_mount_equals_hf():
    """The trainer's prefix tree (flash-attn 2, bf16) under a layer subset (OCTOPUS_MOUNT_LAYERS as the trainer reads it)
    on [B + q + ans, A[:e]] with the mount {B: [(p_B, A, s, e)]} against dense_logits with the same layers (sdpa, bf16),
    on test_live_dense_mount_equals_hf's cases: the answer log-probs (mean |d| within the tree's own error without a
    mount + 0.01) and the gradient of their sum with respect to every input embedding (B's, A's, the answer's; relative
    error within the tree's own without a mount + 0.05). The reference with the mount in every layer is printed for
    scale. Without a subset, A's bf16 gradient in the peer-behind case is ill-conditioned, not wrong: equivalent bf16
    computations of the same function scatter along one direction on a few A tokens (their errors |cos| ≥ 0.96).
    Relative error vs fp32: tree (flash-attn 2) 1.47 (norm 41 vs 20), dense reference with rows A, B, x 0.99, per-node
    memory-efficient sdpa on the tree 0.96, fp32-internal math attention 0.79, the original reference 0.54 on the
    opposite side (norm 19). The fp32 tree and reference differ by 4e-4 on A (1e-3 on its first mounted tokens) vs
    ≤ 6e-5 elsewhere: the same amplification at fp32 rounding. Over 8 cases the tree is not systematically worse (a2:
    0.53 vs 0.26–1.90). Not a tree defect; an A-gradient bar needs the spread of equivalent bf16 algorithms or an fp32
    reference, so this check runs under a subset."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from octopus.rl import prefix_tree
    from octopus.rl.prefix_tree import patch_attention, tree_inputs

    tok = AutoTokenizer.from_pretrained(DENSE)
    model = AutoModelForCausalLM.from_pretrained(DENSE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.requires_grad_(False)
    patch_attention()
    text, q, layers = dense_text(tok), tok("\n\nContinue the mounted passage:\n")["input_ids"], prefix_tree.MOUNT_LAYERS
    embed = model.model.embed_tokens

    def run_tree(seqs, mounts):
        """Answer log-probs and the gradient of their sum per flat token (each sequence's last token excluded)."""
        model.set_attn_implementation("flash_attention_2")
        flat = torch.tensor([sum(seqs, [])], device="cuda")
        cu = torch.tensor(np.cumsum([0] + [len(x) for x in seqs]), dtype=torch.int32, device="cuda")
        tree = tree_inputs(flat, cu, mounts)
        if tree is None:  # one plain sequence: the trainer runs a plain forward
            emb = embed(flat).detach().requires_grad_()
            h, at = model.model(inputs_embeds=emb).last_hidden_state[0], (lambda r: r)
        else:
            meta, kw = tree
            emb = embed(kw.pop("input_ids")).detach().requires_grad_()
            h, at = model.model(inputs_embeds=emb, **kw).last_hidden_state[0], (lambda r: meta.pair_pos[meta.flat_pair[r]])
        rows = torch.arange(len(seqs[0]) - 65, len(seqs[0]) - 1, device="cuda")  # predicting the 64 answer tokens
        lp = model.lm_head(h[at(rows)]).float().log_softmax(-1)
        lp = lp.gather(1, flat[0, rows + 1][:, None])[:, 0]
        lp.sum().backward()
        ends = set(cu[1:].tolist())
        keep = torch.tensor([p for p in range(int(cu[-1])) if p + 1 not in ends], device="cuda")
        return lp.detach(), emb.grad[0][at(keep)].float(), keep

    def run_ref(b, a, s, e, x, nxt, layers, keep):
        """The same from dense_logits, its gradients in the tree's flat order (b + x, then a[:e])."""
        model.set_attn_implementation("sdpa")
        emb = embed(torch.tensor([b + a[:e] + x], device="cuda")).detach().requires_grad_()
        out = dense_logits(model, b, a, s, e, x, nxt, layers, emb)[-65:-1].float().log_softmax(-1)
        lp = out.gather(1, torch.tensor(x[-64:], device="cuda")[:, None])[:, 0]
        lp.sum().backward()
        nb, nx = len(b), len(x)
        order = torch.tensor([*range(nb), *range(nb + e, nb + e + nx), *range(nb, nb + e)], device="cuda")
        return lp.detach(), emb.grad[0][order[keep]].float()

    def compare(t, r, nb, nx):
        (lp, g), (lp_r, g_r) = t, r
        parts = {"B": slice(0, nb), "x": slice(nb, nb + nx - 1), "A": slice(nb + nx - 1, None)}
        return ((lp - lp_r).abs().mean().item(),
                {k: ((g[sl] - g_r[sl]).norm() / g_r[sl].norm()).item() for k, sl in parts.items() if len(g_r[sl])})

    res = {}
    for nb, (a0, na), (s, e) in DENSE_CASES:
        b, a = text[:nb], text[a0:a0 + na]
        x, nxt = q + a[e:e + 64], max(nb, e)
        lp, g, keep = run_tree([b + x, a[:e]], {0: [(nb, 1, s, e)]})
        res.setdefault("mount", []).append(compare((lp, g), run_ref(b, a, s, e, x, nxt, layers, keep), nb, len(x)))
        res.setdefault("every_layer", []).append(compare((lp, g), run_ref(b, a, s, e, x, nxt, None, keep), nb, len(x)))
        lp, g, keep = run_tree([b + x], None)  # the tree's own error: no mount, x right after B
        res.setdefault("no_mount", []).append(compare((lp, g), run_ref(b, [], 0, 0, x, nb, None, keep), nb, len(x)))
        torch.cuda.empty_cache()
    out = {k: {"mean_abs": statistics.mean(d for d, _ in v),
               "grad_rel": {p: max(gr[p] for _, gr in v) for p in v[0][1]}} for k, v in res.items()}
    print(json.dumps({"mount_layers": sorted(layers), **out}, indent=1))
    base = out["no_mount"]
    assert out["mount"]["mean_abs"] <= base["mean_abs"] + 0.01
    for p in ("B", "x"):
        assert out["mount"]["grad_rel"][p] <= base["grad_rel"][p] + 0.05
    assert out["mount"]["grad_rel"]["A"] <= max(base["grad_rel"].values()) + 0.05  # A: no gradient without a mount


HYBRID = os.environ.get("OCTOPUS_HYBRID_MODEL")  # a served hybrid model's HF path (mamba/GDN layers + attention layers)


@pytest.mark.skipif(not (URL and HYBRID), reason="needs OCTOPUS_SGLANG_URL serving OCTOPUS_HYBRID_MODEL")
def test_live_hybrid_mount_equals_hf():
    """A hybrid model (granite-4.0-h: mamba2 layers and attention layers without positional embeddings), on either
    mamba strategy, overlap on or off: the mount request's teacher-forced log-probs (prefill) and its greedy reply
    (decode) vs transformers (fp32), one token at a time (the torch mamba path continues a cached state only token by
    token) at the K2 positions, after B's cache whose attention layers get A's K/V rows [s, e) (computed in A's own
    context) after B's and whose recurrent layers keep B's state. Bar: the server's own error without a mount + 0.01
    nats. That error is large here (SGLang's granite differs from transformers' by ~0.4 nats on any plain request,
    0.87 on the first token), so argmax agreement (~0.83 without a mount) moves by near-ties and is only printed: on
    the old path (no_buffer, overlap off) the mount's is 0.82 vs 0.84. B's and A's prefill-only requests snapshot their recurrent state
    at their end (under extra_buffer otherwise cached only up to a state-chunk boundary, A's rows past it dropped; two
    cases mount up to A's end). Exactness of B's state inside SGLang: B's continuation from it equals one prefill of
    B + q + ans up to prefill-split numerics (0.05-0.06 nats on this model under either strategy, no_buffer's own
    cached state included). After /flush_cache the mount request rebuilds B and A (prefill-only, so
    snapshotted again) and meets the HF bar; nothing stays locked."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HYBRID)
    model = AutoModelForCausalLM.from_pretrained(HYBRID, dtype=torch.float32).cuda()
    text = tok(gzip.decompress((Path(__file__).parent / "fixtures/mount_text.txt.gz").read_bytes()).decode())["input_ids"]
    q = tok("\n\nContinue the mounted passage:\n")["input_ids"]

    def step(cache, ids, start):
        """Feed ids one token at a time after cache (None: from scratch) at positions start, start + 1, ...: the
        cache and each step's logits."""
        rows = []
        with torch.no_grad():
            for i, t in enumerate(ids):
                out = model(input_ids=torch.tensor([[t]], device="cuda"), past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([[start + i]], device="cuda"))
                cache = out.past_key_values
                rows.append(out.logits[0, -1].float())
        return cache, rows

    def spliced(cache_b, cache_a, s, e):
        """B's cache with A's attention K/V rows [s, e) appended in every attention layer; recurrent layers: B's."""
        cache = copy.deepcopy(cache_b)
        for layer, layer_a in zip(cache.layers, cache_a.layers):
            if hasattr(layer, "keys"):
                layer.keys = torch.cat([layer.keys, layer_a.keys[:, :, s:e]], 2)
                layer.values = torch.cat([layer.values, layer_a.values[:, :, s:e]], 2)
        return cache

    def ref(cache, x, nxt):
        """Log-probs of x after q and the argmax before each, x fed after cache at positions nxt, nxt + 1, ..."""
        out = torch.stack(step(copy.deepcopy(cache), x[:-1], nxt)[1]).log_softmax(-1)[len(q) - 1:]
        return out.gather(1, torch.tensor(x[len(q):], device="cuda")[:, None])[:, 0].tolist(), out.argmax(-1).tolist()

    def diff(lp, am, ref_lp, ref_am):
        return [abs(x - y) for x, y in zip(lp, ref_lp)], [x == y for x, y in zip(am, ref_am)]

    st = {k: ([], []) for k in ("mount", "no_mount", "decode", "rebuilt", "snapshot")}
    for nb, (a0, na), (s, e) in ((1500, (8000, 3000), (2000, 2600)), (3000, (8000, 1500), (900, 1500)),
                                 (2000, (5000, 2500), (100, 700)), (2500, (6000, 2000), (1300, 2000))):
        b, a = text[:nb], text[a0:a0 + na]
        M, nxt, ans = e - s, max(nb, e), text[a0 + e:a0 + e + 64]  # K2: after B and after the last mounted key
        assert post("/flush_cache", {})[0] == 200
        cb, ca = cursor_of(b), cursor_of(a)
        mount = [{"cursor": ca, "start": s, "end": e}]
        cache_b = step(None, b, 0)[0]
        cache_m = spliced(cache_b, step(None, a[:e], 0)[0], s, e)
        lp = scored(cb, q + ans, nb + M + len(q), mount)
        ref_k2 = ref(cache_m, q + ans, nxt)
        r = gen({"cursor": cb, "input_ids": q, "mount": mount, "return_logprob": True,
                 "sampling_params": {"max_new_tokens": 32, "temperature": 0}})
        from_state = scored(cb, q + ans, nb + len(q))  # B's state at its end: under extra_buffer its snapshot
        add = {"mount": diff(*lp, *ref_k2), "no_mount": diff(*from_state, *ref(cache_b, q + ans, nb)),
               "decode": diff([x[0] for x in r["meta_info"]["output_token_logprobs"]], r["output_ids"],
                              *ref(cache_m, q + r["output_ids"], nxt))}
        assert post("/flush_cache", {})[0] == 200  # eviction: B's and A's rows and B's state are rebuilt by prefill
        add["rebuilt"] = diff(*scored(cb, q + ans, nb + M + len(q), mount), *ref_k2)
        assert post("/flush_cache", {})[0] == 200
        one = gen({"input_ids": b + q + ans, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                   "return_logprob": True, "logprob_start_len": nb + len(q) - 1, "top_logprobs_num": 1})["meta_info"]
        add["snapshot"] = diff(*from_state, [x[0] for x in one["input_token_logprobs"][1:]],
                               [x[0][1] for x in one["input_top_logprobs"][1:]])
        for k, (dd, ag) in add.items():
            st[k][0].extend(dd)
            st[k][1].extend(ag)
        print(f"p_b {nb} M {M} a_end {e}/{na}: " + ", ".join(
            f"{k} mean |d| {statistics.mean(dd):.4f} argmax {statistics.mean(ag):.3f}" for k, (dd, ag) in add.items()),
            flush=True)
    stats = {k: {"tokens": len(dd), "mean_abs": statistics.mean(dd), "max_abs": max(dd),
                 "argmax_agree": statistics.mean(ag)} for k, (dd, ag) in st.items()}
    stats["leaked_after"] = leaked()
    print(json.dumps(stats, indent=1))
    base = stats["no_mount"]
    assert stats["leaked_after"] == 0
    for k in ("mount", "decode", "rebuilt"):
        assert stats[k]["mean_abs"] <= base["mean_abs"] + 0.01
    assert stats["snapshot"]["mean_abs"] <= 0.1 and stats["snapshot"]["argmax_agree"] >= 0.9


MULTI = os.environ.get("OCTOPUS_MOUNT_MULTI")  # prep-multi's cases with their offline K2 (Probe.k2) references


@pytest.mark.skipif(not (URL and MULTI), reason="needs OCTOPUS_SGLANG_URL and OCTOPUS_MOUNT_MULTI")
def test_live_multi_and_transitive_mounts_equal_offline_k2():
    """kv_read_probe.multi_cases: B mounts two segments (of A and of X) in one request, and B mounts a range of A that
    holds A's own read of X (A: a prefill-only mount request from A's prefix, placeholders inside B's range). Teacher-
    forced log-probs of B's tokens after the mount vs the offline Probe.k2 are within the single mount's error on the
    same tokens (B mounting A's segment alone: + 0.01 nats, - 0.02 argmax agreement, the dense bar's margins); after
    /flush_cache the same request rebuilds everything (A with its own read first) and meets the same bar (other prefill
    splits than the resident run: not bit-identical to it, as on the dense model), and the request sent again on the
    rebuilt rows gives the rebuilt log-probs (<= 1e-3).
    Zero copy: A's read allocates slots for its own tokens only (none for X's mounted rows, which it lends); at idle the
    nothing is locked; after a flush nothing is shared."""
    st, evicted, acct = {}, [], []
    for c in json.loads(gzip.open(MULTI).read()):
        assert post("/flush_cache", {})[0] == 200
        cur, held = {}, {}  # name -> cursor; name -> its mounts' (at, placeholders) on the server

        def ranges(segs):  # a range of a source in the server's tokens: its placeholders before or inside it added
            return [{"cursor": cur[src], "start": s + sum(m for a, m in held[src] if a <= s),
                     "end": e + sum(m for a, m in held[src] if a < e)} for src, s, e in segs]

        def make(name):
            ids, ms = c["seqs"][name], c["mounts"].get(name, [])
            for src in {x[0] for _, segs in ms for x in segs} - set(cur):
                make(src)
            if not ms:
                cur[name], held[name] = cursor_of(ids), []
                return
            [(at, segs)] = ms
            first, before = cursor_of(ids[:at]), kv()
            mount = ranges(segs)
            r = gen({"cursor": first, "input_ids": ids[at:], "mount": mount, "sampling_params": {"max_new_tokens": 0}})
            after, M = kv(), sum(m["end"] - m["start"] for m in mount)
            acct.append({"own": len(ids) - at, "allocated": before["octopus_available"] - after["octopus_available"],
                         "mounted": M, "shared": after["octopus_shared_holds"] - before["octopus_shared_holds"]})
            cur[name], held[name] = r["meta_info"]["id"], [(at, M)]
        for name in c["seqs"]:
            if name != c["reader"] and name not in cur:
                make(name)
        ids, [(at, segs)] = c["seqs"][c["reader"]], c["mounts"][c["reader"]]
        mount = ranges(segs)
        M, first = sum(m["end"] - m["start"] for m in mount), cursor_of(ids[:at])
        score = lambda: scored(first, ids[at:], c["score"] + M, mount)  # noqa: E731
        lp, am = score()
        assert leaked() == 0
        assert post("/flush_cache", {})[0] == 200  # eviction: B, A (with its read of X) and X are rebuilt by prefill
        lp2, am2 = score()
        lp3, _ = score()  # the same request again, on the rebuilt rows
        evicted.append(max(abs(x - y) for x, y in zip(lp2, lp3)))
        for kind, (l_, a_) in ((c["kind"], (lp, am)), (c["kind"] + " rebuilt", (lp2, am2))):
            st.setdefault(kind, ([], []))[0].extend(abs(x - y) for x, y in zip(l_, c["ref_lp"]))
            st[kind][1].extend(x == y for x, y in zip(a_, c["ref_argmax"]))
        print(f"{c['kind']} {[len(x) for x in c['seqs'].values()]} M {M}: mean |d| "
              f"{statistics.mean(abs(x - y) for x, y in zip(lp, c['ref_lp'])):.4f}, rebuilt "
              f"{statistics.mean(abs(x - y) for x, y in zip(lp2, c['ref_lp'])):.4f} (again {evicted[-1]:.4f})", flush=True)
    stats = {k: {"tokens": len(dd), "mean_abs": statistics.mean(dd), "max_abs": max(dd), "argmax_agree":
                 statistics.mean(ag)} for k, (dd, ag) in st.items()}
    assert post("/flush_cache", {})[0] == 200
    stats.update(eviction_max_abs=evicted, source_reads=acct, leaked_after=leaked(),
                 after_flush={k: v for k, v in kv().items() if k.startswith("octopus_")})
    print(json.dumps(stats, indent=1))
    assert all(a["allocated"] == a["own"] and a["shared"] == a["mounted"] for a in acct)
    assert stats["leaked_after"] == 0 and stats["after_flush"]["octopus_shared_holds"] == 0
    assert stats["after_flush"]["octopus_full_protected"] == 0
    base = stats["one"]  # the single mount of A's segment, on the same tokens
    for k in ("two", "transitive", "one rebuilt", "two rebuilt", "transitive rebuilt"):
        assert stats[k]["mean_abs"] <= base["mean_abs"] + 0.01 and stats[k]["argmax_agree"] >= base["argmax_agree"] - 0.02
    assert max(evicted) <= EVICTION


LORA = os.environ.get("OCTOPUS_MOUNT_LORA")  # the adapter the server serves as v1 (--enable-lora --lora-paths v1=<it>)


@pytest.mark.skipif(not (URL and DATA and LORA), reason="needs OCTOPUS_SGLANG_URL, OCTOPUS_MOUNT_DATA, OCTOPUS_MOUNT_LORA")
def test_live_mount_across_a_lora_version_switch():
    """Review of #185: SGLang keys the radix tree by LoRA version, and AReaL loads each version under a new name with no
    flush. B reads A under v1; the same adapter is loaded again as v2; the mount reply's continuation under v2 finds
    nothing in v2's namespace, so B and A are rebuilt there and the mount request goes again with the same
    placeholders. Then other mount points' prefills (v2) push the pool (a small --max-total-tokens) into evicting
    around that continuation, and after each it is scored again: the same log-probs every time (its mounted rows are
    never freed under it, whatever is evicted first; a rebuild after its own eviction repeats them), v2 equals v1 (the
    same weights), and at idle nothing is locked; a flush leaves the whole pool free and nothing shared."""
    rows = sorted(json.loads(gzip.open(DATA).read()), key=lambda r: len(r["b"]) + len(r["a"]))
    r = rows[0]
    b, a, a0, M, q, ans = r["b"], r["a"], r["a0"], r["M"], r["probes"][0]["q"], r["probes"][0]["ans"]
    g = lambda body, v: gen({**body, "lora_path": v})  # noqa: E731
    cb, ca = (g({"input_ids": x, "sampling_params": {"max_new_tokens": 0}}, "v1")["meta_info"]["id"] for x in (b, a))
    r1 = g({"cursor": cb, "input_ids": q, "mount": [{"cursor": ca, "start": a0, "end": a0 + M}],
            "sampling_params": {"max_new_tokens": 0}}, "v1")["meta_info"]["id"]

    def score(v):
        m = g({"cursor": r1, "input_ids": ans, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
               "return_logprob": True, "logprob_start_len": len(b) + M + len(q) - 1}, v)["meta_info"]
        return [x[0] for x in m["input_token_logprobs"][1:]]
    lp1 = score("v1")
    assert post("/load_lora_adapter", {"lora_name": "v2", "lora_path": LORA})[0] == 200
    lp2, drift, evicted = score("v2"), [], []
    for f in rows[1:12]:
        before = kv()["octopus_available"]
        g({"input_ids": f["b"], "sampling_params": {"max_new_tokens": 0}}, "v2")
        evicted.append(before - len(f["b"]) - kv()["octopus_available"] < 0)  # the pool gave back more than it took
        drift.append(max(abs(x - y) for x, y in zip(score("v2"), lp2)))
    stats = {"v1_vs_v2_max_abs": max(abs(x - y) for x, y in zip(lp1, lp2)), "drift_after_each_filler": drift,
             "filler_evicted": evicted, "leaked_after": leaked()}
    assert post("/flush_cache", {})[0] == 200
    stats["after_flush"] = {k: v for k, v in kv().items() if k.startswith("octopus_")}
    print(json.dumps(stats, indent=1))
    assert any(evicted) and max(drift) <= EVICTION and stats["leaked_after"] == 0
    assert stats["after_flush"]["octopus_shared_holds"] == 0 and stats["after_flush"]["octopus_full_protected"] == 0


# Mount points ------------------------------------------------------------------------------------------------------
def prep(results: str, probes: str, tokenizer: str, out: str, n: int = 40):
    """The first n/2 peer-ahead and n/2 peer-behind mount points of a #129 run-2 results file (in id order) with their
    token ids, rebuilt exactly as octopus.tools.kv_read_probe.run does (run from the probe's code directory: agent_dir
    paths are relative to it), and the offline K2 / none per-token answer log-probs and greedy continuations."""
    from transformers import AutoTokenizer

    from octopus.tools.kv_read_probe import QUESTION, materialize, slot
    tok = AutoTokenizer.from_pretrained(tokenizer)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    rows = {json.loads(x)["id"]: json.loads(x) for x in open(probes)}
    res = sorted((json.loads(x) for x in open(results)), key=lambda x: x["id"])
    picked = [x for x in res if x["ahead_by"] >= 0][:n // 2] + [x for x in res if x["ahead_by"] < 0][:n // 2]
    data = []
    for x in picked:
        row = rows[x["id"]]
        m = materialize(tok, row)
        b, gen_ids, pre, mount = enc(m["b_ctx"]), enc(m["gen"]), enc(m["a_pre"]), enc(m["mount"])
        assert (len(b), len(pre), len(mount)) == (x["p_b"], x["a0"], x["M"])
        k2, none = ({f["answer"]: f for f in x["cond"][c]["facts"]} for c in ("K2", "none"))
        ps = []
        for f in row["facts"]:
            pre_ans, ans = slot(enc, f["cue"], f["answer"])
            if ans is None:
                continue
            q = enc("<|im_start|>user\n" + QUESTION.format(a=row["a_id"], cue=f["cue"]) + "<|im_end|>\n") + gen_ids + pre_ans
            g = k2[f["answer"]]
            ps.append({"q": q, "ans": ans, "k2_lp": g["tok_lp"], "k2_gen": g["gen16"],
                       "none_lp": none[f["answer"]]["tok_lp"], "none_gen": none[f["answer"]]["gen16"]})
        data.append({"id": x["id"], "b": b, "a": pre + mount, "a0": len(pre), "M": len(mount), "probes": ps})
    with gzip.open(out, "wt") as fh:
        json.dump(data, fh)
    print(f"{len(data)} mount points, {sum(len(r['probes']) for r in data)} probes -> {out}")


def prep_multi(points: str, model: str, out: str, n: int = 4):
    """kv_read_probe.multi_cases of the mount points (prep's output) with their offline references: Probe.k2's
    log-probs and argmax for the reader's scored tokens (transformers, flash-attn 2 + fla, bf16, as #129's K2)."""
    import torch
    from transformers import AutoModelForCausalLM

    from octopus.tools.kv_read_probe import Probe, multi_cases
    p = Probe(AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2").cuda().eval())
    cases = multi_cases(json.loads(gzip.open(points).read()), n)
    with torch.no_grad():
        for c in cases:
            ids = c["seqs"][c["reader"]]
            lp = p.logprobs(p.k2(c["seqs"], c["mounts"], c["reader"])[0][c["score"] - 1:-1])
            c["ref_lp"] = lp.gather(1, torch.tensor(ids[c["score"]:], device="cuda")[:, None])[:, 0].tolist()
            c["ref_argmax"] = lp.argmax(-1).tolist()
            torch.cuda.empty_cache()
    with gzip.open(out, "wt") as fh:
        json.dump(cases, fh)
    print(f"{len(cases)} cases -> {out}")


def main():
    """`prep <results.jsonl> <probes.jsonl> <tokenizer> <out.json.gz> [n]` or
    `prep-multi <mount points.json.gz> <model> <out.json.gz> [n]`."""
    if sys.argv[1] == "prep":
        prep(*sys.argv[2:6], *map(int, sys.argv[6:7]))
    if sys.argv[1] == "prep-multi":
        prep_multi(*sys.argv[2:5], *map(int, sys.argv[5:6]))


if __name__ == "__main__":
    main()
