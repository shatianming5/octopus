#!/usr/bin/env python3
"""Idempotent SGLang 0.5.19 patch: attention-only mount of other cursors' tokens on /generate and /v1/chat/completions
(docs/cursor-protocol.md "Mount", K2 positions from docs/kv-read-probe.md). Needs apply_sglang_cursor.py first; the
lora patch may come in either order.

/generate takes `"mount": [{"cursor": A, "start": s, "end": e}, ...]` next to `"cursor": B`, /v1/chat/completions
read_thread's `"mount": {"from", "to", "thread"}` (turned into one such segment), streaming or not: both endpoints build
the sequence with octopus_mount.prepare and run it with octopus_mount.generate, which rebuilds missing rows before the
first output. The request's 8
full-attention layers see B's KV + each segment's KV for its source's tokens [s, e) (radix cache, its own pages,
shared with B's continuation without a copy) + its own, in prefill and decode; its GDN layers continue from B's state.
Mechanism and lifecycle: octopus/serving/kv_mount.py (imported from the code dir through the shim srt/octopus_mount.py,
as apply_sglang_cursor.py installs its modules: the code dir goes on PYTHONPATH). Needs the radix
cache and page size 1; overlap scheduling and either mamba strategy work (a model with only full-attention layers
mounts in every layer, unless OCTOPUS_MOUNT_LAYERS names a subset: then only those, on the triton backend); a mount on
another server config, or on a model with
sliding-window attention or another radix cache than UnifiedRadixCache (each scheduler decides at startup from its
own model runner and cache: Scheduler init), is a 400.

SGLang internals touched (one hook each): Req.init_next_round_input (mount after the prefix match),
PrefillAdder.add_one_req (the sources' pages locked for the budget check), PrefillAdder._req_inc_lock_ref (lock them),
UnifiedRadixCache.cache_finished_req (a prefill-only request snapshots its recurrent state; after the insert the kept
mounted rows become shared pages, the sources unlocked) and cache_unfinished_req (no insert while the row holds the
sources' pages), FullComponent.evict_component (an evicted node frees only the pages no other node holds) and
redistribute_on_node_split (the shared mark),
ForwardBatch.init_new (position offset; the mounted rows under a layer subset, a ForwardBatch field), TritonAttnBackend
init_forward_metadata / forward_extend / forward_decode (the KV view without the mounted rows for the attention layers
outside OCTOPUS_MOUNT_LAYERS), ModelRunner.forward (such a batch runs eager), Scheduler init (can it mount),
Scheduler.get_next_batch_to_run (prefill-only requests stay out of the next decode batch), Scheduler.get_internal_state
(the layer setting, KV accounting for leak checks),
TokenizerManager.abort_request (an abort also stops a rebuild), plus the request fields (GenerateReqInput,
ChatCompletionRequest), the /generate handler and OpenAIServingChat: the cursor branch gives the delta, then
octopus_mount.prepare_chat (read_thread's mount as /generate's, reply ids as the AReaL proxy takes them) and
octopus_mount.chat (generate) for both the streaming and the non-streaming path; the reply keeps its breakpoints.
  python3 patches/apply_sglang_mount.py /path/to/site-packages/sglang
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

MARK = "# octopus-mount"
CURSOR_MARK = "# octopus-cursor"
SHIM = runpy.run_path(str(Path(__file__).with_name("apply_sglang_cursor.py")))["SHIM"].format(mark=MARK, name="kv_mount")

EDITS = {
    "srt/managers/io_struct.py": [(
        """    cursor: Optional[str] = None  # octopus-cursor: input_ids are the tokens after this request id's
""",
        f"""    cursor: Optional[str] = None  # octopus-cursor: input_ids are the tokens after this request id's
    mount: Optional[List[Dict[str, Any]]] = None  {MARK}: [{{cursor, start, end}}, ...] read in the attention layers
""")],
    "srt/entrypoints/http_server.py": [
        ("""from sglang.srt.octopus_cursor_store import CURSORS  # octopus-cursor
""",
         f"""from sglang.srt.octopus_cursor_store import CURSORS  # octopus-cursor
from sglang.srt import octopus_mount  {MARK}
"""),
        ("""    if obj.cursor is not None:  # octopus-cursor: input_ids hold only the tokens after the cursor's
        parent = CURSORS.get(obj.cursor)
        if parent is None:
            raise HTTPException(status_code=404, detail=f"cursor {obj.cursor} is unknown or evicted")
        obj.input_ids = parent.tolist() + obj.input_ids
""",
         f"""    octopus_mount.prepare(obj, CURSORS, _global_state.tokenizer_manager)  {MARK}: cursor's tokens, mount, positions
"""),
        ("""                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
""",
         f"""                async for out in octopus_mount.generate(  {MARK}: rebuilds before the first output
                    _global_state.tokenizer_manager, obj, request
                ):
"""),
        ("""            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            if isinstance(ret, dict) and obj.input_ids is not None:  # octopus-cursor: input_ids + output_ids as generated
                CURSORS.put(ret["meta_info"]["id"], obj.input_ids + ret["output_ids"])
""",
         f"""            ret = await octopus_mount.generate(  {MARK}
                _global_state.tokenizer_manager, obj, request
            ).__anext__()
            if isinstance(ret, dict) and obj.input_ids is not None:  # octopus-cursor: input_ids + output_ids as generated
                CURSORS.put(ret["meta_info"]["id"], obj.input_ids + ret["output_ids"], obj.octopus_pos)
"""),
        ("""                    CURSORS.put(out["meta_info"]["id"], obj.input_ids + out["output_ids"])  # the last chunk: all of it
""",
         f"""                    CURSORS.put(out["meta_info"]["id"], obj.input_ids + out["output_ids"], obj.octopus_pos)  {MARK}
"""),
    ],
    "srt/entrypoints/openai/protocol.py": [(
        """    cursor: Optional[str] = None  # octopus-cursor: messages are those after this response id's reply
""",
        f"""    cursor: Optional[str] = None  # octopus-cursor: messages are those after this response id's reply
    mount: Optional[Dict[str, Any]] = None  {MARK}: read_thread's {{"from", "to", "thread"}} (octopus_mount.prepare_chat)
""")],
    "srt/entrypoints/openai/serving_chat.py": [
        ("""from sglang.srt.octopus_cursor_store import CURSORS
""",
         f"""from sglang.srt.octopus_cursor_store import CURSORS
from sglang.srt import octopus_mount  {MARK}
"""),
        ("""            parent = CURSORS.get(request.cursor)
            if parent is None:
                raise HTTPException(status_code=404, detail=f"cursor {request.cursor} is unknown or evicted")
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=parent.tolist() + render_delta(
""",
         f"""            result = MessageProcessingResult(  {MARK}: prepare checks the cursor and puts its tokens before them
                prompt="",
                prompt_ids=render_delta(
"""),
        ("""            return_prompt_token_ids=True,  # octopus-cursor: stored with the reply; the response shows them as requested
        )
""",
         f"""            return_prompt_token_ids=True,  # octopus-cursor: stored with the reply; the response shows them as requested
        )
        octopus_mount.prepare_chat(adapted_request, request, CURSORS, self.tokenizer_manager)  {MARK}: as on /generate
"""),
        ("""            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
""",
         f"""            async for content in octopus_mount.chat(  {MARK}: rebuilds before the first chunk
                self.tokenizer_manager, adapted_request, raw_request
            ):
"""),
        ("""            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
""",
         f"""            ret = await octopus_mount.chat(  {MARK}
                self.tokenizer_manager, adapted_request, raw_request
            ).__anext__()
"""),
        ("""            self._octopus_store(request, ret[0], response)
""",
         f"""            pos = adapted_request.octopus_pos  {MARK}
            self._octopus_store(request, ret[0], response, pos)
            if request.mount is not None:  {MARK}: the tokens mounted, which the client checks (tree)
                response.metadata = {{**(response.metadata or {{}}), "octopus_mount": pos[-1][0] - pos[-1][2][0]}}
"""),
        ("""                self._octopus_store(request, content, self._build_chat_response(request, [content], int(time.time())))
""",
         f"""                self._octopus_store(request, content, self._build_chat_response(request, [content], int(time.time())),
                                    adapted_request.octopus_pos)  {MARK}
"""),
    ],
    "srt/managers/schedule_batch.py": [(
        """            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

    def _compute_max_prefix_len(self, input_len: int) -> int:
""",
        f"""            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )
        from sglang.srt.octopus_mount import resolve  {MARK}: mount after the prefix; abort if it cannot

        if tree_cache is not None and resolve(self, tree_cache):
            self.init_next_round_input(tree_cache, cow_mamba)

    def _compute_max_prefix_len(self, input_len: int) -> int:
""")],
    "srt/managers/schedule_policy.py": [(
        """    def _req_inc_lock_ref(self, req: Req):
        result = self.tree_cache.inc_lock_ref(req.last_node)
""",
        f"""    def _req_inc_lock_ref(self, req: Req):
        result = self.tree_cache.inc_lock_ref(req.last_node)
        from sglang.srt.octopus_mount import lock  {MARK}

        lock(self, req)
"""), (
        """        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
""",
        f"""        from sglang.srt.octopus_mount import check_lock  {MARK}: the budget check sees A's pages locked too

        with self._lock_node(req.last_node), check_lock(self, req):
            # self.rem_total_tokens may decrease after the lock acquisition
""")],
    "srt/mem_cache/unified_radix_cache.py": [
        ("""        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return
""",
         f"""        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return
        from sglang.srt.octopus_mount import finish  {MARK}: a prefill-only request snapshots its state

        finish(self, req, is_insert, kv_len_to_handle)
"""),
        ("""        self._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)
""",
         f"""        self._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)
        from sglang.srt.octopus_mount import lend  {MARK}: the kept mounted rows stay their source's pages

        lend(self, req, result)
"""),
        ("""        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return
""",
         f"""        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return
        from sglang.srt.octopus_mount import unfinished  {MARK}: no insert while the row holds the mounted pages

        if unfinished(self, req):
            return
"""),
    ],
    "srt/mem_cache/unified_cache/components/full_component.py": [(
        """        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
""",
        f"""        ct = self.component_type
        from sglang.srt.octopus_mount import split  {MARK}: a split keeps the shared mark

        split(new_parent, child)
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
"""), (
        """            device_frees[self.component_type].append(cd.value)
            freed = len(cd.value)
""",
        f"""            from sglang.srt.octopus_mount import owned  {MARK}: not the pages another node holds

            own = owned(self.cache, node, cd.value)
            device_frees[self.component_type].append(own)
            freed = len(own)
""")],
    "srt/model_executor/forward_batch_info.py": [(
        """        # Init lora information (None on a draft runner: it is unadapted)
""",
        f"""        from sglang.srt.octopus_mount import hide_mounts, offset_positions  {MARK}

        offset_positions(ret, batch)
        hide_mounts(ret, batch)

        # Init lora information (None on a draft runner: it is unadapted)
"""), (
        """    lora_ids: Optional[List[str]] = None
""",
        f"""    lora_ids: Optional[List[str]] = None
    octopus_hidden: Optional[list] = None  {MARK}: each request's mounted rows (a field: runners copy the batch)
""")],
    "srt/layers/attention/triton_backend.py": [(
        """            lean_locks=lean_locks,
        )

    def init_cuda_graph_state(
""",
        f"""            lean_locks=lean_locks,
        )
        from sglang.srt.octopus_mount import layer_view  {MARK}: the KV rows without the mounted ones

        layer_view(self, forward_batch)

    def init_cuda_graph_state(
"""), (
        """            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None
""",
        f"""            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None
            from sglang.srt.octopus_mount import layer_kv  {MARK}: this layer's view

            kv_indptr, kv_indices = layer_kv(self.forward_metadata, layer, kv_indptr, kv_indices)
"""), (
        """        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
""",
        f"""        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            from sglang.srt.octopus_mount import layer_kv  {MARK}: this layer's view

            kv_indptr, kv_indices = layer_kv(self.forward_metadata, layer, kv_indptr, kv_indices)

        if layer.k_scale is not None and layer.v_scale is not None:
""")],
    "srt/model_executor/model_runner.py": [(
        """                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
""",
        f"""                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
                and not getattr(forward_batch, "octopus_hidden", None)  {MARK}: two KV views, eager
"""), (
        """                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
""",
        f"""                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
                and not getattr(forward_batch, "octopus_hidden", None)  {MARK}: two KV views, eager
""")],
    "srt/managers/tokenizer_manager.py": [(
        """    def abort_request(self, rid: str = "", abort_all: bool = False):
""",
        f"""    def abort_request(self, rid: str = "", abort_all: bool = False):
        from sglang.srt.octopus_mount import abort  {MARK}: also a mount request between attempts (no rid_to_state)

        abort(rid, abort_all)
""")],
    "srt/managers/scheduler.py": [(
        """        ret["startup_time"] = self.startup_time
""",
        f"""        ret["startup_time"] = self.startup_time
        from sglang.srt.octopus_mount import stats  {MARK}: KV accounting, for leak checks

        ret.update(stats(self.tree_cache))
"""), (
        """        self.tree_cache = result.tree_cache
""",
        f"""        self.tree_cache = result.tree_cache
        from sglang.srt.octopus_mount import serve  {MARK}: can this scheduler mount (its cache, the runner's window)

        serve(self.tree_cache, self.tp_worker.model_runner)
"""), (
        """            # Filter batch
            last_bs = last_batch.batch_size()
""",
        f"""            from sglang.srt.octopus_mount import prefill_only  {MARK}: kept out of the next decode batch

            chunked_req_to_exclude.update(prefill_only(last_batch.reqs))
            # Filter batch
            last_bs = last_batch.batch_size()
""")],
}


def patch(root: Path) -> bool:
    if CURSOR_MARK not in (root / "srt/entrypoints/http_server.py").read_text():
        raise SystemExit(f"apply patches/apply_sglang_cursor.py to {root} first")
    (root / "srt/octopus_mount.py").write_text(SHIM)
    changed = False
    for rel, edits in EDITS.items():
        path = root / rel
        text = path.read_text()
        if MARK in text:
            continue
        for needle, new in edits:
            if text.count(needle) != 1:
                raise SystemExit(f"needle not found exactly once in {path}: {needle[:60]!r}")
            text = text.replace(needle, new)
        path.write_text(text)
        changed = True
    return changed


if __name__ == "__main__":
    root = Path(sys.argv[1])
    found = runpy.run_path(str(root / "_version.py"))["__version__"]
    if found != "0.5.19":
        raise SystemExit(f"built for SGLang 0.5.19, found {found}; re-align these hooks: {', '.join(EDITS)}")
    print("patched" if patch(root) else "already patched", root)
