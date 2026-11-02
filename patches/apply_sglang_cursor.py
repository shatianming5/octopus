#!/usr/bin/env python3
"""Idempotent SGLang 0.5.19 patch: a cursor store, so a client never resends history (docs/cursor-protocol.md).

- /v1/chat/completions takes "cursor": <response id>, with messages = only those after that reply. Prompt ids = the
  cursor's tokens (AReaL concat's parent: prompt + output without stop + the reply's terminator, e.g. eos) +
  render_delta(messages); everything
  after prompt building is unchanged. Each single-choice reply, streamed or not, is stored under its response id (not
  one aborted before the scheduler took it: it has no prompt ids, and nothing to continue), with its context_stub for
  templates whose rendering reads the history.
- /generate takes "cursor": <id>, with input_ids = only the tokens after it. Input ids = the cursor's tokens (its
  input_ids + its output_ids as generated) + input_ids. Each single request with input_ids is stored, streamed or not
  (a stream from its last chunk, which holds the whole output; not under --incremental-streaming-output).
- No cursor: today's behaviour. Unknown or evicted cursor: HTTP 404 naming it.

Installs sglang/srt/octopus_delta_render.py and octopus_cursor_store.py as shims that import octopus/serving/delta_render.py
and octopus/serving/cursor_store.py from the code dir, so the server needs that dir on PYTHONPATH (after the overlay) and
runs the trainer's files, never a copy (budget: OCTOPUS_CURSOR_BYTES, default 2 GiB). Applies alongside
apply_sglang_lora_abort_release.py (another file):
  python3 patches/apply_sglang_cursor.py /path/to/site-packages/sglang
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

MARK = "# octopus-cursor"
SHIM = """import sys  {mark}: the code dir's octopus/serving/{name}.py itself (PYTHONPATH), one module, never a copy
from octopus.serving import {name}
sys.modules[__name__] = {name}
"""

EDITS = {
    "srt/entrypoints/openai/protocol.py": [(
        """    session_params: Optional[Dict] = None
    separate_reasoning: bool = True
""",
        f"""    session_params: Optional[Dict] = None
    cursor: Optional[str] = None  {MARK}: messages are those after this response id's reply
    separate_reasoning: bool = True
""")],
    "srt/managers/io_struct.py": [(
        """    session_params: Optional[Dict[str, Any]] = None
""",
        f"""    session_params: Optional[Dict[str, Any]] = None
    cursor: Optional[str] = None  {MARK}: input_ids are the tokens after this request id's
""")],
    "srt/entrypoints/openai/serving_chat.py": [
        ("""from sglang.srt.managers.io_struct import GenerateReqInput
""",
         f"""from sglang.srt.managers.io_struct import GenerateReqInput
from fastapi import HTTPException  {MARK}
from sglang.srt.octopus_cursor_store import CURSORS
from sglang.srt.octopus_delta_render import context_stub, render_delta, terminator
"""),
        ("""        if request.input_ids is not None:
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
""",
         f"""        if request.input_ids is not None or request.cursor is not None:  {MARK}
            prompt_kwargs = {{"input_ids": processed_messages.prompt_ids}}
"""),
        ("""            return_prompt_token_ids=request.return_prompt_token_ids
            or request.return_token_ids,
""",
         f"""            return_prompt_token_ids=True,  {MARK}: stored with the reply; the response shows them as requested
"""),
        ("""        # When input_ids are provided, skip template tokenization entirely;
        # only stop tokens and tool_call_constraint are needed.
        if request.input_ids is not None:
""",
         f"""        if request.cursor is not None:  {MARK}: the cursor's tokens + the new messages' tokens
            parent = CURSORS.get(request.cursor)
            if parent is None:
                raise HTTPException(status_code=404, detail=f"cursor {{request.cursor}} is unknown or evicted")
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=parent.tolist() + render_delta(
                    self.tokenizer_manager.tokenizer,
                    [m.model_dump(exclude_unset=True) for m in request.messages],
                    tools,
                    request.chat_template_kwargs,
                    CURSORS.ctx.get(request.cursor),
                ),
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=request.stop,
            )
        # When input_ids are provided, skip template tokenization entirely;
        # only stop tokens and tool_call_constraint are needed.
        elif request.input_ids is not None:
"""),
        ("""            int(time.time()),
        )

        return response
""",
         f"""            int(time.time()),
        )
        if len(ret) == 1:  {MARK}: the reply is a cursor
            self._octopus_store(request, ret[0], response)

        return response

    def _octopus_store(self, request, out, response, pos=()):  {MARK}
        \"\"\"Store a finished single-choice reply, streamed or not (out: its TokenizerManager output; response: the chat
        response built from it), as a cursor with its context stub; pos: its position breakpoints (a mount). A reply
        aborted before the scheduler took it has no prompt ids: nothing to continue.\"\"\"
        if "prompt_token_ids" not in out:
            return
        tok = self.tokenizer_manager.tokenizer
        reply = [c.message.model_dump(exclude_none=True) for c in getattr(response, "choices", ())]
        ctx = None  # none when the cursor's own ctx was evicted meanwhile: a request from it then raises
        if reply and (request.cursor is None or request.cursor in CURSORS.ctx):
            ctx = context_stub(CURSORS.ctx.get(request.cursor, [])
                               + [m.model_dump(exclude_unset=True) for m in request.messages] + reply)
        end = terminator(tok, reply[-1] if reply else {{}}, request.chat_template_kwargs)
        CURSORS.put_reply(out, tok.eos_token_id, tok.pad_token_id, end, ctx, pos)
"""),
        ("""            # Send finish_reason chunks for each index that completed
""",
         f"""            if len(finish_reasons) == 1 and not self.tokenizer_manager.server_args.incremental_streaming_output:
                {MARK}: a streamed reply is a cursor too; its last chunk holds the whole output (not so under
                # --incremental-streaming-output: those replies are not stored, a request from one is a 404)
                self._octopus_store(request, content, self._build_chat_response(request, [content], int(time.time())))
            # Send finish_reason chunks for each index that completed
"""),
    ],
    "srt/entrypoints/http_server.py": [
        ("""from fastapi.responses import ORJSONResponse, Response, StreamingResponse
""",
         f"""from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang.srt.octopus_cursor_store import CURSORS  {MARK}
"""),
        ('''    """Handle a generate request."""
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
''',
         f'''    """Handle a generate request."""
    if obj.cursor is not None:  {MARK}: input_ids hold only the tokens after the cursor's
        parent = CURSORS.get(obj.cursor)
        if parent is None:
            raise HTTPException(status_code=404, detail=f"cursor {{obj.cursor}} is unknown or evicted")
        obj.input_ids = parent.tolist() + obj.input_ids
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
'''),
        ("""            ).__anext__()
            return orjson_response(ret)
""",
         f"""            ).__anext__()
            if isinstance(ret, dict) and obj.input_ids is not None:  {MARK}: input_ids + output_ids as generated
                CURSORS.put(ret["meta_info"]["id"], obj.input_ids + ret["output_ids"])
            return orjson_response(ret)
"""),
        ("""                    yield b"data: " + dumps_json(out) + b"\\n\\n"
            except ValueError as e:
""",
         f"""                    yield b"data: " + dumps_json(out) + b"\\n\\n"
                if (obj.input_ids is not None and obj.is_single and "status_code" not in out["meta_info"]["finish_reason"]
                        and not _global_state.tokenizer_manager.incremental_streaming_output):  {MARK}: as unstreamed
                    CURSORS.put(out["meta_info"]["id"], obj.input_ids + out["output_ids"])  # the last chunk: all of it
            except ValueError as e:
"""),
    ],
}


def patch(root: Path) -> bool:
    for name in ("delta_render", "cursor_store"):
        (root / f"srt/octopus_{name}.py").write_text(SHIM.format(mark=MARK, name=name))
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
