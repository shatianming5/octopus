#!/usr/bin/env python3
"""Idempotent SGLang 0.5.19 patch: release the LoRA usage counter on queue aborts.

TokenizerManager._handle_abort_req drops a request that the scheduler aborted
from its waiting queue without releasing its LoRA counter (only 500/503 aborts
are released, in _handle_abort_finish_reason). The counter never reaches zero,
so a later /unload_lora_adapter of that version waits forever while holding
lora_update_lock, and the trainer's update_weights call times out after 600 s.
That happens when a pause(abort) catches requests in the queue.

  python3 patches/apply_sglang_lora_abort_release.py /path/to/site-packages/sglang
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

MARK = "# octopus-lora-abort-release"
NEEDLE = """        del self.rid_to_state[recv_obj.rid]

        state.out_list.append(out)
        state.event.set()
"""
FIX = f"""        del self.rid_to_state[recv_obj.rid]
        {MARK}: 500/503 aborts are released in _handle_abort_finish_reason.
        if (
            self.enable_lora
            and state.obj.lora_path
            and finish_reason.get("status_code")
            not in (HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.INTERNAL_SERVER_ERROR)
        ):
            asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

        state.out_list.append(out)
        state.event.set()
"""


def patch(root: Path) -> bool:
    path = root / "srt/managers/tokenizer_manager.py"
    text = path.read_text()
    if MARK in text:
        return False
    if text.count(NEEDLE) != 1:
        raise SystemExit(f"needle not found exactly once in {path}")
    path.write_text(text.replace(NEEDLE, FIX))
    return True


if __name__ == "__main__":
    root = Path(sys.argv[1])
    found = runpy.run_path(str(root / "_version.py"))["__version__"]
    if found != "0.5.19":
        raise SystemExit(f"built for SGLang 0.5.19, found {found}; re-align these hooks: "
                         "srt/managers/tokenizer_manager.py")
    print("patched" if patch(root) else "already patched", root)
