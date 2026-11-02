"""OCTOPUS_CURSOR_AUDIT=<dir>: per-request records of the cursor path on live traffic (docs/cursor-protocol.md
"Live acceptance"), one JSON line each in <dir>/<pid>.jsonl, summarized by scripts/rl_analysis/cursor_audit.py.
patches/areal-cursor.patch calls these only when the variable is set; everything O(L) here runs only then.

- prompt (the proxy, per chat request): a cursor request's prompt against AReaL's own full-mode concat
  (`_concat_prompt_token_ids_with_parent` on the full message list, rendered and tokenized), the build time of each,
  and the request's messages bytes against the full history's.
- hop (the rollout engine, per /generate segment): the bytes sent against all ids, and why a hop sent all ids.
"""

from __future__ import annotations

import json
import os
import time

DIR = os.environ.get("OCTOPUS_CURSOR_AUDIT", "")


def _log(rec: dict) -> None:
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _bytes(x) -> int:
    return len(json.dumps(x, default=str))


def prompt(completion_id, rid, cursor, interaction, concat_messages, tokenizer, tools, extra_body, ids, build_s):
    from areal.experimental.openai.client import _concat_prompt_token_ids_with_parent

    rec = {"t": time.time(), "kind": "prompt", "id": completion_id, "rid": rid, "cursor": cursor,
           "L": len(ids), "build_ms": round(build_s * 1e3, 3)}
    if cursor is not None:
        t = time.perf_counter()
        oracle = _concat_prompt_token_ids_with_parent(concat_messages, interaction.parent, tokenizer, tools=tools,
                                                      extra_body=extra_body)[0]
        r = interaction.parent.model_response
        rec.update(oracle_ms=round((time.perf_counter() - t) * 1e3, 3), match=oracle == ids,
                   delta_tokens=len(ids) - len(r.input_tokens) - len(r.output_tokens_without_stop) - 1,
                   msg_bytes=_bytes(interaction.remaining_messages), full_msg_bytes=_bytes(interaction.messages))
        if oracle != ids:
            rec["first_diff"] = next((i for i, (a, b) in enumerate(zip(oracle, ids)) if a != b),
                                     min(len(oracle), len(ids)))
            rec["oracle_L"] = len(oracle)
    else:
        rec["msg_bytes"] = rec["full_msg_bytes"] = _bytes(interaction.messages)
    _log(rec)


def hop(req, server, payload, full_payload, cursor, resend=False):
    p = req.parent
    why = (None if cursor is not None else "404 resend" if resend else "no parent" if p is None
           else "parent has no cursor" if p.cursor is None else "parent on another server" if p.cursor[0] != server
           else "images" if req.image_data else "parent not a prefix")
    _log({"t": time.time(), "kind": "hop", "rid": req.rid, "server": server, "n_input": len(req.input_ids),
          "ids_sent": len(payload["input_ids"]), "bytes": _bytes(payload), "full_bytes": _bytes(full_payload),
          "cursor": cursor is not None, "full_why": why, "resend_404": resend})
