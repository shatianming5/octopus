"""Dry run of the search environment, CPU only: one episode of octopus.envs.search.agent against an in-process fake
retrieval service (part A's API on a three-document corpus) with a scripted stub policy (an OpenAI-compatible chat
server): main forks two branches, each searches, opens a document and closes with done (its answer and evidence); main
answers from their findings; the main-first control answers without forking. Prints the episode's record
(reward.search_outcome: y, critical path, threads) and its trial dir.

    python -m octopus.envs.search.dry_run [OUT_DIR]   # needs harbor importable (the Harbor environment's python)
"""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DOCS = {
    "d1": {"title": "Zorblax Inc.", "url": "https://example.org/zorblax",
           "text": "Zorblax Inc. is a maker of garden robots. It was founded in 2004 by the engineer Mira Okonkwo."},
    "d2": {"title": "Mira Okonkwo", "url": "https://example.org/okonkwo",
           "text": "Mira Okonkwo (born 1971) is an engineer and the founder of Zorblax Inc."},
    "d3": {"title": "Garden robot", "url": "https://example.org/garden-robot",
           "text": "A garden robot mows, weeds or waters a garden without a person."},
}
QUESTION = {"qid": "dry-1", "question": "In which year was the founder of Zorblax Inc. born?", "answer": "1971"}
# the stub policy's chat template: declared tools as Qwen3.5 shows them (read_thread's card, core.read_card)
STUB_TEMPLATE = ('{%- if tools %}{{- "<tools>" }}{%- for tool in tools %}{{- "\\n" + tool | tojson }}{%- endfor %}'
                 '{{- "\\n</tools>\\n\\n" }}{%- endif %}{%- for m in messages %}{{- m.content }}{%- endfor %}')


@functools.cache
def stub_tokenizer() -> str:
    """The stub policy's tokenizer dir (OCTOPUS_TOKENIZER): its chat template, and a byte-level tokenizer.json (one
    token a UTF-8 byte, more than any real tokenizer's count: core.clip_results)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    path = Path(tempfile.gettempdir()) / f"octopus-stub-tokenizer-{os.getuid()}"  # one dir per user
    path.mkdir(exist_ok=True)
    part = path / f".chat_template.{os.getpid()}"  # written whole, then renamed: a reader never sees a partial file
    part.write_text(STUB_TEMPLATE)
    part.replace(path / "chat_template.jinja")
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
                               merges=[]))
    tok.pre_tokenizer, tok.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    part = path / f".tokenizer.{os.getpid()}"
    tok.save(str(part))
    part.replace(path / "tokenizer.json")
    return str(path)


class _Json(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def body(self) -> dict:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")


class FakeRetrieval(_Json):
    """The retrieval service API (POST /search, POST /open, GET /health) on DOCS; score = query words in the doc."""

    def do_GET(self):
        self.reply(200, {"corpus": "dry-run", "n_docs": len(DOCS), "backend": "bm25"})

    def do_POST(self):
        body = self.body()
        if self.path == "/search":
            words = set(body["query"].lower().split())
            scored = sorted(((sum(w in (d["title"] + " " + d["text"]).lower() for w in words), docid)
                             for docid, d in DOCS.items()), key=lambda x: (-x[0], x[1]))
            self.reply(200, {"results": [{"docid": docid, "title": DOCS[docid]["title"], "url": DOCS[docid]["url"],
                                          "snippet": DOCS[docid]["text"][:300], "score": float(score)}
                                         for score, docid in scored[:5] if score]})
        elif body.get("docid") not in DOCS:
            self.reply(404, {"error": "unknown docid"})
        else:
            doc, offset = DOCS[body["docid"]], int(body.get("offset") or 0)
            self.reply(200, {"docid": body["docid"], "title": doc["title"], "url": doc["url"], "offset": offset,
                             "text": doc["text"][offset:offset + int(body.get("length") or 6000)],
                             "total_length": len(doc["text"])})


def _call(name: str, **args) -> list[dict]:
    return [{"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def stub_reply(thread: str, turn: int, messages: list[dict]) -> list[dict] | None:
    """The scripted policy: tool calls of a thread's turn-th reply (main / control / a branch's thread name)."""
    if thread == "main":
        script = [_call("fork", tasks=["Find who founded Zorblax Inc.", "Find the birth year of Mira Okonkwo"]),
                  _call("done", answer="1971")]
        return script[turn - 1] if turn <= len(script) else None
    if thread == "control":
        script = [_call("search", query="Mira Okonkwo born"), _call("done", answer="1971")]
        return script[turn - 1] if turn <= 2 else None
    task = next(m["content"] for m in reversed(messages) if "Your task:" in str(m["content"]))  # thinking on: a tool result
    query = task.split("Your task: ", 1)[1].split("\n", 1)[0]
    script = [_call("search", query=query), _call("open", docid="d2" if "Okonkwo" in query else "d1"),
              _call("done", answer="1971" if "Okonkwo" in query else "Mira Okonkwo",
                    evidence="d2: Mira Okonkwo (born 1971)" if "Okonkwo" in query else
                    "d1: founded in 2004 by Mira Okonkwo")]
    return script[turn - 1] if turn <= 3 else None


class StubPolicy(_Json):
    """An OpenAI-compatible chat endpoint answering with stub_reply; a thread is the request's session id (a branch),
    or main / control (no fork tool offered)."""
    turns: collections.Counter = collections.Counter()
    lock = threading.Lock()

    def do_POST(self):
        body = self.body()
        tools = {t["function"]["name"] for t in body.get("tools") or []}
        thread = body.get("session_id") or ("main" if "fork" in tools else "control")
        with self.lock:
            self.turns[thread] += 1
            turn = self.turns[thread]
        calls = stub_reply(thread, turn, body["messages"])
        self.reply(200, {"id": f"{thread}#{turn}", "object": "chat.completion",
                         "usage": {"prompt_tokens": len(json.dumps(body["messages"])) // 4, "completion_tokens": 20},
                         "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                                      "message": {"role": "assistant", "content": "" if calls else "no tool call",
                                                  **({"tool_calls": calls} if calls else {})}}]})


def serve(handler) -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def main(out: Path) -> dict:
    from octopus.envs.search.agent import run_trial
    from octopus.envs.search.reward import search_outcome
    os.environ.update(OCTOPUS_SEARCH_URL=serve(FakeRetrieval), OCTOPUS_CHAT_URL=serve(StubPolicy) + "/v1/chat/completions",
                      OCTOPUS_MODEL="stub", OCTOPUS_HTTP_ATTEMPTS="1", OCTOPUS_PLAN_CONTROL="1",
                      OCTOPUS_TOKENIZER=stub_tokenizer())
    StubPolicy.turns.clear()
    trial = asyncio.run(run_trial(QUESTION, out))
    requests = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    outcome = search_outcome(trial, requests)
    summary = {"trial": str(trial), "y": outcome["y"], "forked": outcome["forked"], "C": outcome["C"],
               "C_wall": round(outcome["C_wall"], 3), "control": outcome["control"],
               "branches": [(b["key"], b["ended_with"]) for b in outcome["branches"]],
               "threads": dict(collections.Counter(t for _, t, _, _ in outcome["requests"])),
               "answers": json.loads((trial / "verifier/answers.json").read_text())}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="search-dry-run-")))
