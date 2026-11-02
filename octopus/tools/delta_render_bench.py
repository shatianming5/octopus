"""Per-request prompt cost at L tokens of history: AReaL concat's full re-render today vs render_delta.

The history repeats the recorded bundle-e11 main thread's turns (tests/fixtures/delta_render_episode.json.gz) until it
holds about L tokens; the request is one fixed turn's tool results. Full: the whole conversation rendered and
tokenized, cut after the parent's eos, from a request carrying every message. Delta: render_delta of the new messages,
from a request carrying a cursor and those messages. Both requests carry the tools (7 KB here).

    python -m octopus.tools.delta_render_bench /path/to/Qwen3.5-4B-text
"""

import gzip
import json
import statistics
import sys
import time
from itertools import cycle
from pathlib import Path

from transformers import AutoTokenizer

from octopus.serving.delta_render import parse_tool_arguments, render_delta

EPISODE = Path(__file__).resolve().parents[2] / "tests/fixtures/delta_render_episode.json.gz"


def median_seconds(fn, repeats: int = 5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), out


def main(tokenizer_path: str) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    episode = json.loads(gzip.decompress(EPISODE.read_bytes()))
    tools, kwargs, eos = episode["tools"], episode["chat_template_kwargs"], tok.eos_token_id
    turns = []  # a reply and the messages after it
    for m in episode["main"][2:]:
        if m["role"] == "assistant":
            turns.append([m])
        else:
            turns[-1].append(m)
    rendered = lambda messages, **kw: list(tok.apply_chat_template(parse_tool_arguments(messages), tools=tools,
                                                                  **kwargs, **kw)["input_ids"])
    size = [len(tok.encode(json.dumps(t))) for t in turns]  # JSON tokens, scaled to the rendered thread's
    head = len(rendered(episode["main"][:2]))
    size = [n * (len(rendered(episode["main"])) - head) / sum(size) for n in size]

    def full(messages: list[dict], k: int) -> list[int]:
        ids = rendered(messages, add_generation_prompt=True)
        return ids[[p for p, t in enumerate(ids) if t == eos][k - 1] + 1:]

    print("| L (history tokens) | full render+tokenize ms | delta ms | delta tokens | full request bytes"
          " | delta request bytes |")
    print("|---|---|---|---|---|---|")
    for target in (8_000, 32_000, 128_000):
        messages, total = list(episode["main"][:2]), head
        for turn, n in cycle(zip(turns, size)):
            if total + n >= target:
                break
            messages += turn
            total += n
        messages += turns[0]  # the same request at every L: two tool results after a reply
        reply, delta = len(messages) - len(turns[0]), turns[0][1:]
        parent = rendered(messages[:reply + 1])
        full_s, full_tail = median_seconds(lambda: full(messages, parent.count(eos)))
        delta_s, delta_tail = median_seconds(lambda: render_delta(tok, delta, tools, kwargs))
        assert delta_tail == full_tail
        full_bytes = len(json.dumps({"messages": messages, "tools": tools}))
        delta_bytes = len(json.dumps({"cursor": "chatcmpl-" + "0" * 32, "messages": delta, "tools": tools}))
        print(f"| {len(parent):,} | {full_s * 1e3:.1f} | {delta_s * 1e3:.2f} | {len(delta_tail)} | {full_bytes:,}"
              f" | {delta_bytes:,} |")


if __name__ == "__main__":
    main(sys.argv[1])
