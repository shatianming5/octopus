"""Mount load for slice 6 of docs/cross-branch-read.md: forced reads on a mount-capable SGLang server, straight to its
/generate (cursor + mount, as the AReaL engine sends them), on top of the rollout load of octopus/tools/rollout_load.py.

PAIRS episode-like thread pairs run at a time until DURATION seconds have passed. A pair is a fork: a root request of
8k-40k tokens, then two sibling threads from its reply, each taking 15-30 turns (a tool result of lognormal size,
median 500 tokens, sent from the thread's last reply, then a 40-500-token sampled reply with the policy LoRA, 0.5-8 s
of "tool time" between turns). Every 2-5 turns the first thread reads its sibling (the second one every 6-10 turns,
so it stays a long source): the read turn carries a 60-token tool result and mounts the end of the sibling's own
tokens before its last reply (never its fork prefix or the part up to rows it mounted itself), M lognormal with the
slice-7 audit's median 949 and p90 9,076, capped by what the sibling has. Later turns of the
reader continue from mounted cursors (a 404 there ends the thread, as in the AReaL engine). Each failed request is sent
3 times in all, 1 s apart (the engine's request_retries). One JSON line per request in OUT; a summary on stdout.

usage: python -m octopus.tools.mount_load SERVER TOKENIZER CORPUS_DIR PAIRS DURATION OUT [LORA]
LORA: the adapter name (rq1-v0; "" for none), or @FILE: the name in FILE at each request (rollout_load's version).
"""
import asyncio
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

server, tokenizer, corpus_dir, pairs, duration, out = sys.argv[1:7]
lora = sys.argv[7] if len(sys.argv) > 7 else "rq1-v0"
URL = f"http://{server}/generate"
DEADLINE = time.time() + float(duration)
TOK: list[int] = []
LOG = open(out, "a")


def lognormal(median: float, p90: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, random.lognormvariate(math.log(median), math.log(p90 / median) / 1.2816))))


def text(n: int) -> list[int]:
    i = random.randrange(len(TOK) - n)
    return TOK[i : i + n]


class Thread:
    def __init__(self, cursor: str, n: int):
        self.cursor, self.n, self.last_reply, self.fork = cursor, n, 0, n
        self.own_from = n  # its tokens from here on are its own (no rows it mounted)
        self.mounted = False  # its cursor holds placeholders


async def send(http, pair, who, parent: Thread | None, delta, mount=None):
    """One reply from parent's cursor (or a full send); (cursor, server length, reply length) or None on failure."""
    r = random.randint(40, 500)
    name = open(lora[1:]).read().strip() if lora.startswith("@") else lora or None
    body = {"input_ids": delta, "return_logprob": True, "stream": False, "lora_path": name,
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": r, "ignore_eos": True}}
    if parent is not None:
        body["cursor"] = parent.cursor
    if mount is not None:
        body["mount"] = [mount]
    M = 0 if mount is None else mount["end"] - mount["start"]
    n_in = (parent.n if parent else 0) + M + len(delta)
    row = {"t": round(time.time(), 1), "pair": pair, "thread": who, "kind": "mount" if mount else "turn",
           "cont": bool(parent and parent.mounted), "M": M, "n_in": n_in, "lora": name, "statuses": []}
    t0 = time.time()
    for attempt in range(3):
        try:
            async with http.post(URL, json=body) as resp:
                status = resp.status
                res = await resp.json(content_type=None) if status == 200 else await resp.text()
        except aiohttp.ClientError as err:  # e.g. a reused keep-alive connection reset: the engine retries those too
            status, res = "conn", repr(err)
        row["statuses"].append(status)
        if status == 200:
            break
        row["detail"] = res[:300]
        if attempt < 2:
            await asyncio.sleep(1)
    row["latency"] = round(time.time() - t0, 2)
    ok = status == 200
    if ok:
        row["out"] = len(res["output_ids"])
        row["prompt_ok"] = res["meta_info"]["prompt_tokens"] == n_in
    LOG.write(json.dumps(row) + "\n")
    LOG.flush()
    return (res["meta_info"]["id"], n_in + len(res["output_ids"]), len(res["output_ids"])) if ok else None


async def thread_loop(http, pair, me: Thread, sib: Thread, who: int):
    gaps = (2, 5) if who == 0 else (6, 10)
    gap = random.randint(*gaps)
    for _ in range(random.randint(15, 30)):
        await asyncio.sleep(random.uniform(0.5, 8))
        if time.time() > DEADLINE or me.n > 150_000:
            return
        mount, lo, e = None, sib.own_from, sib.n - sib.last_reply
        if gap <= 0 and e - lo >= 16:
            s = max(lo, e - lognormal(949, 9076, 16, 30_000))
            mount = {"cursor": sib.cursor, "start": s, "end": e}
        n_before = me.n
        got = await send(http, pair, who, me, text(60 if mount else lognormal(500, 3000, 30, 8000)), mount)
        if got is None:
            return
        me.cursor, me.n, me.last_reply = got
        gap -= 1
        if mount:
            me.mounted, me.own_from, gap = True, n_before + (e - s) + 1, random.randint(*gaps)


async def pair_loop(http, pair):
    while time.time() < DEADLINE:
        got = await send(http, pair, "root", None, text(random.randint(8000, 40000)))
        if got is None:
            continue
        a, b = Thread(got[0], got[1]), Thread(got[0], got[1])
        await asyncio.gather(thread_loop(http, pair, a, b, 0), thread_loop(http, pair, b, a, 1))
        pair += 1000


async def main():
    tok = AutoTokenizer.from_pretrained(tokenizer)
    for f in sorted(Path(corpus_dir).rglob("*.py")):
        TOK.extend(tok.encode(f.read_text(errors="ignore")))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as http:
        await asyncio.gather(*(pair_loop(http, p) for p in range(int(pairs))))
    LOG.close()
    rows = [json.loads(x) for x in open(out)]
    final = collections.Counter((r["kind"], r["cont"], r["statuses"][-1]) for r in rows)
    raw = collections.Counter(s for r in rows for s in r["statuses"])
    Ms = sorted(r["M"] for r in rows if r["kind"] == "mount" and r["statuses"][-1] == 200)
    print(json.dumps({"requests": len(rows), "final (kind, from a mounted cursor, status)": {str(k): v for k, v in
                      sorted(final.items(), key=str)}, "raw statuses": raw, "prompt mismatches": sum(r.get("prompt_ok") is False
                      for r in rows), "mounted M median/p90": Ms and [Ms[len(Ms) // 2], Ms[int(len(Ms) * 0.9)]]}))


asyncio.run(main())
