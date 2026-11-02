"""Cross-branch KV read probe: can sibling B use sibling A's turns mounted into B's 8 full-attention layers only?

Qwen3.5-4B has 24 Gated DeltaNet (GDN) and 8 full-attention layers. In the design under test, B "reads" A by appending
the K/V of A's selected turns, as A computed them, to B's full-attention caches; B's GDN layers keep B's own conv and
recurrent state. Conditions at each mount point (docs/kv-read-probe.md):
  none    B alone
  text    A's turns as one user message in B's context (every layer sees them)
  replay  A's turn tokens spliced into B's context (every layer recomputes them from B's state: no private GDN)
  K1      mount at A's original positions
  K2      mount, B continues at max(its next position, last mounted position + 1)
  K3      mount relocated to B's next positions (keys re-rotated by the offset), B continues after it
The mount is surgery on transformers' hybrid cache, which continues a multi-token input from a stored GDN state (the
conv context and the recurrent state as initial state): prefill B once, copy the cache, append, continue.

usage: python -m octopus.tools.kv_read_probe build --tokenizer MODEL --out PROBES.jsonl JOBS_DIR...
       python -m octopus.tools.kv_read_probe run --model MODEL [--adapter LORA] --probes PROBES.jsonl --out RESULTS.jsonl
       python -m octopus.tools.kv_read_probe report RESULTS.jsonl
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from octopus.tools.tree_sft.encode import _for_template

MAX_TURNS = 4  # the mount: A's last completed exchanges (assistant call + tool results) before B's request
MAX_MOUNT_CHARS = 12000
MAX_FACTS = 4
MAX_CTX = 64000  # tokens; longer contexts are skipped
PER_PAIR, PER_TREE = 2, 4  # mount points per ordered sibling pair and per tree
REPLY_TOKENS, GEN_TOKENS = 256, 96
CONDITIONS = ("none", "text", "replay", "K1", "K2", "K3")
QUESTION = ("A sibling branch ({a}) works on another task in parallel. One line of its tool output begins exactly "
            "like this:\n```\n{cue}\n```\nWhat text comes next on that line? Reply with only that text.")
TEXT_READ = "Read-only log of sibling branch {a} (it works on another task in parallel):\n"
IDENT = re.compile(r"[A-Za-z0-9_][\w./:-]{4,59}")
TAG = re.compile(r"\[(?:[^\]\n|]+ \| )?(?:token budget|context) left")  # a tool result's tag (OCTOPUS_READ_TOOL: [main | ...)


def idents(text: str) -> set[str]:
    """File paths, dotted names, test ids, numbered tokens: strings specific enough to say whose work a text is."""
    out = set()
    for m in IDENT.findall(text):
        m = m.rstrip(".:-/")
        if len(m) >= 5 and re.search(r"[0-9_/.]", m) and re.search(r"[A-Za-z]", m):
            out.add(m)
    return out


def facts(tool_texts: list[str], mounted: str, known: str, k: int = MAX_FACTS) -> list[dict]:
    """(cue, answer) pairs from A's tool output lines: the answer is the line's last identifier, absent (as is the line
    up to it) from everything B already has; the cue is unique in the mount, so it names one line."""
    cands, seen = [], set()
    for text in tool_texts:
        for line in text.splitlines():
            if not 12 <= len(line) <= 200 or TAG.match(line):
                continue
            ms = [m for m in IDENT.finditer(line) if m.group().rstrip(".:-/") in idents(m.group())]
            if not ms:
                continue
            m = ms[-1]
            answer, cue = m.group().rstrip(".:-/"), line[:m.start()]
            if (len(cue.strip()) < 8 or answer in seen or answer in known or line[:m.start() + len(answer)] in known
                    or mounted.count(cue) != 1):
                continue
            seen.add(answer)
            cands.append({"cue": cue, "answer": answer})
    step = max(1, len(cands) / k)
    return [cands[int(i * step)] for i in range(min(k, len(cands)))]


def slot(enc, cue: str, answer: str) -> tuple[list[int], list[int] | None]:
    """Completion-style answer slot: B's reply starts with the cue line (trailing spaces stripped); the answer is the
    tokens of the joint encoding after the prefix's tokens, None when those are not a prefix of the joint encoding."""
    pre, joint = enc("```\n" + cue.rstrip(" ")), enc("```\n" + cue + answer)
    return pre, (joint[len(pre):] if joint[:len(pre)] == pre else None)


def render(tok, messages: list[dict], tools: list[dict], gen: bool = False) -> str:
    return tok.apply_chat_template([_for_template(m) for m in messages], tools=tools, tokenize=False,
                                   add_generation_prompt=gen, enable_thinking=False)


def exchanges(msgs: list[dict], lo: int, hi: int) -> list[int]:
    """Start indices of the exchanges (an assistant message and the tool results after it) inside msgs[lo:hi]."""
    return [i for i in range(lo, hi) if msgs[i]["role"] == "assistant"]


def load(agent_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    tree = json.loads((agent_dir / "tree.json").read_text())
    threads: dict[str, list[dict]] = {}
    for line in (agent_dir / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("http_code") == 200:
            threads.setdefault(r["thread"], []).append(r)
    return tree, threads


def materialize(tok, row: dict, tree: dict | None = None) -> dict:
    """The texts of a mount point: B's context, its generation prompt and recorded reply, A's turns before the mount
    and the mount itself (each a suffix of the longer rendering, so token sequences concatenate exactly)."""
    tree = tree or json.loads((Path(row["agent_dir"]) / "tree.json").read_text())
    tools, fork = tree["tools"], tree["forks"][row["fork"]]
    main = tree["main"][:fork["prefix_len"]]
    a_msgs, b_msgs = fork["branches"][row["a"]]["messages"], fork["branches"][row["b"]]["messages"]
    k, s, ja = row["k"], row["s"], row["ja"]
    b_ctx = render(tok, main + b_msgs[:k], tools)
    gen = render(tok, main + b_msgs[:k], tools, gen=True)
    full = render(tok, main + b_msgs[:k + 1], tools)
    a_pre = render(tok, main + a_msgs[:s], tools)
    a_full = render(tok, main + a_msgs[:ja], tools)
    if not (gen.startswith(b_ctx) and full.startswith(gen) and a_full.startswith(a_pre)):
        raise ValueError("the chat template does not render these conversations as prefixes of each other")
    return {"b_ctx": b_ctx, "gen": gen[len(b_ctx):], "reply": full[len(gen):], "a_pre": a_pre,
            "mount": a_full[len(a_pre):], "a_task": a_msgs[0]["content"], "a_msgs": a_msgs[s:ja]}


def build(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(0)
    rows, skipped = [], {}
    dirs = sorted(Path(p).parent for d in args.jobs_dirs for p in glob.glob(f"{d}/**/agent/tree.json", recursive=True))
    for agent_dir in dirs:
        tree, threads = load(agent_dir)
        pairs = [(fi, a, b) for fi, fork in enumerate(tree.get("forks", [])) for a in range(len(fork.get("branches") or []))
                 for b in range(len(fork["branches"])) if a != b]
        rng.shuffle(pairs)
        tree_rows = []
        for fi, a, b in pairs:
            if len(tree_rows) >= PER_TREE:
                break
            tree_rows += pair_rows(tok, agent_dir, tree, threads, fi, a, b, skipped)[:PER_TREE - len(tree_rows)]
        rows += tree_rows
    rng.shuffle(rows)  # run --limit takes a random subset
    for i, r in enumerate(rows):
        r["id"] = i
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} mount points from {len({r['agent_dir'] for r in rows})} trees, "
          f"{sum(len(r['facts']) for r in rows)} retrieval probes; skipped {skipped}")


def pair_rows(tok, agent_dir, tree, threads, fi, a, b, skipped) -> list[dict]:
    """Mount points of ordered sibling pair (A=a, B=b): B's requests at which A has completed exchanges B lacks."""
    fork = tree["forks"][fi]
    A, B = fork["branches"][a], fork["branches"][b]
    req_a, req_b = threads.get(f"fork{fi}/{A['branch_id']}", []), threads.get(f"fork{fi}/{B['branch_id']}", [])
    a_asst, b_asst = exchanges(A["messages"], 0, len(A["messages"])), exchanges(B["messages"], 0, len(B["messages"]))
    if A.get("rewinds") or B.get("rewinds") or len(req_a) < len(a_asst) or len(req_b) < len(b_asst):
        skipped["thread"] = skipped.get("thread", 0) + 1
        return []
    cands = []
    for j in range(1, len(b_asst)):  # B's j-th request, after at least one exchange of its own
        t = float(req_b[j]["t"])
        done = [i for i, r in enumerate(req_a[:len(a_asst)]) if float(r["t"]) <= t]  # A's requests sent by then
        ja = a_asst[done[-1]] if done else 0  # A's messages its latest request carried
        starts = exchanges(A["messages"], 1, ja)[-MAX_TURNS:]
        while starts and sum(len(json.dumps(m)) for m in A["messages"][starts[0]:ja]) > MAX_MOUNT_CHARS:
            starts = starts[1:]
        if starts:
            cands.append((j, t, starts[0], ja, done[-1]))
    # spread over B's run: try its latest request, then the middle one, then their neighbours
    mid = len(cands) // 2
    order = [x for pair in zip(range(len(cands) - 1, mid - 1, -1), range(mid - 1, -1, -1)) for x in pair]
    order += [i for i in range(len(cands) - 1, -1, -1) if i not in order]
    main = tree["main"][:fork["prefix_len"]]

    def offset(msgs, asst, req, i):  # our rendering of a thread's i-th request minus the prompt tokens it recorded
        text = render(tok, main + msgs[:asst[i]], tree["tools"], gen=True)
        return len(tok(text, add_special_tokens=False)["input_ids"]) - int(req[i]["prompt_tokens"])

    out, base = [], {}
    for j, t, s, ja, d in (cands[i] for i in order):
        if len(out) == PER_PAIR:
            break
        row = {"agent_dir": str(agent_dir), "fork": fi, "a": a, "b": b, "a_id": A["branch_id"],
               "b_id": B["branch_id"], "k": b_asst[j], "j": j, "s": s, "ja": ja, "t": t,
               "prompt_tokens": int(req_b[j]["prompt_tokens"])}
        try:
            m = materialize(tok, row, tree)
        except ValueError:
            skipped["template"] = skipped.get("template", 0) + 1
            continue
        # our rendering of the shared prefix is a constant few tokens shorter than the server's (70 at the system
        # prompt in e11); each thread's own turns must render exactly: the offset at B's (A's) request is the one at
        # its first request
        n_ctx = len(tok(m["b_ctx"] + m["gen"], add_special_tokens=False)["input_ids"])
        if "b" not in base:
            base["b"], base["a"] = offset(B["messages"], b_asst, req_b, 0), offset(A["messages"], a_asst, req_a, 0)
        if n_ctx - row["prompt_tokens"] != base["b"] or offset(A["messages"], a_asst, req_a, d) != base["a"]:
            skipped["prompt_mismatch"] = skipped.get("prompt_mismatch", 0) + 1
            continue
        row["offset"] = base["b"]
        if n_ctx > MAX_CTX:
            skipped["long"] = skipped.get("long", 0) + 1
            continue
        tools_out = [x["content"] for x in m["a_msgs"] if x["role"] == "tool" and isinstance(x.get("content"), str)]
        row["facts"] = facts(tools_out, m["mount"], m["b_ctx"])
        if not row["facts"]:
            skipped["no_fact"] = skipped.get("no_fact", 0) + 1
            continue
        a_text = m["a_task"] + m["mount"]
        b_text = B["messages"][0]["content"] + m["b_ctx"] + m["reply"]
        row["a_only"] = sorted((idents(a_text) | {A["branch_id"]}) - idents(m["b_ctx"] + m["reply"]) - {B["branch_id"]})
        row["b_only"] = sorted(idents(b_text) - idents(m["a_pre"] + m["mount"]))
        out.append(row)
    return out


# ---------------------------------------------------------------- GPU side

def multi_cases(points: list[dict], n: int) -> list[dict]:
    """Two-segment and transitive mounts from mount points (tests/test_sglang_mount.py prep: b, a, a0, M, probes with q
    and ans token ids). For each pair (i, j) of the 2n shortest points the reader is i's b + q + ans, scored from q's
    second token on (score: the first scored index; q's first is predicted from B's last row, before the mount). "one"
    mounts i's segment of A alone (the single-mount level on the same tokens), "two"
    mounts i's segment of A (i's a) and j's segment of X (j's a) at the end of b; "transitive" mounts only i's segment
    of A, where A has read j's segment of X at the middle of that segment (Probe.k2's input format)."""
    pts = sorted(points, key=lambda r: len(r["b"]) + len(r["a"]))[:2 * n]
    out = []
    for i, j in zip(pts[::2], pts[1::2]):
        q, ans = i["probes"][0]["q"], i["probes"][0]["ans"]
        seqs, at = {"B": i["b"] + q + ans, "A": i["a"], "X": j["a"]}, len(i["b"])
        seg_a, seg_x = ["A", i["a0"], i["a0"] + i["M"]], ["X", j["a0"], j["a0"] + j["M"]]
        base = {"seqs": seqs, "reader": "B", "score": at + 1}
        out += [{"kind": "one", "mounts": {"B": [[at, [seg_a]]]}, **base},  # the single-mount level on these tokens
                {"kind": "two", "mounts": {"B": [[at, [seg_a, seg_x]]]}, **base},
                {"kind": "transitive", "mounts": {"A": [[i["a0"] + i["M"] // 2, [seg_x]]], "B": [[at, [seg_a]]]}, **base}]
    return out


class Probe:
    """Prefill, cache copy + K/V mount, and continuation on a transformers Qwen3.5 text model."""

    def __init__(self, model):
        import torch

        from octopus.rl.model_hooks import patch_model

        patch_model(model.config)  # fla kernels for the delta rule; takes the cache's initial state
        self.torch, self.model, self.body, self.head = torch, model, model.model, model.lm_head
        self.attn = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]

    def run(self, ids: list[int], start: int, cache=None):
        """Hidden states of ids at positions start.. (continuing cache in place, or a fresh one); returns (hidden, cache)."""
        torch = self.torch
        x = torch.tensor([ids], device="cuda")
        pos = torch.arange(start, start + len(ids), device="cuda")[None]
        out = self.body(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True)
        return out.last_hidden_state[0], out.past_key_values

    def logprobs(self, hidden):
        return self.torch.log_softmax(self.head(hidden).float(), -1)

    def kv(self, cache, a: int, b: int) -> dict:
        return {i: (cache.layers[i].keys[:, :, a:b].clone(), cache.layers[i].values[:, :, a:b].clone()) for i in self.attn}

    def rotate(self, k, delta: int):
        """Keys re-rotated by delta positions: RoPE composes, R(p + delta) = R(delta) R(p) (Hogwild!-style shift)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        torch = self.torch
        kf = k.float()
        cos, sin = self.body.rotary_emb(kf, torch.full((1, k.shape[2]), delta, device=k.device))
        return apply_rotary_pos_emb(kf, kf, cos, sin)[1].to(k.dtype)

    def mounted(self, cache, kv: dict, delta: int):
        c = copy.deepcopy(cache)
        for i, (k, v) in kv.items():
            layer = c.layers[i]
            layer.keys = self.torch.cat([layer.keys, self.rotate(k, delta) if delta else k], dim=-2)
            layer.values = self.torch.cat([layer.values, v], dim=-2)
        return c

    def k2(self, seqs: dict, mounts: dict, name: str, memo: dict | None = None):
        """K2 with any number of segments, transitively: sequence `name` (seqs[name], token ids) as a mounted read
        computes it. Its tokens run in order; at each mount point (mounts[name] = [(at, [(src, s, e), ...]), ...]) every
        segment's K/V rows join its attention layers: src's rows for its tokens [s, e) together with the rows src itself
        mounted strictly inside that range, as src computed them; its next tokens continue at max(own next position,
        each segment's last position + 1). Returns (hidden states of its tokens, cache, cache row and position of each
        token); a source is computed once (memo)."""
        memo = {} if memo is None else memo
        if name not in memo:
            ids, cache, hs, rows, pos, i, nxt, n = seqs[name], None, [], [], [], 0, 0, 0  # n: cache rows so far
            for at, segs in sorted(mounts.get(name, ()), key=lambda m: m[0]) + [[len(ids), ()]]:
                if at > i:
                    h, cache = self.run(ids[i:at], nxt, cache)
                    hs.append(h)
                    rows += range(n, n + at - i)
                    pos += range(nxt, nxt + at - i)
                    n, i, nxt = n + at - i, at, nxt + at - i
                for src, s, e in segs:
                    _, c, r, p = self.k2(seqs, mounts, src, memo)
                    cache = self.mounted(cache, self.kv(c, r[s], r[e - 1] + 1), 0)
                    n, nxt = n + r[e - 1] + 1 - r[s], max(nxt, p[e - 1] + 1)
            memo[name] = (self.torch.cat(hs), cache, rows, pos)
        return memo[name]

    def generate(self, cache, prompt: list[int], start: int, stop: set[int], tok) -> str:
        h, cache = self.run(prompt, start, cache)
        pos, out = start + len(prompt), []
        for _ in range(GEN_TOKENS):
            t = int(self.head(h[-1]).argmax())
            out.append(t)
            if t in stop or tok.decode(out[-8:]).endswith("</tool_call>"):
                break
            h, cache = self.run([t], pos, cache)
            pos += 1
        return tok.decode(out)


def sanity(p: Probe, b_ids, insert, tail, cache_b, start_b, start, n_score) -> dict:
    """Cache continuation (insert at start_b, tail at start: the condition loop's positions) vs one plain forward over
    the concatenated tokens at their natural positions: logprobs of the last n_score tokens."""
    torch = p.torch
    seq = b_ids + insert + tail
    h_full, _ = p.run(seq, 0)
    lp_full = p.logprobs(h_full[len(seq) - n_score - 1:-1])
    c = copy.deepcopy(cache_b)
    if insert:
        _, c = p.run(insert, start_b, c)
    h, _ = p.run(tail, start, c)
    lp = p.logprobs(h[len(tail) - n_score - 1:-1])
    labels = torch.tensor(seq[-n_score:], device="cuda")[:, None]
    d = (lp.gather(1, labels) - lp_full.gather(1, labels)).abs()
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
            "argmax_agree": float((lp.argmax(-1) == lp_full.argmax(-1)).float().mean())}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.cuda().eval()
    torch.set_grad_enabled(False)
    p = Probe(model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    rows = [json.loads(x) for x in open(args.probes)][:args.limit or None]
    tree_answers: dict[str, set[str]] = {}  # distractor pool: the answers of every probe of the same tree
    for row in rows:
        tree_answers.setdefault(row["agent_dir"], set()).update(f["answer"] for f in row["facts"])
    i, n_shards = map(int, args.shard.split("/"))
    rows = rows[i::n_shards]
    done = {json.loads(x)["id"] for x in open(args.out)} if Path(args.out).exists() else set()
    out = open(args.out, "a")
    for n, row in enumerate(rows):
        if row["id"] in done:
            continue
        t0 = time.time()
        m = materialize(tok, row)
        b_ids, gen_ids = enc(m["b_ctx"]), enc(m["gen"])
        reply_ids = enc(m["reply"])[:REPLY_TOKENS]
        pre_ids, mount_ids = enc(m["a_pre"]), enc(m["mount"])
        text_ids = enc("<|im_start|>user\n" + TEXT_READ.format(a=row["a_id"])
                       + m["mount"].replace("<|im_start|>", "").replace("<|im_end|>", "") + "<|im_end|>\n")
        assert tok.decode(mount_ids) == m["mount"]  # the mounted rows are A's selected turns, token for token
        for f in row["facts"]:
            assert f["cue"] + f["answer"] in m["mount"] and f["answer"] not in m["b_ctx"]
        p_b, a0, M = len(b_ids), len(pre_ids), len(mount_ids)
        _, cache_b = p.run(b_ids, 0)
        _, cache_a = p.run(pre_ids + mount_ids, 0)
        kv = p.kv(cache_a, a0, a0 + M)
        del cache_a
        res = {"id": row["id"], "agent_dir": row["agent_dir"], "p_b": p_b, "a0": a0, "M": M,
               "ahead_by": a0 + M - p_b, "a_only": row["a_only"], "b_only": row["b_only"],
               "reply_text": m["reply"], "cond": {}}
        settings = {"none": ([], None, 0, p_b), "text": (text_ids, None, 0, p_b + len(text_ids)),
                    "replay": (mount_ids, None, 0, p_b + M), "K1": ([], kv, 0, p_b),
                    "K2": ([], kv, 0, max(p_b, a0 + M)), "K3": ([], kv, p_b - a0, p_b + M)}
        probes = []  # (fact, question + reply prefix, answer tokens, distractor answer tokens of equal length)
        for fi, f in enumerate(row["facts"]):
            pre, ans = slot(enc, f["cue"], f["answer"])
            if ans is None:
                res["dropped"] = res.get("dropped", 0) + 1
                continue
            pool = sorted(tree_answers[row["agent_dir"]] - {f["answer"]})
            random.Random(row["id"] * 16 + fi).shuffle(pool)
            dis = [d for d in (slot(enc, f["cue"], c)[1] for c in pool) if d is not None and len(d) == len(ans)][:3]
            q = enc("<|im_start|>user\n" + QUESTION.format(a=row["a_id"], cue=f["cue"]) + "<|im_end|>\n") + gen_ids + pre
            probes.append((f, q, ans, dis))
        if n < args.sanity:
            res["sanity"] = {c: sanity(p, b_ids, settings[c][0], gen_ids + reply_ids, cache_b, p_b, settings[c][3],
                                       len(reply_ids)) for c in ("none", "text")}
            if n == 0:  # the K3 shift: keys of a block continued at shifted positions vs re-rotated keys
                delta = p_b - a0
                _, c = p.run(pre_ids, 0)
                _, c = p.run(mount_ids, a0 + delta, c)
                i0 = p.attn[0]
                ref, rot = c.layers[i0].keys[:, :, a0:].float(), p.rotate(kv[i0][0], delta).float()
                res["sanity"]["rotate_rel_err"] = float((ref - rot).norm() / ref.norm())
                del c
        lp_none = None
        for cond, (insert, mkv, delta, start) in settings.items():
            base = p.mounted(cache_b, mkv, delta) if mkv else cache_b
            if insert:
                _, base = p.run(insert, p_b, copy.deepcopy(base))
            r = {"facts": []}
            for f, q, ans, dis in probes:
                h_q, c_q = p.run(q, start, copy.deepcopy(base))
                tot = []  # per-token answer log-probs, then each distractor's total
                for cand in [ans] + dis:
                    h, _ = p.run(cand, start + len(q), copy.deepcopy(c_q))
                    lp = p.logprobs(torch.cat([h_q[-1:], h[:-1]]))
                    tot.append(lp.gather(1, torch.tensor(cand, device="cuda")[:, None])[:, 0].tolist())
                g, h, pos = [], h_q, start + len(q)  # greedy continuation, 16 tokens
                for _ in range(16):
                    g.append(int(p.head(h[-1]).argmax()))
                    h, c_q = p.run([g[-1]], pos, c_q)
                    pos += 1
                r["facts"].append({"answer": f["answer"], "lp": sum(tot[0]) / len(ans), "tok_lp": tot[0], "n": len(ans),
                                   "em": tok.decode(g).startswith(tok.decode(ans)), "gen16": g,
                                   "d_tot": [sum(t) for t in tot[1:]]})
            h, _ = p.run(gen_ids + reply_ids, start, copy.deepcopy(base))
            lp = p.logprobs(h[len(gen_ids) - 1:-1])
            tgt = torch.tensor(reply_ids, device="cuda")[:, None]
            r["nll"] = float(-lp.gather(1, tgt).mean())
            if lp_none is None:
                lp_none = lp
            r["kl"] = float((lp_none.exp() * (lp_none - lp)).sum(-1).mean())
            r["gen"] = p.generate(copy.deepcopy(base), gen_ids, start, stop, tok)
            res["cond"][cond] = r
            del base
        res["sec"] = round(time.time() - t0, 1)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"{n + 1}/{len(rows)} id {row['id']} p_b {p_b} ahead_by {res['ahead_by']} M {M} {res['sec']}s"
              + (f" sanity {res['sanity']}" if "sanity" in res else ""), flush=True)


# ---------------------------------------------------------------- report

def tool_name(text: str) -> str | None:
    m = re.search(r"<function=([\w.-]+)>", text)
    return m.group(1) if m else None


def per_mount(res: dict) -> dict:
    """Per condition: retrieval EM / answer logprob (all answer tokens, and without the first) per fact, 4-way ranking
    accuracy against 3 equal-length distractors (facts that have them), reply NLL, KL vs none, hijack (the generated action names an A-only string and no B-only string), same tool as B's recorded reply."""
    own = tool_name(res["reply_text"])
    out = {}
    for cond, r in res["cond"].items():
        a_hit = any(s in r["gen"] for s in res["a_only"])
        b_hit = any(s in r["gen"] for s in res["b_only"])
        out[cond] = {"em": [f["em"] for f in r["facts"]], "lp": [f["lp"] for f in r["facts"]], "nll": r["nll"],
                     "lp_rest": [sum(f["tok_lp"][1:]) / (f["n"] - 1) for f in r["facts"] if f["n"] > 1],
                     "acc": [float(sum(f["tok_lp"]) > max(f["d_tot"])) for f in r["facts"] if len(f["d_tot"]) == 3],
                     "kl": r["kl"], "hijack": float(a_hit and not b_hit), "same_tool": float(tool_name(r["gen"]) == own)}
    return out


def boot(values_by_cluster: list[list[float]], reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of all values and a 95% CI resampling clusters (trees) with replacement."""
    if not values_by_cluster:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sums = np.array([sum(v) for v in values_by_cluster])
    cnts = np.array([len(v) for v in values_by_cluster])
    idx = rng.integers(0, len(sums), (reps, len(sums)))
    est = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def report(args):
    rows = [json.loads(x) for x in open(args.results)]
    groups = {"all": lambda r: True, "ahead": lambda r: r["ahead_by"] > 0, "behind": lambda r: r["ahead_by"] <= 0}
    pm = [(r, per_mount(r)) for r in rows]

    def cluster(sel, fn):
        by: dict[str, list[float]] = {}
        for r, m in pm:
            if sel(r):
                by.setdefault(r["agent_dir"], []).extend(fn(m))
        return [v for v in by.values() if v]

    def fmt(c):
        return f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"

    print(f"{len(rows)} mount points, {len({r['agent_dir'] for r in rows})} trees, "
          f"{sum(len(r['cond']['none']['facts']) for r in rows)} retrieval probes "
          f"({sum(r.get('dropped', 0) for r in rows)} dropped: answer slot not token-aligned; "
          f"{sum(len(m['none']['acc']) for _, m in pm)} with 3 distractors); "
          + ", ".join(f"{g}: {sum(fn(r) for r in rows)}" for g, fn in groups.items() if g != "all"))
    print("\n| condition | group | EM | EM - none | answer logprob/token | reply NLL - none | KL(none, cond) | hijack | same tool as recorded | rank acc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for g, sel in groups.items():
            em = cluster(sel, lambda m: [float(x) for x in m[cond]["em"]])
            dem = cluster(sel, lambda m: [float(x) - float(y) for x, y in zip(m[cond]["em"], m["none"]["em"])])
            lp = cluster(sel, lambda m: m[cond]["lp"])
            dnll = cluster(sel, lambda m: [m[cond]["nll"] - m["none"]["nll"]])
            kl = cluster(sel, lambda m: [m[cond]["kl"]])
            hj = cluster(sel, lambda m: [m[cond]["hijack"]])
            st = cluster(sel, lambda m: [m[cond]["same_tool"]])
            acc = cluster(sel, lambda m: m[cond]["acc"])
            print(f"| {cond} | {g} | {fmt(boot(em))} | {fmt(boot(dem))} | {fmt(boot(lp))} | {fmt(boot(dnll))} | "
                  f"{fmt(boot(kl))} | {fmt(boot(hj))} | {fmt(boot(st))} | {fmt(boot(acc))} |")
    print("\npaired differences (decision rules, docs/kv-read-probe.md)")
    for g, sel in groups.items():
        for a, b, key in (("K1", "K3", "em"), ("K2", "K3", "em"), ("K1", "K2", "em")):
            d = cluster(sel, lambda m: [float(x) - float(y) for x, y in zip(m[a][key], m[b][key])])
            print(f"  {g}: EM {a} - {b} = {fmt(boot(d))}")
        for k in ("K1", "K2", "K3"):
            d = cluster(sel, lambda m: [float(x) - 0.8 * float(y) for x, y in zip(m[k]["em"], m["text"]["em"])])
            kl = cluster(sel, lambda m: [m[k]["kl"] - m["text"]["kl"]])
            hj = cluster(sel, lambda m: [m[k]["hijack"] - m["none"]["hijack"]])
            print(f"  {g}: EM {k} - 0.8 EM text = {fmt(boot(d))}; KL {k} - KL text = {fmt(boot(kl))}; "
                  f"hijack {k} - none = {fmt(boot(hj))}")
    gate = cluster(groups["all"], lambda m: [float(x) - float(y) for x, y in zip(m["text"]["em"], m["none"]["em"])])
    rank = cluster(groups["all"], lambda m: [x - y for x, y in zip(m["text"]["acc"], m["none"]["acc"])])
    print(f"\npositive-control gate, pooled: EM text - none = {fmt(boot(gate))}; rank acc text - none = {fmt(boot(rank))}")
    print("\nexploratory: paired answer log-prob/token, all answer tokens | without the first")
    for g, sel in groups.items():
        for a, b in (("text", "none"), ("replay", "none"), ("K1", "none"), ("K2", "none"), ("K3", "none"),
                     ("K1", "text"), ("K2", "text"), ("K3", "text"), ("K1", "K3"), ("K2", "K3")):
            d = [boot(cluster(sel, lambda m: [x - y for x, y in zip(m[a][k], m[b][k])])) for k in ("lp", "lp_rest")]
            print(f"  {g}: {a} - {b} = {fmt(d[0])} | {fmt(d[1])}")
    san = [r["sanity"] for r in rows if "sanity" in r]
    if san:
        print("\nsanity:", json.dumps(san))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("jobs_dirs", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--adapter")
    r.add_argument("--probes", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--shard", default="0/1", help="i/n: this process runs rows i, i+n, ...")
    r.add_argument("--sanity", type=int, default=4, help="mount points with the plain-forward checks")
    p = sub.add_parser("report")
    p.add_argument("results")
    args = ap.parse_args()
    {"build": build, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
