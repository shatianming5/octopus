"""Route A: external ways of letting sibling B read sibling A, as conditions of the attention-only KV-mount probe, on
GDN-hybrid, Mamba-2-hybrid (NoPE attention) and dense models. Conditions and papers: baselines/README.md.

The probe set, answer slot, materialized texts and bootstrap are octopus/tools/kv_read_probe.py's (which still
reproduces Run 1/2 of docs/kv-read-probe.md); this module adds the conditions summary and kvcomm_sel, a mount on a layer
subset, any registered model family, and template-agnostic rendering of the messages a condition adds.

usage: python -m baselines.kv_read.route_a run --model MODEL [--adapter LORA] --probes PROBES.jsonl --out RESULTS.jsonl
           [--conditions none,text,...] [--mount-layers all|a-b|i,j,...] [--root-map OLD=NEW ...] [--shard i/n]
       python -m baselines.kv_read.route_a report RESULTS.jsonl [MORE.jsonl ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

from octopus.rl.lm_head import _post_head, _post_head_ops
from octopus.rl.model_hooks import ATTENTION_LAYERS, modeling_module, patch_model
from octopus.serving.delta_render import terminator
from octopus.tools import kv_read_probe
from octopus.tools.kv_read_probe import (GEN_TOKENS, QUESTION, REPLY_TOKENS, TEXT_READ, boot, materialize, render,
                                         slot)

CONDITIONS = ("none", "text", "summary", "replay", "K1", "K2", "K3", "kvcomm_sel")
SUMMARY_TOKENS = 192
SUMMARY_REQUEST = ("Summarize your last few tool calls and their outputs above for a sibling branch that works on "
                   "another task in parallel. Keep exact file paths, names, numbers and error messages. Reply with "
                   "only the summary.")
SUMMARY_READ = "Summary of the recent work of sibling branch {a} (it works on another task in parallel):\n"
SLOT = "\x00"  # stands for the summary, which exists only after A's decode, in its rendered message


def mount_layers(spec: str, attn: list[int]) -> list[int]:
    """--mount-layers: all (every attention layer), a-b (inclusive) or i,j,...; each must be an attention layer."""
    if spec == "all":
        return list(attn)
    a, _, b = spec.partition("-")
    out = list(range(int(a), int(b) + 1)) if b else sorted({int(x) for x in spec.split(",")})
    if not out or set(out) - set(attn):
        raise ValueError(f"--mount-layers {spec}: not a non-empty subset of the attention layers {attn}")
    return out


def remap(path: str, root_map: list[str]) -> str:
    """agent_dir under the first --root-map OLD=NEW whose OLD it starts with."""
    for pair in root_map:
        old, new = pair.split("=", 1)
        if path.startswith(old):
            return new + path[len(old):]
    return path


def tool_name(text: str) -> str | None:
    """The first tool call's name: Qwen3.5's <function=NAME>, or Qwen3's / granite's <tool_call>{"name": NAME, ...}."""
    m = re.search(r'<function=([\w.-]+)>|<tool_call>\s*\{\s*"name"\s*:\s*"([^"]+)"', text)
    return (m.group(1) or m.group(2)) if m else None


def same_tool(gen: str, reply: str) -> float:
    """The generated action calls the tool B's recorded reply called; a reply without a call never matches."""
    own = tool_name(reply)
    return float(own is not None and tool_name(gen) == own)


def appended(tok, ctx: list[dict], extra: list[dict], tools: list[dict], gen: bool = False) -> str:
    """What the chat template renders for messages extra appended to a cached context ctx (+ the generation prompt):
    their rendering after a stub of ctx, its system message and a user query. Not after ctx itself: Qwen3.5's template
    re-renders the assistant turns before a later user query (drops their empty think blocks), which a cache keeps."""
    stub = [m for m in ctx[:1] if m["role"] == "system"] + [{"role": "user", "content": "stub"}]
    base, full = render(tok, stub, tools), render(tok, stub + extra, tools, gen)
    if not full.startswith(base):
        raise ValueError("the chat template does not render the appended messages as a suffix")
    return full[len(base):]


def unmarked(tok, text: str) -> str:
    """text without the tokenizer's control tokens (chat markers)."""
    for t in tok.added_tokens_decoder.values():
        if t.special:
            text = text.replace(t.content, "")
    return text


class Probe:
    """Prefill, cache copy + K/V mount, and continuation on a transformers causal LM of any family model_hooks knows
    (pure attention, GDN or Mamba-2 hybrids). Recurrent layers continue a multi-token input from the cached state (the
    hybrid cache's conv context + recurrent state; tests/test_route_a.py checks it against one plain forward)."""

    def __init__(self, model, tok):
        import torch

        patch_model(model.config)  # GDN: fla kernels for the delta rule, which take the cache's initial state
        self.torch, self.tok, self.body = torch, tok, model.model
        self.device = next(model.parameters()).device
        self.head = model.get_output_embeddings()
        self.post = _post_head_ops(model, self.head)  # granite's / logits_scaling, ...
        text = model.config.get_text_config()
        types = getattr(text, "layer_types", None) or ["full_attention"] * text.num_hidden_layers
        self.attn = [i for i, t in enumerate(types) if t in ATTENTION_LAYERS]
        self.rope = getattr(self.body, "rotary_emb", None)  # None: attention without positions (granite-4.0-h)
        self.apply_rope = modeling_module(model.config).apply_rotary_pos_emb
        eos = model.generation_config.eos_token_id
        self.stop = ({*terminator(tok, {}), *terminator(tok, {"tool_calls": [{}]}), tok.eos_token_id,
                      *(eos if isinstance(eos, list) else [eos])} - {None})

    def run(self, ids: list[int], start: int, cache=None):
        """Hidden states of ids at positions start.. (continuing cache in place, or a fresh one); returns (hidden, cache)."""
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        pos = torch.arange(start, start + len(ids), device=self.device)[None]
        out = self.body(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True)
        return out.last_hidden_state[0], out.past_key_values

    def logits(self, hidden):
        return _post_head(self.post, self.head(hidden))

    def logprobs(self, hidden):
        return self.torch.log_softmax(self.logits(hidden).float(), -1)

    def kv(self, cache, a: int, b: int) -> dict:
        return {i: (cache.layers[i].keys[:, :, a:b].clone(), cache.layers[i].values[:, :, a:b].clone()) for i in self.attn}

    def rotate(self, k, delta: int):
        """Keys re-rotated by delta positions: RoPE composes, R(p + delta) = R(delta) R(p) (Hogwild!-style shift);
        the identity without positional embeddings."""
        if self.rope is None or not delta:
            return k
        kf = k.float()
        cos, sin = self.rope(kf, self.torch.full((1, k.shape[2]), delta, device=k.device))
        return self.apply_rope(kf, kf, cos, sin)[1].to(k.dtype)

    def mounted(self, cache, kv: dict, delta: int):
        c = copy.deepcopy(cache)
        for i, (k, v) in kv.items():
            layer = c.layers[i]
            layer.keys = self.torch.cat([layer.keys, self.rotate(k, delta)], dim=-2)
            layer.values = self.torch.cat([layer.values, v], dim=-2)
        return c

    def generate(self, cache, prompt: list[int], start: int, n: int) -> list[int]:
        """Greedy tokens after prompt (continuing cache in place), at most n, through a stop token or </tool_call>."""
        h, cache = self.run(prompt, start, cache)
        pos, out = start + len(prompt), []
        for _ in range(n):
            out.append(int(self.logits(h[-1]).argmax()))
            if out[-1] in self.stop or self.tok.decode(out[-8:]).endswith("</tool_call>"):
                break
            h, cache = self.run([out[-1]], pos, cache)
            pos += 1
        return out


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
    labels = torch.tensor(seq[-n_score:], device=p.device)[:, None]
    d = (lp.gather(1, labels) - lp_full.gather(1, labels)).abs()
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
            "argmax_agree": float((lp.argmax(-1) == lp_full.argmax(-1)).float().mean())}


def inputs(tok, row: dict, tree: dict, conds: list[str]) -> dict:
    """A mount point's token ids: B's context, generation prompt and recorded reply, A's turns before the mount and the
    mount (materialize), and the messages the conditions append (raises ValueError where the template does not render
    them as suffixes)."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    m = materialize(tok, row, tree)
    fork = tree["forks"][row["fork"]]
    main = tree["main"][:fork["prefix_len"]]
    a_ctx = main + fork["branches"][row["a"]]["messages"][:row["ja"]]
    b_ctx = main + fork["branches"][row["b"]]["messages"][:row["k"]]

    def user(ctx, text, gen=False):
        return appended(tok, ctx, [{"role": "user", "content": text}], tree["tools"], gen)

    x = {"b": enc(m["b_ctx"]), "gen": enc(m["gen"]), "reply": enc(m["reply"])[:REPLY_TOKENS], "reply_text": m["reply"],
         "pre": enc(m["a_pre"]), "mount": enc(m["mount"]), "text": [], "ask": [], "summary": ("", ""), "q": []}
    assert tok.decode(x["mount"]) == m["mount"]  # the mounted rows are A's selected turns, token for token
    for f in row["facts"]:
        assert f["cue"] + f["answer"] in m["mount"] and f["answer"] not in m["b_ctx"]
    if "text" in conds:  # Qwen3.5's template trims content: one token (the mount's last newline) short of run 2's text
        x["text"] = enc(user(b_ctx, TEXT_READ.format(a=row["a_id"]) + unmarked(tok, m["mount"])))
    if "summary" in conds:
        x["ask"] = enc(user(a_ctx, SUMMARY_REQUEST, gen=True))
        head, _, tail = user(b_ctx, SUMMARY_READ.format(a=row["a_id"]) + SLOT).partition(SLOT)
        x["summary"] = (head, tail)
    for f in row["facts"]:
        pre, ans = slot(enc, f["cue"], f["answer"])
        x["q"].append((f, enc(user(b_ctx, QUESTION.format(a=row["a_id"], cue=f["cue"]), gen=True)) + pre, ans))
    return x


def mount_point(p: Probe, row: dict, x: dict, pool: set[str], conds: list[str], sel: list[int], checks: bool,
                rotation: bool) -> dict:
    """One mount point (kv_read_probe.run's row format, plus the summary, copy bytes and the mounted layers). A runs
    first: its K/V is sliced and its summary decoded, and its cache freed, before B's prefill (peak memory)."""
    torch, tok = p.torch, p.tok
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    t0 = time.time()
    b_ids, gen_ids, reply_ids, pre_ids, mount_ids = x["b"], x["gen"], x["reply"], x["pre"], x["mount"]
    p_b, a0, M = len(b_ids), len(pre_ids), len(mount_ids)
    _, cache_a = p.run(pre_ids + mount_ids, 0)
    kv = p.kv(cache_a, a0, a0 + M)
    summary, sum_ids = None, []
    if "summary" in conds:
        t1 = time.time()
        out = p.generate(cache_a, x["ask"], a0 + M, SUMMARY_TOKENS)
        text = tok.decode([t for t in out if t not in p.stop], skip_special_tokens=True).strip()
        sum_ids = enc(x["summary"][0] + text + x["summary"][1])
        summary = {"text": text, "gen_tokens": len(out), "insert_tokens": len(sum_ids),
                   "ms": round((time.time() - t1) * 1000, 1)}
    del cache_a
    res = {"id": row["id"], "agent_dir": row["agent_dir"], "p_b": p_b, "a0": a0, "M": M, "ahead_by": a0 + M - p_b,
           "a_only": row["a_only"], "b_only": row["b_only"], "reply_text": x["reply_text"], "mount_layers": sel,
           "attn_layers": p.attn, "cond": {}}
    if rotation and {"K3", "kvcomm_sel"} & set(conds):
        # S4: A's mount continued at positions shifted by delta vs at its own, re-rotated (same chunking: 0 for NoPE)
        delta = p_b - a0
        _, c = p.run(pre_ids, 0)
        c2 = copy.deepcopy(c)
        _, c = p.run(mount_ids, a0, c)
        _, c2 = p.run(mount_ids, a0 + delta, c2)
        i0 = p.attn[0]
        ref, rot = c2.layers[i0].keys[:, :, a0:].float(), p.rotate(c.layers[i0].keys[:, :, a0:], delta).float()
        res["sanity"] = {"rotate_rel_err": float((ref - rot).norm() / ref.norm())}
        del c, c2
    _, cache_b = p.run(b_ids, 0)
    kv_sel = {i: kv[i] for i in sel}
    settings = {"none": ([], None, 0, p_b), "text": (x["text"], None, 0, p_b + len(x["text"])),
                "summary": (sum_ids, None, 0, p_b + len(sum_ids)), "replay": (mount_ids, None, 0, p_b + M),
                "K1": ([], kv_sel, 0, p_b), "K2": ([], kv_sel, 0, max(p_b, a0 + M)), "K3": ([], kv, p_b - a0, p_b + M),
                "kvcomm_sel": ([], kv_sel, p_b - a0, p_b + M)}
    probes = []  # (fact, question + reply prefix, answer tokens, distractor answer tokens of equal length)
    for fi, (f, q, ans) in enumerate(x["q"]):
        if ans is None:
            res["dropped"] = res.get("dropped", 0) + 1
            continue
        cands = sorted(pool - {f["answer"]})
        random.Random(row["id"] * 16 + fi).shuffle(cands)
        dis = [d for d in (slot(enc, f["cue"], c)[1] for c in cands) if d is not None and len(d) == len(ans)][:3]
        probes.append((f, q, ans, dis))
    if checks:
        res.setdefault("sanity", {}).update(
            {c: sanity(p, b_ids, settings[c][0], gen_ids + reply_ids, cache_b, p_b, settings[c][3], len(reply_ids))
             for c in ("none", "text") if c in conds})
    lp_none, computed = None, {}  # mount: the condition that ran each (layers, rotation, start)
    for cond in conds:
        insert, mkv, delta, start = settings[cond]
        if mkv:
            # without RoPE no layer reads positions (granite: NoPE attention, Mamba-2): only the layers tell mounts apart
            key = (tuple(mkv),) + ((delta, start) if p.rope is not None else ())
            if key in computed:  # e.g. kvcomm_sel on every attention layer is K3
                res["cond"][cond] = {**res["cond"][computed[key]], "same_as": computed[key]}
                continue
            computed[key] = cond
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
                tot.append(lp.gather(1, torch.tensor(cand, device=p.device)[:, None])[:, 0].tolist())
            g, h, pos = [], h_q, start + len(q)  # greedy continuation, 16 tokens
            for _ in range(16):
                g.append(int(p.logits(h[-1]).argmax()))
                h, c_q = p.run([g[-1]], pos, c_q)
                pos += 1
            r["facts"].append({"answer": f["answer"], "lp": sum(tot[0]) / len(ans), "tok_lp": tot[0], "n": len(ans),
                               "em": tok.decode(g).startswith(tok.decode(ans)), "gen16": g,
                               "d_tot": [sum(t) for t in tot[1:]]})
        h, _ = p.run(gen_ids + reply_ids, start, copy.deepcopy(base))
        lp = p.logprobs(h[len(gen_ids) - 1:-1])
        r["nll"] = float(-lp.gather(1, torch.tensor(reply_ids, device=p.device)[:, None]).mean())
        if lp_none is None:
            lp_none = lp
        r["kl"] = float((lp_none.exp() * (lp_none - lp)).sum(-1).mean())
        r["gen"] = tok.decode(p.generate(copy.deepcopy(base), gen_ids, start, GEN_TOKENS))
        if mkv:  # layers x M x kv heads x head dim x 2 (K and V) x dtype bytes copied into B's cache
            r["copy_bytes"] = sum(2 * k.numel() * k.element_size() for k, _ in mkv.values())
        if cond == "summary":
            r["summary"] = summary
        res["cond"][cond] = r
        del base
    res["sec"] = round(time.time() - t0, 1)
    return res


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    asked = args.conditions.split(",")
    if set(asked) - set(CONDITIONS):
        raise SystemExit(f"--conditions: unknown {sorted(set(asked) - set(CONDITIONS))}; known {CONDITIONS}")
    conds = [c for c in CONDITIONS if c == "none" or c in asked]  # none: the KL reference
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.cuda().eval()
    torch.set_grad_enabled(False)
    p = Probe(model, tok)
    sel = mount_layers(args.mount_layers, p.attn)
    rows = [json.loads(x) for x in open(args.probes)][:args.limit or None]
    pool: dict[str, set[str]] = {}  # distractors: the answers of every probe of the same tree
    for row in rows:
        pool.setdefault(row["agent_dir"], set()).update(f["answer"] for f in row["facts"])
    i, n_shards = map(int, args.shard.split("/"))
    rows = rows[i::n_shards]
    done = {json.loads(x)["id"] for x in open(args.out)} if Path(args.out).exists() else set()
    out = open(args.out, "a")
    for n, row in enumerate(rows):
        if row["id"] in done:
            continue
        local = {**row, "agent_dir": remap(row["agent_dir"], args.root_map)}
        tree = json.loads((Path(local["agent_dir"]) / "tree.json").read_text())
        try:
            x = inputs(tok, local, tree, conds)
        except ValueError as e:  # counted by report, the same mount point for every condition
            res = {"id": row["id"], "agent_dir": row["agent_dir"], "skipped": str(e)}
        else:
            res = mount_point(p, row, x, pool[row["agent_dir"]], conds, sel, n < args.sanity, n == 0)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"{n + 1}/{len(rows)} id {row['id']} " + (f"skipped: {res['skipped']}" if "skipped" in res else
              f"p_b {res['p_b']} ahead_by {res['ahead_by']} M {res['M']} {res['sec']}s"
              + (f" sanity {res['sanity']}" if "sanity" in res else "")), flush=True)


# ---------------------------------------------------------------- report

SUBSET = ("K1", "K2", "kvcomm_sel")  # the conditions --mount-layers selects layers for


def layer_tag(layers: list[int]) -> str:
    return f"{layers[0]}-{layers[-1]}" if layers == list(range(layers[0], layers[-1] + 1)) else ",".join(map(str, layers))


def merged(paths: list[str]) -> tuple[dict, dict]:
    """Rows by id over results files of one model (e.g. run 2's and a none+summary backfill, or a run on every
    attention layer and one on a subset), and each file's mount layers. A subset run's K1 / K2 / kvcomm_sel are named
    K2@16-24 etc.; for a condition or check already present the first file wins."""
    by: dict = {}
    files: dict[str, set[str]] = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if "skipped" not in r:
                layers = r.get("mount_layers")  # run 1/2 rows: every attention layer
                tag = "all" if layers is None or layers == r["attn_layers"] else layer_tag(layers)
                files.setdefault(path, set()).add(tag)
                if tag != "all":
                    q = lambda c: f"{c}@{tag}" if c in SUBSET else c  # noqa: E731
                    r["cond"] = {q(c): {**x, "same_as": q(x["same_as"])} if "same_as" in x else x
                                 for c, x in r["cond"].items()}
            have = by.setdefault(r["id"], r)
            if have is r or "skipped" in r:
                continue
            if "skipped" in have:
                by[r["id"]] = r
                continue
            for c, x in r["cond"].items():
                have["cond"].setdefault(c, x)
            for c, x in r.get("sanity", {}).items():
                have.setdefault("sanity", {}).setdefault(c, x)
    return by, files


def per_mount(res: dict) -> dict:
    """kv_read_probe.per_mount, with same_tool that parses JSON tool calls and never matches a reply without one, and
    the copy and summary costs."""
    out = kv_read_probe.per_mount(res)
    for cond, r in res["cond"].items():
        s = r.get("summary") or {}
        out[cond].update(same_tool=same_tool(r["gen"], res["reply_text"]), copy_mb=r.get("copy_bytes", 0) / 2**20,
                         **{k: s[k] for k in ("gen_tokens", "insert_tokens", "ms") if k in s})
    return out


def report(args):
    by, files = merged(args.results)
    rows = [r for r in by.values() if "skipped" not in r]
    skipped = [r for r in by.values() if "skipped" in r]
    names = {c for r in rows for c in r["cond"]}
    conds = [c for base in CONDITIONS for c in [base] + sorted(n for n in names if n.startswith(base + "@")) if c in names]
    same = {}  # conditions that are another one's computation at every mount point (on RoPE models K2 is K1 only
    for c in conds:  # where the peer is behind)
        s = {r["cond"][c].get("same_as") for r in rows if c in r["cond"]}
        if len(s) == 1 and None not in s:
            same[c] = s.pop()
    groups = {"all": lambda r: True, "ahead": lambda r: r["ahead_by"] > 0, "behind": lambda r: r["ahead_by"] <= 0}
    pm = [(r, per_mount(r)) for r in rows]

    def cluster(sel, fn, *need):  # mount points that ran every condition in need
        by: dict[str, list[float]] = {}
        for r, m in pm:
            if sel(r) and all(c in m for c in need):
                by.setdefault(r["agent_dir"], []).extend(fn(m))
        return [v for v in by.values() if v]

    def fmt(c):
        return f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"

    def em_diff(a, b, wb=1.0):
        return lambda m: [float(x) - wb * float(y) for x, y in zip(m[a]["em"], m[b]["em"])]

    print(f"{len(rows)} mount points, {len({r['agent_dir'] for r in rows})} trees, "
          f"{sum(len(r['cond']['none']['facts']) for r in rows)} retrieval probes "
          f"({sum(r.get('dropped', 0) for r in rows)} dropped: answer slot not token-aligned); "
          f"{len(skipped)} mount points skipped (template); conditions "
          + ", ".join(c + (f" (= {same[c]}: the same computation)" if c in same else "") for c in conds) + "; "
          + ", ".join(f"{g}: {sum(fn(r) for r in rows)}" for g, fn in groups.items() if g != "all"))
    print("mount layers (K1 / K2 / kvcomm_sel; K3 every attention layer; X@L: X on layers L only): "
          + "; ".join(f"{path}: {', '.join(sorted(tags))}" for path, tags in files.items()))
    print("\n| condition | group | n | EM | EM - none | answer logprob/token | reply NLL - none | KL(none, cond) | hijack "
          "| same tool as recorded | rank acc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in conds:
        for g, sel in groups.items():
            need = (cond, "none")
            em = cluster(sel, lambda m: [float(x) for x in m[cond]["em"]], *need)
            print(f"| {cond} | {g} | {sum(map(len, em))} | {fmt(boot(em))} | "
                  f"{fmt(boot(cluster(sel, em_diff(cond, 'none'), *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: m[cond]['lp'], *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: [m[cond]['nll'] - m['none']['nll']], *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: [m[cond]['kl']], *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: [m[cond]['hijack']], *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: [m[cond]['same_tool']], *need)))} | "
                  f"{fmt(boot(cluster(sel, lambda m: m[cond]['acc'], *need)))} |")
    print("\n| condition | KV copied (MB/mount) | summary tokens generated | summary tokens inserted | summary wall ms |")
    print("|---|---|---|---|---|")
    for cond in conds:
        cost = [fmt(boot(cluster(groups["all"], lambda m: [m[cond][k]] if k in m[cond] else [], cond)))
                for k in ("copy_mb", "gen_tokens", "insert_tokens", "ms")]
        print(f"| {cond} | " + " | ".join(cost) + " |")
    pairs = [("K1", "K3"), ("K2", "K3"), ("K1", "K2"), ("K2", "summary"), ("kvcomm_sel", "K2"), ("kvcomm_sel", "K3")]
    for c in conds:  # a layer subset vs every attention layer, and vs K3 / K2 on the same subset
        base, _, tag = c.partition("@")
        if tag:
            pairs += [(c, base), (c, "K3")] + ([(c, f"K2@{tag}")] if base == "kvcomm_sel" else [])
    print("\npaired differences (decision rules: docs/kv-read-probe.md; baselines/README.md)")
    for g, sel in groups.items():
        for a, b in pairs:
            if a in conds and b in conds:
                print(f"  {g}: EM {a} - {b} = {fmt(boot(cluster(sel, em_diff(a, b), a, b)))}")
        for k in conds:
            if k.partition("@")[0] in ("K1", "K2", "K3", "kvcomm_sel") and "text" in conds:
                d = cluster(sel, em_diff(k, "text", 0.8), k, "text")
                kl = cluster(sel, lambda m: [m[k]["kl"] - m["text"]["kl"]], k, "text")
                hj = cluster(sel, lambda m: [m[k]["hijack"] - m["none"]["hijack"]], k)
                print(f"  {g}: EM {k} - 0.8 EM text = {fmt(boot(d))}; KL {k} - KL text = {fmt(boot(kl))}; "
                      f"hijack {k} - none = {fmt(boot(hj))}")
    if "text" in conds:
        gate = cluster(groups["all"], em_diff("text", "none"), "text")
        rank = cluster(groups["all"], lambda m: [x - y for x, y in zip(m["text"]["acc"], m["none"]["acc"])], "text")
        print(f"\npositive-control gate, pooled: EM text - none = {fmt(boot(gate))}; rank acc text - none = {fmt(boot(rank))}")
    print("\nexploratory: paired answer log-prob/token, all answer tokens | without the first")
    for g, sel in groups.items():
        for a in conds:
            for b in ("none", "text"):
                if a != b and b in conds and not (a == "none" and b == "text"):
                    d = [boot(cluster(sel, lambda m: [x - y for x, y in zip(m[a][k], m[b][k])], a, b))
                         for k in ("lp", "lp_rest")]
                    print(f"  {g}: {a} - {b} = {fmt(d[0])} | {fmt(d[1])}")
    san = [r["sanity"] for r in rows if "sanity" in r]
    if san:
        print("\nsanity (S1 none / S2 text: cache continuation vs plain forward; S4 rotate_rel_err):", json.dumps(san))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--adapter")
    r.add_argument("--probes", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated subset; none always runs")
    r.add_argument("--mount-layers", default="all", help="K1/K2/kvcomm_sel's layers: all | a-b | i,j,...")
    r.add_argument("--root-map", action="append", default=[], metavar="OLD=NEW",
                   help="read a probe's agent_dir under NEW where it starts with OLD (repeatable)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--shard", default="0/1", help="i/n: this process runs rows i, i+n, ...")
    r.add_argument("--sanity", type=int, default=4, help="mount points with the plain-forward checks")
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+")
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
