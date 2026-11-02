"""Is a mount in the prefix-tree forward the same computation as kv_read_probe's K2 cache surgery (#129)?

Per mount point of the probe set, the oracle prefills B and A, appends A's mounted K/V rows to B's full-attention caches
and continues with B's next reply at max(p_B, a0 + M) (kv_read_probe's K2). The tree is [B's context + reply, A's
context + mounted turns] with the mount {B: [(p_B, A, a0, a0 + M)]}, run once; B's reply log-probs are compared.
Without the mount, the same tree is compared with one plain forward over B's tokens. --backward N: on the N shortest
mount points, a loss on B's reply through the mounted tree must give finite gradients that reach A's mounted tokens'
input embeddings (and none without the mount).
--points MOUNT_POINTS.json.gz --multi N (tests/test_sglang_mount.py prep): N two-segment and N transitive cases
(kv_read_probe.multi_cases) through the tree vs Probe.k2, and the backward of the shortest transitive case: finite
gradients that reach the tokens A mounted (X's) through A's segment, and none once A's read is dropped.

usage: python -m octopus.tools.mount_check --model TEXT_MODEL_DIR [--probes PROBES.jsonl --n 40 --backward 2]
                                           [--points MOUNT_POINTS.json.gz --multi 4]
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from octopus.tools.kv_read_probe import REPLY_TOKENS, Probe, materialize, multi_cases
from octopus.rl.prefix_tree import patch_attention, tree_inputs


def tree_rows(model, seqs, mounts, rows, embeds=False):
    """Final hidden states of the tree at the flat positions `rows` (and the input embeddings when embeds)."""
    flat = torch.tensor([sum(seqs, [])], device="cuda")
    cu = torch.tensor(np.cumsum([0] + [len(s) for s in seqs]), dtype=torch.int32, device="cuda")
    meta, kw = tree_inputs(flat, cu, mounts)
    ids = kw.pop("input_ids")
    if embeds:
        kw["inputs_embeds"] = model.model.embed_tokens(ids).detach().requires_grad_()
    else:
        kw["input_ids"] = ids
    h = model.model(**kw).last_hidden_state[0]
    tree_pos = lambda r: meta.pair_pos[meta.flat_pair[torch.as_tensor(r, device="cuda")]]  # noqa: E731
    return h[tree_pos(rows)], kw.get("inputs_embeds"), tree_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backward", type=int, default=2)
    ap.add_argument("--points")
    ap.add_argument("--multi", type=int, default=4)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    model.requires_grad_(False)
    patch_attention()
    p = Probe(model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    if args.points:
        multi(model, p, json.loads(gzip.open(args.points).read()), args.multi)
    if not args.probes:
        return
    rows = [json.loads(x) for x in open(args.probes)]
    rows = rows[::max(1, len(rows) // args.n)][:args.n]
    deltas = {"K2": [], "none": []}  # per mount point: (|delta logprob| per reply token, argmax agreement, ahead)
    cases = []
    for n, row in enumerate(rows):
        m = materialize(tok, row)
        b_ids, gen_ids, reply_ids = enc(m["b_ctx"]), enc(m["gen"]), enc(m["reply"])[:REPLY_TOKENS]
        pre_ids, mount_ids = enc(m["a_pre"]), enc(m["mount"])
        p_b, a0, M = len(b_ids), len(pre_ids), len(mount_ids)
        cont = gen_ids + reply_ids
        seqs, mount = [b_ids + cont, pre_ids + mount_ids], {0: [(p_b, 1, a0, a0 + M)]}
        rows_b = list(range(p_b + len(gen_ids) - 1, len(seqs[0]) - 1))  # positions predicting B's reply tokens
        labels = torch.tensor(reply_ids, device="cuda")[:, None]
        cases.append((len(seqs[0]) + len(seqs[1]), seqs, mount, rows_b, labels, a0, M))
        with torch.no_grad():
            _, cache_b = p.run(b_ids, 0)
            _, cache_a = p.run(pre_ids + mount_ids, 0)
            kv = p.kv(cache_a, a0, a0 + M)
            del cache_a
            h, _ = p.run(cont, max(p_b, a0 + M), p.mounted(cache_b, kv, 0))
            ref = {"K2": p.logprobs(h[len(gen_ids) - 1:-1])}
            del cache_b, kv, h
            h, _ = p.run(seqs[0], 0)
            ref["none"] = p.logprobs(h[rows_b])
            del h
            for cond, mounts in (("K2", mount), ("none", None)):
                lp = p.logprobs(tree_rows(model, seqs, mounts, rows_b)[0])
                d = (lp.gather(1, labels) - ref[cond].gather(1, labels)).abs()[:, 0]
                agree = (lp.argmax(-1) == ref[cond].argmax(-1)).float()
                deltas[cond].append((d.cpu(), agree.cpu(), a0 + M > p_b))
        print(f"{n + 1}/{len(rows)} id {row['id']} p_b {p_b} a0 {a0} M {M} ahead_by {a0 + M - p_b} reply {len(reply_ids)}"
              + "".join(f" | {c} mean {deltas[c][-1][0].mean():.4f} max {deltas[c][-1][0].max():.3f} argmax "
                        f"{deltas[c][-1][1].mean():.3f}" for c in deltas), flush=True)
        torch.cuda.empty_cache()
    for cond, res in deltas.items():
        for group, keep in (("all", lambda a: True), ("ahead", lambda a: a), ("behind", lambda a: not a)):
            sel = [r for r in res if keep(r[2])]
            if sel:
                d, agree = torch.cat([r[0] for r in sel]), torch.cat([r[1] for r in sel])
                print(f"{cond:4s} vs {'K2 cache surgery' if cond == 'K2' else 'plain forward'} {group:6s}: "
                      f"{len(sel)} mount points, {len(d)} reply tokens, mean |delta logprob| {d.mean():.4f}, "
                      f"max per-mount-point mean {max(float(r[0].mean()) for r in sel):.4f}, "
                      f"argmax agreement {agree.mean():.4f}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF checkpoints layers only in training mode (Qwen3.5 has no dropout)
    for _, seqs, mount, rows_b, labels, a0, M in sorted(cases, key=lambda c: c[0])[:args.backward]:
        a_rows = [len(seqs[0]) + i for i in range(a0, a0 + M - 1)]  # A's mounted tokens (but its last, label-less)
        for cond, mounts in (("K2", mount), ("none", None)):
            h, emb, tree_pos = tree_rows(model, seqs, mounts, rows_b, embeds=True)
            loss = -p.logprobs(h).gather(1, labels).mean()
            loss.backward()
            g = emb.grad[0]
            print(f"backward {cond:4s} ({sum(map(len, seqs))} flat tokens): loss {loss.item():.4f}, grad finite "
                  f"{bool(torch.isfinite(g).all())}, |grad| on A's mounted tokens {g[tree_pos(a_rows)].norm():.3e}, "
                  f"on B's reply tokens {g[tree_pos(rows_b[1:])].norm():.3e}", flush=True)
            del h, emb, g, loss
            torch.cuda.empty_cache()


def multi(model, p, points, n):
    """Two-segment and transitive mounts: the reader's answer log-probs through the tree vs Probe.k2, then a backward."""
    res = {}
    cases = multi_cases(points, n)
    for c in cases:
        names = [c["reader"]] + [k for k in c["seqs"] if k != c["reader"]]
        seqs = [c["seqs"][k] for k in names]
        mounts = {names.index(k): [(at, names.index(src), s, e) for at, segs in ms for src, s, e in segs]
                  for k, ms in c["mounts"].items()}
        rows = list(range(c["score"] - 1, len(seqs[0]) - 1))
        labels = torch.tensor(seqs[0][c["score"]:], device="cuda")[:, None]
        with torch.no_grad():
            ref = p.logprobs(p.k2(c["seqs"], c["mounts"], c["reader"])[0][rows])
            lp = p.logprobs(tree_rows(model, seqs, mounts, rows)[0])
        d = (lp.gather(1, labels) - ref.gather(1, labels)).abs()[:, 0].cpu()
        agree = (lp.argmax(-1) == ref.argmax(-1)).float().cpu()
        res.setdefault(c["kind"], []).append((d, agree))
        c.update(names=names, tree_mounts=mounts, rows=rows, labels=labels, size=sum(map(len, seqs)))
        if c["kind"] == "one":  # the kernel floor on these tokens: the tree without a mount vs a plain forward of B
            with torch.no_grad():
                ref = p.logprobs(p.run(seqs[0], 0)[0][rows])
                lp = p.logprobs(tree_rows(model, seqs, None, rows)[0])
            res.setdefault("none", []).append(((lp.gather(1, labels) - ref.gather(1, labels)).abs()[:, 0].cpu(),
                                               (lp.argmax(-1) == ref.argmax(-1)).float().cpu()))
        print(f"{c['kind']:10s} {[len(x) for x in seqs]} tokens: mean |delta logprob| {d.mean():.4f} max {d.max():.3f} "
              f"argmax {agree.mean():.3f}", flush=True)
        del ref, lp
        torch.cuda.empty_cache()
    for kind, r in res.items():
        d, agree = torch.cat([x[0] for x in r]), torch.cat([x[1] for x in r])
        print(f"{kind} vs Probe.k2: {len(r)} cases, {len(d)} answer tokens, mean |delta logprob| {d.mean():.4f}, max "
              f"per-case mean {max(float(x[0].mean()) for x in r):.4f}, argmax agreement {agree.mean():.4f}", flush=True)
    c = min((c for c in cases if c["kind"] == "transitive"), key=lambda c: c["size"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF checkpoints layers only in training mode (Qwen3.5 has no dropout)
    seqs, (_, x0, x1) = [c["seqs"][k] for k in c["names"]], c["mounts"]["A"][0][1][0]
    x_flat = sum(len(seqs[i]) for i in range(c["names"].index("X")))  # X's first flat position
    x_rows = list(range(x_flat + x0, x_flat + x1 - 1))  # X's tokens A mounted (but its last, label-less)
    for cond, mounts in (("transitive", c["tree_mounts"]), ("A's read dropped", {0: c["tree_mounts"][0]})):
        h, emb, tree_pos = tree_rows(model, seqs, mounts, c["rows"], embeds=True)
        loss = -p.logprobs(h).gather(1, c["labels"]).mean()
        loss.backward()
        g = emb.grad[0]
        print(f"backward {cond} ({c['size']} flat tokens): loss {loss.item():.4f}, grad finite "
              f"{bool(torch.isfinite(g).all())}, |grad| on X's tokens A mounted {g[tree_pos(x_rows)].norm():.3e}",
              flush=True)
        del h, emb, g, loss
        torch.cuda.empty_cache()
    model.eval()
    model.gradient_checkpointing_disable()


if __name__ == "__main__":
    main()
