#!/usr/bin/env python3
"""CLM-style prefix-reuse FLOPs per Search eval episode, from per-request token counts.

Analytic, not measured (as in the CLM paper). For request t of an episode:
P = prompt_tokens, G = completion_tokens, R = prompt tokens assumed served by an
ideal prefix cache, U = P - R, and

    F_t = C_token*(U + G) + C_attn*[(P^2 - R^2)/2 + G*P + G^2/2],   F = sum_t F_t

over every thread (main and branches; thread "control" and non-200 requests are
skipped). R follows the append-only cursor protocol:
  - a thread's later request: R = min(P, P_prev + G_prev) of that thread;
  - a branch's first request (thread "fork<k>/<id>", nested "<parent>/fork<k>/<id>"):
    R = min(P, P + G) of the parent thread's latest request before it;
  - main's first request: R = 0 (no cross-episode reuse).
A thread that rewinds or edits its history would reuse less, so R is then an upper bound.

Model constants come from the HF config.json (Qwen3.5-4B text: 8 full-attention layers,
24 Gated DeltaNet layers):
  C_token = 2 x non-embedding matmul parameters: full-attention q (with output gate)/k/v/o,
            GDN in_proj_qkv/z/a/b and out_proj, MLP gate/up/down, over all layers.
  C_attn  = 2*2*n_heads*head_dim per (query, key) pair x full-attention layers only
            (GDN layers have no quadratic term).
Omitted, as in CLM: GDN conv/state updates, norms, softmax, embeddings and the LM head.
The "+lm" variant adds the LM head (2*hidden*vocab) on every processed token.

Usage: prefix_flops.py CONFIG.json EVAL.jsonl REQUESTS_ROOT [EVAL.jsonl REQUESTS_ROOT ...]
REQUESTS_ROOT holds <job>/<trial>/agent/requests.jsonl at any depth (Val jobs dirs or BCP
copies); eval rows join on the last two components of their "trial" path; error rows skip.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def model_constants(cfg):
    cfg = cfg.get("text_config", cfg)
    d, L, types = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["layer_types"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n_full, n_gdn = types.count("full_attention"), types.count("linear_attention")
    assert n_full + n_gdn == L
    full = d * (nh * hd * (2 if cfg["attn_output_gate"] else 1) + 2 * nkv * hd) + nh * hd * d
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    nv = cfg["linear_num_value_heads"]
    vd = nv * cfg["linear_value_head_dim"]
    gdn = d * (2 * kd + vd + vd + 2 * nv) + vd * d
    params = n_full * full + n_gdn * gdn + L * 3 * d * cfg["intermediate_size"]
    return params, 2 * params, 2 * params + 2 * d * cfg["vocab_size"], 4 * nh * hd * n_full


def episode(rows):
    """Return (sum U, sum G, sum P, sum R, sum quadratic term) of one episode."""
    last, U, G, P, R, A = {}, 0, 0, 0, 0, 0.0
    for r in sorted(rows, key=lambda r: r["t"]):
        th, p, g = r["thread"], r["prompt_tokens"], r["completion_tokens"]
        if th in last:
            src = last[th]
        elif th == "main":
            src = (0, 0)
        else:
            src = last[th.rsplit("/", 2)[0] if th.count("/") >= 2 else "main"]
        c = min(p, sum(src))
        last[th] = (p, g)
        U, G, P, R = U + p - c, G + g, P + p, R + c
        A += (p * p - c * c) / 2 + g * p + g * g / 2
    return U, G, P, R, A


def self_check():
    rows = [
        {"thread": "main", "t": 0, "prompt_tokens": 100, "completion_tokens": 10},
        {"thread": "fork0/b-0", "t": 1, "prompt_tokens": 150, "completion_tokens": 20},  # R=110
        {"thread": "main", "t": 2, "prompt_tokens": 300, "completion_tokens": 5},  # R=110
        {"thread": "fork0/b-0", "t": 3, "prompt_tokens": 200, "completion_tokens": 7},  # R=170
        {"thread": "fork0/b-0/fork1/c", "t": 4, "prompt_tokens": 250, "completion_tokens": 1},  # R=207
    ]
    assert episode(rows[::-1]) == (403, 43, 1000, 597, 71963.0), episode(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("pairs", nargs="+", help="EVAL.jsonl REQUESTS_ROOT, repeated")
    ap.add_argument("--per-question", action="store_true", help="also print per-qid means")
    a = ap.parse_args()
    assert len(a.pairs) % 2 == 0, "pairs must be EVAL.jsonl REQUESTS_ROOT"
    self_check()
    params, c_tok, c_lm, c_attn = model_constants(json.load(open(a.config)))
    print(f"non-embedding matmul params {params:,}  C_token {c_tok:.4e}  "
          f"C_token+lm {c_lm:.4e}  C_attn {c_attn:,}")
    print(f"{'file':40s} {'n':>9s} {'y':>6s} {'PF mean':>8s} {'PF med':>8s} {'PF+lm':>8s} "
          f"{'sumU':>9s} {'sumG':>8s} {'sumP':>9s} {'reuse':>6s}")
    for ev, root in zip(a.pairs[::2], a.pairs[1::2]):
        reqs = {p.parts[-4:-2]: p for p in Path(root).rglob("requests.jsonl")
                if p.parent.name == "agent"}
        evals = [r for r in map(json.loads, open(ev)) if not r.get("error")]
        eps = []
        for r in evals:
            p = reqs.get(Path(r["trial"]).parts[-2:])
            if p:
                rows = [q for q in map(json.loads, open(p))
                        if q["http_code"] == 200 and q["thread"] != "control"]
                eps.append((r, *episode(rows)))
        if not eps:
            print(f"{Path(ev).name:40s} {0:>4d}/{len(evals):<4d}")
            continue
        pf = [(c_tok * (u + g) + c_attn * q) / 1e15 for _, u, g, _, _, q in eps]
        pf_lm = [(c_lm * (u + g) + c_attn * q) / 1e15 for _, u, g, _, _, q in eps]
        mean = lambda i: statistics.mean(e[i] for e in eps)
        print(f"{Path(ev).name:40s} {len(eps):>4d}/{len(evals):<4d} "
              f"{statistics.mean(bool(e[0]['y']) for e in eps):6.3f} "
              f"{statistics.mean(pf):8.3f} {statistics.median(pf):8.3f} "
              f"{statistics.mean(pf_lm):8.3f} {mean(1):9.0f} {mean(2):8.0f} {mean(3):9.0f} "
              f"{sum(e[4] for e in eps) / sum(e[3] for e in eps):6.3f}")
        if a.per_question:
            by = defaultdict(list)
            for (r, *_), f in zip(eps, pf):
                by[r["qid"]].append((bool(r["y"]), f))
            for qid, v in sorted(by.items()):
                print(f"  qid {qid} n {len(v)} y {statistics.mean(x for x, _ in v):.3f} "
                      f"PF {statistics.mean(f for _, f in v):.3f}")


if __name__ == "__main__":
    main()
