"""Task hijack after read_thread in Search episodes, Measure 2 (docs/search-read-hijack.md): a counterfactual probe.

At real mount points (Measure 1's MOUNTED units, octopus/tools/read_hijack.py: R's history through the reply whose read
mounted and that reply's tool results), R's next reply is sampled again from the same checkpoint on a mount-capable
SGLang server (cursor + mount patches, /generate, docs/cursor-protocol.md "Mount"), under three conditions:
  none   R's history as recorded, nothing mounted (the read's result text, with its one line per step, stays)
  text   the same, with S's mounted steps appended in full as text to the read's result
  mount  none plus the mount of S's steps, the runtime's read (K2): S's KV as S computed it, in R's attention layers
All three continue one cursor of R's history (prefill-only /generate), and the mount names a prefill-only cursor of S's
history, over the token range the read named (delta_render.read_range: from the turn of step lo's reply to the turn of
the reply after step hi, or through step hi's reply when S had ended). Besides the mounted rows, mount differs from
none by K2's position jump: R's tokens continue at max(len(B), e), so each point records jump = max(0, e - len(B)) and
the report stratifies by it. Each sample becomes a judge item in Measure 1's form (python -m octopus.tools.read_hijack
judge). Rendering is the chat template with thinking off (kv_read_probe.render), so absolute positions may differ from
the run's by a constant per thread, the same for every condition; range_minus_read records the rendered range minus the
read's token count. Skipped (counted), since this probe rebuilds no other mount: a point whose R had mounted before,
whose S had mounted anything before the range end (a transitive read included), or whose S has rewound.
run appends to ITEMS and skips (point, checkpoint) pairs already there; a failed pair is printed and tried again on the
next run (exit 1 while any failed). report leaves ERROR / UNPARSED / UNLABELLED samples out, counts them, and gives no
verdict (exit 1) while any is left.

usage: python -m octopus.tools.read_hijack_probe run --tokenizer TOK --servers H:P[,H:P...] --lora NAME[,NAME...]
           [--n 300] [--samples 4] UNITS.jsonl ITEMS.jsonl
       python -m octopus.tools.read_hijack_probe report ITEMS.jsonl LABELS.jsonl
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from octopus.tools.kv_read_probe import boot, render
from octopus.tools.read_hijack import (HIJACK, INVALID, LABELS, MOUNTED, action_text, as_action, body, branches, fmt,
                                       labels)

CONDITIONS = ("none", "text", "mount")
STEP = re.compile(r"^\[[^\]\n]*\| step (\d+)\]")
TEXT = "\n\nThe full text of these steps:\n"


class Skip(Exception):
    pass


def text_of(msgs: list[dict]) -> str:
    return "\n".join(f"[{m['role']}] " + (action_text(m) if m["role"] == "assistant" else str(m.get("content") or ""))
                     for m in msgs)


def point(u: dict, tok) -> dict:
    """Token sequences of one mount point (module doc); Skip when it cannot be reproduced."""
    tree = json.loads(Path(u["tree"]).read_text())
    bs, tools = branches(tree), tree["tools"]
    rmsgs, k = bs[u["R"]][1]["messages"], u["read_reply"]
    t = next((j for j in range(k + 1, len(rmsgs)) if rmsgs[j]["role"] != "tool"), len(rmsgs))
    hist = rmsgs[:t]
    mounted_in = lambda msgs: any(m["role"] == "tool" and MOUNTED.match(body(m)) for m in msgs)  # noqa: E731
    if mounted_in(rmsgs[:k]):
        raise Skip("reader mounted before")
    src = bs[u["S"]][1]
    if src.get("rewinds"):
        raise Skip("source rewound")
    smsgs, (lo, hi, n) = src["messages"], u["steps"]
    tags = collections.defaultdict(list)
    for i, m in enumerate(smsgs):
        if m["role"] == "tool" and (mm := STEP.match(str(m.get("content")))):
            tags[int(mm.group(1))].append(i)
    if any(v[-1] - v[0] >= len(v) for v in tags.values()):
        raise Skip("a step tag twice")

    def reply(after: int | None) -> int | None:  # the first reply after history index `after`
        return None if after is None else next((j for j in range(after + 1, len(smsgs))
                                                if smsgs[j]["role"] == "assistant"), None)
    i_lo, to = reply((tags.get(lo - 1) or [None])[-1]), reply((tags.get(hi) or [None])[-1])
    if to is None:  # S had ended: through step hi's reply
        last = reply((tags.get(hi - 1) or [None])[-1])
        if last is None:
            raise Skip("steps not in the source's history")
        a_msgs = smsgs[:last + 1]
    else:
        a_msgs = smsgs[:to]
    if i_lo is None:
        raise Skip("steps not in the source's history")
    if mounted_in(a_msgs):
        raise Skip("source mounted before the range end")

    def ids(msgs, gen=False):
        return tok.encode(render(tok, msgs, tools, gen), add_special_tokens=False)
    a, pre = ids(a_msgs), ids(smsgs[:i_lo])
    if a[:len(pre)] != pre:
        raise Skip("source rendering is not a prefix")
    s, e = len(pre), len(a)
    mounted = smsgs[i_lo:len(a_msgs)]
    j = next(j for j in range(k + 1, t) if MOUNTED.match(body(hist[j])))
    with_text = hist[:j] + [{**hist[j], "content": hist[j]["content"] + TEXT + text_of(mounted)}] + hist[j + 1:]
    b, full, full_text = ids(rmsgs[:k + 1]), ids(hist, True), ids(with_text, True)
    if full[:len(b)] != b or full_text[:len(b)] != b:
        raise Skip("reader rendering is not a prefix")
    return {"a": a, "s": s, "e": e, "b": b, "delta": {"none": full[len(b):], "text": full_text[len(b):]},
            "jump": max(0, e - len(b)), "range_minus_read": e - s - n}


def parse_reply(text: str) -> str:
    """A generated reply (Qwen3.5 tool-call markup) in Measure 1's action form: its text, then name(arguments)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    content, *rest = text.split("<tool_call>", 1)
    lines = []
    for call in re.findall(r"<function=([^>\n]+)>(.*?)</function>", rest[0] if rest else "", re.S):
        args = dict(re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", call[1], re.S))
        lines.append(f"{call[0]}({json.dumps(args, ensure_ascii=False)})")
    return as_action(content, lines)


def post(server: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://{server}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def run(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    units = [u for u in map(json.loads, open(args.units)) if u["arm"] == "mounted"]
    random.Random(0).shuffle(units)
    points, skipped, per_ep = [], collections.Counter(), collections.Counter()
    for u in units:  # in shuffled order until n points, at most 2 per episode
        if len(points) == args.n:
            break
        if per_ep[u["ep"]] == 2:
            continue
        try:
            points.append((u, point(u, tok)))
            per_ep[u["ep"]] += 1
        except Skip as x:
            skipped[str(x)] += 1
    print(f"{len(points)} mount points; skipped {dict(skipped)}", flush=True)
    servers, loras = args.servers.split(","), args.lora.split(",")
    sp = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_tokens}
    old = map(json.loads, open(args.items)) if Path(args.items).exists() else []
    done = {(x["point"], x["ckpt"]) for x in old}  # a pair's rows are written together

    def one(job):
        (u, p), lora, server = job
        gen = lambda body: post(server, {**body, "lora_path": lora})  # noqa: E731
        try:
            ca, cb = (gen({"input_ids": x, "sampling_params": {"max_new_tokens": 0}})["meta_info"]["id"]
                      for x in (p["a"], p["b"]))  # prefill-only: each a cursor of exactly these tokens
            rows = []
            for cond in CONDITIONS:
                req = {"cursor": cb, "input_ids": p["delta"]["text" if cond == "text" else "none"],
                       "sampling_params": sp}
                if cond == "mount":
                    req["mount"] = [{"cursor": ca, "start": p["s"], "end": p["e"]}]
                for i in range(args.samples):
                    raw = gen(req)["text"]
                    names = set(re.findall(r"<function=([^>\n]+)>", raw))
                    rows.append({k: u[k] for k in ("question", "threads", "R", "R_task", "S", "S_task", "relation",
                                                    "run", "cluster")}
                                | {"id": f"{u['id']}|{lora}|{cond}|{i}", "point": u["id"], "ckpt": lora, "cond": cond,
                                   "sample": i, "action": parse_reply(raw), "raw": raw, "jump": p["jump"],
                                   "range_minus_read": p["range_minus_read"],
                                   "skip": bool(names) and names <= {"read_thread", "fork"}})
            return rows
        except Exception as e:  # noqa: BLE001  (one failed pair is retried on the next run, the others go on)
            print(f"failed {u['id']} {lora}: {e!r}"[:400], flush=True)
            return None
    jobs = [(pt, lora, servers[i % len(servers)]) for i, pt in enumerate(points) for lora in loras
            if (pt[0]["id"], lora) not in done]
    failed = 0
    with open(args.items, "a") as out, cf.ThreadPoolExecutor(args.workers * len(servers)) as ex:
        for rows in ex.map(one, jobs):
            failed += rows is None
            out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows or [])
            out.flush()
    print(f"{len(jobs) - failed} (point, checkpoint) pairs written, {failed} failed, {len(done)} there before")
    return 1 if failed else 0


def report(args):
    lab = labels(args.labels)
    items = [it | {"label": lab.get(it["id"], "UNLABELLED")} for it in map(json.loads, open(args.items))]
    bad = collections.Counter(it["label"] for it in items if it["label"] in INVALID)
    print(f"invalid labels left out of every rate: {dict(bad) or 'none'}")
    # per (checkpoint, point, condition): the mean over its validly labelled samples; a read/fork-only reply is never a
    # hijack
    score = {"hijack": lambda it: it["label"] in HIJACK and not it["skip"],
             "root_close": lambda it: it["label"] == "ROOT_CLOSE" and not it["skip"],
             "read_or_fork_only": lambda it: it["skip"]}
    by = collections.defaultdict(list)
    for it in items:
        if it["label"] not in INVALID:
            by[it["ckpt"], it["point"], it["cond"]].append(it)
    ckpts = list(dict.fromkeys(it["ckpt"] for it in items))
    pts = {it["point"]: it for it in items}
    strata = {"all": lambda p: True, "sibling": lambda p: p["relation"] == "sibling",
              "cousin": lambda p: p["relation"] == "cousin", "descendant": lambda p: p["relation"] == "descendant",
              "K2 jump": lambda p: p["jump"] > 0, "no jump": lambda p: p["jump"] == 0}

    def values(ck, f, cond, ref=None, ck0=None, sel=strata["all"]):
        """per cluster, per point: f's mean under cond (minus under ref; minus the same difference at ck0)"""
        out = collections.defaultdict(list)
        for p, it in pts.items():
            m = lambda c, k: sum(map(f, by[k, p, c])) / len(by[k, p, c])  # noqa: E731
            if not sel(it) or not all((k, p, c) in by for k in {ck, ck0} - {None} for c in CONDITIONS):
                continue
            v = m(cond, ck) - (m(ref, ck) if ref else 0)
            if ck0:
                v -= m(cond, ck0) - (m(ref, ck0) if ref else 0)
            out[it["cluster"]].append(v)
        return list(out.values())
    print(f"{len(pts)} mount points ({sum(p['jump'] > 0 for p in pts.values())} with a K2 jump), "
          f"{len({p['cluster'] for p in pts.values()})} question groups, checkpoints {ckpts}; per point the mean over "
          "its samples; 95% bootstrap CIs over question groups (2,000 replicates, seed 0)\n")
    print("| checkpoint | stratum | score | none | text | mount | mount - none | text - none | mount - text |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ck in ckpts:
        for st, sel in strata.items():
            for name, f in score.items():
                if not values(ck, f, "none", sel=sel):
                    continue
                cells = [fmt(boot(values(ck, f, c, sel=sel))) for c in CONDITIONS]
                diffs = [fmt(boot(values(ck, f, a, b, sel=sel))) for a, b in (("mount", "none"), ("text", "none"),
                                                                              ("mount", "text"))]
                print(f"| {ck} | {st} | {name} | " + " | ".join(cells + diffs) + " |")
    print("\nlabel mix (validly labelled samples):")
    for ck in ckpts:
        for cond in CONDITIONS:
            c = collections.Counter(it["label"] for it in items if it["ckpt"] == ck and it["cond"] == cond
                                    and it["label"] not in INVALID)
            n = max(1, sum(c.values()))
            print(f"  {ck} {cond}: " + " ".join(f"{k} {c[k] / n:.2f}" for k in LABELS if c[k]))
    if bad:
        print(f"\nNO VERDICT: {sum(bad.values())} samples have no valid label; rerun judge")
        return 1
    print("\npre-registered decision (docs/search-read-hijack.md), hijack:")
    hj = score["hijack"]
    for ck in ckpts:
        d = boot(values(ck, hj, "mount", "none"))
        verdict = ("the mount hijacks" if d[1] > 0.05 else "no mount hijack" if d[2] <= 0.05 else "inconclusive")
        ref = boot(values(ck, hj, "text", "none"))
        print(f"  {ck}: mount - none {fmt(d)} -> {verdict}; reference text - none {fmt(ref)}")
        if ck != ckpts[0]:
            t = boot(values(ck, hj, "mount", "none", ck0=ckpts[0]))
            print(f"    (mount - none) at {ck} minus at {ckpts[0]}: {fmt(t)} -> "
                  + ("training raised the mount's hijack" if t[1] > 0 else "training lowered it" if t[2] < 0
                     else "no change shown"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--servers", required=True, help="mount-capable SGLang servers, host:port[,host:port...]")
    r.add_argument("--lora", required=True, help="adapter names loaded on every server, comma-separated; the first is "
                   "the reference checkpoint of the report")
    r.add_argument("--n", type=int, default=300, help="mount points (at most 2 per episode)")
    r.add_argument("--samples", type=int, default=4)
    r.add_argument("--temperature", type=float, default=0.7)  # the Search eval's sampling (search rl.EVAL_SAMPLING)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--max-tokens", type=int, default=4096)  # the eval's reply cap
    r.add_argument("--workers", type=int, default=8, help="mount points in flight per server")
    r.add_argument("units")
    r.add_argument("items")
    p = sub.add_parser("report")
    p.add_argument("items")
    p.add_argument("labels")
    args = ap.parse_args(argv)
    return {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
