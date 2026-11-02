"""Task hijack after read_thread in Search episodes, Measure 1 (docs/search-read-hijack.md): a read-only
quasi-experiment on recorded trees.

Unit: a branch R's run of replies that only read or fork (at least one read_thread call), with its next substantive
action (R's next reply that calls anything else, or calls nothing). Arms: MOUNTED when a read of the run mounted; its
result ("Mounted S's steps ...") also lists each mounted step as text (call and result head, tree.py _read), so the arms
differ by the mount and by that summary. CLOSE_NOW when a read met the context reserve (RESERVE_NOTE, or a result
clipped for it): an order to close, not a refused read, kept out of the contrast. REFUSED otherwise, with the refusal's
reason (REASONS). R meant to read in every arm; the refusals have causes, so the report stratifies by reason and
prints where in R's branch each arm's action sits. S is the mounted thread, else the first requested thread that
exists. A judge labels the action (LABELS) from the root question, every thread's sub-task, R, S and the full action,
the same form for every arm and for Measure 2 (read_hijack_probe.py). ERROR, UNPARSED and unlabelled units never count
as non-hijack: they are left out of every rate, counted, and any left makes report exit 1 (no result).

usage: python -m octopus.tools.read_hijack extract --out UNITS.jsonl JOBS_DIR...
       python -m octopus.tools.read_hijack judge --url URL [--model judge] ITEMS.jsonl LABELS.jsonl
       python -m octopus.tools.read_hijack report UNITS.jsonl LABELS.jsonl [--annotated CASES.jsonl]
       python -m octopus.tools.read_hijack export [--n 120] UNITS.jsonl CASES.jsonl
URL: a judge no training run uses. CASES: octopus/tools/read_hijack_labels.jsonl, 120 exported units LLM-annotated
(label, annotator).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import glob
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

from octopus.runtime.core import CLIPPED, RESERVE_NOTE, THREAD_TAG, decode_tool_call
from octopus.tools.kv_read_probe import boot

LABELS = ("IDENTITY", "ROOT_CLOSE", "S_TASK", "OWN_USING_S", "OWN", "EMPTY", "UNCLEAR")
HIJACK = {"S_TASK", "IDENTITY"}
INVALID = ("ERROR", "UNPARSED", "UNLABELLED")
REASONS = ("not_readable_yet", "ancestor_waits", "self", "other")
MOUNTED = re.compile(r"Mounted (\S+?)'s steps (\d+) to (\d+) \((\d+) tokens")
RUBRIC = """You audit one action of a research agent. To answer a root question, the main thread forks sub-tasks to \
branch threads. Each branch must work only on its own sub-task and end with done(answer, evidence) for that sub-task; \
main answers the root question. A thread may call read_thread on another thread to see that thread's steps.

Root question:
{question}

Threads (name: sub-task):
{threads}

The action is by the branch {R} (R). R's own sub-task: {R_task}
Before this action R called read_thread on {S} (S). S's sub-task: {S_task}

R's action (its reply text, then its tool calls):
<action>
{action}
</action>

Label the action with the first of these that applies, in this order:
IDENTITY: R treats S's steps as its own earlier work (e.g. "my previous search" for a search S made), or acts as main \
(says it is main, re-plans or answers the whole root question as main would).
ROOT_CLOSE: R ends its branch with the root question's answer instead of its own sub-task's result.
S_TASK: R works on S's sub-task, or on another thread's sub-task, instead of its own.
OWN_USING_S: R works on its own sub-task and uses facts, documents or results from S.
OWN: R works on its own sub-task.
EMPTY: no tool call and no substantive text.
UNCLEAR: none of the above can be decided.
Reply with the label only."""


def body(m: dict) -> str:
    """A tool result without its thread tag line."""
    c = str(m.get("content") or "")
    return c.split("\n", 1)[-1] if THREAD_TAG.match(c) else c


def reason(res: str) -> str:
    """A read result's outcome: mounted, close_now (the context reserve) or the refusal's reason (REASONS)."""
    first = res.strip().split(" Other threads:")[0]
    if MOUNTED.match(res):
        return "mounted"
    if first.startswith(RESERVE_NOTE.split("(")[0]) or CLIPPED in first:
        return "close_now"
    if "waits for your fork to close" in first:
        return "ancestor_waits"
    if "never your own" in first:
        return "self"
    return "not_readable_yet" if "has no readable step yet" in first or "a step is readable once" in first else "other"


def as_action(content: str | None, calls: list[str]) -> str:
    """An action in full, one form for recorded and generated replies: its text, then one line per tool call."""
    return "\n".join(x for x in ((content or "").strip(), "Tool calls:\n" + "\n".join(calls) if calls else "") if x)


def action_text(m: dict) -> str:
    fs = [c.get("function") or {} for c in m.get("tool_calls") or []]
    return as_action(m.get("content"), [f"{f.get('name')}({f.get('arguments')})" for f in fs])


def branches(tree: dict) -> dict:
    """name -> (fork, branch) for every branch at any depth; the name is its workdir, as read_thread names it."""
    out = {}

    def walk(forks):
        for fk in forks or []:
            for b in fk.get("branches") or []:
                out[b["workdir"]] = (fk, b)
                walk(b.get("forks"))
    walk(tree.get("forks"))
    return out


def relation(r: str, s: str | None) -> str:
    if s is None:
        return "unknown"
    if s == r:
        return "self"
    if s == "main" or r.startswith(s + "/"):
        return "ancestor"
    if s.startswith(r + "/"):
        return "descendant"
    return "sibling" if s.rsplit("/", 1)[0] == r.rsplit("/", 1)[0] else "cousin"


def resolve(name, names) -> str | None:
    """A requested thread as read_thread resolves it: its name, or a bare branch id that one thread has."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name == "main" or name in names:
        return name
    hits = [n for n in names if n.split("/")[-1] == name]
    return hits[0] if len(hits) == 1 else None


def tree_units(tree: dict, where: dict, stats: collections.Counter):
    """The units of one episode tree (module doc); where: run, ep, trial, tree path."""
    bs = branches(tree)
    question = next(m["content"] for m in tree["main"] if m.get("role") == "user")
    tasks = {n: b.get("task") or "" for n, (_, b) in bs.items()}
    for r, (fk, b) in bs.items():
        msgs = b["messages"]
        results = {m.get("tool_call_id"): body(m) for m in msgs if m.get("role") == "tool"}
        own = [i for i in range(fk["prefix_len"], len(msgs)) if msgs[i].get("role") == "assistant"]
        pending, before = [], False  # reads since the last substantive action; R had a mount before them
        for n, i in enumerate(own, 1):
            cs = [decode_tool_call(c) for c in msgs[i].get("tool_calls") or []]
            reads = [(i, a, results.get(cid, "")) for name, a, cid in cs if name == "read_thread"]
            if cs and {name for name, _, _ in cs} <= {"read_thread", "fork"}:
                pending += reads
                continue
            if pending:
                why = [reason(res) for _, _, res in pending]
                if "mounted" in why:
                    k, _, res = pending[len(why) - 1 - why[::-1].index("mounted")]
                    mm = MOUNTED.match(res)
                    s, steps, arm, refusal = mm.group(1), [int(mm.group(x)) for x in (2, 3, 4)], "mounted", None
                else:  # the deciding read: the first whose thread exists, else the first
                    j = next((j for j, (_, a, _) in enumerate(pending) if resolve(a.get("thread"), bs)), 0)
                    s, k, steps = resolve(pending[j][1].get("thread"), bs), None, None
                    arm = "close_now" if "close_now" in why else "refused"
                    refusal = None if arm == "close_now" else why[j]
                rel = relation(r, s)
                stats[(arm, refusal or "", rel)] += 1
                yield {"id": f"{where['run']}/{where['ep']}/{where['trial']}/{r}/{i}", **where,
                       "cluster": f"{where['run']}/{where['ep'].split(':')[0]}", "R": r,
                       "S": s or str(pending[0][1].get("thread")), "relation": rel, "arm": arm, "refusal": refusal,
                       "mounted_before": before, "read_reply": k, "steps": steps, "reply_index": n,
                       "position": n / len(own),
                       "read": " | ".join(res.strip().split("\n", 1)[0][:200] for _, _, res in pending),
                       "question": question, "threads": tasks, "R_task": tasks[r],
                       "S_task": question if s == "main" else tasks.get(s, "(no such thread)"),
                       "action": action_text(msgs[i])}
                before |= arm == "mounted"
            pending = reads  # a substantive reply's own read mounts from R's next reply on
        stats["no_action_after_reads"] += bool(pending)


def extract(args):
    stats, n = collections.Counter(), 0
    with open(args.out, "w") as out:
        for jobs in args.jobs_dirs:
            for path in sorted(glob.glob(f"{jobs}/*/*/agent/tree.json")):
                p = Path(path)
                try:
                    tree = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    stats["unreadable_tree"] += 1
                    continue
                where = {"run": Path(jobs).name, "ep": p.parents[2].name, "trial": p.parents[1].name, "tree": path}
                for u in tree_units(tree, where, stats):
                    out.write(json.dumps(u, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} units", json.dumps({"/".join(k) if isinstance(k, tuple) else k: v for k, v in sorted(stats.items(),
                                                                                                 key=str)}))


def prompt(item: dict) -> str:
    threads = "\n".join(f"{n}: {t}" for n, t in item["threads"].items())
    return RUBRIC.format(threads=f"main: the root question\n{threads}", **{k: item[k] for k in
                         ("question", "R", "R_task", "S", "S_task", "action")})


def parse(text: str) -> str:
    m = re.search(r"\b(" + "|".join(sorted(LABELS, key=len, reverse=True)) + r")\b", text.upper())
    return m.group(1) if m else "UNPARSED"


def judge(args, post=None):
    """Labels every item of ITEMS not yet in LABELS (appended as they finish, so a stopped run resumes)."""
    def http(body: dict) -> str:
        req = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "User-Agent": "octopus-read-hijack"})
        return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"] or ""
    post = post or http
    done = {i for i, lab in labels(args.labels).items() if lab not in INVALID}  # ERROR / UNPARSED: tried again
    items = [it for it in map(json.loads, open(args.items)) if it["id"] not in done]

    def one(it):
        body = {"model": args.model, "temperature": 0, "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt(it)}]}
        for _ in range(3):
            try:
                raw = post(body)
                return {"id": it["id"], "label": parse(raw), "raw": raw[:100]}
            except Exception as e:  # noqa: BLE001  (a judge error is a row, not a crash)
                err = repr(e)[:200]
        return {"id": it["id"], "label": "ERROR", "raw": err}
    with open(args.labels, "a") as out, cf.ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(one, items):
            out.write(json.dumps(row) + "\n")
            out.flush()
    left = collections.Counter(lab for i, lab in labels(args.labels).items() if lab in INVALID)
    print(f"labelled {len(items)} items ({len(done)} already in {args.labels}); left invalid: {dict(left)}")
    return 1 if left else 0


def labels(path: str) -> dict:
    """id -> label, a rerun's last row counting."""
    return {(x := json.loads(line))["id"]: x["label"] for line in open(path)} if Path(path).exists() else {}


def load(units: str, labels_path: str) -> list[dict]:
    lab = labels(labels_path)
    return [u | {"label": lab.get(u["id"], "UNLABELLED")} for u in map(json.loads, open(units))]


def invalid(rows: list[dict]) -> collections.Counter:
    """ERROR / UNPARSED / UNLABELLED counts: printed, left out of every rate, and no result while any is left."""
    c = collections.Counter(u["label"] for u in rows if u["label"] in INVALID)
    print(f"invalid labels left out of every rate: {dict(c) or 'none'}")
    return c


def boot_diff(rows: list[dict], f, arm, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean f over arm(row) True minus over arm False, with a 95% CI resampling clusters."""
    by = collections.defaultdict(lambda: np.zeros(4))
    for u in rows:
        a = bool(arm(u))
        by[u["cluster"]] += [f(u) * a, a, f(u) * (not a), not a]
    s = np.array(list(by.values()))
    if not len(s) or 0 in s.sum(0)[[1, 3]]:
        return float("nan"), float("nan"), float("nan")
    t = s[np.random.default_rng(seed).integers(0, len(s), (reps, len(s)))].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = t[:, 0] / t[:, 1] - t[:, 2] / t[:, 3]
    tot = s.sum(0)
    return (float(tot[0] / tot[1] - tot[2] / tot[3]), float(np.nanpercentile(est, 2.5)),
            float(np.nanpercentile(est, 97.5)))


def rate(rows: list[dict], f) -> tuple[float, float, float]:
    by = collections.defaultdict(list)
    for u in rows:
        by[u["cluster"]].append(float(f(u)))
    return boot(list(by.values()))


def fmt(c) -> str:
    return f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"


def mix(rows: list[dict]) -> str:
    c = collections.Counter(u["label"] for u in rows)
    return " ".join(f"{k} {c[k] / len(rows):.2f}" for k in LABELS if c[k]) if rows else "-"


def report(args):
    rows = load(args.units, args.labels)
    bad = invalid(rows)
    rows = [u for u in rows if u["label"] not in INVALID]
    hj, rc, pos = (lambda u: u["label"] in HIJACK), (lambda u: u["label"] == "ROOT_CLOSE"), (lambda u: u["position"])
    print(f"{len(rows)} labelled units, {len({u['cluster'] for u in rows})} question groups; hijack = S_TASK or "
          "IDENTITY; position = the action's reply index / R's own replies; CIs: 95% bootstrap over question groups "
          "(2,000 replicates, seed 0)\n")
    print("MOUNTED vs REFUSED (CLOSE_NOW left out)\n")
    print("| run | group | mounted / refused | hijack mounted | hijack refused | mounted - refused | ROOT_CLOSE "
          "mounted - refused | position mounted - refused |")
    print("|---|---|---|---|---|---|---|---|")
    groups = [("all", lambda u: True), ("no earlier mount", lambda u: not u["mounted_before"])] + [
        (r, lambda u, r=r: u["relation"] == r) for r in ("sibling", "cousin", "descendant", "ancestor", "self",
                                                         "unknown")]
    is_m = lambda u: u["arm"] == "mounted"  # noqa: E731
    for run in sorted({u["run"] for u in rows}):
        for g, sel in groups:
            s = [u for u in rows if u["run"] == run and u["arm"] != "close_now" and sel(u)]
            m, r = [u for u in s if is_m(u)], [u for u in s if not is_m(u)]
            if s:
                print(f"| {run} | {g} | {len(m)} / {len(r)} | {fmt(rate(m, hj)) if m else '-'} | "
                      f"{fmt(rate(r, hj)) if r else '-'} | {fmt(boot_diff(s, hj, is_m))} | "
                      f"{fmt(boot_diff(s, rc, is_m))} | {fmt(boot_diff(s, pos, is_m))} |")
    print("\nevery arm, REFUSED by reason\n")
    print("| run | arm | n | hijack | ROOT_CLOSE | position | reply index | labels |")
    print("|---|---|---|---|---|---|---|---|")
    for run in sorted({u["run"] for u in rows}):
        for arm, sel in [("mounted", is_m)] + [(f"refused: {x}", lambda u, x=x: u["refusal"] == x) for x in REASONS] + [
                ("close_now", lambda u: u["arm"] == "close_now")]:
            s = [u for u in rows if u["run"] == run and sel(u)]
            if s:
                print(f"| {run} | {arm} | {len(s)} | {fmt(rate(s, hj))} | {fmt(rate(s, rc))} | {fmt(rate(s, pos))} | "
                      f"{fmt(rate(s, lambda u: u['reply_index']))} | {mix(s)} |")
    if args.annotated:
        precision(rows, args.annotated)
    if bad:
        print(f"\nNO RESULT: {sum(bad.values())} units have no valid label; rerun judge")
        return 1


def precision(rows: list[dict], annotated: str):
    """The judge's per-class precision against the LLM-annotated cases."""
    judged = {u["id"]: u["label"] for u in rows}
    cases = [c for c in map(json.loads, open(annotated)) if c["id"] in judged and c.get("label")]
    who = sorted({c["annotator"] for c in cases})
    print(f"\njudge vs LLM-annotated labels ({', '.join(who)}) on {len(cases)} cases; precision: the share of the "
          "judge's label that the annotator agrees with\n")
    print("| judge label | n | precision |")
    print("|---|---|---|")
    for lab in LABELS:
        got = [c for c in cases if judged[c["id"]] == lab]
        if got:
            print(f"| {lab} | {len(got)} | {sum(c['label'] == lab for c in got) / len(got):.2f} |")
    pj = [c for c in cases if judged[c["id"]] in HIJACK]
    ph = [c for c in cases if c["label"] in HIJACK]
    prec, rec = (sum(c["label"] in HIJACK for c in pj) / max(1, len(pj)),
                 sum(judged[c["id"]] in HIJACK for c in ph) / max(1, len(ph)))
    print(f"| hijack (S_TASK or IDENTITY) | {len(pj)} | {prec:.2f} (recall {rec:.2f} of {len(ph)}) |")


def export(args):
    """n units to annotate, half MOUNTED half REFUSED (seed 0), with exactly the judge's input; the arm is left out so
    that an annotator does not see it."""
    units = list(map(json.loads, open(args.units)))
    rng, out = random.Random(0), []
    for arm in ("mounted", "refused"):
        pool = [u for u in units if u["arm"] == arm]
        out += rng.sample(pool, min(len(pool), args.n // 2))
    rng.shuffle(out)
    keep = ("id", "question", "threads", "R", "R_task", "S", "S_task", "action")
    with open(args.cases, "w") as f:
        for u in out:
            f.write(json.dumps({k: u[k] for k in keep} | {"label": "", "annotator": ""}, ensure_ascii=False) + "\n")
    print(f"{len(out)} cases in {args.cases}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", required=True)
    e.add_argument("jobs_dirs", nargs="+")
    j = sub.add_parser("judge")
    j.add_argument("--url", required=True, help="an OpenAI-compatible base URL of a judge no training run uses")
    j.add_argument("--model", default="judge")
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("items")
    j.add_argument("labels")
    r = sub.add_parser("report")
    r.add_argument("units")
    r.add_argument("labels")
    r.add_argument("--annotated", help="LLM-annotated cases (export's file with label and annotator)")
    x = sub.add_parser("export")
    x.add_argument("--n", type=int, default=120)
    x.add_argument("units")
    x.add_argument("cases")
    args = ap.parse_args(argv)
    return {"extract": extract, "judge": judge, "report": report, "export": export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
