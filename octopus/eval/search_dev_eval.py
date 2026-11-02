"""Evaluation of a search run's policy (a saved adapter, or v0: the run's initial one) on the episodes RL trains on, k
samples per question: the search counterpart of octopus/eval/dev_eval.py, with the same run arguments (--run or
--config, --lora, --env, config overrides: config.add_run_args). Each episode is training's (config.workflow,
octopus/envs/search/rl.SearchEpisodeAgent, on the training row of its question) through training's AReaL proxy
(octopus/rl/proxy.py) over --servers. Appends one JSON line per episode to --out (a rerun skips the
episodes already there, and runs those with an error again: an episode's last row counts): task_dir (search-<qid>),
qid, sample, lora, y, forked, C, C_wall, control {kind, y, C, C_wall, wall_s, calls, prompt / completion tokens}
(the main-first control from main's fork: rl.plan_control, which search training runs without; rl.plan_control=true
adds it), J, P, ended_with, reminded {main, branches} (core.REMIND_NOTEs), rewinds {calls, steps, note_chars}
(OCTOPUS_REWIND_TOOL: the tree's rewinds, the steps they removed, their notes' characters), calls, prompt / completion
tokens, branches, trial, wall_s (calls and tokens: the episode tree's, the control's own under control; wall_s: the
episode's minus the control's own run time, grading it is still included); or error (an infra failure: not scored).
scripts/eval/summary.py reads these rows.

Both sets are graded by octopus/envs/search/reward.grade (exact match, else the BrowseComp-Plus judge at --judge-url;
no --judge-url: exact match only; browsecomp-plus then needs --keep-trials for the bcp re-grade), the judge at temperature 0
as in training; the policy samples at EVAL_SAMPLING (temperature 0.7, top_p 0.95: octopus/envs/search/rl.py),
never through OCTOPUS_JUDGE_CACHE (its keys hold the question).
--set validation: the 500 OpenResearcher validation questions (data/search/openresearcher_split.json; --dataset-dir =
  the downloaded OpenResearcher__OpenResearcher-Corpus-Gold-Doc).
--set browsecomp-plus: the 830 BrowseComp-Plus queries, decrypted in memory (--dataset-dir = Tevatron__browsecomp-plus);
  rows add recall (the official one: evidence docs among every docid search returned), open_recall and
  open_gold_recall (evidence / gold docs the fork tree opened). `python -m octopus.eval.bcp OUT.jsonl ...`
  re-grades the same trials with the judge alone.
Rows hold ids and scores only, never a question or gold answer; the trial dirs hold the conversation, so the question.
validation: in --jobs-dir, on the lab host. browsecomp-plus (decrypted in memory only, never on disk): in a private
mode-700 dir on tmpfs (/dev/shm), removed when the run ends unless --keep-trials (then it stays there, e.g. for an
bcp re-grade, until removed or reboot); --jobs-dir is refused. SIGTERM / SIGHUP / Ctrl-C still remove it; a
request in flight then may recreate its trial's agent/requests.jsonl (ids and token counts, no text), and SIGKILL or
the OOM killer leave the whole dir in RAM until reboot: rm -rf /dev/shm/bcp-trials-*. OCTOPUS_SEARCH_URL: the retrieval service of the set's corpus (BM25 is the
main setting for both sets, as in training; a dense service is the option); OCTOPUS_JUDGE_KEY: the judge's bearer key,
if any.

usage: OCTOPUS_SEARCH_URL=http://HOST:PORT python -m octopus.eval.search_dev_eval --set validation|browsecomp-plus
       --dataset-dir DIR --servers 127.0.0.1:30300,... --lora rq1-vN --out OUT.jsonl (--run RUN_LOG_DIR | --config YAML)
       [--tokenizer DIR] [--env KEY=VALUE ...] [--judge-url URL/v1] [--judge-model M] [--k 2] [--parallel 12 per server]
       [--jobs-dir DIR | --keep-trials] [--offset K] [--limit N] [--qids FILE] [key=value config overrides ...]
--env OCTOPUS_LINEAR=1: a single agent (octopus/envs/search/agent.py), no fork.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

from octopus.rl.config import add_run_args, eval_config, workflow_kwargs
from octopus.envs.search import data
from octopus.eval import bcp
from octopus.envs.search.defaults import DEFAULTS
from octopus.envs.search.rl import EVAL_SAMPLING, SearchEpisodeAgent, search_row


def questions(name: str, dataset_dir: str) -> list[dict]:
    """{qid, question, answer} per question of the set (browsecomp-plus: also evidence_docids, gold_docids)."""
    if name == "validation":
        return [{"qid": str(r["query_id"]), "question": r["query"], "answer": r["answer"]}
                for r in data.openresearcher_split("validation", dataset_dir)]
    return [{**r, "qid": str(q), "question": r["query"]} for q, r in data.browsecomp_plus(dataset_dir).items()]


def outcome(o: dict) -> dict:
    """The row fields of an episode's record (octopus/envs/search/reward.search_outcome: the agent's, which raised on an
    infra failure)."""
    trial = Path(o["trial"])
    requests = [json.loads(x) for x in (trial / "agent/requests.jsonl").read_text().splitlines()]
    row = {k: o[k] for k in ("y", "forked", "C", "C_wall", "J", "P", "ended_with", "trial", "rewinds",
                                  "reminded")}
    row["judge_unparsed"] = o.get("judge_unparsed", 0)

    def totals(rows: list[dict]) -> dict:
        return {"calls": len(rows), **{k: sum(r.get(k) or 0 for r in rows)
                                       for k in ("prompt_tokens", "completion_tokens")}}
    row["control"] = None
    if o["control"]:  # its own totals: the row's are the episode tree's, comparable with a linear run's
        row["control"] = {**{k: o["control"][k] for k in ("kind", "y", "C", "C_wall")},
                          "wall_s": json.loads((trial / "agent/tree.json").read_text())["control"]["wall_s"],
                          **totals([r for r in requests if r.get("thread") == "control"])}
    threads = {}  # per thread (not the control): the first and the last prompt, in request order
    for r in requests:
        if r.get("thread") != "control":
            threads.setdefault(r.get("thread"), []).append(r.get("prompt_tokens") or 0)
    growth = sum(p[-1] - p[0] for p in threads.values())  # the context tokens the threads added (thread_budget's measure)
    return {**row, **totals([r for r in requests if r.get("thread") != "control"]), "branches": len(o["branches"]),
            "thread_prompt_growth": growth}


def summary(rows: list[dict]) -> str:
    rows = list({(r["qid"], r["sample"]): r for r in rows}.values())  # an episode's last row: a retry supersedes
    ok = [r for r in rows if "y" in r]
    mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731
    d = [r["y"] - r["control"]["y"] for r in ok if r["forked"] and r.get("control")]
    out = f"n={len(ok)} accuracy {mean([r['y'] for r in ok]):.3f} | forked {mean([r['forked'] for r in ok]):.0%}"
    if d:
        out += f" | fork - control {mean(d):+.3f} n={len(d)}"
    for k in ("recall", "open_recall", "open_gold_recall"):
        if any(r.get(k) is not None for r in ok):
            out += f" | {k} {mean([r[k] for r in ok if r.get(k) is not None]):.3f}"
    if any((r.get("rewinds") or {}).get("calls") for r in ok):  # per episode
        out += " | rewinds " + ", ".join(f"{k} {mean([(r.get('rewinds') or {}).get(k, 0) for r in ok]):.2f}"
                                         for k in ("calls", "steps", "note_chars"))
    if unparsed := sum(r.get("judge_unparsed") or 0 for r in ok):  # answers the judge gave no verdict on: incorrect
        out += f" | judge_unparsed {unparsed}"
    return out + f"\ninfra failures (not scored): {len(rows) - len(ok)}"


async def main_async(args, config, qs: list[dict]) -> None:
    from areal.utils.dynamic_import import import_from_string

    if not issubclass(import_from_string(config.workflow), SearchEpisodeAgent):  # e.g. --config's repo default
        raise SystemExit(f"workflow {config.workflow} runs no search episodes: a search run's --run, or the override "
                         "workflow=octopus.envs.search.rl.SearchEpisodeAgent")
    os.environ.pop("OCTOPUS_JUDGE_CACHE", None)
    if args.judge_url:
        os.environ.update(OCTOPUS_JUDGE_URL=args.judge_url, OCTOPUS_JUDGE_MODEL=args.judge_model,
                          OCTOPUS_JUDGE_TEMPERATURE="0")  # as training's (CLM's), not BCP's 0.7
    else:
        os.environ.pop("OCTOPUS_JUDGE_URL", None)
    if args.qids:  # exactly these questions, in the file's order
        by_id = {q["qid"]: q for q in qs}
        qs = [by_id[line.strip()] for line in open(args.qids) if line.strip()]
    qs = qs[args.offset:][: args.limit]
    out = Path(args.out)
    if args.set == "browsecomp-plus":  # plaintext questions stay off persistent disk
        jobs = Path(tempfile.mkdtemp(prefix="bcp-trials-", dir="/dev/shm"))
    else:
        jobs = Path(args.jobs_dir or "/home/user/harbor-udocker/rl/jobs-search-eval")
        jobs.mkdir(parents=True, exist_ok=True)
    try:
        await run_all(args, config, qs, out, jobs)
    finally:
        if args.set == "browsecomp-plus" and not args.keep_trials:
            shutil.rmtree(jobs)
        elif args.set == "browsecomp-plus":
            print(f"trial dirs kept on tmpfs: {jobs}", flush=True)


async def run_all(args, config, qs: list[dict], out: Path, jobs: Path) -> None:
    from areal.utils.dynamic_import import import_from_string

    from octopus.rl import proxy

    done = {(r["qid"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())
            if "error" not in r} if out.exists() else set()  # an infra failure is run again
    todo = [(q, s) for s in range(args.k) for q in qs if (q["qid"], s) not in done]
    print(f"{args.set}: {len(qs)} questions x {args.k}: {len(todo)} episodes to run ({len(done)} already in {out})",
          flush=True)
    servers = args.servers.split(",")
    agent = import_from_string(config.workflow)(**{**workflow_kwargs(config), "jobs_dir": str(jobs),
                                                   "parallel": args.parallel * len(servers), **EVAL_SAMPLING})
    gate = asyncio.Semaphore(args.parallel * len(servers))

    async def one(q: dict, sample: int, url: str) -> None:
        async with gate:
            start = time.time()
            try:
                rec = outcome(await proxy.episode(agent, search_row(config, q), url, f"search-{q['qid']}:{sample}"))
                if "evidence_docids" in q:
                    rec.update(bcp.recalls(bcp.trial_run(rec["trial"]), q))
            except Exception as e:  # infra failure (retrieval, model request, judge): not a policy outcome
                rec = {"error": f"{type(e).__name__}: {e}"[:300]}
        ran = (rec.get("control") or {}).get("wall_s") or 0
        rec["wall_s"] = round(time.time() - start - ran)  # minus the control's run (under it)
        with out.open("a") as f:
            f.write(json.dumps({"task_dir": f"search-{q['qid']}", "kind": "search", "set": args.set, "qid": q["qid"],
                                "sample": sample, "lora": args.lora, "k": args.k,
                                    "linear": int(bool(os.environ.get("OCTOPUS_LINEAR"))),
                                    "main_max_calls": os.environ.get("OCTOPUS_MAIN_MAX_CALLS",
                                                                     DEFAULTS["OCTOPUS_MAIN_MAX_CALLS"]), **rec}) + "\n")

    async with proxy.serve(config, servers, args.lora) as url:
        await asyncio.gather(*(one(q, s, url) for q, s in todo))
    print(summary([json.loads(x) for x in out.read_text().splitlines()]), flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["validation", "browsecomp-plus"])
    ap.add_argument("--dataset-dir", required=True, help="the set's downloaded HF dataset dir")
    ap.add_argument("--servers", required=True, help="host:port,... of the SGLang servers serving --lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default="", help="OpenAI-compatible base URL ending in /v1 (empty: exact match only)")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=12, help="concurrent episodes per server")
    ap.add_argument("--jobs-dir", help="validation's trial dirs (default /home/user/harbor-udocker/rl/jobs-search-eval)")
    ap.add_argument("--keep-trials", action="store_true", help="browsecomp-plus: keep the tmpfs trial dirs")
    ap.add_argument("--qids", help="file of qids, one per line: run exactly these, in order")
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET questions (to split a run)")
    ap.add_argument("--limit", type=int, help="only the first LIMIT questions (smoke runs)")
    add_run_args(ap)
    args = ap.parse_intermixed_args(argv)
    if args.set == "browsecomp-plus" and args.jobs_dir:
        ap.error("--jobs-dir: browsecomp-plus trial dirs stay on tmpfs (--keep-trials keeps them)")
    if args.set == "browsecomp-plus" and not args.judge_url and not args.keep_trials:
        ap.error("browsecomp-plus without --judge-url: --keep-trials, so that bcp can grade the trials")
    return args


def main():
    args = parse()
    config = eval_config(args)
    for sig in (signal.SIGTERM, signal.SIGHUP):  # as Ctrl-C: main_async's finally removes the tmpfs trial dirs
        if signal.getsignal(sig) is not signal.SIG_IGN:  # nohup's ignored SIGHUP stays ignored
            signal.signal(sig, signal.default_int_handler)
    asyncio.run(main_async(args, config, questions(args.set, args.dataset_dir)))


if __name__ == "__main__":
    main()
