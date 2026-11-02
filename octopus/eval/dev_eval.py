"""Dev-split evaluation of an RL run's policy (a saved adapter, or v0: the run's initial one) on the episodes RL trains
on, k samples per task: training's episodes, through training's code. The run config (AReaL's config.yaml in the
run's log dir) gives the episode agent (config.workflow with training's arguments and dataset rows) and the AReaL proxy
its episodes call (octopus/rl/proxy.py: training's client, engine and sampling over --servers, which serve the
adapter as --lora); train.py's episode_settings.json gives the episodes' settings (config.eval_config). Appends one JSON
line per episode to --out (a rerun skips the episodes already there, and runs those with an error again: an episode's
last row counts) and prints a summary. Infra failures are counted, not scored.
A run trained before episode_settings.json existed: `RL_ENV_FILE=<its env file> scripts/rl_env/rl_run.sh settings
DIR <its EXTRA overrides>` writes one from the run's env file; copy it next to the run's config.yaml. A run whose
config.yaml has keys this code lacks (trained before PR #213: rl.plan_lambda, plan_alpha, plan_cost_unit) fails to
load: evaluate it with its own commit. The no-fork control is the override rl.plan_control=true (training never runs it).

usage: python -m octopus.eval.dev_eval --servers 127.0.0.1:30300,... --lora rq1-vN --out OUT.jsonl
       --tasks-dir DATASET_DIR (--run RUN_LOG_DIR | --config YAML) [--tokenizer DIR] [--env KEY=VALUE ...] [--k 2]
       [--parallel 12 per server] [--jobs-dir DIR] [--offset K] [--limit N] [key=value config overrides ...]
Each task dir is a plan episode (the model forks on its own); the run records each episode's model calls, prompt /
completion / cached tokens and forks. A bundle dataset (tests/tasks.json in each task dir) as
--tasks-dir adds per episode n, the per-task vector, solved fraction, branches per task and each
branch's credit {task, solved, landed: its task solved in the final tree}. ended_with is how the main
thread ended: done, budget, step cap, context, no tool call or request failed. Every row has its wall time. Calls
and tokens are the episode tree's, the no-fork control's own under control (the override rl.plan_control=true;
branch_control when nested); wall_s is the episode's minus the control's own run time (grading it is still included).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import time
from pathlib import Path

from octopus.rl.config import add_run_args, eval_config, plan_row, workflow_kwargs
from octopus.runtime.core import tree_branches


def usage(trial: str) -> dict:
    """Model calls, prompt / completion / prefix-cached tokens, forks and branches (at any depth) of one episode's tree,
    and under control the control's own calls, tokens and wall_s (rl.plan_control)."""
    reqs = [json.loads(x) for x in (Path(trial) / "agent/requests.jsonl").read_text().splitlines()]
    tree_f = Path(trial) / "agent/tree.json"
    tree = json.loads(tree_f.read_text()) if tree_f.exists() else {}
    forks = tree.get("forks") or []

    def totals(rows: list[dict]) -> dict:
        return {"calls": len(rows), **{k: sum(r.get(k) or 0 for r in rows)
                                       for k in ("prompt_tokens", "completion_tokens", "cached_tokens")}}
    nodes = list(tree_branches(forks))
    return {**totals([r for r in reqs if r.get("thread") != "control"]),
            "forks": len(forks) + sum(len(b.get("forks") or []) for _, _, b in nodes), "branches": len(nodes),
            "control": {**totals([r for r in reqs if r.get("thread") == "control"]),
                        "wall_s": (tree.get("control") or {}).get("wall_s")}}


def bundle_row(rec: dict) -> dict:
    """n, solved fraction, branches per task ("-" = none named) and branch credit of one bundle episode."""
    return {"n": len(rec["tasks"]), "fraction": rec["y"],
            "branches_per_task": dict(collections.Counter(b["task"] or "-" for b in rec["branches"])),
            "branch_credit": [{"task": b["task"], "solved": b["passed"], "landed": b["landed"]} for b in rec["branches"]]}


def summary(rows: list[dict]) -> str:
    rows = list({(r["task_dir"], r["sample"]): r for r in rows}.values())  # an episode's last row: a retry supersedes
    out = []
    plans = [r for r in rows if r.get("kind") == "plan" and "y" in r]
    if plans:
        n = len(plans); avg = lambda k: sum(r.get(k) or 0 for r in plans) / n  # noqa: E731
        out.append(f"plan: n={n} passes {sum(r['y'] == 1 for r in plans) / n:.1%} | forked {sum((r.get('forks') or 0) > 0 for r in plans) / n:.0%}"
                   f" | per episode: calls {avg('calls'):.0f}, prompt {avg('prompt_tokens'):.0f} (cached {avg('cached_tokens'):.0f}),"
                   f" completion {avg('completion_tokens'):.0f}, branches {avg('branches'):.1f}")
        if any("n" in r for r in plans):
            out.append(f"bundle: solved fraction {avg('fraction'):.3f} | wall {avg('wall_s'):.0f} s")
        by_kind = collections.defaultdict(list)  # rl.plan_control: y - the no-fork control's y from the same fork call
        for r in plans:
            if r.get("control"):
                by_kind[r["control"]["kind"]].append(r["y"] - r["control"]["y"])
        for kind, d in sorted(by_kind.items()):
            se = math.sqrt(sum((x - sum(d) / len(d)) ** 2 for x in d) / max(len(d) - 1, 1) / len(d))
            out.append(f"fork - control ({kind}): n={len(d)} {sum(d) / len(d):+.4f}±{se:.4f}")
    out.append(f"infra failures (not scored): {sum('error' in r for r in rows)}")
    return "\n".join(out)


async def main_async(args, config) -> None:
    from areal.utils.dynamic_import import import_from_string

    from octopus.rl import proxy

    tasks = [str(d) for d in sorted(Path(args.tasks_dir).iterdir()) if d.is_dir()][args.offset:][: args.limit]
    servers = args.servers.split(",")
    agent = import_from_string(config.workflow)(**{**workflow_kwargs(config), "jobs_dir": args.jobs_dir,
                                                   "parallel": args.parallel * len(servers)})
    out = Path(args.out)
    done = {(r["task_dir"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())
            if "error" not in r} if out.exists() else set()  # an infra failure is run again
    todo = [(t, s) for s in range(args.k) for t in tasks if (t, s) not in done]
    print(f"{len(tasks)} tasks x {args.k}: {len(todo)} episodes to run ({len(done)} already in {out})", flush=True)
    gate = asyncio.Semaphore(args.parallel * len(servers))

    async def one(task: str, sample: int, url: str) -> None:
        async with gate:
            start = time.time()
            try:
                rec = await proxy.episode(agent, plan_row(config, task), url, f"{Path(task).name}:{sample}")
                if rec.get("tasks") is not None:
                    rec.update(bundle_row(rec))  # before usage(), whose "branches" is a count
                use = usage(rec["trial"])
                own = use.pop("control")  # onto the control's row (main's or a branch's), if one ran
                (rec.get("control") or rec.get("branch_control") or {}).update(own)
                rec.update(use)
            except Exception as e:  # infra failure (harness exception, failed request): not a policy outcome
                rec = {"error": f"{type(e).__name__}: {e}"[:300]}
        ran = (rec.get("control") or rec.get("branch_control") or {}).get("wall_s") or 0
        rec["wall_s"] = round(time.time() - start - ran)  # minus the control's run (under it)
        with out.open("a") as f:
            f.write(json.dumps({"task_dir": task, "kind": "plan", "sample": sample, "lora": args.lora, **rec}) + "\n")

    async with proxy.serve(config, servers, args.lora) as url:
        await asyncio.gather(*(one(t, s, url) for t, s in todo))
    print(summary([json.loads(x) for x in out.read_text().splitlines()]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servers", required=True, help="host:port,... of the SGLang servers serving --lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=12, help="concurrent episodes per server")
    ap.add_argument("--jobs-dir", default="/home/user/harbor-udocker/rl/jobs-dev-eval")
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET tasks (to split a run across machines)")
    ap.add_argument("--limit", type=int, help="only the first LIMIT tasks (smoke runs)")
    ap.add_argument("--tasks-dir", required=True, help="the task dirs to run (kind plan)")
    add_run_args(ap)
    args = ap.parse_intermixed_args()
    asyncio.run(main_async(args, eval_config(args)))


if __name__ == "__main__":
    main()
