"""BrowseComp-Plus grading: judged accuracy, the official retrieval recall, and the recall of the docs the agent opened.

Input: a JSONL run, one line per query: {"query_id": str, "response": str (final answer text), "retrieved_docids":
[str] (every docid a search returned), "opened_docids": [str]}, or the results JSONL of octopus/eval/search_dev_eval.py,
whose rows name their trial dirs (trial_run converts them; infra-error rows are left out).
The judge is BrowseComp-Plus's (octopus/envs/search/reward.judge: the official grader prompt and parser, Qwen3.5-27B at
temperature 0 by default, as CLM; BCP's is Qwen3-32B at 0.7; enable_thinking false) on any OpenAI-compatible
/chat/completions endpoint; an empty response is incorrect without a judge call, as in the official scripts_evaluation/evaluate_run.py.
Per query, averaged over the run: recall = |retrieved & evidence| / |evidence| (the official Recall),
open_recall = |opened & evidence| / |evidence|, open_gold_recall = |opened & gold| / |gold|.

  python -m octopus.eval.bcp run.jsonl --dataset-dir <root>/datasets/Tevatron__browsecomp-plus \
      --base-url http://127.0.0.1:30000/v1 --model Qwen/Qwen3.5-27B [--out judged.jsonl]

--base-url / --model default to OCTOPUS_JUDGE_URL / OCTOPUS_JUDGE_MODEL, the key is OCTOPUS_JUDGE_KEY (reward.py's
names). --out keeps only query_id, verdict, confidence and recalls: no decrypted question or answer is ever written.
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from octopus.runtime.core import tree_branches
from octopus.envs.search.data import browsecomp_plus
from octopus.envs.search.reward import judge, rewound

HIT = re.compile(r"^\[\d+\] docid=(.*?) \|", re.M)  # a result line of octopus.envs.search.agent.search_text


def trial_run(trial):
    """A octopus.envs.search.agent trial dir as a run line: main's answer and the docids the fork tree (main and every
    branch, not the control) got from search and opened, each in first-seen order, also by calls a rewind removed."""
    trial = Path(trial)
    tree = json.loads((trial / "agent/tree.json").read_text())
    branches = [b for _, _, b in tree_branches(tree.get("forks") or [])]
    messages = tree["main"] + rewound(tree.get("rewinds", [])) + [m for b in branches
                                                          for m in b["messages"] + rewound(b.get("rewinds", []))]
    searches, retrieved, opened = set(), {}, {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") in searches:
            retrieved.update(dict.fromkeys(HIT.findall(m.get("content") or "")))
        for call in m.get("tool_calls") or []:
            if call["function"]["name"] == "search":
                searches.add(call["id"])
            if call["function"]["name"] == "open":
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except ValueError:  # the policy's malformed arguments: the call opened nothing
                    continue
                opened[str(args.get("docid"))] = None
    answers = json.loads((trial / "verifier/answers.json").read_text())
    return {"query_id": json.loads((trial / "result.json").read_text())["task_name"],
            "response": answers["main"]["answer"] or "", "retrieved_docids": list(retrieved),
            "opened_docids": list(opened)}


def recall(docids, relevant):
    return len(set(docids) & set(relevant)) / len(relevant) if relevant else None


def recalls(run_line, truth_row):
    """The official recall (searched docids) and ours (opened docids) of one query."""
    opened = run_line.get("opened_docids", [])
    return {"recall": recall(run_line.get("retrieved_docids", []), truth_row["evidence_docids"]),
            "open_recall": recall(opened, truth_row["evidence_docids"]),
            "open_gold_recall": recall(opened, truth_row["gold_docids"])}


def evaluate(run, truth, base_url, model, api_key=None, workers=16):
    """run: [{"query_id", "response", "retrieved_docids", "opened_docids"}]; truth: browsecomp_plus() ->
    (summary, per-query rows)."""
    def one(r):
        t = truth[str(r["query_id"])]
        correct, conf = False, None
        if r["response"]:
            try:
                correct, conf = judge(t["query"], r["response"], t["answer"], base_url, model, api_key)
            except Exception:  # endpoint failure, a reply without choices: a judge failure row (no verdict: likewise)
                correct = None
        return {"query_id": str(r["query_id"]), "correct": correct, "confidence": conf, **recalls(r, t)}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, run))
    mean = lambda k: (lambda v: sum(v) / len(v) if v else None)([r[k] for r in rows if r[k] is not None])
    return {"n": len(rows), "accuracy": sum(r["correct"] is True for r in rows) / max(len(rows), 1),
            "judge_failures": sum(r["correct"] is None for r in rows),
            **{k: mean(k) for k in ("recall", "open_recall", "open_gold_recall")}}, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dataset-dir", default=os.path.expanduser("~/octo-env/datasets/Tevatron__browsecomp-plus"))
    ap.add_argument("--base-url", default=os.environ.get("OCTOPUS_JUDGE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--model", default=os.environ.get("OCTOPUS_JUDGE_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", help="per-query verdicts and recalls (no decrypted text)")
    a = ap.parse_args()
    os.environ.pop("OCTOPUS_JUDGE_CACHE", None)  # its keys hold the decrypted question and gold answer
    run = [trial_run(r["trial"]) if "trial" in r else r for r in (json.loads(l) for l in open(a.run) if l.strip())
           if "error" not in r]
    summary, rows = evaluate(run, browsecomp_plus(a.dataset_dir),
                             a.base_url, a.model, os.environ.get("OCTOPUS_JUDGE_KEY"), a.workers)
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(summary, indent=1))
