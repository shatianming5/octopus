"""gate_dist.py JOBS_DIR (PYTHONPATH=CODE, the trainer venv): rl.branch_credit=gate (search's rule) on every finished search episode of a
jobs dir: branches by how they ended, ok (search_branch_ok) or gated, own search/open or none."""
import collections, glob, json, sys
from pathlib import Path
from octopus.envs.search.reward import search_outcome
from octopus.rl.group import search_branch_ok
c, n = collections.Counter(), 0
for t in glob.glob(sys.argv[1] + "/search-*/*/agent/tree.json"):
    trial = Path(t).parent.parent
    if not (trial / "verifier/answers.json").exists():
        continue
    rec = search_outcome(trial, [json.loads(l) for l in open(trial / "agent/requests.jsonl")])
    n += 1
    for b in rec["branches"]:
        c[(b["ended_with"], "ok" if search_branch_ok(b) else "gated", "searched" if b["searched"] else "no-search")] += 1
print(n, "episodes;", sum(c.values()), "branches"); [print(k, v) for k, v in sorted(c.items(), key=lambda x: -x[1])]
