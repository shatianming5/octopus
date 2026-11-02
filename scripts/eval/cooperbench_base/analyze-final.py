# Current accepted data is cooperbench-v2. This file stays on v1 for the recorded cooperbench-v1 run;
# do not point it at v2.
# final per-repo summary of the base train6 run (merged output): scored bundles, mean reward (solved fraction),
# both-solved share (fraction == 1), >=1 solved, all-zero flag, infra failures, ended by budget
import json, collections, pathlib, sys
B = pathlib.Path("/home/user/harbor-udocker")
SIX = ["pallets_jinja", "samuelcolvin_dirty_equals", "openai_tiktoken", "pillow", "llama_index", "huggingface_datasets"]
out = pathlib.Path(sys.argv[1])
last = {}
for line in out.read_text().splitlines():
    r = json.loads(line)
    last[(r["task_dir"], r["sample"])] = r  # an episode's last row counts
manifest = [json.loads(l) for l in open(B / "datasets/cooperbench-v1-train-valid.manifest.jsonl") if l.strip()]
planned = collections.Counter(r["repo"] for r in manifest if r["repo"] in SIX)
stats = {repo: collections.defaultdict(float) for repo in SIX}
for (td, _), r in last.items():
    repo = pathlib.Path(td).name.split(".task")[0]
    s = stats[repo]
    if r.get("error") or "y" not in r:
        s["infra"] += 1
        continue
    s["scored"] += 1
    s["reward"] += r["y"]
    s["any"] += r["y"] > 0
    s["both"] += r["y"] == 1
    s["budget"] += r.get("ended_with") == "budget"
    s["nonzero"] += r["y"] > 0
tot = collections.defaultdict(float)
for repo in SIX:
    s = stats[repo]; n = int(s["scored"])
    line = f"{repo}: bundles {planned[repo]} | scored {n} | mean reward {s['reward']/n:.3f} | both-solved {int(s['both'])}/{n} = {s['both']/n:.3f} | >=1 solved {int(s['any'])}/{n} = {s['any']/n:.3f} | all-zero {'YES' if s['any']==0 else 'no'} | infra {int(s['infra'])} | ended by budget {int(s['budget'])}" if n else f"{repo}: bundles {planned[repo]} | scored 0 | infra {int(s['infra'])}"
    print(line)
    for k in ("scored", "reward", "both", "any", "infra", "budget"):
        tot[k] += s[k]
n = int(tot["scored"])
print(f"TOTAL: bundles {sum(planned.values())} | scored {n} | mean reward {tot['reward']/n:.3f} | both-solved {int(tot['both'])}/{n} = {tot['both']/n:.3f} | >=1 solved {int(tot['any'])}/{n} = {tot['any']/n:.3f} | infra {int(tot['infra'])} | ended by budget {int(tot['budget'])}")
