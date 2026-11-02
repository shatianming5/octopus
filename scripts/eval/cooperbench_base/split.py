# Current accepted data is cooperbench-v2. This file stays on v1 for the recorded cooperbench-v1 run;
# do not point it at v2.
# split the 311 Train bundles into shards by repo: 134 = llama_index + openai_tiktoken, 136 = pallets_jinja + pillow + samuelcolvin_dirty_equals
import json, pathlib
B = pathlib.Path("/home/user/harbor-udocker")
SHARD = {"134": ["llama_index", "openai_tiktoken"], "136": ["pallets_jinja", "pillow", "samuelcolvin_dirty_equals"]}
rows = [json.loads(l) for l in open(B / "datasets/cooperbench-v1-train-valid.manifest.jsonl") if l.strip()]
for host, repos in SHARD.items():
    names = sorted(r["bundle"] for r in rows if r["repo"] in repos)
    (B / f"s1b/shard{host}.txt").write_text("\n".join(names) + "\n")
    d = B / f"s1b/tasks-{host}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():
        if old.is_symlink(): old.unlink()
    for n in names:
        (d / n).symlink_to(B / "datasets/cooperbench-v1-train-valid" / n)
    print(host, len(names), "bundles")
