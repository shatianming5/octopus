# Current accepted data is cooperbench-v2. This file stays on v1 for the recorded cooperbench-v1 run;
# do not point it at v2.
# build tasks dir (symlinks) for the six Train repos from the accepted train-valid bundle dir + manifest
import json, os, sys, pathlib
SIX = {"pallets_jinja", "samuelcolvin_dirty_equals", "openai_tiktoken", "pillow", "llama_index", "huggingface_datasets"}
B = pathlib.Path("/home/user/harbor-udocker")
src = B / "datasets/cooperbench-v1-train-valid"
out = B / "s1b/tasks-train6"
rows = [json.loads(l) for l in open(B / "datasets/cooperbench-v1-train-valid.manifest.jsonl") if l.strip()]
want = {r["bundle"]: r["repo"] for r in rows if r.get("repo") in SIX}
out.mkdir(parents=True, exist_ok=True)
for old in out.iterdir():
    if old.is_symlink():
        old.unlink()
have = {p.name for p in src.iterdir() if p.is_dir()}
missing = sorted(b for b in want if b not in have)
for b in sorted(want):
    if b in have:
        (out / b).symlink_to(src / b)
import collections
print("manifest rows for six repos:", len(want), "dirs linked:", len(list(out.iterdir())), "missing dirs:", len(missing), missing[:5])
print(collections.Counter(want[b] for b in want if b in have))
