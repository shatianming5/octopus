"""unlabelled.py NAME: exit 1 (and print the count) while a task of datasets/NAME has no judge verdict; a fix that is not
a plain text edit counts as done (reduce keeps it as it is), and so do tasks in labels/NAME.giveup (BUNDLE:K lines: the
gateway refused them on every try; finalize.py leaves their bundles out)"""
import json, os, sys
W = "/mnt/disk6/user/coupled-s1a"
ds = "/home/user/harbor-udocker/datasets/" + sys.argv[1]
lab = f"{W}/labels/{sys.argv[1]}.jsonl"
done = {(r["bundle"], r["k"]) for r in map(json.loads, open(lab)) if "verdict" in r or r.get("units", 0) is None} \
    if os.path.exists(lab) else set()
give = f"{W}/labels/{sys.argv[1]}.giveup"
done |= {(b, int(k)) for b, k in (l.rsplit(":", 1) for l in open(give).read().split())} if os.path.exists(give) else set()
todo = [(r["bundle"], t["k"]) for r in map(json.loads, open(f"{ds}.manifest.jsonl")) for t in r["tasks"] if (r["bundle"], t["k"]) not in done]
print(sys.argv[1], "unlabelled", len(todo), flush=True)
sys.exit(1 if todo else 0)
