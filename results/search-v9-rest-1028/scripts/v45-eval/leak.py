# leak check: decrypt the pilot's BCP questions/answers in memory, search files modified since the pilot start for any
# 40-char slice of them; prints paths only. argv: rows.jsonl start.epoch roots...  (stdin mode "patterns": print slices)
import json, os, sys
from octopus.envs.search import data
rows, epoch = sys.argv[1], float(open(sys.argv[2]).read())
qids = {json.loads(l)["qid"] for l in open(rows)}
bcp = data.browsecomp_plus("/home/user/octo-env/datasets/Tevatron__browsecomp-plus")
pats = set()
for q in qids:
    r = bcp[q]
    for t in (r["query"], str(r["answer"])):
        t = " ".join(t.split())
        if len(t) >= 40:
            pats.update(t[i:i + 40].encode() for i in range(0, len(t) - 40, 60))
        elif len(t) >= 12:
            pats.add(t.encode())
if sys.argv[3] == "-":  # print slices for a remote grep (stdout pipe only)
    sys.stdout.write("\n".join(p.decode() for p in pats)); sys.exit()
hits, n = [], 0
for p in sys.stdin.read().split("\0"):  # files from find -newermt -print0 (argv[3] "files")
    if not p: continue
    try:
        n += 1
        b = open(p, "rb").read()
        if any(x in b for x in pats): hits.append(p)
    except OSError: pass
print(f"questions {len(qids)} slices {len(pats)} files scanned {n} hits {len(hits)}", *hits, sep="\n")
