# Read-only (python3 - JOBS_DIR TRAIN_QIDS): per qid, group dirs (search-<qid>-<idx>) and their creation time; how many
# qids have >1 group (re-drawn: the dataloader wrapped), groups created in the last 60 min that are repeats, and the qids
# of the train list never drawn
import os, sys, time, collections
J, Q = sys.argv[1:3]
want = set(l.strip() for l in open(Q) if l.strip())
g = collections.defaultdict(set); first = {}
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not (d.startswith("search-") and os.path.isdir(p)): continue
    key = d.split(":")[0]; qid = key.split("-")[1]
    g[qid].add(key); t = os.path.getctime(p); first[key] = min(first.get(key, t), t)
now = time.time()
recent = [k for k, t in first.items() if now - t < 3600]
rep = [k for k in recent if len(g[k.split("-")[1]]) > 1]
print("qids drawn", len(g), "of", len(want), "| never drawn", len(want - set(g)), "| qids drawn more than once", sum(len(v) > 1 for v in g.values()))
print("groups created in the last 60 min", len(recent), "of which re-draws of a qid drawn before", len(rep))
print("sample never drawn:", sorted(want - set(g))[:10])
