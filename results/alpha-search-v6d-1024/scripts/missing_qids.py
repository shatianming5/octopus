# Read-only (python3 - JOBS_DIR TRAIN_QIDS COVERED_CACHE, the search_finished_qids.py cache): the train qids not yet "finished": do they have trial dirs at all
# (dispatched), how many, and what their results say (exception_info / answers.json present)
import os, sys, json, collections, time
J, Q, C = sys.argv[1:4]
want = [l.strip() for l in open(Q) if l.strip()]
cov = set(l.split()[1] for l in open(C) if l.split()[1:] and l.split()[1] != "-") if os.path.exists(C) else set()  # search_finished_qids.py cache: "dir qid|-"
miss = [q for q in want if q not in cov]
dirs = collections.defaultdict(list)
for d in os.listdir(J):
    if os.path.isdir(os.path.join(J, d)) and d.startswith("search-"):
        dirs[d.split(":")[0].split("-")[1]].append(d)
print("train qids", len(want), "covered", len(cov & set(want)), "missing", len(miss))
stat = collections.Counter(); ex = collections.Counter(); newest = []
for q in miss:
    ds = dirs.get(q, [])
    stat["dispatched" if ds else "never dispatched"] += 1
    for d in ds:
        p = os.path.join(J, d)
        for t in os.listdir(p):
            r = os.path.join(p, t, "result.json"); a = os.path.join(p, t, "verifier/answers.json")
            if os.path.exists(a): stat["has answers.json"] += 1
            if os.path.exists(r):
                e = (json.load(open(r)).get("exception_info") or {})
                ex[str(e.get("exception_type") or e.get("exception_message") or "none")[:60]] += 1
            newest.append(os.path.getmtime(p))
print(dict(stat)); print("exceptions:", dict(ex.most_common(6)))
if newest: print("newest dir of a missing qid:", time.strftime("%H:%M", time.localtime(max(newest))))
print("sample missing:", miss[:8])
