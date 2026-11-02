"""search_coverage.py JOBS_DIR MAIN_LOG QIDS: distinct train qids rolled out so far (episode job dirs search-<qid>-...,
started before each "Train step N done"), per step and in all, against the train split (QIDS file)."""
import os, re, sys
from datetime import datetime
jobs, log, qids = sys.argv[1:4]
train = set(open(qids).read().split())
done = [(datetime.strptime(m[1], "%Y%m%d-%H:%M:%S").timestamp(), int(m[2])) for m in
        re.finditer(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: .*Train step (\d+)/", open(log).read())]
eps = sorted((os.stat(os.path.join(jobs, d)).st_mtime, d.split("-")[1]) for d in os.listdir(jobs)
             if d.startswith("search-") and os.path.isdir(os.path.join(jobs, d)))
seen = set(); i = 0
for t, step in done:
    while i < len(eps) and eps[i][0] < t:
        seen.add(eps[i][1]); i += 1
    print(f"step {step}: {len(seen & train)} of {len(train)} train qids rolled out")
seen.update(q for _, q in eps[i:])
print(f"now: {len(seen & train)} of {len(train)} ({len(seen - train)} outside the train split)")
