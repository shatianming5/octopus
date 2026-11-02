"""search_finished_qids.py JOBS_DIR CACHE [QIDS]: train qids with at least one finished episode (tree.json main_end not a
failed request), cached in CACHE (one line per scanned job dir: name and qid or -); prints the count."""
import json, os, sys
jobs, cache = sys.argv[1:3]
train = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else None  # QIDS file: count train qids only
seen, fin = set(), set()
if os.path.exists(cache):
    for l in open(cache):
        n, q = l.split(); seen.add(n); fin.add(q) if q != "-" else None
new = []
for d in os.listdir(jobs):
    p = os.path.join(jobs, d)
    if d in seen or not d.startswith("search-") or not os.path.isdir(p):
        continue
    trees = [os.path.join(p, t, "agent/tree.json") for t in os.listdir(p)]
    trees = [t for t in trees if os.path.exists(t)]
    if not trees:
        continue  # still running (or no trial): scanned again next time
    try:
        m = json.load(open(trees[0])).get("main_end")
    except ValueError:
        continue
    end = m.get("ended_with") if isinstance(m, dict) else m
    q = d.split("-")[1] if end and end != "request failed" else "-"
    new.append(f"{d} {q}\n"); fin.add(q) if q != "-" else None
open(cache, "a").writelines(new)
print(len((fin - {"-"}) & train if train is not None else fin - {"-"}))
