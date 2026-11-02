"""search_audit.py JOBS_DIR [MAIN_LOG]: step-3 audit numbers from the trial's episodes (tree.json) and logs; run on Alpha with python3.
Episodes: main_end distribution, forks/branches (at any depth), rewind calls/steps/note chars (main + branches), done-with-evidence, request failures."""
import json, os, sys, collections, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the checkout's octopus/, run as plain python3
from octopus.runtime.core import tree_branches
jobs = sys.argv[1]; C = collections.Counter(); R = collections.Counter(); n = 0
rew_eps = 0; calls = []
for d in sorted(os.listdir(jobs)):
    p = os.path.join(jobs, d)
    if not d.startswith("search-") or not os.path.isdir(p): continue
    for t in os.listdir(p):
        f = os.path.join(p, t, "agent/tree.json")
        if not os.path.exists(f): continue
        try: tr = json.load(open(f))
        except ValueError: C["unreadable tree"] += 1; continue
        n += 1; m = tr.get("main_end"); C["main_end=" + str(m.get("ended_with") if isinstance(m, dict) else m)] += 1
        fk = tr.get("forks") or []; C["forked"] += bool(fk)
        rw = list(tr.get("rewinds", [])); nb = 0
        for _, _, b in tree_branches(fk):
            nb += 1; rw += b.get("rewinds", []) or []
            C["branch_end=" + str(b.get("ended_with"))] += 1
        C["branches"] += nb; C["episodes_with_rewind"] += bool(rw)
        R["rewind_calls"] += len(rw); R["rewind_note_chars"] += sum(len(r.get("note", "")) for r in rw)
        R["rewound_assistant_steps"] += sum(mm.get("role") == "assistant" for r in rw for mm in r.get("messages", []))
print("episodes", n); [print(" ", k, v) for k, v in sorted(C.items())]; [print(" ", k, v) for k, v in sorted(R.items())]
if len(sys.argv) > 2:
    L = open(sys.argv[2], errors="replace").read()
    for pat in ["no requests.jsonl row", "is in no leaf and no branch extends it", "requests in no leaf", "has no graded branch", "refused", "Traceback"]:
        print(" log", pat, len(re.findall(pat, L)))
