"""search_audit2.py JOBS_DIR: protocol markers over all episodes of a trial (run with PYTHONPATH=code dir): per thread's OWN messages (main; each branch after its
fork prefix_len; every rewound message too): reserve refusals, clipped results, fork-alone refusals, fork/read refusals, done calls with evidence
(main / branch), done answer empty, tool call counts by name, rewinds (note length, steps removed), request error rows, per-episode calls."""
import json, os, sys, collections, re
from octopus.envs.search.reward import tree_branches
VALID_ = {"search", "open", "done", "fork", "rewind", "read_thread"}
J = sys.argv[1]; C = collections.Counter(); calls = collections.Counter(); per_ep = []; notes = []; steps_removed = []
def scan(msgs, who):
    for m in msgs:
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}; n = f.get("name"); calls[who + ":" + str(n)] += 1
                if n == "done":
                    try: a = json.loads(f.get("arguments") or "{}")
                    except ValueError: C[who + " done args unparsable"] += 1; continue
                    C[who + " done"] += 1; C[who + " done with evidence"] += bool(str(a.get("evidence") or "").strip()); C[who + " done empty answer"] += not str(a.get("answer") or "").strip()
        elif m.get("role") == "tool":
            t = str(m.get("content"))
            for k, p in (("reserve refusals", "Refused: your context is nearly used up"), ("clipped results", "[clipped: context nearly used up]"),
                         ("close notes", "Your context is nearly used up. Close now"), ("fork-alone refusals", "refused: a reply whose fork started"),
                         ("fork refused", "fork refused"), ("read refused", "read_thread refused"), ("rewind refused", "rewind refused"), ("any 'refused'", "efused")):
                C[who + " " + k] += p in t
EP = collections.Counter(); n = 0
for d in os.listdir(J):
    p = os.path.join(J, d)
    if not d.startswith("search-") or not os.path.isdir(p): continue
    for t in os.listdir(p):
        f = os.path.join(p, t, "agent/tree.json")
        if not os.path.exists(f): continue
        try: tr = json.load(open(f))
        except ValueError: continue
        before = (C["main reserve refusals"] + C["branch reserve refusals"], C["main clipped results"] + C["branch clipped results"], sum(calls.values()), sum(v for k, v in calls.items() if k.split(":", 1)[1] not in VALID_))
        n += 1; scan(tr.get("main") or [], "main")
        for r in tr.get("rewinds") or []:
            scan(r.get("messages") or [], "main"); notes.append(len(r.get("note", ""))); steps_removed.append(sum(m.get("role") == "assistant" for m in r.get("messages", [])))
        for _, fork, b in tree_branches(tr.get("forks") or []):
            scan(b["messages"][fork["prefix_len"]:], "branch")
            for r in b.get("rewinds") or []:
                scan(r.get("messages") or [], "branch"); notes.append(len(r.get("note", ""))); steps_removed.append(sum(m.get("role") == "assistant" for m in r.get("messages", [])))
        after = (C["main reserve refusals"] + C["branch reserve refusals"], C["main clipped results"] + C["branch clipped results"], sum(calls.values()), sum(v for k, v in calls.items() if k.split(":", 1)[1] not in VALID_))
        EP["reserve"] += after[0] > before[0]; EP["clip"] += after[1] > before[1]; EP["invalid"] += after[3] > before[3]; EP["rewind"] += bool(tr.get("rewinds")) or any(b.get("rewinds") for _, _, b in tree_branches(tr.get("forks") or []))
        rq = os.path.join(p, t, "agent/requests.jsonl")
        if os.path.exists(rq):
            rows = [json.loads(l) for l in open(rq) if l.strip()]
            C["request rows"] += len(rows); C["request error rows"] += sum("error" in r for r in rows); per_ep.append(len(rows))
print("trees", n); [print(" ", k, v) for k, v in sorted(C.items())]
VALID = {"search", "open", "done", "fork", "rewind", "read_thread"}
for who in ("main", "branch"):
    ok = {k.split(":", 1)[1]: v for k, v in calls.items() if k.startswith(who + ":") and k.split(":", 1)[1] in VALID}
    bad = sum(v for k, v in calls.items() if k.startswith(who + ":") and k.split(":", 1)[1] not in VALID)
    print(" tool calls", who, dict(sorted(ok.items())), "INVALID-NAME calls", bad, "(%.2f%%)" % (100 * bad / max(1, bad + sum(ok.values()))))
print(" episodes with >=1 reserve refusal:", EP["reserve"], "of", n, "; with clipped result:", EP["clip"], "; with rewind:", EP["rewind"], "; with invalid tool name:", EP["invalid"])
if notes: print(" rewind notes: n=%d mean chars %.0f max %d; mean steps removed %.1f" % (len(notes), sum(notes) / len(notes), max(notes), sum(steps_removed) / len(steps_removed)))
if per_ep: print(" requests/episode: mean %.1f max %d" % (sum(per_ep) / len(per_ep), max(per_ep)))
