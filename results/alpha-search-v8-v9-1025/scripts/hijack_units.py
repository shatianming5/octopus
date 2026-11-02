# Read-only (python3 hijack_units.py LABELS.jsonl JOBS_DIR > units.tsv): joins hijack_judge.py's labels back to the mounted
# reads of the trial (same extraction as hijack_judge.py) and prints one TSV row per labelled read: ep, R, S, relation of S
# to R (sib = same fork, desc = S under R, anc, other = cousin), pre_label, post_label, post_is_read (R's next reply calls
# read_thread again), pre_is_fork (R's last action before the read was a fork call).
import glob, json, re, sys
LAB, J = sys.argv[1], sys.argv[2]


def own(fk, b): return (b.get("messages") or [])[fk.get("prefix_len") or 0:]


def walk(forks, parent):
    for j, fk in enumerate(forks or []):
        for b in fk.get("branches") or []:
            name = b.get("workdir") or (parent + "/" if parent != "main" else "") + f"fork{j}/{b['branch_id']}"
            yield name, fk, b
            yield from walk(b.get("forks"), name)


def names(m): return [(c.get("function") or {}).get("name") for c in m.get("tool_calls") or []]


def action(m):
    calls = [f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})" for c in m.get("tool_calls") or []]
    return ((m.get("content") or "").strip()[:1200] + ("\nTool calls: " + "; ".join(calls) if calls else ""))[:1600]


def rel(R, S):
    if S.startswith(R + "/"): return "desc"
    if R.startswith(S + "/"): return "anc"
    return "sib" if R.rsplit("/", 1)[0] == S.rsplit("/", 1)[0] else "other"


labs = [json.loads(x) for x in open(LAB) if x.strip()]
labs = [x for x in labs if not x.get("summary")]
eps = {x["ep"] for x in labs}
recs = []
for q in glob.glob(J + "/*/*"):
    if q.split("/")[-2] not in eps: continue
    try: tr = json.load(open(q + "/agent/tree.json"))
    except Exception: continue
    tasks, threads = {}, []
    for name, fk, b in walk(tr.get("forks"), "main"):
        tasks[name] = b.get("task") or ""; threads.append((name, own(fk, b)))
    for rname, msgs in threads:
        for i, m in enumerate(msgs):
            if m.get("role") != "tool" or "Mounted " not in str(m.get("content") or ""): continue
            mm = re.search(r"Mounted (\S+?)'s steps", str(m["content"]))
            if not mm or mm.group(1) not in tasks or mm.group(1) == rname: continue
            nxt = next((x for x in msgs[i + 1:] if x.get("role") == "assistant"), None)
            pre = next((x for x in reversed(msgs[:i]) if x.get("role") == "assistant" and "read_thread" not in names(x)), None)
            if nxt is None or pre is None: continue
            recs.append({"ep": q.split("/")[-2], "R": rname, "S": mm.group(1), "post": action(nxt)[:300],
                         "post_is_read": "read_thread" in names(nxt), "pre_is_fork": "fork" in names(pre)})
used = set()
for L in labs:
    k = next((k for k, r in enumerate(recs) if k not in used and (r["ep"], r["R"], r["S"], r["post"]) == (L["ep"], L["R"], L["S"], L["post"])), None)
    if k is None:
        print("NOJOIN", L["ep"], L["R"], L["S"], file=sys.stderr); continue
    used.add(k); r = recs[k]
    print("\t".join(map(str, (r["ep"], r["R"], r["S"], rel(r["R"], r["S"]), L["pre_label"], L["post_label"], int(r["post_is_read"]), int(r["pre_is_fork"])))))
