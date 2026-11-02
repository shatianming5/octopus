# Read-only on the run: python3 - JOBS_DIR N_SAMPLE JUDGE_URL SEED. For a random sample of MOUNTED read_thread calls made
# by a branch R of another thread S (search), the judge (temperature 0) labels R's next action (its next assistant reply:
# text + tool-call arguments) OWN / OTHER / UNCLEAR against R's own task and S's task; control: R's last action before the
# read, same rubric. Hijack = OTHER. Prints one JSON line per read, then a summary line.
import json, sys, glob, re, random, urllib.request, concurrent.futures as cf
J, N, URL, SEED = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
def own(fk, b): return (b.get("messages") or [])[fk.get("prefix_len") or 0:]
def walk(forks, parent):
    for j, fk in enumerate(forks or []):
        for b in fk.get("branches") or []:
            name = b.get("workdir") or (parent + "/" if parent != "main" else "") + f"fork{j}/{b['branch_id']}"
            yield name, fk, b
            yield from walk(b.get("forks"), name)
def action(m):
    calls = [f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})" for c in m.get("tool_calls") or []]
    return ((m.get("content") or "").strip()[:1200] + ("\nTool calls: " + "; ".join(calls) if calls else ""))[:1600]
def is_read(m): return any((c.get("function") or {}).get("name") == "read_thread" for c in m.get("tool_calls") or [])
recs = []
for q in glob.glob(J + "/*/*"):
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
            nxt = next((x for x in msgs[i+1:] if x.get("role") == "assistant"), None)
            pre = next((x for x in reversed(msgs[:i]) if x.get("role") == "assistant" and not is_read(x)), None)
            if nxt is None or pre is None: continue
            recs.append({"ep": q.split("/")[-2], "R": rname, "S": mm.group(1), "R_task": tasks[rname], "S_task": tasks[mm.group(1)],
                         "post": action(nxt), "pre": action(pre)})
random.Random(SEED).shuffle(recs); total = len(recs); recs = recs[:N]
RUBRIC = """You audit a research agent that splits a question into sub-tasks, one per thread. Thread R just read (attended to) another thread S's work.
R's own task: {rt}
S's task (the thread R read): {st}
R's action to judge:
---
{act}
---
Is this action working on R's OWN task (it may use facts or evidence found by S, as long as it serves R's task), or has R switched to S's task / another thread's task instead of its own (OTHER)? Reply with exactly one word: OWN, OTHER, or UNCLEAR."""
def judge(rt, st, act):
    body = {"model": "judge", "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": RUBRIC.format(rt=rt, st=st, act=act)}]}
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "User-Agent": "octopus-hijack"})
    for _ in range(3):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip().upper()
            return next((w for w in ("OTHER", "OWN", "UNCLEAR") if w in out), "UNPARSED")
        except Exception: pass
    return "ERROR"
def run(r):
    r["post_label"] = judge(r["R_task"], r["S_task"], r["post"]); r["pre_label"] = judge(r["R_task"], r["S_task"], r["pre"]); return r
with cf.ThreadPoolExecutor(4) as ex: done = list(ex.map(run, recs))
for r in done: print(json.dumps({k: r[k] for k in ("ep", "R", "S", "post_label", "pre_label")} | {"R_task": r["R_task"][:150], "S_task": r["S_task"][:150], "post": r["post"][:300]}, ensure_ascii=False))
def rate(k): n = sum(r[k] in ("OWN", "OTHER") for r in done); return sum(r[k] == "OTHER" for r in done), n
(a, n1), (b, n2) = rate("post_label"), rate("pre_label")
print(json.dumps({"summary": True, "reads_available": total, "sampled": len(done), "post_other": a, "post_n": n1, "pre_other": b, "pre_n": n2,
                  "post_rate": round(a / max(1, n1), 4), "pre_rate": round(b / max(1, n2), 4),
                  "labels_post": {l: sum(r["post_label"] == l for r in done) for l in ("OWN", "OTHER", "UNCLEAR", "UNPARSED", "ERROR")},
                  "labels_pre": {l: sum(r["pre_label"] == l for r in done) for l in ("OWN", "OTHER", "UNCLEAR", "UNPARSED", "ERROR")}}))
