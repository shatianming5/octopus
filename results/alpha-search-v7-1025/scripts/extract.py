# Read-only (python3 - JOBS_DIR MAIN_LOG): one JSON line per episode with what the trainer weights it by:
# trained completion tokens per thread kind (requests.jsonl; readctl/* and control rows are exported with loss mask 0),
# branch count / max depth (tree.json), fork and turn counts per thread kind, y (verifier/answers.json), group key,
# answers.json mtime. Last line: {"steps": [[t, step], ...]} from main.log "Train step" lines.
import json, os, re, sys
from datetime import datetime
J, M = sys.argv[1], sys.argv[2]


def own(fork, b):
    return (b.get("messages") or [])[fork.get("prefix_len") or 0:] + [m for r in b.get("rewinds") or [] for m in r.get("messages") or []]


def walk(forks, d):
    for fk in forks or []:
        for b in fk.get("branches") or []:
            yield d, fk, b
            yield from walk(b.get("forks"), d + 1)


def calls(msgs):
    turns = forks = 0
    for m in msgs:
        if m.get("role") == "assistant":
            turns += 1
            forks += sum((c.get("function") or {}).get("name") == "fork" for c in m.get("tool_calls") or [])
    return turns, forks


for g in os.listdir(J):
    p = os.path.join(J, g)
    if not os.path.isdir(p):
        continue
    for t in os.listdir(p):
        q = os.path.join(p, t)
        try:
            tr = json.load(open(os.path.join(q, "agent/tree.json")))
            ans = json.load(open(os.path.join(q, "verifier/answers.json")))
            reqs = [json.loads(x) for x in open(os.path.join(q, "agent/requests.jsonl")) if x.strip()]
        except Exception:
            continue
        T = {"main": 0, "br": 0, "readctl": 0, "control": 0}
        Td = {}
        err = 0
        for r in reqs:
            th = r.get("thread") or "main"
            ct = int(r.get("completion_tokens") or 0)
            err += r.get("http_code") != 200
            if th.startswith("readctl/"):
                T["readctl"] += ct
            elif th == "control":
                T["control"] += ct
            elif th == "main":
                T["main"] += ct
            else:
                T["br"] += ct
                d = th.count("fork")
                Td[d] = Td.get(d, 0) + ct
        bs = list(walk(tr.get("forks"), 1))
        mt, mf = calls(list(tr.get("main") or []) + [m for r in tr.get("rewinds") or [] for m in r.get("messages") or []])
        bt = bf = 0
        for _, fk, b in bs:
            a, f = calls(own(fk, b))
            bt += a; bf += f
        me = tr.get("main_end")
        print(json.dumps({"g": g, "y": bool(ans["main"].get("correct")), "mt": os.path.getmtime(os.path.join(q, "verifier/answers.json")),
                          "nb": len(bs), "md": max([d for d, _, _ in bs], default=0), "T": T, "Td": Td, "err": err,
                          "turns": [mt, bt], "forkcalls": [mf, bf], "nforks_main": len(tr.get("forks") or []),
                          "end": me.get("ended_with") if isinstance(me, dict) else me,
                          "nd": {d: sum(1 for x, _, _ in bs if x == d) for d in {x for x, _, _ in bs}},
                          "bf": [sum(1 for x, _, b in bs if x == d and b.get("forks")) for d in range(1, 1 + max([x for x, _, _ in bs], default=0))]}))
steps = []
for line in open(M, errors="replace"):
    m = re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \d+/\d+ Step (\d+)/\d+ Train step", line)
    if m:
        steps.append([datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp(), int(m.group(2))])
print(json.dumps({"steps": steps}))
