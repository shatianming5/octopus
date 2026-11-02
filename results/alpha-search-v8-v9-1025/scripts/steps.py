# Read-only (python3 - TRAIN_LOG JOBS_DIR): one JSON line per training step of a Search RL trial.
# Step ends and AReaL stats come from the trainer log (ANSI stripped; the stats block follows "Train step N/.. done").
# Episodes are bucketed by agent/tree.json mtime into the first step that ended at or after it; "step" = last + 1 holds
# the episodes finished after the last logged step (not yet trained). Per bucket: episodes, fork rate, branches/ep,
# mu (children per branch thread), share with >= 10 branches (the cap), share reaching depth >= 3, main reminders
# (#313: episodes reminded, of which main then ended with done), branch reminders (and of those branches ended done),
# share of mains ending "no tool call", rewind shares (any rewind; main; branch), accuracy, episodes with an HTTP 400 in
# agent/requests.jsonl.
import collections, glob, json, os, re, sys
from datetime import datetime
L, J = sys.argv[1], sys.argv[2]
KEYS = ["task_reward/avg", "train_infer/n_valid_tokens", "update/grad_norm", "update/importance_weight/max", "train_infer/kl_k1"]
steps, cur = {}, None
for line in open(L, errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: .*Train step (\d+)/\d+ done", line)
    if m:
        cur = steps.setdefault(int(m.group(2)), {})
        cur["end"] = m.group(1)
        continue
    if cur is not None:
        for k in KEYS:
            for mm in re.finditer(r"ppo_actor/" + re.escape(k) + r"\s*│\s*([-+0-9.e]+)", line):
                cur[k] = float(mm.group(1))
S = sorted(steps)
T = [datetime.strptime(steps[s]["end"], "%Y%m%d-%H:%M:%S").timestamp() for s in S]


def walk(forks, d):
    for fk in forks or []:
        for b in fk.get("branches") or []:
            yield d, b
            yield from walk(b.get("forks"), d + 1)


B = collections.defaultdict(collections.Counter)
for f in glob.glob(J + "/*/*/agent/tree.json"):
    q = os.path.dirname(os.path.dirname(f))
    try:
        tr = json.load(open(f))
        y = bool((json.load(open(q + "/verifier/answers.json")).get("main") or {}).get("correct"))
    except Exception:
        continue
    mt = os.path.getmtime(f)
    k = next((s for s, t in zip(S, T) if mt <= t), (S[-1] if S else 0) + 1)
    c = B[k]
    bs = list(walk(tr.get("forks"), 1))
    me = tr.get("main_end"); me = me.get("ended_with") if isinstance(me, dict) else me
    c["eps"] += 1; c["y"] += y; c["forked"] += bool(bs); c["branches"] += len(bs)
    c["kids"] += sum(len(fk.get("branches") or []) for _, b in bs for fk in b.get("forks") or [])
    c["cap10"] += len(bs) >= 10; c["d3"] += max([d for d, _ in bs], default=0) >= 3
    c["notool"] += me == "no tool call"
    if tr.get("reminded"):
        c["rem_eps"] += 1; c["rem_done"] += me == "done"
    rb = [b for _, b in bs if b.get("reminded")]
    c["brem"] += len(rb); c["brem_done"] += sum(str(b.get("ended_with")) == "done" for b in rb)
    rm, rbr = bool(tr.get("rewinds")), any(b.get("rewinds") for _, b in bs)
    c["rw_any"] += rm or rbr; c["rw_main"] += rm; c["rw_branch"] += rbr
    try:
        c["http400"] += any(json.loads(x).get("http_code") == 400 for x in open(q + "/agent/requests.jsonl") if x.strip())
    except Exception:
        pass
prev = None
for k in sorted(B):
    c, n = B[k], B[k]["eps"]
    st = steps.get(k, {})
    t = T[S.index(k)] if k in steps else None
    row = {"step": k, "end": st.get("end"), "minutes": round((t - prev) / 60, 1) if t and prev else None,
           "eps": n, "fork": c["forked"] / n, "br_ep": c["branches"] / n, "mu": c["kids"] / c["branches"] if c["branches"] else None,
           "cap10": c["cap10"] / n, "depth3": c["d3"] / n, "rem_eps": c["rem_eps"], "rem_then_done": c["rem_done"],
           "branch_rem": c["brem"], "branch_rem_then_done": c["brem_done"], "notool": c["notool"] / n,
           "rewind_any": c["rw_any"] / n, "rewind_main": c["rw_main"] / n, "rewind_branch": c["rw_branch"] / n,
           "acc": c["y"] / n, "http400_eps": c["http400"]}
    row.update({k2: v for k2, v in st.items() if k2 != "end"})
    prev = t or prev
    print(json.dumps(row))
