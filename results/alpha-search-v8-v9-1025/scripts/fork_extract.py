# Read-only (python3 - JOBS_DIR > TRIAL.jsonl): one compact line per Search RL episode for the fork diagnosis (from the
# 10-25 independent review's extractor): main ending, forks, done answer, graded answer, last main message is the join
# note ("branches finished:"; the text reply that followed is not stored in tree.json), depth-1 branch endings.
import glob, json, os, sys
J = sys.argv[1]


def calls(m):
    return [((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")) for c in m.get("tool_calls") or []]


for tp in glob.glob(J + "/*/*/agent/tree.json"):
    ep = os.path.dirname(os.path.dirname(tp))
    r = {"ep": os.path.relpath(ep, J), "gdir": os.path.basename(os.path.dirname(ep)), "mt": os.path.getmtime(tp)}
    try:
        t = json.load(open(tp))
    except Exception as e:
        r["terr"] = str(e); print(json.dumps(r)); continue
    main = t.get("main") or []
    r["main_end"] = t.get("main_end"); r["nforks"] = len(t.get("forks") or [])
    r["ndone"] = sum(n == "done" for m in main if m.get("role") == "assistant" for n, _ in calls(m))
    r["last_is_join"] = bool(main) and main[-1].get("role") == "user" and str(main[-1].get("content") or "").startswith("branches finished:")
    r["ends"] = [str(b.get("ended_with")) for fk in t.get("forks") or [] for b in fk.get("branches") or []]
    try:
        a = json.load(open(ep + "/verifier/answers.json")).get("main") or {}
        r["ans"] = a.get("answer"); r["y"] = a.get("correct")
    except Exception:
        r["ans"] = r["y"] = None; r["noans"] = True
    print(json.dumps(r, ensure_ascii=False))
