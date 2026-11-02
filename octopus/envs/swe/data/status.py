"""One-line status of a Harbor job. usage: status.py <job name>"""
import collections, glob, json, os, sys, time
from datetime import datetime

N = "/home/user/harbor-udocker"
JOB = sys.argv[1]
J = f"{N}/jobs/{JOB}"

def ts(x):
    return datetime.fromisoformat(x.replace("Z", "")).timestamp()

try:
    st = json.load(open(J + "/result.json"))["stats"]
    s = f"c/e/r/p={st['n_completed_trials']}/{st['n_errored_trials']}/{st['n_running_trials']}/{st['n_pending_trials']}"
except Exception as e:
    s = f"no result.json yet ({type(e).__name__})"
rw, ex, envs = collections.Counter(), collections.Counter(), []
for f in glob.glob(J + "/*/result.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    r = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if r is not None:
        rw[r] += 1
    e = (d.get("exception_info") or {}).get("exception_type")
    if e:
        ex[e] += 1
    es = d.get("environment_setup") or {}
    if es.get("started_at") and es.get("finished_at"):
        envs.append((es["started_at"], round(ts(es["finished_at"]) - ts(es["started_at"]))))
now, codes = time.time(), collections.Counter()
for f in glob.glob(J + "/*/agent/requests.jsonl"):
    if now - os.path.getmtime(f) > 600:
        continue
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if now - r.get("t", 0) <= 600:
            codes[r.get("http_code", "err")] += 1
pid_file = f"{N}/jobs/{JOB}.pid"
alive = os.path.exists(pid_file) and os.system(f"kill -0 {open(pid_file).read().strip()} 2>/dev/null") == 0
trials = len([x for x in os.listdir(N + "/containers") if not x.startswith(".")])
print(f"{time.strftime('%H:%M')} [{JOB}] harbor={'up' if alive else 'EXITED'} {s} "
      f"reward={dict(rw)} exc={dict(ex)} api_10min={dict(codes)} "
      f"env_setup_last={[d for _, d in sorted(envs)[-4:]]} containers={trials} "
      f"load={open('/proc/loadavg').read().split()[0]}")
