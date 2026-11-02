# per step window of the SWE full trial: episodes, infra errors (exception in result.json), episodes whose verifier or
# agent hit an exec timeout ("timed out after"), verifier wall (max/median s). usage: swe_health.py JOBS MAIN_LOG
import glob, json, os, re, statistics, sys
from datetime import datetime
jobs, log = sys.argv[1:3]
done = [datetime.strptime(m[1], "%Y%m%d-%H:%M:%S").timestamp() for m in re.finditer(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: .*Train step", open(log).read())]
w = {}
for res in glob.glob(f"{jobs}/*/*/result.json"):
    t = os.path.getmtime(res); s = next((i + 1 for i, d in enumerate(done) if t < d), len(done) + 1)
    d = os.path.dirname(res); r = json.load(open(res))
    a = w.setdefault(s, {"eps": 0, "infra": 0, "timeout": 0, "vwall": []})
    a["eps"] += 1; a["infra"] += bool(r.get("exception_info"))
    txt = ""
    for f in glob.glob(f"{d}/verifier/*.txt") + glob.glob(f"{d}/agent/*.log"):
        try: txt += open(f, errors="replace").read()[-20000:]
        except OSError: pass
    a["timeout"] += "timed out after" in txt
    vr = (r.get("verifier") or {})
    try:
        st, en = vr.get("started_at"), vr.get("finished_at")
        if st and en: a["vwall"].append((datetime.fromisoformat(en.replace("Z", "+00:00")) - datetime.fromisoformat(st.replace("Z", "+00:00"))).total_seconds())
    except Exception: pass
for s in sorted(w):
    a = w[s]; v = a["vwall"]
    print(f"s{s}{'*' if s > len(done) else ''} eps {a['eps']} infra {a['infra']} timeout-eps {a['timeout']}" + (f" verifier s max {max(v):.0f} med {statistics.median(v):.0f}" if v else ""))
