"""Per-step timing from AReaL main.log (timeperf/rollout = wait for rollouts, train_step, recompute_logp), every 4th step
and the mean over steps >= 5. usage: python step_timing.py TRIAL... (logs under /home/user/areal-exp on node-94)"""
import re, statistics as st
import sys
for t in sys.argv[1:]:
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(f"/home/user/areal-exp/logs/user/rq1-rl/{t}/main.log", errors="ignore").read())
    blocks = re.split(r"Train step (\d+)/\d+ done", txt)
    rows = []
    for i in range(1, len(blocks), 2):
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        v = dict((k, float(x)) for k, x in re.findall(r"timeperf/(\w+)\s*│\s*([-0-9.e+]+)", body))
        if v: rows.append((int(blocks[i]), v))
    print(t, "steps with timing", len(rows))
    for s, v in rows[::4]:
        print("  step %3d  rollout_wait %6.0f  train_step %6.0f  recompute %5.0f" % (s, v.get("rollout", 0), v.get("train_step", 0), v.get("recompute_logp", 0)))
    late = [v for s, v in rows if s >= 5]
    print("  mean step>=5: rollout_wait %.0f  train_step %.0f  recompute %.0f  steps waiting >60s on rollout: %d/%d" % (
        st.mean(v.get("rollout", 0) for v in late), st.mean(v.get("train_step", 0) for v in late),
        st.mean(v.get("recompute_logp", 0) for v in late), sum(v.get("rollout", 0) > 60 for v in late), len(late)))
