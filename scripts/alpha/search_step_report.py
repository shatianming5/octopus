"""search_step_report.py MAIN_LOG JUDGE_LOG SERVER_LOG...: per step timeperf (s), reward, and mean/max running + queued requests per rollout server and
the judge in each step's window (window = between the previous step's table and this one's). kl_k1 is -update/behave_approx_kl/avg (rollout minus
proximal logp, over the tokens rejection sampling keeps): ≈ train_infer/kl_k1, which needs compute_logp (docs/trainer-memory.md
"Proximal logprobs"); in older logs too."""
import re, sys, datetime as dt, collections
main, judge, servers = sys.argv[1], sys.argv[2], sys.argv[3:]
txt = open(main, errors="replace").read()
ts = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
ends = [dt.datetime.strptime(m, "%Y%m%d-%H:%M:%S") for m in re.findall(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: .*Train step \d+/\d+ done", txt)]
tabs = re.split(r"timeperf/rollout", txt)[1:]
rows = []
for t in tabs:
    d = {k: float(v) for k, v in re.findall(r"timeperf/(\w+)\s*│\s*(-?[\d.]+e[+-]\d+)", "timeperf/rollout" + t)}
    d["rollout"] = float(re.match(r"\s*│\s*(-?[\d.]+e[+-]\d+)", t).group(1))
    for k, p in (("reward", r"ppo_actor/task_reward/avg\s*│\s*(-?[\d.]+e[+-]\d+)"), ("seq_len", r"ppo_actor/seq_len/avg\s*│\s*(-?[\d.]+e[+-]\d+)"),
                 ("kl_k1", r"ppo_actor/update/behave_approx_kl/avg\s*│\s*(-?[\d.]+e[+-]\d+)")):
        m = re.search(p, t); d[k] = float(m.group(1)) * (-1 if k == "kl_k1" else 1) if m else float("nan")
    rows.append(d)
def stats(path):
    out = []
    for l in open(path, errors="replace"):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*#running-req: (\d+), #queue-req: (\d+)", l)
        if m: out.append((ts(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out
S = {p.rsplit("/", 1)[-1]: stats(p) for p in servers}; J = stats(judge)
print("step | wall_min | rollout adv train_step upd_w ckpt (s) | reward seq_len kl_k1 | run/queue avg(max) per server | judge run/queue")
prev = None
for i, d in enumerate(rows):
    e = ends[i] if i < len(ends) else None
    wall = (e - ends[i - 1]).total_seconds() / 60 if e and i else float("nan")
    def win(st):
        w = [x for x in st if (prev is None or x[0] > prev) and (e is None or x[0] <= e)]
        if not w: return "-"
        return "%.0f/%.0f(%d/%d)" % (sum(x[1] for x in w) / len(w), sum(x[2] for x in w) / len(w), max(x[1] for x in w), max(x[2] for x in w))
    print(i + 1, "|", "%.1f" % wall, "|", " ".join("%.0f" % d.get(k, float("nan")) for k in ("rollout", "compute_advantage", "train_step", "update_weights", "checkpoint_for_recover")),
          "|", "%.3f %.0f %.2e" % (d["reward"], d["seq_len"], d["kl_k1"]), "|", " ".join(win(v) for v in S.values()), "| judge", win(J))
    prev = e
