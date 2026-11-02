"""step_stats.py MAIN_LOG [KEY ...]: per AReaL "Train step N done" the stats table's values of KEYs (default: the
train/infer mismatch and the step's time split), one row per step. The mismatch comes from the loss's stats, logged with
or without compute_logp (docs/trainer-memory.md "Proximal logprobs"): behave_approx_kl/avg ≈ -kl_k1 (over the tokens
rejection sampling keeps)."""
import re, sys
keys = sys.argv[2:] or ["ppo_actor/update/behave_approx_kl/avg", "ppo_actor/update/logp_abs_diff/avg",
                        "ppo_actor/update/logp_abs_diff/max", "ppo_actor/update/behave_imp_weight/avg",
                        "timeperf/rollout", "timeperf/train_step"]
step, rows = None, {}
for line in open(sys.argv[1], errors="replace"):
    if m := re.search(r"Train step (\d+)/\d+ done", line):
        step = int(m[1]); rows[step] = {}
    elif step is not None and "│" in line:
        for k, v in re.findall(r"│ ([\w/.-]+)\s+│\s+(-?[\d.]+e[-+]\d+)", line):
            rows[step][k] = float(v)
print("step " + " ".join(k.split("/", 1)[1] for k in keys))
for s, r in rows.items():
    print(s, " ".join(f"{r.get(k, float('nan')):.3g}" for k in keys))
