"""Per-step training stats of an AReaL trial from its main.log: task reward, entropy, grad norm, clip ratio,
train/infer logprob gap, sequences.

usage: python -m octopus.tools.step_stats FILEROOT/logs/<user>/rq1-rl/<trial>/main.log
"""
import re, sys
keys = ["ppo_actor/task_reward/avg", "ppo_actor/update/entropy/avg", "ppo_actor/update/grad_norm", "ppo_actor/update/clip_ratio/avg",
        "ppo_actor/update/logp_abs_diff/avg", "ppo_actor/n_seqs"]
text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1], errors="ignore").read())
blocks = re.split(r"Train step (\d+)/\d+ done", text)  # each step's stats table follows its "done" line
print("step | " + " | ".join(k.split("/", 1)[1] if "/" in k else k for k in keys))
seen = {}
for i in range(1, len(blocks), 2):
    step, body = int(blocks[i]), blocks[i + 1][:60000]
    vals = []
    for k in keys:
        m = re.findall(re.escape(k.split("ppo_actor/")[-1]) + r"\s*│\s*([-0-9.e+]+)", body)
        vals.append(f"{float(m[-1]):.4g}" if m else "-")
    seen[step] = vals
for step in sorted(seen):
    print(f"{step} | " + " | ".join(seen[step]))
