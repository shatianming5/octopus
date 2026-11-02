# Read-only (python3 - MAIN_LOG): per "Train step" block, the AReaL stats that show the loss weighting:
# n_groups (dispatch units), group_loss_weight avg/max/min (loss-mask tokens per unit), train_infer n_valid_tokens,
# task_reward avg, advantage +/- fractions, correct/incorrect n_seqs, grad norm, entropy, importance weight avg/max, kl_k1.
import json, re, sys
KEYS = ["ppo_actor/n_groups", "ppo_actor/group_loss_weight/avg", "ppo_actor/group_loss_weight/max",
        "ppo_actor/group_loss_weight/min", "ppo_actor/train_infer/n_valid_tokens", "ppo_actor/task_reward/avg",
        "ppo_actor/advantage_positive_fraction", "ppo_actor/advantage_negative_fraction", "ppo_actor/correct_n_seqs",
        "ppo_actor/incorrect_n_seqs", "ppo_actor/update/grad_norm", "ppo_actor/update/entropy/avg",
        "ppo_actor/update/importance_weight/avg", "ppo_actor/update/importance_weight/max", "ppo_actor/train_infer/kl_k1"]
cur = None
out = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \d+/\d+ Step (\d+)/\d+ Train step", line)
    if m:
        cur = {"t": m.group(1), "step": int(m.group(2))}
        out.append(cur)
        continue
    if cur is None:
        continue
    for k in KEYS:
        for mm in re.finditer(re.escape(k) + r"\s*│\s*([-+0-9.e]+)", line):
            cur[k.split("ppo_actor/")[1]] = float(mm.group(1))
for c in out:
    print(json.dumps(c))
