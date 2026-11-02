# Analysis A (our verification agent, 10-25 ~15:05 EDT; unreviewed). Scripts/data: scratchpad/branch-weight/ (extract.py, stepstats.py, analyze2.py, analyze3.py; v7.jsonl, v6d.jsonl, v7steps.jsonl, v6dsteps.jsonl)

Conclusion: the mechanism is real, the 58x figure is exaggerated, the causal claim fails. Over steps 1-11, token weighting gives 10+ branch and depth>=3 episodes NEGATIVE mass (more negative than episode weighting), so it suppresses rather than drives the runaway. Branch growth comes mainly from main's fork rate doubling (same trend in v6d with the same loss) and from branch threads forking when depth is unlimited. 1/N_branches not recommended.

(a) Loss: A_branch = A_main (group.py:91; v7 branch_credit outcome). Per micro-batch: sum of valid-token loss / batch tokens x local/global tokens = global token mean per step; ppo_n_minibatches 1 (cursor-18f4fe7/areal functional.py:571, trainer/ppo/actor.py:844, fsdp_engine.py:2282, train_engine.py:82). Inherited prefixes masked by own_start (workflow.py:103); readctl/control untrained. v7 steps 1-5: main+branch completion tokens from requests.jsonl match main.log n_valid_tokens within <=3 tokens. dp_balance: one episode per unit, normalisation unchanged. => an episode's gradient weight is proportional to A x its trained tokens: unbiased REINFORCE.

(b) Trained tokens per episode (v7): steps 1-11 nb=0 878/484 (mean/median), nb=1 1393/762, nb=10 12856/10065 -> 10-branch / 1-branch = 9.2x / 13.2x. Per step range: 1-3 7.1/7.5x; 4-6 12.2/17.2x; 7-9 7.7/12.5x; 10-11 8.2/12.6x. 58x only for the extreme pair (step 11 max 40,841 vs nb=1 median 762).

(c) Signed gradient mass, mixed groups, LOO A, R=2y-1: share of sum(A*w) over total weight, token-weighted / episode-weighted; brN = branch tokens / N_br:

| class | 1-3 | 4-6 | 7-9 | 10-11 | 1-11 | 1-11 brN | in-group acc |
|---|---|---|---|---|---|---|---|
| nb=0 | -.053/-.049 | -.053/-.056 | +.006/-.038 | -.013/-.016 | -.019/-.042 | -.045 | 0.52 |
| nb=1 | -.004/+.004 | +.001/+.010 | -.009/+.021 | -.003/-.015 | -.004/+.007 | -.010 | 0.66 |
| 2-3 | +.031/+.029 | +.059/+.031 | -.004/-.002 | +.008/+.023 | +.016/+.020 | +.023 | 0.63 |
| 4-9 | +.045/+.024 | -.002/+.014 | +.009/+.025 | +.010/+.030 | +.013/+.023 | +.019 | 0.61 |
| 10+ | -.047/-.007 | +.038/.000 | -.019/-.006 | -.117/-.023 | -.053/-.008 | -.025 | 0.40 |
| d1 | .000/+.016 | +.028/+.021 | -.003/+.016 | -.003/-.004 | +.002/+.014 | .000 | 0.62 |
| d2 | +.022/+.029 | +.001/+.016 | +.002/+.019 | -.036/-.010 | -.011/+.015 | +.012 | 0.64 |
| d>=3 | +.002/+.004 | +.068/+.019 | -.021/+.004 | -.062/+.030 | -.020/+.013 | -.004 | 0.50 |

- Only steps 4-6 support B: token weighting amplified deep/10+ positive signal; in-branch fork push +0.80 token vs +0.13 episode.
- Steps 7-11 and overall: negative for 10+ and d>=3, more negative than episode weighting, yet branch count kept rising.
- "d>=3 accuracy 0.25" is difficulty confounding: 53% of d>=3 episodes are in all-wrong groups (A=0). In mixed groups d>=3 0.47 vs shallow 0.55; d>=3 mean in-group A +0.080.
- v6d (depth 1): same loss; fork rate 0.31 -> 0.96, branches/ep 0.68 -> 1.61, almost no 10-branch. Steps 1-55 mass token/episode: nb=0 -.018/-.018, nb=1 -.019/+.003, 2-3 +.006/+.010, 4-9 +.013/+.004.

(d) Decomposition (exact identity): branches/ep = fork rate x direct branches per main fork / (1 - children per branch thread)

| step | fork rate | direct/fork | children/branch | branches/ep |
|---|---|---|---|---|
| 1 | 0.34 | 2.26 | 0.48 | 1.46 |
| 6 | 0.44 | 1.88 | 0.54 | 1.77 |
| 9 | 0.64 | 1.91 | 0.64 | 3.38 |
| 11 | 0.74 | 2.13 | 0.64 | 4.37 |

1. main fork rate x2.2 is the main driver: in mixed groups forked episodes 0.61-0.66 vs unforked 0.52 -> main fork push positive in 9 of 11 steps; normal episode-level GRPO signal, also in v6d.
2. children per branch 0.48 -> 0.64 contributes x1.45; already 0.48 at step 1 -> base policy forks in branches when allowed; in-branch fork push ~0 overall (+0.08); guess: shared-policy transfer from main (unverified).
3. minor: <=2 versions off-policy lag; token-weighted sum of A mostly negative per step (-0.114 at steps 10-11) while entropy 0.36 -> 0.54; possibly related (unverified).

(e) Do not use 1/N_branches or 1/T_i: the current form is unbiased; per-episode normalisation reintroduces the length bias Dr. GRPO identified. On our data 1/N_br halves the negative mass on 10+ (-0.053 -> -0.025) and raises main's fork push from +0.24 to +0.57 (x1e-4 per unit weight), possibly making branch count grow faster. Only benefit: weakens steps 4-6 in-branch positive push. Also changes the loss (not comparable to v6d/v7). To avoid runaway at unlimited depth: limit the branching process (depth 2 or lower OCTOPUS_MAX_BRANCHES) and monitor the 3 factors per step.

Caveats: steps assigned as every 16 mixed groups in completion order; steps 1-5 match logs exactly; 6-11 after a restart, token ratio to logs 0.61-1.18, possibly +-1 step off. 16 groups per step, noisy. Fork push = REINFORCE approximation with pooled fork probability p. Read credit ignored.
