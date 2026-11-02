# alpha-search-v7-1025: Search v7 on Cluster Alpha (unlimited fork depth, stopped at step 11)

Search RL trial `search-v7-1025` on Cluster Alpha (H200), from adapter v0, outcome credit R = 2y-1, batch 16 x 8.
`OCTOPUS_FORK_DEPTH=0` means unlimited depth. Settings are otherwise those of v6d (`../alpha-search-v6d-1024/`). It was
stopped on 10-25 for v8. Record written 10-25; times are EDT.

ROOT = `/projects/cluster/octopus/octopus`. Numbers were recomputed on 10-25 from Alpha's raw files
with `scripts/`; the outputs are in `out/`. This record has not been independently reviewed. The branch-weight
analyses it summarises are marked as such below.

## Run record

The run used code-18f4fe7 (#305). It includes #304 (fork depth >= 2: window guard, general read offer, nested counts) but
not #306 or #309. Sources: `ROOT/search/pilot/hold-*/queue/{20,01-rollout}-search-v7-1025.sh.*` and the trainer log
`ROOT/search/areal-exp/logs/user/rq1-rl/search-v7-1025/main.log`.

| segment | trainer | rollout + judge | EPISODES_PER_SERVER | time (10-25) | steps |
|---|---|---|---|---|---|
| 1 | 148016 (alphagpu22, 4 GPUs) | 147315 (alphagpu19, 3 servers, 10.0.4.25) | 8 | 05:20-13:45 | 1-11 |
| 2 | 148016 | 147853 (3 servers, 10.0.4.26) | 4 | 13:58-14:58 | none |

- Segment 1 was stopped at the step-11 boundary (13:43) to move the rollout. That freed 147315 for the depth-0 Val.
- Segment 2 loaded the step-11 recover at 14:02. It finished no step before it was stopped at 14:58 for v8.
- Step ends from `main.log`: 05:46, 05:57, 06:13, 06:27, 06:39, 08:16, 08:55, 10:55, 11:35, 12:24 and 13:44. Steps 6
  and 8 took 96 and 119 minutes while the rollout slowed (see KV below).
- The jobs dir is `ROOT/search/rl/jobs-search-v7-1025`, with 5,283 graded episodes.

## What drifted

Source: `python3 - JOBS_DIR MAIN_LOG 1 < scripts/v7_depth.py` (`out/v7_depth.txt`). Episodes are bucketed by the step
during which they finished (`answers.json` mtime), with every finished episode included. "no tool" is the share whose
main ended with "no tool call". "cap ref" counts branch-cap refusals ("would pass the episode's limit") per episode.

| step | eps | fork | br/ep | max depth >= 3 | cap ref / ep | request timeout | no tool | acc | acc of no-tool endings |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 763 | 0.23 | 0.99 | 0.07 | 0.20 | 0.000 | 0.06 | 0.51 | 0.51 |
| 2 | 259 | 0.26 | 1.11 | 0.07 | 0.18 | 0.000 | 0.08 | 0.38 | 0.45 |
| 3 | 387 | 0.24 | 1.12 | 0.09 | 0.35 | 0.000 | 0.04 | 0.42 | 0.53 |
| 4 | 338 | 0.30 | 1.29 | 0.09 | 0.26 | 0.000 | 0.09 | 0.47 | 0.62 |
| 5 | 124 | 0.40 | 1.88 | 0.11 | 0.56 | 0.000 | 0.10 | 0.38 | 0.50 |
| 6 | 540 | 0.41 | 2.21 | 0.16 | 0.66 | 0.063 | 0.16 | 0.50 | 0.72 |
| 7 | 422 | 0.53 | 2.76 | 0.22 | 1.06 | 0.021 | 0.26 | 0.57 | 0.68 |
| 8 | 741 | 0.57 | 3.17 | 0.26 | 1.33 | 0.077 | 0.36 | 0.44 | 0.49 |
| 9 | 374 | 0.60 | 3.16 | 0.25 | 1.28 | 0.019 | 0.53 | 0.46 | 0.51 |
| 10 | 357 | 0.75 | 4.93 | 0.42 | 2.36 | 0.036 | 0.55 | 0.36 | 0.35 |
| 11 | 597 | 0.78 | 4.72 | 0.40 | 2.11 | 0.027 | 0.60 | 0.43 | 0.45 |
| after 11 | 381 | 0.79 | 4.64 | 0.41 | 2.68 | 0.016 | 0.75 | 0.48 | 0.51 |

The "after 11" row holds the episodes that finished after step 11, segment 2 included. They were not trained.

The rollout KV figure came from `python3 - '2026-10-25 05:19:00' LOGS < scripts/kv_usage.py` (`out/v7_kv.txt`). LOGS
are the segment-1 SGLang logs `ROOT/search/rl-state-search/serve/sglang-alphagpu19-g{1,2,3}.log`. Segment 2's server logs
were overwritten by v8's servers.

| server | full-token KV mean / median / p90 | hourly mean KV | #queue-req mean / max |
|---|---|---|---|
| g1 | 0.80 / 0.92 / 0.98 | 0.57-0.96 | 6.2 / 56 |
| g2 | 0.82 / 0.91 / 0.98 | 0.55-0.94 | 7.7 / 72 |
| g3 | 0.79 / 0.89 / 0.98 | 0.59-0.91 | 5.7 / 45 |

KV reaches 0.98 at the p90 and 0.91-0.96 (per server) in the peak hours; queues reach 72.

## Root cause of the "no tool call" drift

- On code-18f4fe7 a main that ended on a plain reply without a tool call still handed an answer. It was its last text,
  `CursorRuntime._final_text` from #178, and that answer was graded.
- In v7 those endings were graded correct 35-72% of the time (last column above), so ending without `done` was rewarded.
  Their share grew from 6% to 60%.
- The `\boxed{}` suffix of about 29% of the OpenResearcher questions also contributes. Before #309, exact match did not
  unwrap `\boxed{}`. This contribution comes from the analyses of this session and is not recomputed here.
- Fixes:
  - #306: a thread's result comes only from its closing tool. A main without `done` hands no answer and is graded
    incorrect, in training and in eval.
  - #309: `reward.normalize` unwraps `\boxed{}` and `\text{}`.
- All v6d/v7 evaluations predate both fixes.

## Step-9 importance weight

At step 9, `ppo_actor/update/importance_weight/avg` is 6.9103e+08 and `/max` is 3.9893e+14. Every other step's average is
1.0001-1.0008 (`scripts/stepstats.py`, `out/v7steps.jsonl`).

- The max divided by step 9's 577,300 trained tokens (`train_infer/n_valid_tokens`) is 6.91e+08. That equals the
  average, so one token carries it.
- The step's grad_norm (0.039) and kl_k1 (1.8e-3) are in line with steps 7-11.
- "A recompute glitch on one token, harmless" is the reading of this session. It agrees with these numbers, but which
  token it was was not identified.

## Branch weight: does the full A_main per branch drive the branch growth?

Under the token-mean loss every branch takes the episode's full A_main, so an episode with many branches carries more
gradient. The two analyses are `analysis-A.md` and `analysis-B.md` in this directory (A's "steps 6-11 after a restart" is wrong: main.log and the queue logs show no restart in steps 1-11).

- **Analysis B** was pasted by the user and made with another tool. It proposed A_branch = A_main / N_branches, PR #310.
  - On v7 steps 1-6, within mixed groups, episodes of depth >= 3 were 9.6% of the episodes but carried 62.6% of the
    positive token-weighted gradient.
  - They also carried 88% of the positive branch-fork push.
  - In v6d a branch's fork is refused at depth 1, which gives negative feedback. In v7 it is allowed, which gives
    positive feedback.
  - Its strongest point (Part III): with #306 applied (main without `done` -> y = 0), the depth 3-4 episodes of steps
    1-6 still give +97.9 branch-fork push and +49,082 positive token-weighted gradient.
- **Analysis A** was made by this session's agent and has not been reviewed.
  - The loss is unbiased REINFORCE. An episode's weight is proportional to A times its trained tokens. Main and branch
    completion tokens match `n_valid_tokens` to within 3 tokens on steps 1-5.
  - Token weighting amplified the positive signal of deep and 10+ branch episodes in steps 4-6.
  - In steps 7-11 it gave them negative mass, more negative than episode weighting would.
  - Branch growth comes mainly from the main fork rate doubling, which is the same trend as v6d with the same loss. It
    also comes from branch threads forking when depth is unlimited.
  - A_main / N would reintroduce a length bias and could speed up main's forking.
- **Consistency.** On 10-25 the user found the two analyses consistent. Positive amplification in steps 4-6 and a
  negative brake in steps 7-11 are both present.
- **Decision (user, 10-25).** Keep the full A_main, because it is unbiased. The loss is unchanged; v8 differs from v6d in depth and in
  #303-#309 (incl. #306/#309). PR
  #310 (A_main / N) was closed unmerged.

`scripts/analyze2.py` on a fresh extract reproduces analysis A's tables (b) and (c) (`out/v7_analyze2.txt`). For
example, for 10+ branch episodes over steps 1-11, the token-weighted mass is -0.053 and the episode-weighted mass is
-0.008. For depth >= 3 the values are -0.020 and +0.013.

### Branching-process decomposition

The decomposition is an exact identity:

branches/ep = main fork rate x direct branches per forking main / (1 - mu)

Here mu = children per branch thread, which equals deeper branches / all branches. `scripts/analyze3.py` was re-run on a
fresh extract (`out/v7_analyze3.txt`). It uses trained episodes only: each step takes the next 16 mixed groups in
completion order, so its numbers differ from the all-episode table above.

| step | main fork rate | direct branches / forking main | mu | branches/ep |
|---|---|---|---|---|
| 1 | 0.34 | 2.26 | 0.48 | 1.46 |
| 2 | 0.24 | 2.35 | 0.52 | 1.18 |
| 3 | 0.34 | 1.77 | 0.57 | 1.39 |
| 4 | 0.23 | 1.83 | 0.52 | 0.90 |
| 5 | 0.28 | 1.61 | 0.50 | 0.91 |
| 6 | 0.44 | 1.88 | 0.54 | 1.77 |
| 7 | 0.48 | 2.11 | 0.51 | 2.07 |
| 8 | 0.60 | 1.79 | 0.55 | 2.39 |
| 9 | 0.64 | 1.91 | 0.64 | 3.38 |
| 10 | 0.64 | 1.95 | 0.57 | 2.90 |
| 11 | 0.74 | 2.13 | 0.64 | 4.37 |

Steps 1-5 are placed exactly: their FIFO-placed tokens match `n_valid_tokens`. For steps 6-11 the ratio of FIFO-placed
tokens to `n_valid_tokens` is 1.182 / 1.149 / 0.951 / 1.046 / 0.611 / 0.936, so the decomposition for those steps (step 11
included) is approximate.

## v8 (search-v8-1025)

- Launched 10-25 from about 15:02 at code-f3af572 (main), from v0, with `OCTOPUS_FORK_DEPTH=0` and the full A_main.
- Rollout and judge run on 147853 with EPISODES_PER_SERVER 8; the trainer is on 148016.
- Its settings are in `ROOT/search/pilot/search-v8-1025/run`.
- v8 differs from v6d in depth and in the changes since 2edfb2d (#303-#309, including #306 and #309). It differs from v7
  in #306-#309.
- Plan: watch mu per step with `analyze3.py` and stop at mu >= 0.8.
- Results will be added to this directory later.

## Reproduce

```
R=/projects/cluster/octopus/octopus; J=$R/search/rl/jobs-search-v7-1025
M=$R/search/areal-exp/logs/user/rq1-rl/search-v7-1025/main.log
ssh cluster-login "python3 - $J $M 1" < scripts/v7_depth.py
ssh cluster-login "python3 - '2026-10-25 05:19:00' $R/search/rl-state-search/serve/sglang-alphagpu19-g{1,2,3}.log" < scripts/kv_usage.py
ssh cluster-login "python3 - $M" < scripts/stepstats.py > out/v7steps.jsonl          # v6d: same with search-v6d-1024
ssh cluster-login "python3 - $J $M" < scripts/extract.py > v7.jsonl                 # 1.5 MB (v6d: 7 MB), not committed
python3 scripts/analyze2.py v7.jsonl 1-3,4-6,7-9,10-11
python3 scripts/analyze3.py v7.jsonl 1-1,2-2,3-3,4-4,5-5,6-6,7-7,8-8,9-9,10-10,11-11
```

Everything here only reads from Alpha.
