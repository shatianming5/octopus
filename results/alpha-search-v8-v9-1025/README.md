# alpha-search-v8-v9-1025: Search v8 and v9 on Cluster Alpha (night of 10-25/26)

Search RL trials `search-v8-1025` and `search-v9-1025` on Cluster Alpha (H200), both from adapter v0 with
`OCTOPUS_FORK_DEPTH=0` (unlimited depth), outcome credit R = 2y-1 and the full A_main on every branch. Record written
10-26 at about 02:20; times are EDT. v9 is still training, so its rows are a snapshot taken at about 02:05.

Later (10-31): v9 ran on Cluster to step 41; search-v9-rest continued it from step 41 on the lab cluster and Together
and ended at step 45 (10-31), so v45 is the final adapter ([../search-v9-rest-1028/README.md](../search-v9-rest-1028/README.md)).
Its BCP result is that README's "v45 evaluation (10-31)"; its Val width/depth sweep is in progress. "What's next" below
has no training ablation arm on record.

ROOT = `/projects/cluster/octopus/octopus`. Every number here was recomputed on 10-26 from Alpha's
raw files with `scripts/`; the outputs are in `out/`. Numbers that were not reproduced are marked as such. The
exception is the hijack labels: they come from the 10-25 judge pass and are not committed (see that section). This
record has not been independently reviewed. The v8 fork diagnosis and the hijack audit were reviewed independently on 10-25.

## v8 (`search-v8-1025`): done-only grading, no reminder

- The code is f3af572, which includes #306: a main that does not end with `done` hands no answer and scores 0. It has
  no close reminder. This makes v8 the done-only, no-reminder baseline.
- The trainer log has 17 step ends: 15:32 to 18:01 on 10-25. It was stopped after that.
- Step 10 took 54 minutes. This is the adapter-unload deadlock at 16:41 (the v10 adapter). After it, LORA_KEEP was
  raised to 40 and the trial was resumed (`ROOT/search/pilot/search-v8-1025/stopped-170708`). The 16:41 time and
  the value 40 come from the session's notes and were not re-read from logs.

Source: `python3 - TRAIN_LOG JOBS_DIR < scripts/steps.py` (`out/v8_steps.jsonl`). Episodes are bucketed by the step
during which their `tree.json` was last written. "min" is the wall time since the previous step end. "cap10" is the
share of episodes with at least 10 branches, which is the branch cap. mu = children per branch thread. "rewind" is the
share of episodes with any rewind.

| step | end | min | eps | fork | br/ep | mu | cap10 | depth>=3 | no tool | rewind | acc | task_reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15:32 | - | 657 | 0.24 | 0.98 | 0.48 | 0.032 | 0.064 | 0.058 | 0.23 | 0.46 | -0.020 |
| 2 | 15:43 | 11 | 277 | 0.28 | 1.29 | 0.53 | 0.047 | 0.090 | 0.040 | 0.25 | 0.44 | +0.195 |
| 3 | 15:57 | 14 | 433 | 0.20 | 0.88 | 0.50 | 0.035 | 0.055 | 0.025 | 0.30 | 0.37 | -0.231 |
| 4 | 16:05 | 8 | 356 | 0.14 | 0.62 | 0.52 | 0.025 | 0.045 | 0.034 | 0.19 | 0.46 | +0.078 |
| 5 | 16:10 | 6 | 280 | 0.09 | 0.33 | 0.46 | 0.011 | 0.021 | 0.018 | 0.23 | 0.44 | +0.013 |
| 6 | 16:20 | 9 | 481 | 0.06 | 0.17 | 0.38 | 0.004 | 0.010 | 0.015 | 0.31 | 0.46 | -0.088 |
| 7 | 16:24 | 4 | 347 | 0.03 | 0.11 | 0.59 | 0.003 | 0.012 | 0.020 | 0.18 | 0.52 | +0.021 |
| 8 | 16:27 | 3 | 190 | 0.01 | 0.02 | 0.50 | 0.000 | 0.000 | 0.011 | 0.08 | 0.62 | +0.272 |
| 9 | 16:34 | 6 | 409 | 0.02 | 0.04 | 0.29 | 0.000 | 0.005 | 0.005 | 0.17 | 0.31 | -0.020 |
| 10 | 17:28 | 54 | 1651 | 0.02 | 0.04 | 0.19 | 0.001 | 0.002 | 0.006 | 0.11 | 0.42 | -0.034 |
| 11 | 17:32 | 4 | 229 | 0.01 | 0.04 | 0.00 | 0.000 | 0.000 | 0.004 | 0.10 | 0.40 | -0.033 |
| 12 | 17:38 | 6 | 459 | 0.02 | 0.04 | 0.22 | 0.000 | 0.000 | 0.011 | 0.10 | 0.46 | +0.096 |
| 13 | 17:41 | 4 | 235 | 0.01 | 0.07 | 0.62 | 0.004 | 0.004 | 0.009 | 0.08 | 0.44 | +0.026 |
| 14 | 17:48 | 6 | 553 | 0.02 | 0.05 | 0.22 | 0.000 | 0.002 | 0.005 | 0.06 | 0.43 | -0.088 |
| 15 | 17:51 | 3 | 238 | 0.00 | 0.00 | - | 0.000 | 0.000 | 0.008 | 0.05 | 0.43 | +0.029 |
| 16 | 17:56 | 5 | 399 | 0.01 | 0.04 | 0.27 | 0.000 | 0.003 | 0.005 | 0.06 | 0.47 | -0.180 |
| 17 | 18:01 | 5 | 500 | 0.01 | 0.05 | 0.46 | 0.002 | 0.002 | 0.002 | 0.04 | 0.48 | -0.076 |

- The main fork rate fell from 0.24 at step 1 to 0.00-0.02 from step 8 on.
- The rewind share fell with it, from 0.18-0.31 in steps 1-7 to 0.04-0.06 in steps 14-17.
- With so few forks, the mu and cap10 values of steps 8-17 rest on a handful of branches.

### Why the fork rate collapsed (independently reviewed on 10-25)

Source: `python3 scripts/fork_gap.py DIR search-v6d-1024 search-v7-1025 search-v8-1025` on fresh extracts
(`out/fork_gap.txt`). It uses steps 1-3 of each trial, near v0. "Gap" is the within-group difference in accuracy,
forked minus unforked, over question groups that have both, ± 1 SE.

| | v6d | v7 | v8 |
|---|---|---|---|
| forked mains that ended in text ("no tool call") | 38/485 = 0.078 | 26/338 = 0.077 | 24/320 = 0.075 |
| of these, right after the join note | 23 | 14 | 12 |
| graded accuracy of those text endings | 0.47 | 0.65 | 0 (#306) |
| gap as graded | +0.051 ± 0.019 | +0.070 ± 0.019 | -0.049 ± 0.027 |
| gap, done-only grading | +0.008 ± 0.021 | +0.022 ± 0.026 | -0.049 ± 0.027 |
| gap, done-only, text endings left out | +0.028 ± 0.020 | +0.051 ± 0.021 | -0.014 ± 0.024 |

Pooled over the three trials (inverse-variance): done-only -0.003 ± 0.014; done-only with the text endings left out
+0.025 ± 0.012.

- About 7.5% of forked mains ended with a text reply instead of `done`. Half of these (12 of 24 in v8) came right
  after the join note (`last_is_join` in `fork_extract.py`).
- Under #306 each of those scores 0. In v7, which still graded the text, 65% of them were graded correct.
  - Whether v8's own text replies were mostly correct is not established. A 10-25 strict check (normalized gold
    answer as a substring of the reply) found the gold in 5 of v8's 31 forked text replies over steps 1-9, a lower
    bound. Its script and gold file are not committed, so this figure is not reproduced here.
- The 0 reached the fork tokens through A_branch = A_main. In v8 at steps 1-3, the forked mains' mean leave-one-out
  advantage is -0.044, and the text-ended ones account for all of it: the rest is -0.000.
- Near v0, forking is neutral under done-only grading (-0.003 ± 0.014). It turns mildly positive when the text endings
  are left out (+0.025 ± 0.012). v7's positive gap came mostly from grading the text. As graded it is +0.065, +0.140
  and +0.104 at steps 1, 2 and 3, and +0.070 over steps 1-3; done-only it is +0.022 (`out/fork_gap.txt`).

## v9 (`search-v9-1025`): with the #313 close reminder

- The code is 8b73a48, which adds #313: a thread that replies without a tool call is reminded once. From 01:40 on 10-26
  it is 59517af (`ROOT/search/CODE.pre-59517af-10260140`), the deployed pre-merge head of PR #316. Main has #316 as
  the squash 63206b9, which differs only in comments, an assert and the test, so the training logic is the same. v9
  starts from v0, with the same settings as v8 otherwise.
- The trainer log has 24 step ends so far: 18:25 on 10-25 to 00:47 on 10-26. Steps 22 and 23 took 62 and 89
  minutes, across trainer restarts.

Source: `out/v9_steps.jsonl`, with the same script and definitions as v8. "rem main" counts episodes whose main was
reminded / of which main then ended with `done`. "rem branch" counts reminded branches / of which ended with `done`.
"rw main" and "rw br" are the shares of episodes with a rewind in main and in any branch.

| step | end | min | eps | fork | br/ep | mu | cap10 | depth>=3 | rem main | rem branch | no tool | rewind | rw main | rw br | acc | task_reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 18:25 | - | 797 | 0.23 | 0.88 | 0.47 | 0.026 | 0.058 | 27 / 27 | 28 / 26 | 0.006 | 0.25 | 0.22 | 0.04 | 0.49 | +0.245 |
| 2 | 18:37 | 12 | 547 | 0.25 | 1.29 | 0.57 | 0.062 | 0.113 | 20 / 19 | 46 / 45 | 0.004 | 0.33 | 0.28 | 0.07 | 0.39 | -0.293 |
| 3 | 18:50 | 13 | 432 | 0.17 | 0.63 | 0.49 | 0.023 | 0.039 | 14 / 13 | 10 / 9 | 0.012 | 0.25 | 0.22 | 0.03 | 0.45 | -0.265 |
| 4 | 18:56 | 6 | 314 | 0.17 | 0.60 | 0.44 | 0.013 | 0.032 | 10 / 10 | 6 / 6 | 0.010 | 0.32 | 0.29 | 0.04 | 0.41 | -0.214 |
| 5 | 19:02 | 6 | 347 | 0.22 | 0.69 | 0.46 | 0.020 | 0.052 | 21 / 20 | 23 / 21 | 0.009 | 0.30 | 0.27 | 0.04 | 0.51 | -0.131 |
| 6 | 19:11 | 9 | 427 | 0.19 | 0.60 | 0.44 | 0.019 | 0.033 | 29 / 28 | 15 / 10 | 0.007 | 0.26 | 0.25 | 0.04 | 0.52 | -0.143 |
| 7 | 19:15 | 4 | 196 | 0.15 | 0.58 | 0.52 | 0.015 | 0.031 | 4 / 3 | 9 / 8 | 0.036 | 0.28 | 0.26 | 0.03 | 0.50 | +0.080 |
| 8 | 19:22 | 7 | 257 | 0.26 | 1.17 | 0.51 | 0.058 | 0.078 | 13 / 11 | 16 / 14 | 0.016 | 0.34 | 0.28 | 0.11 | 0.31 | +0.045 |
| 9 | 19:34 | 11 | 710 | 0.20 | 0.75 | 0.45 | 0.028 | 0.042 | 38 / 37 | 28 / 24 | 0.010 | 0.31 | 0.27 | 0.07 | 0.41 | +0.139 |
| 10 | 19:44 | 10 | 551 | 0.19 | 0.65 | 0.44 | 0.025 | 0.031 | 27 / 27 | 13 / 13 | 0.016 | 0.32 | 0.27 | 0.06 | 0.42 | -0.039 |
| 11 | 19:52 | 9 | 418 | 0.20 | 0.60 | 0.35 | 0.014 | 0.029 | 15 / 13 | 5 / 5 | 0.019 | 0.32 | 0.28 | 0.07 | 0.40 | +0.474 |
| 12 | 20:04 | 12 | 379 | 0.24 | 0.64 | 0.36 | 0.024 | 0.034 | 19 / 18 | 1 / 1 | 0.018 | 0.28 | 0.26 | 0.05 | 0.48 | +0.148 |
| 13 | 20:13 | 9 | 285 | 0.25 | 0.85 | 0.37 | 0.014 | 0.042 | 14 / 13 | 5 / 5 | 0.021 | 0.35 | 0.27 | 0.11 | 0.39 | -0.188 |
| 14 | 20:17 | 5 | 152 | 0.20 | 0.60 | 0.32 | 0.013 | 0.013 | 10 / 10 | 1 / 1 | 0.000 | 0.32 | 0.26 | 0.07 | 0.46 | -0.013 |
| 15 | 20:28 | 11 | 523 | 0.36 | 1.11 | 0.39 | 0.029 | 0.057 | 38 / 38 | 11 / 11 | 0.011 | 0.31 | 0.22 | 0.13 | 0.46 | -0.068 |
| 16 | 20:41 | 12 | 432 | 0.36 | 1.24 | 0.43 | 0.035 | 0.081 | 26 / 25 | 14 / 13 | 0.012 | 0.34 | 0.25 | 0.14 | 0.50 | -0.203 |
| 17 | 20:51 | 10 | 311 | 0.49 | 1.47 | 0.34 | 0.035 | 0.064 | 29 / 27 | 10 / 10 | 0.010 | 0.39 | 0.24 | 0.20 | 0.45 | -0.217 |
| 18 | 21:07 | 16 | 553 | 0.43 | 1.37 | 0.37 | 0.045 | 0.063 | 55 / 50 | 21 / 18 | 0.027 | 0.34 | 0.24 | 0.14 | 0.46 | -0.360 |
| 19 | 21:19 | 12 | 283 | 0.54 | 1.92 | 0.36 | 0.053 | 0.078 | 20 / 19 | 12 / 12 | 0.021 | 0.33 | 0.16 | 0.21 | 0.45 | -0.067 |
| 20 | 21:32 | 13 | 350 | 0.50 | 1.91 | 0.46 | 0.089 | 0.114 | 24 / 21 | 15 / 15 | 0.017 | 0.35 | 0.24 | 0.19 | 0.34 | +0.052 |
| 21 | 21:41 | 9 | 299 | 0.60 | 1.75 | 0.39 | 0.047 | 0.087 | 43 / 43 | 5 / 5 | 0.010 | 0.31 | 0.20 | 0.16 | 0.47 | +0.247 |
| 22 | 22:44 | 62 | 585 | 0.61 | 2.24 | 0.44 | 0.089 | 0.135 | 78 / 75 | 34 / 30 | 0.014 | 0.35 | 0.22 | 0.17 | 0.43 | +0.382 |
| 23 | 00:13 | 89 | 845 | 0.61 | 2.08 | 0.43 | 0.065 | 0.118 | 118 / 116 | 22 / 20 | 0.014 | 0.33 | 0.16 | 0.21 | 0.49 | +0.223 |
| 24 | 00:47 | 34 | 735 | 0.75 | 2.82 | 0.44 | 0.110 | 0.163 | 113 / 109 | 16 / 13 | 0.011 | 0.38 | 0.18 | 0.28 | 0.44 | -0.126 |

- Unlike v8, the fork rate did not collapse. It held at 0.15-0.26 through step 14, then rose to 0.75 by step 24.
  Branches/ep rose to 2.82.
- mu stayed at 0.32-0.57, well below the 0.8 stop line. The cap-10 share reached 0.11 and depth >= 3 reached 0.16 at
  step 24.
- Reminders: over steps 1-24, 805 mains were reminded and 772 (96%) then ended with `done`. 366 branches were reminded
  and 335 (92%) ended with `done`. "No tool call" main endings are 1.3% of the episodes; v8 had 1.6%.
- Rewind stayed at 25-39% of the episodes. Over the steps the main share fell (about 0.27 to
  0.18) and the branch share rose (0.04 to 0.28), as forking rose.
- The 244 episodes that finished after step 24 (step 25 in progress at 02:05) are left out.

### Val (new code, k=1, 500 questions)

Source: `python3 scripts/alpha/search_ablate_table.py` (main) on `ROOT/search/pilot/eval/val-search-v9-1025-v{0,10,20}-fork-k1.jsonl`,
run on Alpha (`out/v9_val.txt`). The difference is paired by question against v0, with a bootstrap 95% CI.

| adapter | acc | forked | branches | max fork depth (eps at 0 / 1 / 2 / >=3) | calls | C | diff vs v0 |
|---|---|---|---|---|---|---|---|
| v0 | 0.416 | 14% | 0.50 | 431 / 36 / 19 / 14 | 15.4 | 856 | baseline |
| v10 | 0.442 | 12% | 0.35 | 439 / 49 / 8 / 4 | 16.6 | 1426 | +0.026 [-0.004, +0.056] |
| v20 | 0.466 | 55% | 1.15 | 225 / 238 / 26 / 11 | 26.7 | 2245 | +0.050 [+0.016, +0.084] |

There are no infra failures. v20 is the first Val clearly above v0. Its fork rate (55%) is close to training's at
steps 19-21.

## Infrastructure incidents of the night

These come from the session's operations notes. Only the parts marked "seen" were re-checked on Alpha for this record.

| incident | fix or rule |
|---|---|
| The Haiku supervisor acted on stale plans. | Re-read the live run state before acting on a plan. |
| Job 148450 was left in `ROOT/search/HOLD` (seen in `HOLD.pre-233905`). | Taken out of HOLD (not in the current HOLD: seen). |
| 148575 was submitted without `search.sbatch`'s code argument, so it failed at start. | Always pass the code argument. |
| The adapter unload in `rl_run.sh start` hung 3 times in v9: requests left in flight by a stopped trainer leak the servers' LoRA count. | Restart the servers after a trainer stop instead of unloading old adapters. |
| Beta trainer startup hit AReaL's 60 s fork-ready timeout. | #316 gives forked proxy-rollout workers 300 s. |
| Beta's `search.env` forced `branch_credit=gate`. | Caught before use. |
| Some episodes pass the 262144-token context and get HTTP 400 (<1%). The window guard does not catch them. | Open. |

On the last row, the v9 trainer log shows 5 trajectories rejected with "a model request failed" (19:05 to 00:09). The
episode `requests.jsonl` files that are still graded show no HTTP 400, so the 262144 cause and the <1% share were not
reproduced here.

## Read_thread hijack (preliminary, audited)

**Preliminary.** These numbers come from one observational judge pass and an audit with a single annotator. The measures
of PR #312 (mounted vs refused, counterfactual mount) have not been run yet.

- The labels come from the 10-25 judge pass and are not committed: `hijack_judge.py` with SEED 0, N 400 for v8 and
  553 for v6d (all 460 reads), judge at `http://10.0.4.26:32500/v1`. v8's read pool was 1,423 then and is 1,612
  today (count from the 10-26 review), and the sample depends on the pool, so `out/hijack_units-*.tsv` cannot be
  regenerated from raw data. They are the committed record; `hijack_net.py` reproduces the table from them.
- The judge (`scripts/hijack_judge.py`, temperature 0) labels a branch R's action as OWN or OTHER task. It does so just
  before R mounted another thread S with `read_thread` (pre) and just after (post).
- Net switch = (OWN->OTHER - OTHER->OWN) / judged reads. The ± values are 95% half-widths (1.96 x iid SE).
- "Corrected" leaves out two kinds of read, as the 10-25 audit did:
  - reads whose post reply is itself another `read_thread` call (188 of v8's 400), since it judges a read, not R's
    work;
  - reads whose pre action is a `fork` call, often the fork that created S.

Source: `python3 scripts/hijack_net.py out/hijack_units-*.tsv` (`out/hijack_net.txt`).

| trial | reads | raw net switch | corrected reads | corrected net | siblings | cousins | S under R |
|---|---|---|---|---|---|---|---|
| v6d (all 460 mounted reads) | 460 | +0.254 ± 0.053 | 250 | +0.292 ± 0.072 | +0.451 ± 0.128 (82) | +0.214 ± 0.085 (168) | - (0) |
| v8 early (400 of 1,423 reads) | 400 | -0.060 ± 0.055 | 158 | +0.089 ± 0.085 | +0.073 ± 0.098 (109) | +0.087 ± 0.170 (46) | +0.667 ± 0.653 (3) |

- The audit reported v8 as +0.089 ± 0.043. That ± is 1 SE; this record writes the 95% half-width.
- Label precision is 48-80%. The audit hand-labelled 40 judged OWN->OTHER cases (25 from v6d, 15 from v8). 19/40 (48%)
  are a real hijack; 32/40 (80%) if ambiguous cases count. One annotator did this, and the labels were not saved, so
  this was not reproduced.

## What's next

1. Finish v9's pass over the train questions. Then run Val and BCP on its final adapter.
2. Run the 6 training ablation arms in reverse order: w20 -> w5 -> w2 -> d3 -> d2 -> d1. Each arm is a fresh trial from
   v0 that runs one pass and then Val 500 k=1.

## Reproduce

```
R=/projects/cluster/octopus/octopus
for t in v8 v9; do ssh cluster-login "python3 - $R/search/rl-state-search/train-search-$t-1025.log $R/search/rl/jobs-search-$t-1025" < scripts/steps.py > out/${t}_steps.jsonl; done
E=$R/search/pilot/eval; ssh cluster-login "cd /tmp && python3 - $E/val-search-v9-1025-v{0,10,20}-fork-k1.jsonl" < ../../scripts/alpha/search_ablate_table.py > out/v9_val.txt
# fork diagnosis: extracts (v6d 26k lines) are not committed; step ends are out/TRIAL.steps
for t in search-v6d-1024 search-v7-1025 search-v8-1025; do
  ssh cluster-login "python3 - $R/search/rl/jobs-$t" < scripts/fork_extract.py > D/$t.jsonl
  ssh cluster-login "grep -aoE '[0-9]{8}-[0-9:.]+ StatsLogger INFO: Epoch [0-9]+/[0-9]+ Step [0-9]+/[0-9]+ Train step [0-9]+' $R/search/areal-exp/logs/user/rq1-rl/$t/main.log" | awk '{print $NF, $1}' | uniq > out/$t.steps
done
cp out/*.steps D/; python3 scripts/fork_gap.py D search-v6d-1024 search-v7-1025 search-v8-1025 > out/fork_gap.txt
# hijack: the 10-25 judge pass (labels not committed; today's read pool differs, so a re-run samples other reads)
ssh cluster-login "python3 - $R/search/rl/jobs-search-v8-1025 400 http://10.0.4.26:32500/v1 0" < scripts/hijack_judge.py > labels-v8.jsonl
python3 scripts/hijack_units.py labels-v8.jsonl JOBS_COPY > out/hijack_units-v8-1025.tsv   # JOBS_COPY: the judged episodes' trees
python3 scripts/hijack_net.py out/hijack_units-*.tsv > out/hijack_net.txt                   # from the committed TSVs
```

Everything here only reads from Alpha.
