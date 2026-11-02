# alpha-search-v6d-1024: Search v6d on Cluster Alpha (depth 1, 55 steps)

Search RL trial `search-v6d-1024` on Cluster Alpha (H200), outcome credit R = 2y-1, fork depth 1, batch 16 questions x 8
samples, from adapter v0 (`models/adapter-v0-4b-text-r64`). Ended by hand at step 55. Record written 10-25. Times are EDT.

ROOT = `/projects/cluster/octopus/octopus`. Every number below was recomputed on 10-25 from the raw
files on Alpha with the scripts in `scripts/`; their outputs are in `out/`. The record itself has not had an
independent review yet.

## Run record

### Restarts before v6d

Each restart was a new trial from v0 with new code. Code dirs are `ROOT/code-<sha>`.

| trial | code | change | ran (10-24) | steps |
|---|---|---|---|---|
| search-v6-1024 | f8467a0 (#296) | full-run launch | 07:01-08:20 | 6 |
| search-v6c-1024 | 56ae681 (#297), then c5f78a9 (#298) | rewind cap `OCTOPUS_REWIND_MAX=2`; least-loaded routing | 08:25-13:20, 14:09-15:01 | 24 |
| search-v6d-1024 | b67d164 (#300), then 2edfb2d (#302) | branch turns 20 -> 50; trainer on 4 GPUs, judge in a rollout job | 10-24 15:08 to 10-25 05:15 | 55 |

Episode settings (`ROOT/search/pilot/search-v6d-1024/run/episode_settings.json`): `OCTOPUS_CURSOR=1`, `OCTOPUS_READ_TOOL=1`,
`OCTOPUS_REWIND_TOOL=1`, `OCTOPUS_REWIND_MAX=2`, `OCTOPUS_TOOL_FORMAT=qwen3_coder`. Fork depth 1 is the default.
Launch: `BATCH=16 STEPS=190 SAVE_EVERY=10 EPISODES_PER_SERVER=8`.

### v6d segments

Sources: the queue logs `ROOT/search/pilot/hold-*/queue/{20-search-v6d-1024,01-rollout-search-v6d-1024}.sh.*`, the
stop markers and `epoch-stop-*.log` in `ROOT/search/pilot/search-v6d-1024/`, and the trainer log
`ROOT/search/areal-exp/logs/user/rq1-rl/search-v6d-1024/main.log`.

| segment | trainer job | rollout / judge | code | start | stop | steps |
|---|---|---|---|---|---|---|
| 1 | 145533 (alphagpu20; judge GPU0, trainer GPUs 1-3) | 146854, 4 servers (10.0.4.28) | b67d164 | 10-24 15:08 | 16:48, migration (`stopped-1024-step9`) | 1-9 |
| 2 | 147315 (alphagpu19, trainer 4 GPUs) | 146854 (3 servers + judge, 10.0.4.28) and 145533 (4 servers, 10.0.4.26) | 2edfb2d | 16:55 | 21:59 (`stopped-1024-step37`) | 10-37 |
| 3 | 147315 | 146854 (3 servers + judge, 10.0.4.28) | 2edfb2d | 22:11 | 10-25 01:31 (`stopped-1024-step46-to147853`) | 38-46 |
| 4 | 147315 | 147853 (3 servers + judge, 10.0.4.26) | 2edfb2d | 01:40 | 05:09 (`stopped`) | 47-55 |

- Each segment resumed from the previous recover checkpoint. The step-55 boundary was at 05:09:16. The final adapter is
  `ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v6d-1024/final-step55` (r 64, alpha 64, 10 targets,
  `adapter_model.safetensors` sha256 711610b96325cbd1...).
- Coverage: the epoch-stop log shows 2973, 2975, 2976 and 2979 of the 3040 train qids finished after steps 51-54. The
  run was stopped by hand on this plateau. Recomputed with `scripts/repeats.py` and `scripts/missing_qids.py` on the
  jobs dir (`out/v6d_repeats.txt`, `out/v6d_missing.txt`):
  - At the stop, 2981 qids were covered.
  - 37 qids (1.2%) were never drawn.
  - 22 qids were dispatched but have no finished episode. Their newest dir is from 01:30, which is the step-46
    restart.
  - 289 qids were drawn more than once.
  - Attributing the 37 never-drawn qids to AReaL's resume (its dataloader skips ahead) is not checked here.

## Per-step training

Source: `python3 - JOBS_DIR MAIN_LOG < scripts/step_table.py` (`out/v6d_step_table.txt`). The arguments are
JOBS_DIR = `ROOT/search/rl/jobs-search-v6d-1024` (25,812 graded episodes) and MAIN_LOG = the trainer's `main.log`.

The episode columns bucket each episode by the training step during which it finished. That is, its `answers.json`
mtime falls between two "Train step" times. This is a proxy, because rollouts are up to 2 versions off-policy. Each
bucket holds every finished episode, including groups the trainer filters out.

Columns:

- end: the step's end time from the log clock.
- min: minutes since the previous step. Restart gaps are included.
- task_reward: the trainer's `ppo_actor/task_reward/avg` over the trained samples.
- fork: the share of episodes that forked.
- br/ep: branches per episode.
- main rw/ep: accepted main rewinds per episode.
- br rw/br: accepted rewinds per branch.
- acc: episode accuracy.

| step | end (day hh:mm) | min | task_reward | eps | fork | br/ep | main rw/ep | br rw/br | acc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 15:26 | - | +0.027 | 781 | 0.27 | 0.62 | 0.360 | 0.272 | 0.50 |
| 2 | 24 15:38 | 12 | -0.215 | 554 | 0.34 | 0.79 | 0.439 | 0.416 | 0.38 |
| 3 | 24 15:48 | 10 | -0.083 | 321 | 0.28 | 0.56 | 0.274 | 0.372 | 0.46 |
| 4 | 24 15:55 | 7 | +0.038 | 257 | 0.42 | 0.96 | 0.467 | 0.364 | 0.37 |
| 5 | 24 16:03 | 8 | -0.172 | 300 | 0.36 | 0.80 | 0.490 | 0.418 | 0.48 |
| 6 | 24 16:12 | 9 | +0.073 | 333 | 0.47 | 1.07 | 0.444 | 0.539 | 0.50 |
| 7 | 24 16:22 | 10 | +0.062 | 446 | 0.48 | 1.04 | 0.312 | 0.546 | 0.54 |
| 8 | 24 16:29 | 7 | -0.060 | 217 | 0.43 | 0.94 | 0.364 | 0.527 | 0.39 |
| 9 | 24 16:47 | 17 | -0.132 | 675 | 0.36 | 0.87 | 0.501 | 0.566 | 0.41 |
| 10 | 24 17:22 | 35 | -0.072 | 1158 | 0.29 | 0.66 | 0.481 | 0.610 | 0.41 |
| 11 | 24 17:31 | 9 | -0.401 | 394 | 0.24 | 0.49 | 0.335 | 0.411 | 0.51 |
| 12 | 24 17:37 | 7 | -0.111 | 313 | 0.24 | 0.53 | 0.613 | 0.599 | 0.41 |
| 13 | 24 17:44 | 6 | -0.145 | 346 | 0.26 | 0.55 | 0.338 | 0.309 | 0.51 |
| 14 | 24 17:54 | 10 | -0.154 | 346 | 0.27 | 0.53 | 0.431 | 0.632 | 0.45 |
| 15 | 24 18:01 | 7 | -0.244 | 310 | 0.34 | 0.71 | 0.426 | 0.545 | 0.44 |
| 16 | 24 18:08 | 7 | +0.008 | 336 | 0.38 | 0.89 | 0.393 | 0.498 | 0.51 |
| 17 | 24 18:19 | 10 | -0.172 | 496 | 0.31 | 0.54 | 0.464 | 0.487 | 0.46 |
| 18 | 24 18:24 | 6 | +0.174 | 246 | 0.42 | 0.81 | 0.402 | 0.420 | 0.41 |
| 19 | 24 18:33 | 8 | -0.233 | 447 | 0.46 | 0.84 | 0.418 | 0.520 | 0.43 |
| 20 | 24 18:39 | 7 | -0.273 | 311 | 0.46 | 0.73 | 0.518 | 0.493 | 0.38 |
| 21 | 24 18:47 | 8 | +0.309 | 409 | 0.56 | 0.99 | 0.416 | 0.548 | 0.45 |
| 22 | 24 18:58 | 11 | +0.093 | 537 | 0.63 | 1.17 | 0.438 | 0.618 | 0.39 |
| 23 | 24 19:11 | 13 | -0.175 | 572 | 0.67 | 1.13 | 0.336 | 0.530 | 0.41 |
| 24 | 24 19:19 | 8 | +0.039 | 361 | 0.76 | 1.30 | 0.332 | 0.629 | 0.45 |
| 25 | 24 19:34 | 14 | -0.301 | 662 | 0.82 | 1.28 | 0.285 | 0.503 | 0.47 |
| 26 | 24 19:43 | 10 | -0.386 | 409 | 0.74 | 1.04 | 0.377 | 0.602 | 0.40 |
| 27 | 24 19:49 | 6 | -0.021 | 285 | 0.83 | 1.20 | 0.218 | 0.382 | 0.54 |
| 28 | 24 20:03 | 14 | +0.128 | 633 | 0.88 | 1.34 | 0.365 | 0.625 | 0.46 |
| 29 | 24 20:21 | 17 | -0.189 | 531 | 0.89 | 1.45 | 0.341 | 0.630 | 0.47 |
| 30 | 24 20:29 | 8 | +0.022 | 372 | 0.90 | 1.17 | 0.277 | 0.405 | 0.44 |
| 31 | 24 20:37 | 8 | -0.093 | 357 | 0.90 | 1.31 | 0.381 | 0.542 | 0.50 |
| 32 | 24 20:49 | 12 | +0.029 | 539 | 0.91 | 1.36 | 0.288 | 0.456 | 0.51 |
| 33 | 24 20:57 | 8 | -0.244 | 284 | 0.91 | 1.45 | 0.236 | 0.590 | 0.52 |
| 34 | 24 21:08 | 11 | -0.056 | 471 | 0.93 | 1.49 | 0.253 | 0.558 | 0.51 |
| 35 | 24 21:22 | 13 | +0.119 | 571 | 0.92 | 1.42 | 0.292 | 0.524 | 0.58 |
| 36 | 24 21:37 | 15 | +0.393 | 564 | 0.96 | 1.60 | 0.410 | 0.608 | 0.48 |
| 37 | 24 21:59 | 21 | -0.122 | 690 | 0.95 | 1.68 | 0.454 | 0.749 | 0.36 |
| 38 | 24 22:50 | 51 | -0.011 | 974 | 0.94 | 1.33 | 0.182 | 0.465 | 0.55 |
| 39 | 24 23:06 | 16 | -0.038 | 311 | 0.94 | 1.38 | 0.251 | 0.499 | 0.50 |
| 40 | 24 23:31 | 25 | +0.025 | 629 | 0.95 | 1.46 | 0.243 | 0.490 | 0.56 |
| 41 | 24 23:47 | 16 | +0.155 | 271 | 0.92 | 1.57 | 0.203 | 0.545 | 0.50 |
| 42 | 25 00:10 | 24 | +0.288 | 465 | 0.99 | 1.66 | 0.181 | 0.519 | 0.50 |
| 43 | 25 00:38 | 28 | -0.171 | 532 | 0.94 | 1.78 | 0.199 | 0.562 | 0.41 |
| 44 | 25 00:52 | 14 | -0.139 | 338 | 0.97 | 1.52 | 0.092 | 0.396 | 0.57 |
| 45 | 25 01:04 | 12 | +0.243 | 302 | 0.98 | 1.62 | 0.093 | 0.298 | 0.57 |
| 46 | 25 01:31 | 27 | +0.134 | 544 | 0.95 | 1.71 | 0.145 | 0.432 | 0.49 |
| 47 | 25 02:21 | 50 | +0.084 | 809 | 0.97 | 1.70 | 0.096 | 0.378 | 0.52 |
| 48 | 25 02:34 | 13 | +0.069 | 303 | 0.96 | 1.67 | 0.079 | 0.308 | 0.55 |
| 49 | 25 02:53 | 20 | +0.031 | 398 | 0.98 | 1.70 | 0.075 | 0.362 | 0.50 |
| 50 | 25 03:17 | 24 | -0.208 | 535 | 0.99 | 1.92 | 0.077 | 0.329 | 0.50 |
| 51 | 25 03:33 | 16 | -0.153 | 405 | 0.95 | 1.73 | 0.044 | 0.332 | 0.56 |
| 52 | 25 03:47 | 14 | -0.286 | 386 | 0.98 | 1.63 | 0.018 | 0.241 | 0.59 |
| 53 | 25 04:06 | 18 | -0.218 | 370 | 0.98 | 2.02 | 0.073 | 0.294 | 0.49 |
| 54 | 25 04:35 | 30 | +0.045 | 661 | 0.99 | 2.16 | 0.038 | 0.248 | 0.48 |
| 55 | 25 05:09 | 33 | -0.134 | 735 | 0.98 | 2.00 | 0.026 | 0.239 | 0.55 |

Step 1's bucket holds every episode that finished before the first update. Ten more episodes finished after step 55 and
were not trained. Steps 10, 38 and 47 include restarts. Train/infer kl_k1 and the other trainer stats for every step
are in `../alpha-search-v7-1025/out/v6dsteps.jsonl`, produced by `stepstats.py`.

## Evaluation

Every row was computed with `python3 scripts/alpha/search_ablate_table.py BASE.jsonl ARM.jsonl ...` (main's script) on
`ROOT/search/pilot/eval/{val,test}-search-v6d-1024-*.jsonl`. The snapshot is from 10-25 15:35 EDT and is saved in
`out/evals.txt`.

- Infra means rows without a score. Any arm above 2% infra is NOT COMPARABLE.
- All rows use k=1 and the judge at 0.7.
- Val is the 500 OpenResearcher validation questions. BCP is the 830 BrowseComp-Plus questions.
- vN is the checkpoint after N steps: v10 is `epoch0epochstep9globalstep9`, and v55 is `final-step55`.
- All runs predate #306 and #309 (see `../alpha-search-v7-1025/`), so a main ending without a tool call was still graded
  from its last text. The rows are comparable with each other, not with runs after #306.

| eval | arm | code / EVAL_PARALLEL | n scored | infra | acc | forked | branches/ep |
|---|---|---|---|---|---|---|---|
| Val v0 | fork | b67d164 / default | 500 | 0 | 0.416 | 16% | 0.32 |
| Val v10 | fork | 2edfb2d / 40 | 500 | 0 | 0.462 | 18% | 0.37 |
| Val v20 | fork | 2edfb2d / 55 | 500 | 0 | 0.482 | 50% | 0.75 |
| Val v30 | fork | 2edfb2d / 55 | 500 | 0 | 0.484 | 96% | 1.12 |
| Val v40 | fork | 2edfb2d / 50 | 500 | 0 | 0.492 | 97% | 1.33 |
| Val v55 | fork | 2edfb2d / 55 | 500 | 0 | 0.496 | 98% | 1.90 |
| Val v0 | lin32 (linear, `OCTOPUS_MAIN_MAX_CALLS=100`, default thread budget) | 2edfb2d / 64 | 500 | 0 | 0.400 | 0% | 0 |
| Val v0 | linbig (linear, c14 t27648) | 2edfb2d / 72 | 500 | 0 | 0.400 | 0% | 0 |
| BCP v0 | fork | e883240 / 56 | 830 | 0 | 0.248 | 14% | 0.34 |
| BCP v0 | lin32 | 2edfb2d / 56 | 830 | 0 | 0.246 | 0% | 0 |
| BCP v0 | linbig (c29 t37888) | 2edfb2d / 60 | 830 | 0 | 0.260 | 0% | 0 |
| BCP v55 | fork | 2edfb2d / 55 | 447 | 383 (46.1%) | (0.620) | 100% | 1.95 |

Paired Val differences against v0 fork (bootstrap 95% CI, 500 qids):

| checkpoint | difference [95% CI] |
|---|---|
| v10 | +0.046 [+0.016, +0.076] |
| v20 | +0.066 [+0.034, +0.098] |
| v30 | +0.068 [+0.034, +0.102] |
| v40 | +0.076 [+0.044, +0.108] |
| v55 | +0.080 [+0.048, +0.112] |
| lin32 | -0.016 [-0.046, +0.016] |

BCP v0 linbig against v0 fork: +0.012 [-0.016, +0.040].

BCP v55 is NOT COMPARABLE. 383 of its 830 episodes were infra failures at EVAL_PARALLEL=55, and the 447 that were scored
are a biased subset. Its rerun at EVAL_PARALLEL=20 is queued on 147853 as
`34-bcp-v55-rerun-search-v6d-1024.sh.deferred-v7`; it is not running.

## Depth ablation (eval only, v55)

The ablation re-evaluates `final-step55` on Val with code 18f4fe7. That code includes #304's fork depth >= 2 support but
not #306 or #309. The run settings are `ROOT/search/pilot/search-v6d-1024/run-newcode-d{0,1}`, which set
`OCTOPUS_FORK_DEPTH` = 0 (unlimited) or 1.

| arm | EVAL_PARALLEL | n scored | infra | acc | branches/ep | max depth reached (episodes) | state |
|---|---|---|---|---|---|---|---|
| newcode-d1 | 55 | 491 | 9 (1.8%) | 0.521 | 1.70 | d0 11, d1 480 | done 13:25 |
| newcode-d0 | 40, then 12 | 397 | 103 (20.6%) | (0.572) | 2.34 | d0 9, d1 298, d2 47, d3 20, d4 12, d5 7, d6+ 4 | running, snapshot 15:35 |

- newcode-d0 is NOT COMPARABLE and incomplete.
  - Its first two attempts ran at EVAL_PARALLEL 40. They were ended at 13:51 on 147853 and at 14:42 on 147315.
  - On first attempts, 243 of 499 episodes were infra failures (48.7%). This was counted with the first row per
    `(qid, sample)` of the jsonl. The ablation script keeps the last row instead.
  - The rerun at EVAL_PARALLEL 12 (`hold-147315/queue/35-val-d0-p12.sh`) started at 14:49. It retries the failed
    episodes, so the infra count in the table is still falling.
- newcode-d1 against v55 is not a depth effect. The code changed from 2edfb2d to 18f4fe7, and EVAL_PARALLEL is the
  same, 55.
- The BCP arms of the ablation (`37-bcp-d1`, `38-bcp-d0`) are deferred and have not run.

## Analyses

These analyses were reviewed this session; the recomputation here has not been reviewed. Each finding gives its script
and its output, and every script reads the v6d jobs dir unless stated.

- **Rewind declines because the model reaches the reserve less often and picks rewind less there.**
  - Script: `scripts/rw_drop.py` (`TZ=America/New_York`), output `out/v6d_rw_drop.txt`, per hour of finish.
  - Mains reaching the answer reserve fell from 0.41-0.49 (10-24 15-18h) to 0.07-0.17 (10-25 03-05h).
  - P(rewind | reserve) fell from 0.54-0.60 to 0.17-0.24.
  - Accepted main rewinds per episode fell from 0.39-0.45 to 0.01-0.05.
- **No detectable accuracy gain from rewind.**
  - Script: `scripts/rw_effect.py`, output `out/v6d_rw_effect.txt`.
  - Among mains that reached the reserve, rewound minus closed, paired within the question group: -0.014 [-0.028,
    +0.000] over 1186 groups.
  - For branch rewinds, episode accuracy: +0.005 [-0.006, +0.015] over 1469 groups.
- **Branch rewinds hurt branch completion much more at 20 branch turns than at 50.**
  - Script: `scripts/rw_effect.py`. The 20-turn run is v6c (`ROOT/search/rl/jobs-search-v6c-1024`, before #300); the
    50-turn run is v6d.
  - Branches that reached the reserve, paired done-rate difference, rewound minus closed:

    | branch turns | run | difference [95% CI] | groups | output |
    |---|---|---|---|---|
    | 20 | v6c | -0.360 [-0.422, -0.294] | 188 | `out/v6c_rw_effect.txt` |
    | 50 | v6d | -0.041 [-0.055, -0.026] | 1469 | `out/v6d_rw_effect.txt` |

- **Branch fork attempts at depth 1: about 0.8 per branch, with no benefit.**
  - Scripts: `scripts/fork_limits.py` (`out/v6d_fork_limits.txt`) and `scripts/branch_fork_value.py` on the
    `extract.py` output (`out/v6d_branch_fork_value.txt`).
  - 25,260 branch fork calls in 32,577 branches is 0.78 per branch.
  - At depth 1 the runtime refuses a branch's fork with "Forking is not available at this depth". 34.8% of branches got
    at least one such refusal.
  - In mixed groups, episodes whose branches called fork compare with episodes whose branches did not:

    | branches | n | acc | mean LOO advantage |
    |---|---|---|---|
    | called fork | 2151 | 0.448 | +0.012 |
    | did not | 3048 | 0.572 | +0.030 |

## Reproduce

```
R=/projects/cluster/octopus/octopus; J=$R/search/rl/jobs-search-v6d-1024
M=$R/search/areal-exp/logs/user/rq1-rl/search-v6d-1024/main.log; Q=$R/search-data/search/train-qids.txt
ssh cluster-login "python3 - $J $M" < scripts/step_table.py
ssh cluster-login "python3 - $J $M" < scripts/fork_limits.py
ssh cluster-login "TZ=America/New_York python3 - $J" < scripts/rw_drop.py
ssh cluster-login "python3 - $J" < scripts/rw_effect.py; ssh cluster-login "python3 - $R/search/rl/jobs-search-v6c-1024" < scripts/rw_effect.py
ssh cluster-login "python3 - $J $Q" < scripts/repeats.py
ssh cluster-login "python3 - $J $Q $R/search/analysis-covered-search-v6d-1024.txt" < scripts/missing_qids.py
ssh cluster-login "python3 - $J $M" < ../alpha-search-v7-1025/scripts/extract.py > v6d.jsonl; python3 scripts/branch_fork_value.py v6d.jsonl
```

All of these only read. Each jobs-dir scan takes about 6 minutes on the Alpha login node.
