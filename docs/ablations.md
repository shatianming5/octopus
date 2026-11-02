# Search depth and width ablations on Cluster Alpha

How to run fork **depth** and **width** ablations of the Search environment without the authors.

- **Depth** is `OCTOPUS_FORK_DEPTH`. With `1` (the search default) only main forks, with `D` forks nest D levels below
  main, and with `0` there is no depth limit (the serving window ends deep chains).
- **Width** is `OCTOPUS_MAX_BRANCHES`, the total number of branches one episode may make at every depth. The search
  default is 10 and `0` means no cap. `OCTOPUS_MAX_THREADS` (16) separately caps how many threads run at once (main plus
  open branches). For a width above 15, raise it too if one fork should be able to start every branch.

[search-env.md](search-env.md) "Limits" has every knob. The protocol stays fixed for every arm: search v6+, `done(answer,
evidence)` only, a 32K per-thread growth budget, an answer reserve of 2048, a rewind cap of 2, 50 branch turns and a pure
outcome reward.

`ROOT=/projects/cluster/octopus/octopus`. Every command runs on Alpha (`ssh cluster-login`) unless it is
marked otherwise.

## Prerequisites

- **A held 4-GPU job.** A `search.sbatch` job whose id is in `ROOT/search/HOLD` only holds its allocation and runs
  `ROOT/search/pilot/hold-JOB/queue/*.sh` one at a time in name order (`lib.sh hold`). A task's state is shown by its
  suffix: `.running-JOB`, `.done` or `.failed`, with the output in `.log`. One way to get such a job:
  `j=$(sbatch --parsable --hold -o $ROOT/logs/search-%j.log $ROOT/code-SHA/scripts/alpha/search.sbatch $ROOT/code-SHA); echo $j >> $ROOT/search/HOLD; scontrol release $j`.
  An eval task needs all 4 GPUs. A task already running in that queue, such as a trial's `01-rollout-*.sh`, which serves
  until it is stopped, makes every later task wait.
- **The code deployed** as `ROOT/code-SHA`, with its cursor overlay `ROOT/cursor-SHA`. From the lab host, as
  `search_launch.sh` deploys it:
  `git archive SHA -- . ':!results' | ssh cluster-login "mkdir -p $ROOT/code-SHA.tmp && tar x -C $ROOT/code-SHA.tmp && mv $ROOT/code-SHA.tmp $ROOT/code-SHA"`,
  then `ssh cluster-login ln -s cursor-508e235 $ROOT/cursor-SHA`. The link is only valid while `patches/` is unchanged since
  508e235; otherwise rebuild the overlay (docs/cursor-protocol.md "Deployment").
- **The right SHA.** SHA must include #304 (d4b45c8: fork depth >= 2, `OCTOPUS_FORK_DEPTH=0` = no limit, the window
  guard). Every arm of one table, the baseline included, must run with the same SHA: since #306 (4159155) a thread's
  answer comes only from its closing tool (a text-only answer no longer scores), so an older baseline is not comparable.
- **The adapter and the run's settings dir.** The settings dir holds `config.yaml` and `episode_settings.json`, as
  `rl_run.sh settings` writes them. Examples:
  - v6d final: adapter `ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v6d-1024/final-step55` (NAME `v55`),
    run `ROOT/search/pilot/search-v6d-1024/run`.
  - The base: adapter `ROOT/models/adapter-v0-4b-text-r64` (NAME `v0`), the same run dir.

## Eval-only ablation of one checkpoint

`scripts/alpha/search_ablate.sh [--rerun] QUEUE NAME ADAPTER RUN SETS K LABEL ARM...`

- **Arms.** Each ARM is `A:KEY=VALUE[,KEY=VALUE...]`. Write the baseline as `A:` (the run's own settings) and drop a
  key with `KEY=`. KEY must be one of the episode knobs in the script's `KNOBS` (`OCTOPUS_FORK_DEPTH`,
  `OCTOPUS_MAX_BRANCHES`, `OCTOPUS_MAX_THREADS`, `OCTOPUS_BRANCH_MAX_TURNS`, `OCTOPUS_MAIN_MAX_CALLS`,
  `OCTOPUS_SEARCH_BUDGET`, `OCTOPUS_ANSWER_RESERVE`, `OCTOPUS_THREAD_BUDGET`, `OCTOPUS_CONTEXT_WINDOW`,
  `OCTOPUS_READ_TOOL`, `OCTOPUS_REWIND_TOOL`, `OCTOPUS_REWIND_MAX`, `OCTOPUS_LINEAR`): a typo, a host setting or a
  credential fails before anything is made. Every argument is checked before the first settings dir or task is written.
- **Settings dir.** Each arm gets a copy of RUN at `dirname(RUN)/run-LABEL-A`, with its changes in
  `episode_settings.json`. This is how `run-newcode-d0` was made. `search_dev_eval` takes the arm from that file
  (`config.run_episode_settings`).
- **Tasks.** For each set the arm gets the task `NN-val-LABEL-A-NAME.sh` or `NN+1-bcp-LABEL-A-NAME.sh` (written to
  `.tmp` and moved in). NN defaults to 60, so every Val task runs before any BCP task. The tasks run `search_eval.sh`
  with LABEL `LABEL-A`. Another adapter (NAME) under the same LABEL and arms gets its own tasks.
- **`EVAL_PARALLEL`.** Set in the caller's environment, it is written into the tasks.
- **Re-running.** Running the script again is safe. Queued, running, done, `.skip-*` and `.deferred-*` tasks are
  skipped, a failed one is queued again, and an arm whose settings dir already exists with other settings is refused.
- **`--rerun`.** A run whose episodes failed for infra reasons still ends `.done` (search_dev_eval exits 0). When the
  table shows an arm `NOT COMPARABLE` or `INCOMPLETE`, run the same command with `--rerun` to queue the given arms'
  done tasks again; search_dev_eval then runs only the episodes without a scored row.

```bash
C=$ROOT/code-SHA; Q=$ROOT/search/pilot/hold-JOB/queue; R=$ROOT/search/pilot/search-v6d-1024/run
A=$ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v6d-1024/final-step55
# depth
EVAL_PARALLEL=20 bash $C/scripts/alpha/search_ablate.sh $Q v55 $A $R both 1 depth \
  d1: d2:OCTOPUS_FORK_DEPTH=2 d3:OCTOPUS_FORK_DEPTH=3 d0:OCTOPUS_FORK_DEPTH=0
# width (w10 has depth's d1 settings: drop it when that ran, and use depth-d1 as the table's baseline)
EVAL_PARALLEL=20 bash $C/scripts/alpha/search_ablate.sh $Q v55 $A $R both 1 width \
  w10: w2:OCTOPUS_MAX_BRANCHES=2 w5:OCTOPUS_MAX_BRANCHES=5 w20:OCTOPUS_MAX_BRANCHES=20,OCTOPUS_MAX_THREADS=21
# depth x width grid, Val only; d1w10 is the baseline (MAX_THREADS 21 changes nothing below 16 branches at once)
bash $C/scripts/alpha/search_ablate.sh $Q v55 $A $R val 1 grid \
  $(for d in 1 2 0; do for w in 5 10 20; do echo d${d}w$w:OCTOPUS_FORK_DEPTH=$d,OCTOPUS_MAX_BRANCHES=$w,OCTOPUS_MAX_THREADS=21; done; done)
```

### Runtimes

One task occupies the whole job.

- **Val 500:** 15–35 minutes.
- **BCP 830:** 20 minutes to 1.5 hours. Fork-heavy arms are slower, because they make more requests per question and
  have longer critical paths.

A run's duration depends on how many episodes run at once and how often the arm forks.

`EVAL_PARALLEL` (default 32) is an upper bound on the episodes each of the 3 eval servers runs at once
(`EVAL_PARALLEL` x 3 in all). Adaptive admission is on (`docs/search-env.md` "Admission"): a new episode starts only on
a server below `OCTOPUS_ROUTE_KV` (0.75, `scripts/alpha/search.env`) with no queued request, and `OCTOPUS_ROUTE_CAP`
(32 requests in flight per server, `search_eval.sh`) bounds it. Raise `EVAL_PARALLEL` only after a run's route peak
lines show room. Before admission, fixed values were hand-tuned: fork-heavy checkpoints on BCP needed about 20 (the
deferred d0/d1 BCP tasks used 20), and past server saturation timeouts changed outcomes as infra failures: the
newcode-d0 Val run at `EVAL_PARALLEL=40` lost 47% of its episodes that way (table below).

### Outputs

- **Rows.** `ROOT/search/pilot/eval/val-LABEL-A-NAME-fork-kK.jsonl` for Val and `.../test-LABEL-A-NAME-fork-kK.jsonl`
  for BCP. Each file has one row per (qid, sample), and in each row `error` marks an infra failure.
- **Logs.** The `.log` next to each file, and the task's `.log` in the queue.
- **Trial dirs.** Val keeps them in `eval/jobs-val-...`. BCP removes them, because its questions are decrypted in
  memory only.

### Stop and resume

- **Before a task starts:** rename it, e.g. `mv $Q/61-bcp-depth-d0-v55.sh $Q/61-bcp-depth-d0-v55.sh.skip-user` (the
  script never queues it again).
- **While a task runs:**
  1. Find the exact pid of its `search_eval.sh`:
     `srun --jobid=JOB --overlap -n1 -c1 ps -u $USER -o pid,args | grep "search_eval.sh .* v55 .* depth-d0$"`.
  2. Stop it with `srun --jobid=JOB --overlap -n1 -c1 kill -TERM PID`. The exit trap stops the servers it started and
     removes its `/dev/shm` copy.
  3. The task ends as `.failed`.

  Never use `pgrep -f` or `pkill`: they also match the hold, the other jobs and your own shell.
- **Resume:** run the same `search_ablate.sh` command again. It queues the failed task again, and `search_dev_eval`
  re-runs only the episodes that have no row yet or failed for infra reasons.

## Reading results

`python3 scripts/alpha/search_ablate_table.py BASE.jsonl ARM.jsonl ...` takes the baseline first. It needs only the
standard library and runs anywhere the files are.

For each arm it prints:

- the episodes scored and the infra failures (rows without `y`);
- accuracy, the forked share and the mean number of branches;
- `forks_depth`: the fork depth each episode reached, read from the trial's `agent/tree.json` (Val only: BCP's trial dirs are gone);
- the mean calls, C and C_wall;
- the paired accuracy difference against the baseline over the qids both arms scored, with a 95% bootstrap CI (seed 0,
  10,000 resamples, as `paired.py`).

An arm with more than 2% infra failures is marked `NOT COMPARABLE` (the baseline too, said so), and one with fewer
episodes than the largest arm `INCOMPLETE`. Queue it again with `--rerun` until neither holds.

```bash
E=$ROOT/search/pilot/eval
python3 $C/scripts/alpha/search_ablate_table.py $E/val-depth-d1-v55-fork-k1.jsonl $E/val-depth-*-v55-fork-k1.jsonl
```

Example: the v55 depth arms d1 and d0, both run with code-18f4fe7 (before #306, so not comparable with runs of later
code), read 2026-10-25 14:31 EDT while d0 was being run again. d0's Val ran at `EVAL_PARALLEL=40` and the servers
saturated, hence its 47% infra failures (d1 ran at 55 with 1.8%):

```
arm | n | infra | acc | forked | branches | forks_depth | calls | C | C_wall | diff vs base [95% CI] (qids)
val-search-v6d-1024-newcode-d1-v55-fork-k1 | 491 | 9 (1.8%) | 0.521 | 98% | 1.70 | d0:11 d1:480 | 23.5 | 4670 | 536 | baseline
val-search-v6d-1024-newcode-d0-v55-fork-k1 | 259 | 234 (47.5%) | 0.726 | 97% | 1.64 | d0:7 d1:218 d2:24 d3:7 d4:1 d5:2 | 12.6 | 2604 | 268 | +0.012 [-0.019, +0.043] (257) NOT COMPARABLE (infra > 2%) INCOMPLETE (493 of 500 episodes)
```

## Training ablation

To train under different limits, start one fresh trial per arm with `scripts/alpha/search_launch.sh`, its episode
settings as `OCTOPUS_X=VALUE` arguments after IP (#306; several allowed). They go into the trial's
`P/run/episode_settings.json` and onto the trainer task's `run_clean` line. Each arm needs its own held jobs, because
the launch refuses queues with a running task. Run the guards as that script's header describes.
`search_epoch_stop.sh` then queues Val k=1 and BCP (none with `BCP_K=0`) on RJ, with the trial's own `P/run`, which
carries the arm. Compare the trials with `search_ablate_table.py` on `$E/val-TRIAL-vK-fork-k1.jsonl`.

```bash
# lab host; one line per arm, every trial from the same SHA (the baseline arm too: no settings)
bash scripts/alpha/search_launch.sh SHA search-d1-MMDDHHMM TJ RJ IP
bash scripts/alpha/search_launch.sh SHA search-d2-MMDDHHMM TJ2 RJ2 IP2 OCTOPUS_FORK_DEPTH=2
bash scripts/alpha/search_launch.sh SHA search-d0w20-MMDDHHMM TJ3 RJ3 IP3 OCTOPUS_FORK_DEPTH=0 OCTOPUS_MAX_BRANCHES=20 OCTOPUS_MAX_THREADS=21
```

Check `P/run/episode_settings.json` of each trial before its first step.

### Split layout: the trainer on Alpha, rollout and judge on Beta

The layout of v9 (`search-v9-1025`). TJ is a held Alpha job whose 4 GPUs all train. Two held Beta jobs serve: J1 the
judge on GPU 0 and rollout on GPUs 1-3, J2 rollout on GPUs 0-3. `RJ = TJ` tells `search_launch.sh` there is no Alpha
rollout job (`P/jobs` "TJ TJ"), so the guards stop only TJ and queue the Val on TJ. The Beta servers are yours to stop.

1. **Beta.** Deploy `code-SHA` to `BR=/projects/cluster/octopus/beta/octopus` and build its overlay
   with `scripts/beta/install_env.sh BR BR/code-SHA`. Hold two 4-GPU jobs with `scripts/beta/hold.sbatch`, each on its
   own `OCTO_QUEUE`, and get each node's IP with `srun --jobid=J --overlap -n1 -c1 hostname -I`. Then queue fresh
   servers:

   ```bash
   C=$BR/code-SHA; S=$BR/serve/TRIAL
   JUDGE=0 bash $C/scripts/beta/tasks/serve-rollout.sh $BR/queue-J1 $S-j1 IP1 1,2,3   # judge IP1:32500, rollout IP1:32301-32303
   bash $C/scripts/beta/tasks/serve-rollout.sh $BR/queue-J2 $S-j2 IP2                 # rollout IP2:32300-32303
   ```

   Each task's `.log` ends with `servers ...` once its servers answer.
2. **Launch** from the lab host:

   ```bash
   EXTRA_ROLLOUT=IP1:32301,IP1:32302,IP1:32303,IP2:32300,IP2:32301,IP2:32302,IP2:32303 JUDGE_HOST=IP1 \
     bash scripts/alpha/search_launch.sh SHA TRIAL TJ TJ - OCTOPUS_X=VALUE ...
   # e.g. the w20 arm: ... SHA search-w20-MMDDHHMM TJ TJ - OCTOPUS_FORK_DEPTH=1 OCTOPUS_MAX_BRANCHES=20 OCTOPUS_MAX_THREADS=21
   ```

3. **Guards** on Alpha, as the header of `search_launch.sh` says, with `JUDGE_LOG=$S-j1/sglang-judge.log` in their
   environment (the judge gate and the step-3 check read the judge's log), `SERVER_LOGS="$S-j1 $S-j2"` for the step-3
   report's per-server columns, and `BCP_K=0` for the epoch stop:
   `bash $ROOT/code-SHA/scripts/alpha/search_epoch_stop.sh TRIAL DEADLINE 0`.
4. **After the arm**, stop the Beta servers in their jobs:
   `srun --jobid=J1 --overlap -n1 -c1 bash $C/scripts/beta/serve_rollout.sh stop $S-j1`, and the same for J2.

Start the servers fresh for every trainer start, a resume of the same trial too. Do not reuse them and unload the old
adapters: requests the stopped trainer left in flight leak the servers' LoRA count, and the unload in `rl_run.sh start`
then hangs while it holds the LoRA lock. This happened 3 times in v9.

### The six training arms

Every arm trains a fresh trial from v0 (`ADAPTER0`) with the same latest-main SHA, runs one pass over the train
questions and then Val 500 k=1 (`BCP_K=0`). Another person runs them forward (d1, d2, d3, w2, w5, w20), so we run
them in reverse:

| order | arm | settings after IP |
|---|---|---|
| 1 | w20 | `OCTOPUS_FORK_DEPTH=1 OCTOPUS_MAX_BRANCHES=20 OCTOPUS_MAX_THREADS=21` |
| 2 | w5 | `OCTOPUS_FORK_DEPTH=1 OCTOPUS_MAX_BRANCHES=5` |
| 3 | w2 | `OCTOPUS_FORK_DEPTH=1 OCTOPUS_MAX_BRANCHES=2` |
| 4 | d3 | `OCTOPUS_MAX_BRANCHES=10 OCTOPUS_FORK_DEPTH=3` |
| 5 | d2 | `OCTOPUS_MAX_BRANCHES=10 OCTOPUS_FORK_DEPTH=2` |
| 6 | d1 | `OCTOPUS_MAX_BRANCHES=10 OCTOPUS_FORK_DEPTH=1` |

d1 is also the width table's w10, so it is the baseline of both tables. An episode's caps are `OCTOPUS_MAX_BRANCHES`
and `OCTOPUS_MAX_THREADS` (16 by default: main plus open branches; search's fork takes any number of tasks). w20 raises
the latter to 21, or it would be w15. Every other arm has at most 11 threads at once (w5 6, w2 3), so 16 caps none.

Report per arm, from its Val (`E=$ROOT/search/pilot/eval`, files `val-TRIAL-vK-fork-k1.jsonl`, trial dirs
`jobs-val-TRIAL-vK-fork-k1`):

- **Val acc**, with its paired difference against d1: `search_ablate_table.py` (`acc`, `diff vs base`).
- **Fork rate** and **branches/ep**: the same table (`forked`, `branches`).
- **Main and branch rewind rate**: `PYTHONPATH=$N $ROOT/areal/.venv/bin/python $N/scripts/alpha/search_audit2.py
  $E/jobs-val-TRIAL-vK-fork-k1` (`N=$ROOT/code-SHA`). Its
  `tool calls main` / `tool calls branch` lines count the `rewind` calls; divide them by the episodes and by the
  branches.
- **Critical-path C / total tokens**: `C` from the table, and total tokens as each row's `prompt_tokens` +
  `completion_tokens` (the episode tree's).
