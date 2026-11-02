# Task pool (`pool-v1`)

Bundle RL (the current runs) uses the task-bundle dataset instead: see [task-bundles.md](task-bundles.md).

Training and eval use one versioned Harbor task pool:

`/home/user/harbor-udocker/pool-v1`

| file | what |
|---|---|
| `tasks/<instance_id>/` | Harbor task (`task.toml`, `solution/fix.patch`, `pool.json`) |
| `manifest.jsonl` | one row per task: kind, bug_type, repo, image, split, source_batch, f2p, … |
| `splits/train.txt` | train instance ids |
| `splits/eval.txt` | eval instance ids |

The locked **train+eval mix is 2020 tasks at 1:3 single:multi (505 + 1515)**. `select_pool4000.py` (seed `Random(0)`) drew that subset from remaining `*-final` after both screening queues finished and every new batch had live-F2P unsound=0: drop tiny patches, multi-bug tasks whose statement does not name every patched hunk (`statement_covers`: enclosing def/class, a changed def/class, or the file stem must appear in the statement; 1243 dropped), and the longest 5% by statement+patch length, then even kind quotas, even difficulty/length terciles inside each kind, and a 5% per-repo cap. 1515 multi-bug remained, so single was shrunk to 505 to keep 1:3. A final quality check then moved 18 more out (11 with a FAIL_TO_PASS test deselected as known-failing, 3 whose statement quotes at least half of the fix's added lines, 4 sharing a statement) to `/home/user/harbor-udocker/legacy/pool-v1-dropped-20261008/` with `reasons.json`: **pool-v1 is 2002 tasks (503 single + 1499 multi)**. Extra remaining-final tasks stay on disk under `datasets/swesmith-*-final` and were not imported. Lock files (the 2020 selection): `/home/user/pool-build/pool4000/{selected,single,multi}.txt`, `cover.json`, and `octopus/envs/swe/data/pool-build/pool4000-report.json`.

Eval is **200 single-bug + 100 multi-bug** (train 1702), drawn with `random.Random(0)`, stratified by `(kind, repo)` with largest-remainder quotas (`build_pool.py`). Rebuilding after adding tasks re-draws eval over the full set; do not rebuild mid-run.

Earlier pool-v1 versions are kept, not deleted: the 3596-task mix whose multi-bug half included 1204 tasks with statements naming only some of the bugs at `/home/user/harbor-udocker/legacy/pool-v1-3596-uncovered/` (its lock files at `/home/user/pool-build/pool4000-3596/`), and the 1428-task pool at `/home/user/harbor-udocker/legacy/pool-v1-1428-pre4000/`. Older dataset/spec layouts live in `/home/user/harbor-udocker/legacy/pre-pool-v1/`. Do not point new train/eval jobs at `datasets/swesmith-n560-valid`, `swesmith-mb750`, `swesmith-mb-spare`, or `forkeval/specs-n560*`.

## Screening pipeline (2026-10-06 bar)

Canonical copies: `octopus/envs/swe/data/pool-build/`. Live copies: `/home/user/pool-build` on node-94 (select/regen/finalize) and node-134 (relay into node-183). Do not change the drop rules.

```
select_swesmith.py → regen_tasks.py → baseline_tasks.py
  → oracle+nop → select_valid.py → oracle+nop
  → finalize_batch.py (drop F2P-deselected + oracle pytest > 60s + pass-2 flip + shared statement)
  → audit_tasks.py (live F2P must fail on the bug and pass on the reference; nop must show pytest failures)
     then move unsound instance dirs to datasets/swesmith-NAME-unsound/
  (multi-bug also mb_cover.py: never committed, only a working copy on node-94; its rule is
   pool-build/select_pool4000.py statement_covers)
```

node-183 is missing some goldens: `partition.py` moves those instance dirs to `datasets/tasks-no183` and they are screened on node-94 (`batch.sh no183 src -`) **after** every 183-bound batch has partitioned.

Concurrency: node-94 `CONC=24`; relay `CONC183=48 SLOTS=8 END=revalidate` on 183, then finalize/audit on 94 (parquet lives on 94).

Kinds `func_pm_flip_operators` and `lm_modify` in this parquet are Go (`Test*` FAIL_TO_PASS, repos like bleve/caddy/atlas). Harbor `verifier_shell` is pytest-only, so they stay out of pool-v1 even if their images are imported. That is ineligibility, not a drop of screened Python candidates.

## Rebuild after a finished `*-final` batch

The 2020-task selection (2002 after the 18 drops) is already imported. Further `pool_import.py add` of a whole `*-final` batch would grow past the lock; re-run `select_pool4000.py` and rebuild from `pool4000/selected.txt` instead.

On node-94, with `source /home/user/harbor-udocker/scripts/env.sh`:

```bash
AP=/mnt/disk1/user/areal/.venv/bin/python
POOL=/home/user/harbor-udocker/pool-v1
N=/home/user/harbor-udocker
B=/home/user/pool-build

$AP $B/pool_import.py add $POOL $N/datasets/swesmith-NAME-final NAME $N/jobs/v2-NAME-nop \
  [$N/datasets/swesmith-NAME-final.manifest.json]   # last arg only for multi-bug

$AP $B/build_pool.py $POOL
```

`queue.sh QUEUEFILE` pops the first line before each batch so lines can be appended or held while it runs. `[RUNNER=relay183.sh] queue.sh queue-183.txt` is the node-134 entry. Never start a second `queue.sh` against the same file; never kill a live Harbor/batch.
