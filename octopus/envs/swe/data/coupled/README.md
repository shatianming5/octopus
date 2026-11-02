# Same-file coupled bundles, track A (node-134 pipeline)

Scripts that build the track-A datasets with `build_bundles.py build --rule samefile` on node-134
(work dir `/mnt/disk6/user/coupled-s1a`, whose `scripts/` links here and whose `code/` is a deployed commit).

| script | step |
|---|---|
| `pipeline.sh` | baselines -> builds (train 25 per repo, eval 45) with judge loops alongside -> `stage2.sh` eval, then train |
| `base.sh` | `build_bundles baseline` (udocker trials, nice 19) |
| `build.sh` | `build_bundles build ... --sizes 2,2,3,4 --rule samefile` (32 processes, nice 19) |
| `judge.sh`, `judge_loop.sh`, `unlabelled.py` | Grok 4.6 `statement_coverage judge`, batch by batch while the build grows, until every task is labelled |
| `stage2.sh` | reduce -> refresh -> reduce -> accept -> timing; container stages sharded by repo (`shard.py`) over node-134 and node-138 (node-68 dropped: no overlay, goldens on a full HDD) (`shard_agent.sh` on each) |
| `accept_timing.sh` | sharded accept, then timing of the accepted bundles (called by `stage2.sh`) |
| `recheck.sh` | accepted bundles copied to NAME-r3 with the R1 healthcheck (`build_bundles clean`), then `accept_timing.sh` |
| `affected.py`, `recheck_fix.sh` | after `clean_globs` grew: bundles whose cleanup now deletes other paths, re-accepted and re-timed, results appended (last wins) |
| `reduce_subset.sh` | reduce -> refresh -> reduce -> accept -> timing of given bundles only (node-134), e.g. ones an older reduce dropped |
| `drop_budget.sh` | docs "Dropping the old budget" on the three final splits, with the grep check |
| `finalize.py` | scan (leaks in /testbed and in copies outside it that the healthcheck leaves, change notes, >= 250 deleted lines, coupling on the shipped fixes), R1 drops, timing <= 180 s, per-repo caps (train 10, eval 20, plan order), val/test split of the eval repos, datasets `OUT-{train,val,test}` + report |

One-off tools kept out of the repo: image fetch through Docker Hub mirrors with sha256-checked blobs (`fetch_image.py`,
`fetch_eval.py`, `load_tar.sh`), golden copies between hosts (rsync of golden dirs, `udocker import` of a golden ROOT),
coverage checks.

Published coupled-a (node-134, after `pipeline.sh`, `recheck.sh`, `recheck_fix.sh`, `reduce_subset.sh`):

```
scripts/finalize.py coupled-b-train-r3,coupled-b-train-fix5-r2 coupled-b-eval-r3,coupled-b-nik-r3,coupled-b-eval-fix5-r2 \
    coupled-a --keep-split /mnt/disk6/user/coupled-s1a/split-v5.json
scripts/drop_budget.sh coupled-a
```
`split-v5.json` is the report whose val / test repo split (the 2^N one without pygments) is kept.
