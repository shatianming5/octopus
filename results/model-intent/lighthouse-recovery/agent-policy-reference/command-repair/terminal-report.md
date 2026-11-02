# Agent7 command-repair same-Lighthouse quality result

**Scope:** `software-agent7-command-repair-quality-v1`

**Terminal status:** `complete_with_protocol_validation_failure`

**Official quality:** `QUALITY_FAIL` (`resolved=false`, patch `0` bytes). The original official test `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` remained `FAILED`.

The one authorized episode completed all 50 proposals on the pinned OpenHands-7B-Agent actor and separate C03 tokenizer using commit `bc80be2764cebf362d98f80f5910ce5afaacc121`. The prior 43 `No escaped character` failures fell to zero. The repaired interface produced five physical tool executions: three missing-path views and two successful repository-root views. It produced no source edit, test, finish or product patch. After the second root view, the model again selected the nonexistent `GoogleChrome/lighthouse` path; after its third physical failure, the controller blocked the same effective view for the final 37 proposals. This is the decisive quality failure.

The fresh initial-input record was written before inference, but terminal validation found that it omitted the deterministic 36-token empty `PREFIX_CAPSULE`: recorded `1403` tokens / `295830cc...`, actual `1439` tokens / `ee81d650...`. The actual request used the required pinned tokenizer and remained within all bounds, but exact prospective equality failed. The record is preserved unchanged; no retry or post hoc rewrite occurred.

The run used 50 requests, 109,753 actual prefill tokens, 2,195 decode tokens, 65,056 adopted-prefix tokens, no preemptions/evictions/allocation rejections, and maximum scheduler concurrency 1. GPU engine scope was 83.370 seconds of the 1,200-second phase cap; wrapper wall time was 114 seconds; official grading was 9.031 seconds of the 1,800-second evaluator cap. The active sample was 61% GPU utilization and 43,128 MiB on GPU 0.

Owned runner, GPU process, tmux session and evaluator container are closed. Verified weights and the task-owned source deployment are retained; raw trace and official evidence are preserved under `execution/solver/`.

**Artifacts:** `protocol.json`, `source-identity.json`, `freeze.json`, `initial-input-record.json`, `initial-input-validation.json`, `first-12-prefix.json`, `metrics.json`, `terminal.json`, `cleanup.json`, and `execution/solver/{episode.json,grade.json,result.json,official.log.gz}`.

**Scientific implication:** the generic command repair removed the old parser symptom but did not pass official quality. The gate remains closed, and this result does not authorize a second episode or another model-visible treatment.
