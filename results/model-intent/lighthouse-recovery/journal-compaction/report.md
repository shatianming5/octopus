# Journal-Compaction Episode Report

The single frozen `journal_compaction:deduplicate_v1` episode completed normally, but the official outcome was **QUALITY_FAIL** (`resolved=false`). The original `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` FAIL_TO_PASS test remained failed. The solver reached the 12-action limit with zero edits, zero patch bytes, no verification action, and no done action.

## Publishable metrics

The episode retained 12 raw proposals and 12 raw action results in the full audit. Hashing the original parsed action objects produced 7 distinct raw proposals. Eleven actions were physically submitted; one repeated failed-action proposal was rejected by the controller before external execution. Canonical physical hashes use `execution_action` when present and otherwise the parsed `action`; this episode recorded no separate `execution_action` fields. There was one physical failure (`missing_path`), zero canonical physical resubmissions of a known failure, and therefore zero effective failed repeats. Journal projection is not part of this execution metric.

The episode returned 8 nonempty physical observation outputs. Using those 8 outputs as the denominator and byte-exact returned output content as the uniqueness key, 5 contents were unique.

Per-turn `deduplicated_journal_entries` were:

`[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3]`

Per-turn ordinary oldest-turn budget drops, which are separate from compaction, were:

`[0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2]`

Recorded native work was 41,828 actual prefill tokens and 1,497 actual decode tokens with zero preemptions. The episode did not record a separate recomputed-token counter, so no recomputed amount is inferred. Execution remained sequential, with maximum simultaneous scheduler requests of 1, zero native allocation rejections, and zero cache evictions. The 12 request records and 67 consumer lifecycle records are retained as audit records, not reported as work amounts. GPU engine scope was 82.751 seconds, solver end-to-end time was 84.885 seconds, child-process wall time was 99.263 seconds, official grading took 10.229 seconds, and terminal elapsed time was 114 seconds.

## Comparison with deliberative14

Both episodes shared the same first nine actions (indices 0-8), retained 12 raw proposals, had 7 distinct raw proposal hashes, returned 5 unique byte-exact nonempty output contents, made no edit, performed no verification, and failed the same official test. Their late behavior differed:

- `deliberative14` physically submitted all 12 actions and repeatedly read `lighthouse-core/lib/stack-packs.js` on turns 9-11. It produced 10 total nonempty observations, no journal deduplication, and reached 4 ordinary budget drops.
- `journal-compaction` searched `lighthouse-core/report` on turn 9, rejected a repeated missing-path proposal without physical submission on turn 10, and searched the repository root on turn 11. It produced 8 total nonempty observations, reached 3 compacted journal entries in the final prompt context, and held ordinary budget drops to 2.

The projection therefore changed both the late action trajectory and context accounting. It also used 10.110 fewer GPU-engine seconds and finished 15 seconds sooner than `deliberative14`. These are single-episode descriptive differences, not evidence of a general performance improvement.

## Interpretation and limits

Journal compaction reduced repeated context entries and changed subsequent model behavior, but it did **not** achieve official quality. Smaller or differently projected context is not a pass condition, and this result does not establish the cause of the official assertion failure.

The projection retained unique external observations and the latest representative/explanation for identical action-observation pairs inside read-only segments. It did **not** preserve all older model reasoning text. Full raw proposals, action results, and persistent failure memory remain available in `execution/solver/episode.json`; the projection did not alter that audit.

Cleanup remains proven by `cleanup.json`: GPU 9 returned to 2 MiB and 0% utilization with no owned model process, and the local `octopus-lighthouse-` Docker namespace was empty. Verified model assets were intentionally retained.
