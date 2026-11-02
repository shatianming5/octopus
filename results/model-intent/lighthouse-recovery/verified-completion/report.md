# Failure-aware v3 Lighthouse verification result

**Official result: `resolved=false` (`QUALITY_FAIL`).** The separately frozen
`failure_aware_v3` episode completed all 12 action slots and stopped at the
action limit. The original official evaluator completed successfully, but its
`FAIL_TO_PASS` test remained failed. The retained evidence does not establish
the underlying cause of that official assertion failure.

The model made 12 proposals / 7 distinct. Seven were physically submitted:
one list, one search, one read, one successful edit, and three failed visible
tests. Five later proposals repeated the exact third failed test and were
rejected before physical execution, so there were zero known-failed physical
repeats. No `done` action was proposed or accepted. In particular, `done` was
never accepted while verification was required.

No visible verification succeeded after the edit. The first serial Jest run
timed out after reporting real passing-suite progress. The second used
`yarn jest --runInBand --offline` and failed because Jest did not recognize
`--offline`. The third serial Jest run also timed out. Its raw proposal
`yarn jest --runInBand` differed from the first raw proposal `yarn jest`, so
there were zero exact raw failed-test command repeats among physical
submissions. Default serial-argument injection made both effective commands
identical (`yarn jest --runInBand`), however, with no intervening edit. Across
the three failed physical tests there were therefore two distinct effective
commands and one effective failed-test command repeat. Final failure memory
recorded `verification_required=true`; the remaining five exact raw proposals
were rejected as `repeated_failed_action`. Thus there was neither a successful
visible verification nor an accepted `done`; the run exhausted its action
budget instead.

The patch remained the observed 481-byte redundant `id: lib.id` edit from the
preceding episode. The official result establishes only that the issue remained
unresolved, not that this semantic no-op caused the failed assertion. A visible
test pass, completion-constraint satisfaction, or nonempty patch would not by
itself meet the frozen quality gate.

The solver completed 12 model requests with 12,732 actual prefill tokens and
287 decode tokens. There were zero preemptions, allocation rejections, or cache
evictions; maximum simultaneous scheduler requests was one. GPU engine scope
was 94.725 seconds of the 1,170-second execution budget on one RTX 4090 D in
BF16. Solver end-to-end time was 97.381 seconds and wrapper terminal time was
129 seconds.

All owned compute resources were released. GPU
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` returned to 4 MiB and 0%
utilization with no compute process, and no owned model process remained.
Exact UUID-generated solver/evaluator container names
were not captured, and the wrapper signal is not treated as a container
identity. A read-only check at `2026-09-16T03:35:42Z` found the complete
`octopus-lighthouse-` container namespace empty. Verified model assets were
retained.

Raw evidence is preserved under `execution/`: `solver/episode.json`,
`solver/prediction.json`, `solver/grade.json`, the lossless
`solver/official.log.gz`, solver logs, start and terminal receipts. Frozen
execution source remains under `source-provenance/`. Prior accepted evidence is
retained at `../coder14-reference/download-terminal.json` and
`../coder14-reference/quality/actual-result.json`.

Evidence SHA-256 values:

- Freeze: `feb2c0c79fcb3fcd1d0ee7a3f34e9ad29ce2c6bf0574291aadaa9e5e228faa48`
- Episode: `442fa24c883abea284e92ca3a3745ef0f09eb55c903ae052b72368c1ac399ee4`
- Grade: `1bf37b1abd5960ce30849b68bf185e4e5180dc10781495fd965d06d39f208ca2`
- Prediction: `72156ef58251b31bfe81c8c58ddcf5cc012967184041d6b0353568a2420a2b70`
- Compressed official log:
  `de8e97bfff8647965b64a257afe113a322da8ce414331e75a372d074fecf8f36`
- Decompressed official log:
  `7edf70fd7fd5d093017df605134abfa23cfbb38db2e8c2856fb6d8e291181129`

The official quality failure ends this frozen episode. No additional episode,
outcome-driven runtime change, or multibranch launch is authorized.
