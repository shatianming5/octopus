# Explicit-regex Lighthouse quality episode

The single frozen `search_pattern_syntax:regex_v1` episode completed the
original official evaluator and **failed the quality gate**:
`official_resolved=false`. The solver stopped at the 12-action limit with no
edit, patch, test, or `done` action. The official `FAIL_TO_PASS` result remained
failed for
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`. Only
`official_resolved=true` passes quality; this negative outcome closes this
episode without closing the parent pipeline.

| Metric | Explicit regex | Closed directory discovery |
|---|---:|---:|
| Raw proposals / distinct | 12 / 4 | 12 / 5 |
| Physical submissions / failures / effective failed repeats | 12 / 0 / 0 | 12 / 1 / 0 |
| Byte-exact nonempty outputs, total / unique | 6 / 2 | 2 / 2 |
| Edit / test / done actions | 0 / 0 / 0 | 0 / 0 / 0 |
| Patch bytes | 0 | 0 |
| Native prefill / decode tokens | 17,723 / 1,485 | 12,076 / 1,197 |
| GPU-engine / solver E2E seconds | 77.815 / 80.766 | 68.512 / 71.030 |
| Cache evictions / max scheduler concurrency | 0 / 1 | 0 / 1 |
| Official resolved | false | false |

The executor's canonical `action_hash` is used for physical identity and failed
repeat accounting; hashes of the original parsed actions are reported
separately. Nonempty evidence is counted by byte-exact output hash. The six
nonempty observations were one root listing plus the same file content returned
five times, yielding only two unique outputs.

Regex mode was enabled, but **nonempty regex-search evidence was not obtained**:
all six search actions returned zero bytes, and the model submitted no pattern
using a regex operator. The extra nonempty-output count over directory discovery
therefore reflects repeated reads, not improved search evidence. Journal
deduplication rose to seven entries, while ordinary budget drops and omitted
optional evidence remained zero.

The episode retained all 12 action, raw-model-text, and reasoning records.
Native accounting recorded 12 completed requests, 17,723 prefill tokens, 1,485
decode tokens, no preemption, no allocation rejection, no eviction, and maximum
scheduler concurrency one. The selected GPU returned to 2 MiB and 0%
utilization, with no selected-device compute process. The local
`octopus-lighthouse-` container namespace is empty. Random solver/evaluator
container names were not persisted in the run logs and could not be recovered
after successful removal. An active `nvidia-smi` sample was not captured before
the short episode ended; actual GPU execution is established by the completed
native requests, engine accounting, and 77.815-second GPU-engine scope rather
than a retroactive utilization claim.
