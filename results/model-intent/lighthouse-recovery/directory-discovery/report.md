# Directory-discovery Lighthouse quality episode

The single frozen `directory_search_format:matching_files_v1` episode completed
the original official evaluator and **failed the quality gate**:
`official_resolved=false`. The solver stopped at the 12-action limit with no
edit, no patch, no verification action, and no `done` action. The official
`FAIL_TO_PASS` result remained failed for
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`. This is a valid
negative result and ends this episode; only `official_resolved=true` would pass.

| Metric | Directory discovery | Closed journal compaction |
|---|---:|---:|
| Raw proposals / distinct | 12 / 5 | 12 / 7 |
| Physical submissions / controller rejections | 12 / 0 | 11 / 1 |
| Physical failures / effective failed repeats | 1 / 0 | 1 / 0 |
| Byte-exact nonempty evidence, total / unique | 2 / 2 | 8 / 5 |
| Edit / verification / done actions | 0 / 0 / 0 | 0 / 0 / 0 |
| Patch bytes | 0 | 0 |
| Native prefill / decode tokens | 12,076 / 1,197 | 41,828 / 1,497 |
| GPU-engine / solver E2E seconds | 68.512 / 71.030 | 82.751 / 84.885 |
| Official resolved | false | false |

All 12 action, raw model text, and reasoning records are retained. Raw hashes in
`metrics.json` are computed from each original parsed action. Physical failure
and repeat accounting uses the executor's canonical `action_hash`, not the
journal representation. The one physical failure was a missing explicit path
on turn 7 and was not physically repeated. The two unique evidence items are
the byte-exact outputs from turns 1 and 6; empty successful search results do
not count as nonempty evidence.

Native accounting records 12 completed requests, 12,076 actual prefill tokens,
1,197 actual decode tokens, zero cache evictions, zero allocation rejections,
and maximum scheduler concurrency one. Request records are accounting records,
not additional native work. The terminal pool had all 4,787 blocks free, with
zero active or request-owned blocks.

Relative to the closed journal-compaction episode, this one used 29,752 fewer
prefill tokens and finished the GPU-engine scope 14.239 seconds earlier, but
both episodes were officially unresolved. These paired single-episode
measurements are descriptive only and do **not** support a general speedup or
performance claim. The private official assertion cause is not inferred beyond
the observable failed test.
