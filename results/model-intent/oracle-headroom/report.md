# Stage1A native oracle headroom

**PASS_TO_1B.** All24 oracle choices and native winner retentions passed. The
paired structural/oracle cohort contains48 actual native CPU scheduling episodes,
using the24 semantic S1 views without model inference or CUDA initialization.

| Quantity | Structural | Oracle |
|---|---:|---:|
| Candidate construction scheduled prefill | 12,288 | 12,288 |
| Pressure scheduled prefill | 6,144 | 6,144 |
| Selected continuation scheduled prefill | 3,096 | 24 |
| Total scheduled prefill | 21,528 | 18,456 |
| Full selected-prefix retention | 12/24 | 24/24 |

Total work fell3,072tokens, or14.2698%. Structural continuation work above the
mandatory one new token per view is3,072tokens, so the oracle realizes100% of
the theoretical available saving, exceeding the80% gate. Twelve already-hit
structural views have zero headroom and remain in the full denominator; their
individual saving ratio is not invented as a positive observation. Background
pressure work is unchanged, so this is not a foreground/background token transfer.

## Native boundary

The fixture uses real vLLM0.28 Scheduler.schedule, finish_requests, checkpoint
capture, native block allocation/hit lookup and public Octopus.apply_model_intents.
Only model execution is replaced by the existing NativeEngine CPU test fixture.
There are33total blocks including one null block; each distinct candidate has
256tokens/16blocks. The256-token pressure request leaves16blocks, enough for
exactly one candidate. Every arm records committed request metrics and scheduled
ranges, and asserts that their prefill totals reconcile.

These are **actual committed native scheduled-prefill counts**, not executed GPU
kernels, HBM tensor measurements or latency evidence. The fixture's request clock
is synthetic; per-request times in the raw metrics are not performance data.
The native metadata working set is deliberately controlled and equal-length.
The whole cohort took2.161579CPU wall seconds after imports/setup; this is not
the complete validation/preparation cost.

The queued-prefix-priority and held-owner/rejected-reset native regression probes
both passed. All arms ended with32free blocks and zero retention-only blocks.
The expected rejected-reset warning in the ownership probe is not a run failure.
The focused native suite passed4tests. An initial harness call to nonexistent
Octopus.gather was corrected to the existing step API before the cohort ran;
no model call or scientific input was changed.

## Reproduction and next gate

The implementation/fixture change and complete raw outcome are in commit46e6315;
the remote native integration base is01a98ce9 with the stage1A module, unchanged
semantic S1 source/fixture and the parameterized native scheduler test helper
overlaid. Native source/output remains at
`/home/user/octopus-oracle-headroom-20260914`; the exact table, view mappings,
draws, counters and checks are in [native.json](native.json).

This result authorizes the already-approved new64-decision direct-choice1B test.
It does not bypass1B, authorize unconditional training, or claim serving speedup.
