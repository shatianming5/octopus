# Stable soft admission: extra FG loss removed, useful gain not established

The CPU-only lifetime refinement is complete at library
`53ec5a2e1a698042800e824d6ffd92896df57903`, native snapshot
`5b1d8aef6c28ffe16a08d08e4fa4d82cd254011e`. It eliminates the previous
1,264 / 3,952 additional FG recompute tokens relative to actual fixed.
However, the pooled conditional protection reduction is only
35.604030 block-seconds, or 0.005052633% of the same-state fixed shadow,
with no BG prefill recovery or FG work improvement beyond fixed.
This is not a demonstrated useful R2 gain or a GPU-readiness result.
No new GPU run, pressure expansion or formal run is authorized by this result.

## Rule and native mechanism

The native red case has four online KEEP copies. A later forecast cap of zero
previously withdrew all four despite 14 copies of actual native draw room
and an unchanged semantic promise.

Only a genuine-binding online KEEP now creates a logical leading-prefix grant
for its live checkpoint/consumer promise. Forecast or physical-copy changes
cannot revoke that grant. Nonbinding selection remains exactly fixed and
does not mint grants. Ungranted candidates may compete for the remaining
forecast room after valid grants. A newly sharing promise receives its own
logical grant without charging the shared physical copy again.

Grants remain soft preferences, not refs, owners, leases or hard reservations.
Queued/native priority and mandatory eviction still win. Logical grants map
only to currently cached, prefix-closed copies; missing ancestors and reused
physical IDs cannot confer protection. Completed, cancelled or closed future
promises and queue/native handoff cease future applicability; successful
cache reset clears grants. Resolve events that preserve an existing B6 future
obligation do not prematurely cancel it. Existing node/key snapshots are
reused, with no reranking of resolved/no-room sets.

Forecast excess is explicit rather than incorrectly asserting the old total
forecast cap. No measured reference labels, producer IDs, group sizes,
future outcomes or quality data enter the online rule.

## Two-stream conditional result

Both streams use the actual `807d46de` allocation opportunities, outputs and
release times. Work savings compare with their matched actual fixed and both
APC/queue controls.

| Metric | Seed1601 | Seed1702 |
|---|---:|---:|
| Stable parent recompute savings | 24,240 | 18,272 |
| Actual fixed parent recompute savings | 24,240 | 18,272 |
| Extra parent recompute versus fixed | 0 | 0 |
| BG prompt work change versus fixed | 0 | 0 |
| Recurring BG body recompute | 13,376 | 13,376 |
| Same-state fixed shadow, block-seconds | 388,237.164617 | 316,425.706765 |
| Stable conditional protection, block-seconds | 388,237.164617 | 316,390.102735 |
| Postdecision omitted protection, block-seconds | 0 | 35.604030 |
| Cross-step unpreferred occurrences | 0 | 38 |
| Same-prestate native victim-difference draws | 1 | 3 |
| Stable-only / fixed-shadow-only selected occurrences | 34 / 34 | 76 / 76 |
| Ranking draws | 1 | 1 |
| New admission copy occurrences | 2,084 | 2,042 |
| Reused grant copy occurrences | 1,316,657 | 696,861 |
| Forecast-excess draws / maximum blocks | 13 / 477 | 14 / 592 |
| Native forced-grant eviction occurrences | 569 | 900 |

The 38 cross-step occurrences in seed1702 survive the scheduling-wave end,
the boundary before the next allocation and the boundary after it.
All omitted protection has positive current commitment and zero positive
fixed-reference incremental saving versus both controls; unknown/ancestor
omitted mass is zero. That classification is computed only after policy
replay. Already-sunk exposure removed is zero.

The pooled fixed shadow is 704,662.871382 block-seconds. The reduction is
0.005052633% pooled and 0.011251940% within seed1702, with no reduction in
seed1601. These are small, real conditional distinctions, not idle-only
labels, but their magnitude and unchanged work do not establish the useful
protection/background improvement needed to justify another serving run.
This screen does not prove that GPU timings must be identical.

## Measurement boundary

Stable and blanket-fixed protection views share the same stable-counterfactual
native state at recorded event endpoints, collapsing intra-event time.
The fixed view is a read-only shadow, not a separately evolving fixed replay.
Exact victim comparison uses the existing QueueRetention selector on that
same pre-draw native queue with all eligible semantic keys. No physical IDs
are joined across runs.

Consequently these block-seconds are neither an actual counterfactual GPU
live integral nor recovered historical live exposure. The post-run reference
association is not a causal DROP effect, and does not replace prospective
R2-B serving evidence. The previous actual `807d46de` BG and waste failures
remain unchanged.

## Validation, cost and artifacts

The native lifetime/protection gate passes eight cases; the additional exact
same-state victim-shadow gate passes one case. Coverage includes forecast-only
changes, genuine forced eviction, queue precedence, different-copy recache,
shared/held ancestors, semantic lifecycle/reset, nonbinding parity, omitted
protection and exclusion of sunk exposure. Original red and corrected fixture
failures remain in the evidence.

The previously unrun existing changed-surface regressions also pass on the
same native source: seven value-admission cases, two future-prefix lifecycle
cases and 16 native-selector/exact-audit cases, 25 passed in 6.42 seconds.
They ran once, without repeating the nine new cases, changing source or
launching another replay. Fourteen upstream deprecation warnings remain.
`regression-supplement.json` records exact selectors and source identity;
`evidence/existing-regression-supplement-tests.txt` retains the log.

There are two distinct value streams and three conditional executions.
The preliminary1601 execution lacked exact same-state victim evidence; a
tested read-only shadow was added without changing policy, and that stream
was rerun. Its original output remains preserved. Processing CPU cost is
14.346425 + 15.471089 + 14.683349 = 44.500862 seconds. Serialization costs
have separate sidecars. These are offline costs, not serving CPU or GPU time.
There were zero GPU runs and no all-16 replay.

Full local root:
`/home/ubuntu/experiments/octopus/r2-stable-admission-20260912-d3c687/`.
The matching remote root is
`/mnt/disk2/user/octopus-contention/r2-stable-admission-20260912-d3c687/`.
The terminal report is `result.json`; final outputs are
`evidence/stable-1601-victim-audited.json` and `evidence/stable-1702.json`.
The preliminary `evidence/stable-1601.json`, native test logs,
`evidence/stable-admission-source.bundle` and serialization sidecars are
retained there.

## Parent decision

Accept the implementation/mechanism and terminal CPU evidence, not the full
R2 claim. The work-preservation screen passes, but useful admission improvement
over fixed is not established. No supported next policy variant or ready
GPU matrix follows from these results; do not invent one to keep running.
R1 remains passed, R2 remains unaccepted, and R3/R4 remain unperformed.
The original full R1--R4 objective is incomplete.
