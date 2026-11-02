# Runtime-owned observation state

**Closed: original official `resolved=false`.** All twelve actual tool outcomes
were committed, and all eleven following capsules contained the exact preceding
output. The model still repeated successful inspection and produced no patch.
The preceding task-state episode emitted no citations. This follow-up removes that
dependency on model cooperation without changing root instructions, the model
decision schema, action validity, source/query hints, or the output/action budget.

With `observation_memory=latest_v1`, every delivered tool or controller result
immediately becomes an immutable reducer fact containing the actual action,
result metadata and complete delivered output. A failed/rejected outcome stays
explicitly failed inside the fact; successful commit application means valid
recording, not successful execution or task resolution. The next capsule
requires the latest observation plus any model-cited historical evidence.
Historical observations remain in reducer history and raw audit, rather than
being overwritten or all forced into every bounded model input.

The [CPU replay](cpu-replay.json) reuses every actual model decision and tool
observation from the closed task-state episode, including all twelve empty
citation arrays. It produces twelve observation commits and eleven subsequent
capsules containing the exact preceding output. All capsule token segments
roundtrip with the pinned tokenizer; maximum input length is 5,196 tokens.
No synthetic quote, plan, action, or model output is substituted. The generic
repair fixture covers both modes and preserves citation/verification rejection.
These are dataflow facts, not a quality or behavioral-improvement result.

## Actual same-task outcome

The [terminal report](terminal-report.md) records one list, one read and ten
searches: 12 proposals / 3 distinct, 12 successful physical tools, 9 successful
inspection repeats, no rejection, edit, test, done or patch. Model citations
remained empty, but runtime observation commits and history facts were 12/12.
All twelve actual capsule token segments roundtripped with the pinned tokenizer;
all eleven subsequent capsules preserved their exact preceding tool output.
The initial request token IDs were identical to the prior task-state episode.

Native work was 23,742 prefill / 1,418 decode tokens; engine scope was 75.745
seconds and wrapper wall was 108 seconds. One scheduler request was active at
a time, with no eviction. An engine-resident memory sample was captured, not a
reliable active-compute utilization measure. Compute and actual containers were
released. Missing observation transport/persistence is therefore not a sufficient
explanation for the loop under this setup; no global model/KV claim follows.

The first supervisor admission failed before any freeze, deployment or job.
After proving its exact receiver and CLI absent and its preparation read-only,
one startup recovery admitted the actual execution owner. This was one model
episode, not a repeated quality attempt. The old failed-admission record is
retained for lifecycle reconciliation; it owns no live compute. The original
quality, multibranch and formal objective remains unachieved.
