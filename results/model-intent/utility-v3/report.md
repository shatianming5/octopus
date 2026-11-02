# Utility-first v3: native value exists; online choice overhead fails utility

**Terminal decision: `E2E_UTILITY_NO_GO`.** Frozen-output replay passed and the
merged-LoRA/native integration passed4/4. All six authorized new-workload GPU
traces completed, but both seeds missed the net-work and latency gates. Formal
confirmation, additional seeds and additional training were therefore **not run**.
PR22's historical `LEARNED_7B_ROUTE_NO_GO` is unchanged.

## 1. Frozen-output native replay: pass

All128 stored PR22 responses, including the invalid-ID response, were replayed
through384 real native CPU allocator arms. No model execution or CUDA context
was used. The invalid answer remained an explicit no-advisory case, not repaired
or excluded. Queued-priority, held ownership and complete owner release passed.

| Admission signal | Candidate prefill | Pressure prefill | Continuation prefill | Total prefill |
|---|---:|---:|---:|---:|
| Structural | 65,536 | 32,768 | 16,512 | 114,816 |
| Frozen learned choice | 65,536 | 32,768 | 4,224 | 102,528 |
| Oracle | 65,536 | 32,768 | 128 | 98,432 |

Learned intent saved12,288tokens: **75% of16,384oracle-available tokens** and
**10.7023% of total scheduled prefill**. Construction and pressure were identical.
Every valid learned choice retained its chosen native prefix. This establishes
mechanism utility of the imperfect stored signal, not new held-out accuracy or
GPU latency. Raw evidence: [native.json](replay/native.json), [terminal](replay/terminal.json).

## 2. Same-weight native integration: pass

The exact PR22 final adapter was merged once, without training, into Qwen2.5-7B
revision`a09a35458c702b33eeacc393d103063234e8bc28`. Adapter SHA256:
`afbaddcb5387cbd084c58efeb596b0be0091664ebf75445b8bd038d8c63cc4d3`.
All later arms used this same merged path/weights, bf16 and the same physical
RTX4090D GPU (`3e940a5a-a24d-b615-dc69-df141ced7af0`).

All four new cases produced one valid choice, traversed the public bridge and
retained all2048tokens of the chosen prefix under real native pressure. Choice
accuracy was4/4; the separate continuation-answer quality was only2/4. The smoke
gate is pipeline functionality, **not** proof of task-quality readiness.
Evidence: [trace](integration/smoke/trace.json), [merge](integration/merge.json).

Before any new model/GPU outcome, an explicit [integration amendment](integration-freeze.json)
corrected the original control order: choose using short logical branch views
before materializing two prefixes that already fill the cache. Otherwise the
choice request itself would evict a candidate before advisory application.
The original [freeze](freeze.json) is preserved. Families, seeds, weights,
capacity, token lengths and thresholds were not changed. All choice work and
delay count against both FG completion and waiting BG pressure; this is not
free preprocessing. See the [protocol](../../../docs/utility-first-v3.md).

## 3. Six paired GPU traces: valid no-go

Seeds5101/5202, three arms each,12episodes per trace: six new semantic operation
families with opposed goals. All72episodes completed. Each pair had identical
candidate token identities, generated branch outputs, pressure/construction
work, merged weights and physical GPU. Both seeds retained exactly one complete
candidate under pressure; no checkpoint owners remained after completion.

| Required metric | Gate | Seed5101 | Seed5202 |
|---|---:|---:|---:|
| Model selection | >=80% | 11/12 (91.67%) | 11/12 (91.67%) |
| Structural -> model total prefill | reduction>=10% | 86,150 ->81,772 (5.082%) | 86,150 ->81,764 (5.091%) |
| Net oracle-available saving | >=50% | 35.909% | 35.974% |
| BG prefill / scheduled decode | no increase | 24,576 /0, unchanged | 24,576 /0, unchanged |
| FG mean E2E vs structural | <=0% change | +35.512% | +33.400% |
| BG P95 vs APC | <=+5% | +103.946% | +102.142% |
| Continuation quality | nonregression | 3/12 for every arm | 3/12 for every arm |

The physical mapping worked: each seed saved10,160continuation-prefill tokens,
**83.33% of the12,192available**. However, one-shot choice consumed5,782/5,774
additional prefill tokens and270/266scheduled decode tokens. Net saving fell to
4,378/4,386tokens, and critical-path model/control delay increased both latency
metrics. This is an observed **model-call/control-overhead failure**, not a
reason to retrain the adapter or tune native eviction on these same outcomes.
The16-token shared header explains why available GPU saving is12,192rather than
12,288; the denominator comes from actual committed native ranges.

Quality did not merely tie in aggregate: all12continuation texts were identical
across the three arms within each seed. Nevertheless **25% absolute quality is
poor**. This study does not demonstrate useful production-level task quality,
generalized serving advantage or a satisfactory code-execution assistant.
No claim of benefit at high quality is supported.

The workload is deliberately staged: candidate materialization and pressure
wait behind model choice, with that wait included in BG latency. It is not a
concurrent natural-agent serving benchmark. There are two independent paired
seeds, not72independent observations; no formal confidence interval is claimed.
Full decision and role-attributed work: [score](e2e/score.json). Complete raw
traces, one-shot answers, environments and per-request ranges are in`e2e/`.

## Cost and stop

CPU-native replay:25seconds; CPU merge plus GPU integration assignment:67seconds
(merge function14.20seconds); six GPU traces including startup:205seconds.
These are process/job wall times, not integrated GPU-busy measurements. The one
in-run GPU sample was79% utilization; terminal verification showed4MiB,0% and
no remaining compute process. No GPU reservation or experiment job remains.

The four formal seeds and both formal regions were frozen but never queried.
No answer was retried and no new LoRA was trained.
The authorized conditional route is complete at this valid negative gate.
