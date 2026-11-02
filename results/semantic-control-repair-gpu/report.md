# Minimal semantic GPU confirmation

Status: **complete**. 3/3 diagnostic traces; 12/12 tasks correct; 288 background requests.

`semantic_soft` **passes the fixed 5% P95 TTFT point and one-sided-bound gate on this one diagnostic block**. This is not evidence of general SLO safety or a statistically established foreground speedup.

| Method | Foreground mean E2E s | Change vs APC | Background P95 TTFT s | P95 change | One-sided upper | Gate | Trunk recompute tokens |
|---|---:|---:|---:|---:|---:|---|---:|
| apc | 17.676 | +0.00% | 8.291 | +0.00% | +0.00% | pass | 17392 |
| queue_aware | 17.623 | -0.30% | 8.358 | +0.81% | +1.25% | pass | 14320 |
| semantic_soft | 17.636 | -0.23% | 8.324 | +0.40% | +1.70% | pass | 14304 |

Semantic minus queue-aware foreground E2E: **+0.012697 s**. No foreground confidence interval is inferred from one seed or from four serially dependent tasks.

## What was exercised

The new real-GPU consumer-binding/preemption/resume test passed (1 test, zero failures/errors/skips). It checked no checkpoint lease, no retention-only blocks, completion after actual preemption, and final owner cleanup.
Across the measured semantic trace, retention-only sampled capacity and retention-owner block time were zero.
There were 526 future evaluations, 4 positive evaluations over 2 checkpoint identities, and 2 allocation snapshots with nonempty future preference. The future path participated, but only sparsely. Reasons: `{"native_request_consumer": 96, "no_live_gpu_consumer": 23, "positive_apc_loss_proxy": 4, "unknown_adoption_estimate": 403}`.

Observed queue-aware minus semantic trunk recompute: 16 tokens in this block. That is not an independently identified causal semantic savings estimate. Preferred blocks reclaimed by allocation are not protected-hit counts.

## Protocol and measurement boundaries

Runtime policy source: `fd582d0126958aa1a056e65684d0e31d7f7f1276`. Qwen2.5-7B-Instruct and the existing independent matching-GPU6 cost/real-tool priors were reused. Physical GPU: `GPU-aa7be32a-43d1-b13e-45dd-ef46033f7631`. Fresh process/engine and common warmup/cache reset for each method; predeclared order was APC, queue-aware, semantic-soft. This single block cannot counterbalance method order.
All methods used the same intended arrivals, task identities, fixed 96-request background cohort and physical engine settings. P95 is computed from intended arrival. The one-sided 95% bound uses the existing paired circular moving-block bootstrap (block size 8, 1,000 resamples, seed 17). It describes this fixed trace, not workload-wide uncertainty.
Seed 101 was used for diagnosis/design and is not held-out. No extra seeds, sweep, model or predictor fitting were performed. Old 81-trace results remain unchanged and are not a contemporaneous comparator.
New allocation/adoption/preferences are recorded in full for semantic-soft; the older baselines do not have equivalent full allocation logging. The observed end-to-end result includes that method-path overhead; it does not isolate the policy from instrumentation. Adapter/selector/provider timers overlap and cannot be added; final JSON serialization is outside the trace timers.

## Resources and reproduction

The logged lifecycle-plus-three-trace chain took approximately 221 s (second-resolution log, including startup). Summed in-trace makespan was 143.373 s, not GPU kernel time. The post-run cleanup record contains the device state and confirms no visible commands matching this verification's script, test or output selectors.
The raw as-run script is archival; do not rerun it in place over these results. Recompute this report without GPU work:

```bash
PYTHONPATH=src .venv/bin/python -m benchmarks.semantic_gpu_report
```
