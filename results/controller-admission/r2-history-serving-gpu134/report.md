# Frozen next-reuse serving: complete, not accepted

**Two ten-trace matrices are complete, each with 4,240 official predictions.
The observer repair substantially reduces overhead, but the repaired
implementation still fails the R2 latency/protection criteria.**
The original fixed comparator also has a measured zero no-saving-body
exposure floor under the existing definition. R3/R4 remain unrun.

## Identity and execution

First-matrix native source: `95481254b87a8bf9b0ffe5fc84f511bc97ebbde0`; model/tokenizer
revision: `a09a35458c702b33eeacc393d103063234e8bc28`. The five frozen methods,
424 requests, intended arrivals, single-GPU topology and 9,865,003,008-byte
KV capacity are unchanged. Each seed's methods ran sequentially on the same
RTX 4090 D: seed1601 on GPU1 and seed1702 on GPU2 of `gpu.134`.
The two seed streams ran concurrently.

The source-host `ninja` PATH failure and relocated-model revision failure
remain separate invalid attempts with zero executed requests. The latter
was repaired by using the actual Hugging Face snapshot directory, not by
weakening the revision check. Four weight shards were the same inodes on the
source host and seven configuration/tokenizer files were byte-identical.
The target passed the native revision and all424 prompt-token checks before
inference. Neither policy nor checkpoint changed during migration.

Runtime and weights transferred directly between the hosts over campus IPv6.
The target launch occurred at 2026-09-13T18:41:12Z. Both original launchers
exited successfully, all10 compact exports completed, and both owned GPUs
returned to zero allocated MiB. GPU0 belongs to another task and was untouched.

## First matrix: actual serving and official quality

Times are seconds; protection is measured live future-only block-seconds.
Answer F1 is the official aggregate over all424 predictions in each trace.

| Seed | Method | FG E2E | BG P95 | Parent recompute tokens | Protection | Answer F1 | Full-credit questions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1601 | APC | 254.113 | 52.886 | 33,888 | 0 | 0.467277 | 131 |
| 1601 | Queue | 257.705 | 54.848 | 33,888 | 0 | 0.467277 | 131 |
| 1601 | Semantic fixed | 266.680 | 65.337 | 0 | 459,701.372 | 0.467040 | 131 |
| 1601 | Next-reuse value | 297.919 | 95.124 | 0 | 500,787.888 | 0.467284 | 131 |
| 1601 | History blanket | 293.722 | 89.574 | 2,304 | 654,749.604 | 0.467047 | 131 |
| 1702 | APC | 249.907 | 49.395 | 33,888 | 0 | 0.466783 | 132 |
| 1702 | Queue | 251.341 | 50.131 | 33,888 | 0 | 0.466783 | 132 |
| 1702 | Semantic fixed | 263.588 | 63.512 | 0 | 461,413.514 | 0.467145 | 132 |
| 1702 | Next-reuse value | 291.878 | 90.266 | 4,416 | 472,915.384 | 0.466971 | 131 |
| 1702 | History blanket | 291.555 | 88.577 | 4,416 | 649,201.915 | 0.466971 | 131 |

All10 traces have zero missing predictions. Candidate mean Answer F1 across
the two seeds is0.467127, versus0.467030 for APC/queue, but its full-credit
total is262 versus263. Thus the observed full-credit non-regression criterion
is not met; a slightly higher aggregate F1 does not waive it. This is discovery
quality evidence, not a four-seed confirmation or an all-correct result.
Gold and the official evaluator remained on the separate CPU host.

Candidate BG mean TTFT is42.956/40.666, versus APC14.255/13.277 and
semantic-fixed23.336/22.352. No TTFT improvement is established.

## What the comparison establishes

1. Candidate FG E2E is17.24%/16.79% slower than APC and15.60%/16.13%
   slower than queue. BG P95 is79.87%/82.74% worse than APC and45.59%/42.12%
   worse than semantic fixed. Across these two discovery seeds, the descriptive
   mean and sample SD are17.02% +/-0.31 percentage points for FG versus APC,
   and81.30% +/-2.04 percentage points for BG. These are not confidence bounds.
2. Candidate retains100%/86.97% of semantic fixed's realized FG savings,
   exceeding75%, but protection rises8.94%/2.49% versus that original fixed
   control. Protection falls23.51%/27.15% only versus history blanket.
   That alternative comparator cannot replace the original requirement.
3. Both candidate traces record2,955 initial eligible labels, allKEEP.
   There is no initial DROP group supporting an initial-ranking claim.
   Live candidate-state episode counts are20,696/21,830. Their full prospective
   reference associations remain separate from these aggregate point results;
   conditional shared-event ordinals must not be reused across serving clocks.

## Measured observer overhead

| Seed | Method | Serving-loop CPU seconds | Live-observer CPU seconds |
|---:|---|---:|---:|
| 1601 | APC | 14.614 | 0.103 |
| 1601 | Queue | 15.529 | 0.656 |
| 1601 | Semantic fixed | 33.446 | 21.161 |
| 1601 | Next-reuse value | 73.342 | 58.742 |
| 1601 | History blanket | 68.983 | 53.859 |
| 1702 | APC | 14.625 | 0.104 |
| 1702 | Queue | 14.827 | 0.627 |
| 1702 | Semantic fixed | 31.688 | 20.203 |
| 1702 | Next-reuse value | 67.282 | 53.756 |
| 1702 | History blanket | 68.243 | 53.582 |

Candidate selector CPU is only1.451/1.515 seconds, and history maintenance
is0.108/0.103. The observer counter is about80% of candidate serving-loop
CPU. Adapter and observer accounting overlap; they must not be summed.
These counters identify the next profiling target, not a causal estimate of
the latency recoverable by subtraction.

These measurements motivated an observation-only implementation repair.
The original measurements remain valid; observation was not disabled and
none of these wall times were rescaled.

## Observer-only repair and same-protocol rerun

Library commit `f4ff41427ff83e6245002f42f8401d6cc99375a6` reuses unchanged
candidate-state and history-support evidence. Native commit
`6108c2c62226918d0a65c95f758f0767027b9c5d` contains that same repair.
The policy, all request/arrival values, model, capacities and method ordering
were unchanged. Both full input files were compared after restoring only
their previous source-commit field.

The same recorded candidate seed1601 tape measured observer CPU
81.508 to11.809 seconds, an85.5% reduction; total replay CPU was92.195 to21.936.
Twenty-six native tests passed and21 full non-CPU export comparisons were
exact. This covered the recorded tape and five-arm live CPU fixtures, not
an equivalence claim about historical GPU wall clocks. In particular, the
replay exposure500,897.311339 must not replace live exposure500,787.888.

The authorized same-protocol rerun started2026-09-13T20:01:23Z on the same
GPUs1/2. It continued independently through the Herdr maintenance window and
finished naturally: no experiment was stopped or restarted for maintenance.
All10 traces and10 compact exports passed native validation.

| Seed | Method | FG E2E | BG P95 | Parent recompute tokens | Protection |
|---:|---|---:|---:|---:|---:|
| 1601 | APC | 254.551 | 52.434 | 33,888 | 0 |
| 1601 | Queue | 255.496 | 53.681 | 33,888 | 0 |
| 1601 | Semantic fixed | 254.553 | 55.720 | 0 | 460,684.344 |
| 1601 | Next-reuse value | 266.572 | 64.285 | 0 | 503,154.771 |
| 1601 | History blanket | 266.917 | 64.319 | 2,304 | 638,319.425 |
| 1702 | APC | 252.436 | 50.727 | 33,888 | 0 |
| 1702 | Queue | 251.481 | 50.160 | 33,888 | 0 |
| 1702 | Semantic fixed | 248.585 | 50.493 | 0 | 460,565.741 |
| 1702 | Next-reuse value | 260.900 | 60.317 | 4,416 | 467,213.796 |
| 1702 | History blanket | 265.280 | 64.130 | 4,416 | 628,270.439 |

Candidate FG is still4.72%/3.35% slower and BG P95 still22.60%/18.91%
worse than its same-run APC controls. The descriptive two-seed mean/sample
SD are4.04% +/-0.97 percentage points and20.75% +/-2.61 percentage points.
These are not confidence intervals. Candidate protection remains9.22%/1.44%
higher than semantic fixed, despite being21.18%/25.63% lower than history
blanket. Savings remain100%/86.97% of fixed, above75%.

All4,240 prediction texts are identical to their first-matrix counterparts.
The official evaluator was rerun on the complete predictions, with zero
missing results. The original F1/full-credit values above therefore remain
unchanged: candidate262 versus APC/queue263 full-credit predictions.
The repeated implementations must not be pooled as additional independent seeds.

The actual live observer now costs17.798/16.121 CPU seconds, versus total
serving-loop33.813/30.639. Semantic-fixed observer CPU is3.401/3.254 and
queue is0.697/0.655. The residual observer cost is still substantial; the
recorded replay's85.5% reduction was not a promised GPU-wall improvement.
A bounded CPU-only residual diagnosis is separate from scientific acceptance
and authorizes no additional GPU run.

## Original-fixed reference floor

The existing `r2_reference_density` consumer contract, native adoption proof
and live-exposure partition were applied to the repaired fixed/APC/queue
traces. This is the existing whole-declared-consumer-episode definition,
not a newly chosen interval or denominator.

| Seed | Exposed body units with positive saving | Known body saving tokens | Zero/negative/unknown body exposure | Separate mandatory ancestor exposure |
|---:|---:|---:|---:|---:|
| 1601 | 2,118 | 33,888 | 0 / 0 / 0 block-seconds | 11.269572 block-seconds |
| 1702 | 2,118 | 33,888 | 0 / 0 / 0 block-seconds | 11.294380 block-seconds |

The findings agree against both APC and queue, and total mapped/separate
exposure reconciles exactly with the live scalar. Full results are in
[`fixed-reference-floor.json`](fixed-reference-floor.json).

The requested strict decline in this no-saving-body criterion cannot be
established below its original-fixed zero floor on this cohort. This does
not prove every preceding protected second causally necessary, nor an
impossibility for other workloads or an information-matched history control.
Those would be different comparisons. Do not manufacture improvement by
relabeling earlier benefit, clipping it to FREE-state intervals, treating
mandatory ancestors as removable waste, or silently changing the definition.
This comparison limit and the remaining latency/protection regressions keep
the full R2 claim unaccepted.

## Durable evidence

Remote root: `/home/user/octopus-r2-20260913/attempt003` on `gpu.134`.
`serving-raw/` holds9,946.02 MiB and `serving-compact/` holds2,207.09 MiB.
`terminal/seed-{1601,1702}.json` records exact commands, exits and timestamps;
`analysis/summary.json` contains scalar measurements and paired comparisons.
The immutable native bundle is retained under `../integration/`.

Retrieved summary, minimal prediction/timing input and official score report:
`/home/ubuntu/experiments/octopus/r2-history-serving-20260913/gpu134-attempt003/`.
The full31,903,277-byte prediction/metadata projection remains remote;
the474,363-byte scoring projection preserves the same complete predictions.
The incomplete local transfer of the larger projection was removed.

The existing `benchmarks.controller_quality` scored all10 traces using the
retained official evaluator and gold, checking aggregate/per-question
agreement and binding scores to exact prediction text.
Trace-process wall intervals total3,342.608 seconds across a1,683.461-second
paired window:99.28% occupancy of the two reserved process slots.
This is not average GPU-kernel utilization. The in-run GPU snapshot showed
both owned cards at100% utilization with25,984 MiB each.

Repaired-run artifacts are under the sibling remote `attempt004/`, with
retrieved summary, minimal prediction input, official scores and a complete
native-source bundle at the sibling local `gpu134-attempt004/`.
Both original repaired-run launchers finished naturally, and both assigned
GPUs were observed with zero allocated MiB after completion. No new GPU
work was launched merely to fill capacity.
