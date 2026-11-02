# Current-admission primary GPU screen: exposure improves, BG gate fails

All eight traces at native `807d46de552db10e274d2910e3676b1e62bd9934`
are complete, native-validated, live-exposure-instrumented and officially
scored. They use the unchanged 424 requests and 9,216 physical KV blocks
(8,455,716,864 bytes). Methods share a physical GPU within each seed.
Prelaunch mirrored orders are APC/queue/fixed/value for seed1601 on GPU0
and value/fixed/queue/APC for seed1702 on GPU3.

**The predeclared primary screen fails.** Summed future-only FREE exposure
decreases 6.987672%, and both positive matched savings fractions exceed 75%.
However, mean paired BG P95 versus fixed increases 1.057716%, rather than
decreasing. No second-pressure or formal expansion is authorized.

## Complete actual outcomes

All time columns are seconds. Official F1 covers all 424 predictions.

| Seed | Method | FG E2E | BG P95 | BG TTFT mean / P95 | Parent recompute tokens | Official F1 | Full-credit questions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1601 | APC | 241.421873 | 43.061419 | 8.511588 / 15.701378 | 33888 | 0.467277403 | 131 / 424 |
| 1601 | queue | 241.128870 | 43.207284 | 8.544737 / 15.422218 | 33888 | 0.467277403 | 131 / 424 |
| 1601 | fixed | 238.834847 | 43.459245 | 8.804149 / 15.935123 | 9648 | 0.467040306 | 131 / 424 |
| 1601 | value | 238.291442 | 43.184587 | 8.515328 / 15.318577 | 10912 | 0.467283642 | 131 / 424 |
| 1702 | APC | 249.775719 | 49.612249 | 13.407341 / 24.299519 | 33888 | 0.466782528 | 132 / 424 |
| 1702 | queue | 250.011022 | 49.924086 | 13.619652 / 24.557925 | 33888 | 0.466782528 | 132 / 424 |
| 1702 | fixed | 248.351527 | 49.387116 | 13.722540 / 24.900785 | 15616 | 0.467144816 | 132 / 424 |
| 1702 | value | 250.115847 | 50.743989 | 14.500744 / 26.219498 | 19568 | 0.467144816 | 132 / 424 |

| Seed | Fixed / value future-only block-seconds | Exposure change | Fixed / value saved tokens vs both controls | Preserved savings | BG P95 change vs fixed |
|---:|---:|---:|---:|---:|---:|
| 1601 | 388244.665427 / 374908.172858 | -3.435074% | 24240 / 22976 | 94.785479% | -0.631990% |
| 1702 | 316835.023495 / 280902.861522 | -11.340969% | 18272 / 14320 | 78.371278% | +2.747423% |

Total fixed/value exposure is 705,079.688922 / 655,811.034380 block-seconds.
Value binds in 11/952 and 15/953 draws. The recorded KEEP/DROP observations
are 18,130/1,996 and 22,909/2,367, respectively; these repeated node counts
are not unique retained blocks or block-seconds.

The additional FG work is 1,264 / 3,952 tokens, matching the conditional
diagnostic. All recurring-BG test requests still recompute 13,376 body tokens.
No BG-body work is recovered. The timing directions split by seed; these
two-seed point results are not formal confidence intervals.

Fixed/value output texts agree on 422/424 questions for seed1601 and 424/424
for seed1702. Full-credit counts are unchanged, but strict all-correct remains
false. This is observed official quality evidence, not a formal quality gate.

## R2-B distinction and required analysis

Aggregate saved-token/actual-exposure density declines from 0.062434857 to
0.061284340 for seed1601 and from 0.057670392 to 0.050978477 for seed1702.
These are **arm-level** quantities, not the requested KEEP-versus-DROP ranking.
Actual DROP benefit divided by its policy-protected exposure remains undefined
by construction; that null is not evidence that a matched-reference ranking
must also be undefined.

The required same-fixed-reference comparison has not yet been computed.
Before reducing per-group outcomes, the diagnostic analysis rule is now fixed:
one logical semantic body block within its original checkpoint/consumer
episode, labeled by its first genuine binding KEEP/DROP observation.
Later label flips are reported separately. Both groups use the same matched
fixed-arm exposure and fixed-versus-control actual adoption/recompute benefit;
repeated decisions cannot duplicate that reference.

Report per-unit mean and pooled reference density, group/parent mix,
unmatched/zero-denominator units and shared ancestors. Separately measure
each arm's actual positive exposure without demonstrable incremental savings,
including formerly protected keys later dropped, as both block-seconds and
an exposure-weighted rate. Unknown and closure-required ancestor strata remain
separate. This rule is specified after aggregate discovery outcomes but before
the group reduction; it is diagnostic, not independent confirmation or a
causal effect estimate for DROP.
Reference benefit already realized before the first binding label must be
partitioned explicitly. A whole-episode ratio is not strictly subsequent-use
evidence if that partition is nonzero; recorded remaining-consumer IDs may
support a separately identified future-benefit component, otherwise that
coverage limit remains unresolved.

The reducer completion cannot change the independently failed BG point gate.
Do not replace that gate or manufacture an aggregate-density substitute for
the requested reference ranking.

## CPU accounting does not establish a dominant hot-path cause

Measured serving-side counters, in seconds:

| Seed | Method | Adapter CPU | Selector CPU | Live observer CPU | Deferred audit CPU |
|---:|---|---:|---:|---:|---:|
| 1601 | fixed | 1.996952 | 0.228811 | 0.705194 | 1.501024 |
| 1601 | value | 2.171023 | 0.379303 | 0.656829 | 2.097336 |
| 1702 | fixed | 2.128974 | 0.277085 | 0.804056 | 1.178830 |
| 1702 | value | 2.137484 | 0.356868 | 0.848362 | 1.498721 |

On the failing seed1702, the value-minus-fixed selector excess is 0.079783
seconds and adapter excess 0.008510 seconds. These measurements do not
establish a dominant implementation bottleneck explaining the BG regression.
Deferred audit, raw serialization and offline reduction must not be counted
as BG serving delay. Timers can overlap; do not sum them as disjoint costs.
No unchanged-policy GPU rerun is justified solely by assuming a CPU cause.

## Terminal cost and artifacts

Cost is 2,254.496098 GPU process-seconds, with 97.481590% active utilization
across the two allocated cards. There are zero failed/running traces and zero
owned GPU processes. All eight raw files and eight sidecars are archived,
totaling 4,369,300,675 bytes.
Utilization here is process-active occupancy over allocated campaign time,
not a sampled GPU SM utilization measurement.

Full result, analysis, official scores, commands, compact/matched reports and
raw archive are in
`/home/ubuntu/experiments/octopus/r2-current-admission-gpu-20260912/`.
Policy/conditional evidence is in `../r2-current-admission/report.md`.
The earlier 16-trace result and its missing-exposure defect remain preserved.
