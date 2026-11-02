# R2 physical-pressure discovery: real decisions, no demonstrated value gain

All 16 traces are native-validated and complete at source
`3e7cf232f8aa254fd63712d9d75fa6fb486cf1be`: four methods, two physical KV
capacities and discovery seeds 1601/1702, with the unchanged 424-request input.
Methods are paired on the same physical GPU within each seed/pressure group.
The two regions used different GPUs; absolute between-region latencies are
not a paired causal estimate of capacity.

**R2 is not accepted.** The lower capacity produces real KEEP/DROP decisions
and preserves all positive matched fixed savings. It does not demonstrate
less background interference or extra recurring-BG reuse. Required live FREE
exposure was omitted from the launch freeze, so its value and the R2-B
reference-density/false-retention comparison are unavailable, not zero.
R3/R4 have not run.

## Complete recorded outcomes

All times are seconds. FG is complete-paper mean E2E; BG uses all 412 requests.
TTFT lists mean / P95. Official F1 is scored on all 424 predictions.

| Blocks | Seed | Method | FG E2E | BG P95 | BG TTFT mean / P95 | FG recompute tokens | Official F1 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 9216 | 1601 | APC | 240.832401 | 42.727984 | 8.166808 / 15.120232 | 33888 | 0.467277403 |
| 9216 | 1601 | queue | 240.760037 | 43.021816 | 8.375225 / 15.080049 | 33888 | 0.467277403 |
| 9216 | 1601 | fixed | 238.201977 | 43.129491 | 8.482990 / 15.314328 | 9648 | 0.467040306 |
| 9216 | 1601 | value | 237.849948 | 42.994768 | 8.327248 / 15.004822 | 9648 | 0.467040306 |
| 9216 | 1702 | APC | 245.162694 | 46.419769 | 11.240572 / 20.198615 | 33888 | 0.466782528 |
| 9216 | 1702 | queue | 244.797812 | 46.127298 | 10.986072 / 19.854778 | 33888 | 0.466782528 |
| 9216 | 1702 | fixed | 242.524384 | 46.233829 | 11.103376 / 19.766373 | 15616 | 0.467144816 |
| 9216 | 1702 | value | 243.180938 | 46.551869 | 11.343466 / 20.382003 | 15616 | 0.467144816 |
| 10752 | 1601 | APC | 253.474267 | 52.116425 | 13.584651 / 26.693239 | 33888 | 0.467277403 |
| 10752 | 1601 | queue | 249.133737 | 48.635608 | 12.358008 / 22.871857 | 33888 | 0.467277403 |
| 10752 | 1601 | fixed | 247.722712 | 49.849888 | 12.764115 / 25.146485 | 0 | 0.467040306 |
| 10752 | 1601 | value | 246.481502 | 49.129135 | 12.663949 / 24.132826 | 0 | 0.467040306 |
| 10752 | 1702 | APC | 256.781152 | 53.686563 | 15.840519 / 30.316773 | 33888 | 0.466782528 |
| 10752 | 1702 | queue | 257.782641 | 54.502521 | 16.347876 / 31.272254 | 33888 | 0.466782528 |
| 10752 | 1702 | fixed | 250.555223 | 51.899898 | 15.375700 / 28.772972 | 0 | 0.467144816 |
| 10752 | 1702 | value | 253.142387 | 53.504085 | 15.954816 / 30.780220 | 0 | 0.467144816 |

## Admission and savings

| Blocks | Seed | Value binding / total draws | KEEP / DROP node decisions at binding | Fixed savings vs both APC and queue | Value / fixed savings |
|---:|---:|---:|---:|---:|---:|
| 9216 | 1601 | 8 / 951 | 14417 / 603 | 24240 tokens | 100% |
| 9216 | 1702 | 10 / 953 | 15525 / 976 | 18272 tokens | 100% |
| 10752 | 1601 | 0 / 949 | Not binding | 33888 tokens | 100% |
| 10752 | 1702 | 0 / 951 | Not binding | 33888 tokens | 100% |

Decision counts are repeated per-draw observations, not unique retained blocks
or block-seconds. The physical tensors were verified at exactly 8,455,716,864
and 9,865,003,008 bytes. The lower region meets R2-A's real-decision condition;
the upper point is the nonbinding control, not another admission effect.
All matched fixed savings are positive; the 75% outcome criterion passes
without substituting the historical 33,888 baseline at the tighter point.

Every trace recomputes all 13,376 recurring-BG body tokens at its test request,
including APC and queue. Value therefore adds no recurring-BG recovery.
Fixed and value have equal cache-eviction counts, FG recompute and all 424
output texts within each group. Equal counts alone do not prove identical
victim identities or complete allocation trajectories. There are no
preemptions or rejected allocations.

## Descriptive timing and quality

Value-minus-fixed relative changes favor value for seed1601 and fixed for
seed1702 at both capacities. With two seed strata, mean and sample SD of
per-seed relative changes are:

| Blocks | FG E2E change, mean +/- SD | BG P95 change, mean +/- SD |
|---:|---:|---:|
| 9216 | +0.061465% +/- 0.295926 percentage points | +0.187764% +/- 0.707293 percentage points |
| 10752 | +0.265762% +/- 1.084434 percentage points | +0.822540% +/- 3.207983 percentage points |

These are descriptive discovery statistics, not formal confidence bounds,
proof of equivalence or proof of regression. No consistent background
improvement is established.

All 16 traces have official CPU-local scores and complete predictions.
Fixed/value predictions are identical within every group. Full-credit counts
are 131/424 for seed1601 and 132/424 for seed1702; strict all-correct remains
false. Small per-seed Answer F1 differences versus APC/queue remain visible.
This is observed quality evidence, not formal non-inferiority.

## Measurement defect and bounded next action

The source-relocated launch freeze omitted
`serving_measurement.live_free_exposure: true`. All 16 raw files lack
`physical.policy_protected_live_free_exposure`; their compact exports correctly
state `exposure_evidence.status = not_instrumented`. The original raw and
compact evidence are retained unchanged.

The implementation owner's native counterexample establishes that exact live
recovery is not identifiable: the same emitted allocation, preference, owner,
adoption, free-state and lifecycle records permit a membership transition at
time 2 or 3 before the same recorded allocation end at 4, producing 20 versus
24 block-seconds. Intra-allocation membership/observation timestamps were not
recorded. A recorded-end replay would define a coarsened offline integral,
not recover the historical live one. No full-matrix reconstruction is claimed.

The bounded within-draw identity join finds all 1,579 DROP observations across
18 binding draws in that same draw's actual victims: 603 for seed1601 and 976
for seed1702. None survives the draw as an unpreferred retained copy; no KEEP
copy is selected in those draws. On these observations, value selects
unavoidable current victims rather than establishing a persistent reduction
in protection. This is not a cross-run victim-equivalence claim or a theorem
about every future pressure/workload.

The dedicated measurement helper is corrected at library `9101fc7` /
native repair snapshot `62fe9c3b`: it always collects required live exposure,
without forcing instrumentation into production policy or changing frozen
serving source `3e7cf232`. The actual omitted-flag wrapper reproduced red;
three affected native checks pass after repair, including cleanup on failure
and duplicate-copy/export behavior. The timestamp counterexample passed
separately. The two-export DROP join consumed 5.992 CPU seconds; no full-matrix
reconstruction or repeated archive scan was performed.

No automatic GPU rerun is authorized: exposure instrumentation alone cannot
supply the missing background/recompute gain, and no new useful scientific
candidate has been established. Repair evidence and source bundle are at
`/home/ubuntu/experiments/octopus/r2-exposure-recovery-20260912-d3c687/`.

## Cost and artifacts

The campaign consumed 4,563.827416 GPU process-seconds. Active utilization of
its four allocated GPUs was 95.584890%; all 16 traces completed, none failed,
and the supervisor confirmed zero owned GPU processes.

Full artifacts are under
`/home/ubuntu/experiments/octopus/r2-physical-pressure-discovery-20260912/`:
`execution-result.json`, `analysis.json`, `quality.json`,
`combined-quality-input.json.gz`, `execution-commands.json`, source-bound
freezes in `input/`, raw files in `raw-archive/trace-data/`, and native-validated
exports in `export/`. Policy source/formula and readiness are recorded in
`../r2-value-policy/report.md`. Gold stayed on the isolated CPU evaluator host.
