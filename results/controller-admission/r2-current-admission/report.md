# Current-admission commitment: CPU support, negative primary GPU screen

The refinement is implemented at library
`230424c97a363a381dbc254545685e8743ceba59` and immutable native source
`807d46de552db10e274d2910e3676b1e62bd9934`, with configuration
`benchmarks/configs/r2_current_admission.json`. This passes the bounded
structural/conditional gate, not R2 or an efficiency gate.

At the actual `get_new_blocks` boundary, native fit checks have passed for
the currently committing first request, although its scheduler status can
still be WAITING. RUNNING-only accounting omitted this current prompt
residual. The refined commitment is the union of RUNNING unfinished prefills
and this current fit-checked allocation request, deducting its own current
draw once. Other waiting requests, future arrivals and decode forecasts are
excluded. Held/reused/pending slots use native arithmetic; an already-RUNNING
current request is not counted twice.

The capacity remains `max(0, F - n - Q - C_extra)`. The original semantic
candidate universe, analytic work ranking, queued priority and native owner
semantics are unchanged. Required measurement capture remains mandatory.

## Bounded evidence

The real WAITING fit-checked regression first failed with zero commitment
where 11 residual blocks were required. Ten affected native checks then pass,
covering current-request deduplication, other-waiter exclusion, own-draw
subtraction, held/shared/pending memory and failure cleanup.

Only the same two recorded 9,216-block streams were conditionally evaluated:

| Seed | Cross-wave / next-allocation persistence observations | Conditional saved / actual fixed saved tokens | Retained savings | Extra FG prefill tokens | BG body recompute |
|---:|---:|---:|---:|---:|---:|
| 1601 | 1314 | 22976 / 24240 | 94.785% | 1264 | 13376 |
| 1702 | 988 | 14320 / 18272 | 78.371% | 3952 | 13376 |

Counts are repeated physical-key observations, not unique retained blocks
or block-seconds. Unlike the RUNNING-only variant, this changes conditional
work and shows a meaningful distinction across scheduling boundaries.
However, the changed work is **additional FG recomputation**, not recovered
BG work. No BG, net-compute or latency benefit is claimed.

These streams freeze actual allocation opportunities, output tokens and
release times. Their 21.376 processing CPU seconds establish only conditional
feasibility, not counterfactual GPU timings, quality or missing historical
live exposure. Sources `54bc5489` and `3e7cf232` and their negative artifacts
remain unchanged. Full evidence/source bundle:
`/home/ubuntu/experiments/octopus/r2-current-admission-20260912-d3c687/`.

## Strongest actual-GPU falsifier first

Authorize only the primary 9,216-block point first: eight traces, four fresh
arms (APC, queue, fixed, value) and seeds1601/1702. The two seed groups run
independently on available GPUs, with all methods within a seed on the same
physical GPU. Preserve the exact 424 requests/tokens/arrivals, model/generation
controls and 600+10-second watchdog. Freeze native `807d46de`; require actual
live exposure in every arm and fresh complete official CPU scores.

Before outcomes, define the primary discovery screen as: strictly smaller
summed actual future-only FREE exposure for value than fixed across the two
seed pairs; a negative mean paired relative BG P95 change versus fixed;
and at least 75% of positive fresh matched fixed savings versus both APC and
queue in each seed. Report each seed without subset selection. These are
discovery point screens, not formal confidence bounds.

R2-B still additionally requires useful KEEP-over-DROP matched-fixed
reference benefit density with valid denominators, group/mix disclosure and
reduced unnecessary protection; missing/ambiguous references do not pass.
Actual DROP benefit/exposure remains undefined. Binding, persistence or the
CPU savings fraction alone cannot establish R2.

The second pressure point and R3/R4 are contingent on this actual result.
Do not spend another full matrix merely to repeat a negative BG tradeoff.
The new campaign root is
`/home/ubuntu/experiments/octopus/r2-current-admission-gpu-20260912/`.

At 15:24 UTC the supervisor confirmed launch on GPUs 0 and 3, native process
IDs 1633576 and 1633575, each using 24,400 MiB. The two-group design was
counterbalanced before launch: seed1601 runs APC/queue/fixed/value, while
seed1702 runs value/fixed/queue/APC. This replaces the incomplete two-row
subset of the earlier four-group Latin order, without inspecting outcomes.
All eight traces and their live-exposure reports are now complete.
Exposure decreases 6.987672% and both savings fractions exceed 75%, but mean
paired BG P95 versus fixed increases 1.057716%, failing the predeclared screen.
The second pressure point and formal stages have not launched. Complete
results and the remaining reference-analysis distinction are in
`../r2-current-admission-gpu/report.md`.
