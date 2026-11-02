# R3 confirmation result

**Terminal outcome: `R3_NOT_ACCEPTED`.** This is a valid negative result from the completed conditional controlled-probe matrix, not an instrumentation failure. No R4, tuning, extra seed, or co-design run was started.

All 16 traces completed with 424 native-valid requests at source `beb50062f7bf9603f2b4ec657072c8ff763559ee`, 9,865,003,008 KV bytes / 10,752 blocks, the frozen next-reuse policy, lean native counters, and official CPU-only scoring of all 6,784 predictions.

## Controlled input construction

The amended deterministic rule was applied before outcomes. Seeds2302/2403/2504 retained their unmodified generated background mapping. Seed2201 alone required the latest feasible ordinary-BG slot swap: the controlled target moved from191.5s to155.5s and request `d4b84f48460517bc0a6d4e0c38f6853c58081166` moved to191.5s. All424 question/token records, foreground arrivals, and the original timestamp multiset remained unchanged. Conclusions are conditional on this controlled-probe construction, not unconstrained QASPER arrivals.

## Formal R3 gates

**Formal performance gate: fail.**

- Candidate FG mean versus APC: `+3.3145s`; one-sided95% upper `+22.4967s` — fail.
- Candidate FG mean versus queue: `+4.7765s`; one-sided95% upper `+16.5352s` — fail.
- Candidate BG P95 versus APC: `+2.9735%`; one-sided95% upper `+6.2241%`, exceeding the `+5%` bound — fail.

**Official quality gate: fail.**

| Method | Aggregate Answer F1 | Full-credit questions | Missing |
|---|---:|---:|---:|
| projected_apc | 0.466894349 | 524 / 1,696 | 0 |
| b6_queue_aware | 0.466888906 | 524 / 1,696 | 0 |
| b6_semantic_fixed | 0.466871355 | 524 / 1,696 | 0 |
| b6_semantic_value | 0.466864692 | 524 / 1,696 | 0 |

The candidate's full-credit count is equal, but its Answer F1 is strictly below APC and queue.

## User-condition reconfirmation

| Seed | FG savings retained | BG tokens saved vs fixed | Total tokens saved vs fixed | Candidate BG P95 vs APC | Result |
|---|---:|---:|---:|---:|---|
| 2201 | 0.9674 | 13,376 | 12,272 | 89.9042s vs 94.9461s | all pass |
| 2302 | 0.7384 | 13,376 | 4,512 | 95.3794s vs 74.0506s | fail retention and latency |
| 2403 | 1.0000 | 0 | 0 | 100.6394s vs 108.4245s | fail strict BG/total savings |
| 2504 | undefined; fixed saved 0 | 0 | 0 | 100.1600s vs 89.7578s | fail all four |

The four-seed user-condition aggregate therefore fails.

## TTFT and output parity

Candidate TTFT (foreground mean/P95; background mean/P95, seconds):

| Seed | Foreground TTFT | Background TTFT |
|---|---:|---:|
| 2201 | 44.6967 / 73.4393 | 38.2589 / 72.4110 |
| 2302 | 47.8574 / 78.0331 | 41.0821 / 77.0308 |
| 2403 | 53.3560 / 86.8053 | 45.1307 / 82.2520 |
| 2504 | 51.0945 / 83.0290 | 37.3253 / 73.8982 |

Candidate exact text/token-ID parity out of424:

| Seed | vs APC | vs queue |
|---|---:|---:|
| 2201 | 422 / 422 | 422 / 422 |
| 2302 | 423 / 423 | 424 / 424 |
| 2403 | 423 / 423 | 421 / 421 |
| 2504 | 424 / 424 | 424 / 424 |

## Execution and artifacts

The campaign ran in two counterbalanced two-seed waves on shared `gpu.183` GPU8/9. Co-running jobs were not stopped or reconfigured; latency remains qualified as shared-load measurement. All four launchers and owned model processes exited naturally.

- Formal report: [`formal.json`](formal.json)
- Runtime requirements: [`runtime-gates.json`](runtime-gates.json)
- Official scores: `/home/ubuntu/.copilot/session-state/a7b703df-c72b-4d54-bfc7-d6007622dc13/files/r3-confirmation.quality.json`
- Small combined export: `/home/ubuntu/.copilot/session-state/a7b703df-c72b-4d54-bfc7-d6007622dc13/files/r3-confirmation.small.json.gz`
- Remote raw evidence: `/home/user/octopus-r3-confirmation-beb50062` on `gpu.183`

The earlier two-seed `PASS_TO_R3` result remains unchanged. This confirmation
does not support accepting the runtime-only route, but it is not a universal
impossibility proof or proof that missing model intent is the sole cause.
No further runtime-only tuning or R4 was performed. The requested
Model x vLLM intent-emission direction remains a subsequent implementation,
not code or validated performance included in this result.
