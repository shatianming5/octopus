# RUNNING-prefill commitment: native support, recorded-workload falsification

The capacity-only candidate is implemented at library
`d9f96f65593dd22314c325533558184bcc8bd262`, native
`54bc548922554d55042b71abd3f7cfaa8983cd5a`, with configuration
`benchmarks/configs/r2_prefill_commitment.json`. It is not GPU-ready or an R2
pass. The original semantic universe and analytic work ranking are unchanged.

Native full-sequence-fit checks do not allocate the complete prompt: a
512-token prompt requires 32 blocks, but its first step allocates one. Before
the next one-block RUNNING draw, one block is held and 30 additional prompt
blocks remain beyond the draw. The native coordinator accounts for held,
reused and allocated-but-not-computed slots. Only RUNNING unfinished prefills
contribute; the current draw offsets only its own included contribution, never
another request's commitment. Other waiting requests and decode forecasts
remain excluded.

A native CPU fixture with a 192-token prompt and 16-token step budgets keeps
four rejected FREE copies across two steps without adding owners. This
establishes a possible native distinction, not its presence on the workload.

## Bounded conditional evidence

Only the two actual lower-pressure value streams were used. Actual allocation
opportunities, outputs and release times remain fixed; this is conditional
work accounting, not a counterfactual GPU/scheduler or quality result.

| Seed | Positive commitment draws | Max additional blocks | Binding draws | Immediate unselected DROP observations | Conditional FG saved / matched fixed | BG body recompute | Changed work records |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1601 | 266 | 814 | 8 | 0 | 24240 / 24240 | 13376 | 0 |
| 1702 | 262 | 675 | 13 | 156 | 18272 / 18272 | 13376 | 0 |

No qualifying cross-wave distinction is demonstrated. Seed1601 has no
immediately unselected DROP at all. For seed1702, the wave-end and
before-next-step checks find no rejected cached FREE copy that retains a
declared future consumer and a positive current RUNNING commitment.
This does not claim all 156 observations were immediate victims, nor replace
missing historical live exposure with an offline integral. An idle-only
preference change rearmed before the next allocation would not establish the
requested allocation benefit.

The initial audit checked only after the next allocation, which could miss
copies surviving until that allocation. A focused native regression now
checks wave-end and pre-next-allocation state. The corrected seed1702 audit
was rerun once; the preliminary result is preserved. Seed1601 needed no rerun
because it had zero immediately unselected DROP observations.

Five targeted native accounting/parity checks passed in 10.07 seconds, and
the separate boundary regression passed in 11.21 seconds. The two conditional
streams plus the necessary seed1702 audit refinement consumed 30.403657
processing CPU seconds; serialization has separate sidecars. No GPU,
16-trace replay, old R1/normal experiment or held-out inspection occurred.

Full evidence, configuration, commands and native source bundle:
`/home/ubuntu/experiments/octopus/r2-prefill-commitment-20260912-d3c687/`.

## Narrow remaining transaction-boundary question

Before expanding to any waiting-queue forecast, check one exact boundary:
the current first-admission request may still be marked WAITING when
`get_new_blocks` runs after native fit checks. Its own currently committing
prompt residual might therefore be omitted by the RUNNING-only set.
The next CPU-only refinement must first establish that call ordering, then
count only this current request unioned with RUNNING requests, subtracting
its own draw once. It must not count arbitrary queued requests or infer later
success/outcomes. The same persistence, changed-work and positive matched
savings gates apply before any GPU consideration.
