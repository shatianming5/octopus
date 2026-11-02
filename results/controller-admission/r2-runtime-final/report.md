# Final minimal runtime-only R2: PASS_TO_R3

**Both seeds pass all four user-defined conditions.** This is the authorized
last runtime-only screen, not R3 confirmation or proof of isolated-GPU latency.
At this screen the immediate Model x vLLM failure branch was not taken;
the policy advanced to R3.

## Frozen execution

Native source: `beb50062f7bf9603f2b4ec657072c8ff763559ee`.
Initial gate implementation: `180dcf58682f57e09cd976f74e889ef88c707fbe`.
The unchanged per-seed comparator was subsequently extracted at
`a32aaf46be8c406090227cdab147bafbc8afabf2`; its output on the actual completed
matrix is exactly equal to the retained decision.

The current next-reuse policy, model,424 questions, intended arrival schedules
and9,865,003,008-byte/10,752-block KV point remained fixed. Only the required
native request-work and timing path was used:
`serving_measurement.live_free_exposure=false` bypasses the heavy
live-exposure/free-state reconstruction. Native policy callbacks and
`RequestMetrics` remain enabled. The focused native gate passed15 tests.

Two seed streams ran on `gpu.183` GPUs8/9, with methods sequential on one GPU
per seed and counterbalanced order. Shared GPU use was explicitly authorized.
Co-runners used about1.1GiB each and were not stopped or reconfigured.
External utilization sampled at trace boundaries varied45-74% on GPU8 and
37-70% on GPU9. Timing conclusions are qualified to this shared-load run;
they must not be presented as isolated-GPU causal effects.

Invalid startup attempts remain separate. The valid `attempt003` used
`CUDA_HOME=/usr/local/cuda-12.9` and a startup-memory guard of0.96. These were
operational repairs; explicit KV bytes and computation were unchanged.

## Actual work and latency

Work is executed prefill tokens from each request's nested
`RequestMetrics.actual_prefill_tokens`, checked against committed
`scheduled_ranges`. Repeated prefill after preemption is counted, not unioned
away. Nullable top-level fields, inherited tokens and cache-hit estimates
were not substituted. Tokens are not a FLOP-equivalence claim.

| Seed | Method | FG parent prefill | BG prefill | Total prefill | BG P95 seconds | FG E2E seconds |
|---:|---|---:|---:|---:|---:|---:|
| 1601 | APC | 33,888 | 1,997,442 | 2,065,860 | 111.4073 | 324.5805 |
| 1601 | Queue | 33,888 | 1,997,442 | 2,065,860 | 92.3208 | 305.1643 |
| 1601 | Semantic fixed | 0 | 1,997,442 | 2,031,972 | 102.1897 | 309.4256 |
| 1601 | Value | 0 | 1,984,066 | 2,018,596 | 96.1143 | 303.6745 |
| 1702 | APC | 33,888 | 1,997,442 | 2,065,860 | 107.1074 | 314.7914 |
| 1702 | Queue | 33,888 | 1,997,442 | 2,065,860 | 96.8918 | 304.1265 |
| 1702 | Semantic fixed | 0 | 1,997,442 | 2,031,972 | 110.1937 | 314.9866 |
| 1702 | Value | 4,416 | 1,984,066 | 2,023,012 | 91.6632 | 297.0134 |

| Required condition, each seed | 1601 | 1702 |
|---|---:|---:|
| Fixed FG savings retained versus queue, at least75% | 100.00% | 86.97% |
| BG prefill strictly below fixed | 13,376 fewer | 13,376 fewer |
| Total prefill strictly below fixed | 13,376 fewer | 8,960 fewer |
| BG P95 at most APC | 96.1143 <=111.4073 | 91.6632 <=107.1074 |

The exact8-trace/424-request matrix was native-validated. The committed
`--phase r2-final` evaluator returned `PASS_TO_R3`, exit0, with every gate
true in both seeds. No averaging across seeds hid a failed condition.
FG wall speed, original-fixed protection/waste and quality were not silently
added as extra gates to this screen. Earlier results under the older
measurement/acceptance protocol remain retained, not relabeled invalid.

## Next authorized route

R3 uses the same policy and lean counter path, with the predeclared genuinely
new confirmation seeds2201/2302/2403/2504 and the same four methods.
It must not rename or reuse the discovery traces as confirmation.
The existing formal confidence/official-quality checks and reconfirmation of
these same four user conditions are separate outputs; both must pass before
R3 acceptance. The [subsequent R3 confirmation](../r3-confirmation/report.md)
completed but was not accepted. That result does not rewrite this screen's pass.
No new runtime-only parameter search, fifth method, R4 or co-design experiment
was introduced by this decision.

## Durable artifacts and cleanup

The exact evaluator output is [`decision.json`](decision.json).
Raw evidence and seed terminals remain under
`/home/user/octopus-r2-runtime-final-20260913/attempt003` on `gpu.183`.
The small combined input, decision and seed-terminal copies are preserved in
the parent session's `files/final-runtime-r2.*` artifacts.
Both original launchers and model processes exited naturally; the co-runners
were preserved. Launching the R3 scope does not restart those completed jobs.
