# Checkpoint retention under real tool waits and competing inference

This report is generated from the frozen 30-trace matrix. Negative deltas mean Octopus was lower/faster than matched APC; positive deltas mean it was higher/slower. Three paired seeds are exploratory evidence, so each cell reports every direction and the full observed range rather than a production-stability claim.

## Bounded answer

- **No tested cell established that checkpoint retention lowers complete foreground task cost without shifting cost to competing requests.** The mechanism removed foreground trunk recomputation under constrained KV, but the end-to-end and background effects were seed-dependent.
- At 512 MiB/high load, foreground delta had median -1.247s [-1.497, +1.887] (2/3 faster, 1/3 slower). The fixed common-overlap background P95 delta had median +3.119s [+0.385, +4.985] (3/3 slower, 0/3 faster). Makespan was higher in 2/3 and lower in 1/3.
- At 2 GiB/high load, native APC already preserved every foreground trunk (zero recomputation in all seeds). Octopus foreground latency and makespan were lower in 3/3 traces (foreground median -0.441s), but whole-window and common-overlap background P95 were each higher in 1/3. With no recomputation mechanism difference and only three seeds, this is promising timing evidence, not a stable retention benefit or no-harm result.
- At 2 GiB/low load, Octopus foreground and both background P95 views were higher in 3/3 traces (foreground median +0.211s).
- At 512 MiB/low load, foreground and background directions were mixed (foreground median -0.199s; whole-window background P95 median +0.030s). Low-load makespan was dominated by late planned arrivals after the foreground checkpoint had already been released.
- The real tools were short on this repository: source/AST tools were normally about 0.02-0.03s and state tests about 0.20s. This study therefore bounds behavior for short real waits, not long external tools.

## Paired matched traces

| KV | Load | Seed | FG APC s | FG Octopus s | FG delta % | BG P95 delta % | Common-overlap P95 delta s | Makespan delta s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512m | low | 101 | 16.473 | 12.403 | -24.71% | +1.83% | -0.107 | +0.002 |
| 512m | low | 202 | 12.946 | 12.747 | -1.54% | +1.05% | -0.210 | -0.066 |
| 512m | low | 303 | 11.214 | 11.582 | +3.28% | -1.92% | +0.005 | -0.019 |
| 512m | high | 101 | 30.531 | 32.418 | +6.18% | +24.68% | +4.985 | +4.577 |
| 512m | high | 202 | 29.657 | 28.410 | -4.21% | +19.67% | +3.119 | +3.521 |
| 512m | high | 303 | 18.748 | 17.251 | -7.99% | +2.66% | +0.385 | -0.068 |
| 2g | low | 101 | 13.320 | 13.531 | +1.58% | +0.02% | +0.047 | -0.022 |
| 2g | low | 202 | 13.238 | 13.352 | +0.86% | +1.33% | +0.047 | +0.181 |
| 2g | low | 303 | 11.506 | 12.557 | +9.14% | +0.54% | +0.205 | -0.038 |
| 2g | high | 101 | 25.415 | 23.049 | -9.31% | +1.90% | +0.172 | -0.121 |
| 2g | high | 202 | 22.358 | 22.344 | -0.06% | -0.18% | -0.017 | -0.047 |
| 2g | high | 303 | 18.398 | 17.958 | -2.40% | -2.63% | -0.213 | -0.003 |

## Cell uncertainty

| KV | Load | FG median [range] s | FG directions -/0/+ | BG P95 median [range] s | Common-overlap median [range] s | Makespan directions -/0/+ |
|---|---|---:|---:|---:|---:|---:|
| 512m | low | -0.199 [-4.070, +0.368] | 2/0/1 | +0.030 [-0.038, +0.064] | -0.107 [-0.210, +0.005] | 2/0/1 |
| 512m | high | -1.247 [-1.497, +1.887] | 2/0/1 | +3.128 [+0.260, +4.985] | +3.119 [+0.385, +4.985] | 1/0/2 |
| 2g | low | +0.211 [+0.114, +1.052] | 0/0/3 | +0.011 [+0.000, +0.031] | +0.047 [+0.047, +0.205] | 2/0/1 |
| 2g | high | -0.441 [-2.366, -0.014] | 3/0/0 | -0.010 [-0.099, +0.172] | -0.017 [-0.213, +0.172] | 3/0/0 |

## Native-default APC at high load

These deltas are native-default APC minus matched synchronous/eager APC on the same arrival trace. Native-default actually used `async_scheduling=true` and `enforce_eager=false`; its unavailable internal scheduler counters remain null.

| KV | Seed | FG delta s | BG P95 delta s | TTFT P95 delta s | Makespan delta s |
|---|---:|---:|---:|---:|---:|
| 512m | 101 | -0.913 | -1.618 | -1.517 | -1.611 |
| 512m | 202 | -1.923 | -1.908 | -1.443 | -1.110 |
| 512m | 303 | +0.604 | -0.946 | -0.907 | -0.241 |
| 2g | 101 | -3.798 | -0.786 | -0.890 | -1.218 |
| 2g | 202 | -1.546 | -0.921 | -0.805 | -0.181 |
| 2g | 303 | -0.960 | -0.749 | -0.428 | -0.088 |

Native-default APC improved foreground in 5/6 traces, and improved background P95, TTFT P95, and makespan in 6/6, 6/6, and 6/6. This is an execution-mode comparison, not evidence that APC changed its cache semantics.

## Mechanism and lifecycle evidence

| KV | Load | APC foreground trunk recompute by seed | Octopus trunk recompute by seed | Peak active-block delta by seed |
|---|---|---|---|---|
| 512m | low | 4624, 1008, 0 | 0, 0, 0 | +0, +1, +2 |
| 512m | high | 16896, 16832, 9056 | 0, 0, 0 | +2, +0, +0 |
| 2g | low | 0, 0, 0 | 0, 0, 0 | +0, +0, +0 |
| 2g | high | 0, 0, 0 | 0, 0, 0 | +62, +69, +29 |

- Every Octopus trace recorded four captures and four releases. The final checkpoint was released at foreground completion, 5.609s to 85.441s before full background drain.
- At 512 MiB APC's foreground trunk recomputation varied with traffic as shown above; Octopus recomputed zero. At 2 GiB both methods recomputed zero, explaining the absence of a retention mechanism advantage.
- Independently of these latency traces, the native allocator boundary precludes an unconditional no-background-harm guarantee: with `B=16` usable blocks, a dormant checkpoint pin `P=8`, and an unrelated background request requiring `R=12`, ordinary APC admits, retention does not, and releasing the lease restores admission. In general, `B - P < R <= B` is sufficient for this admission harm. This is a CPU-only allocator fact, not a measured GPU latency result.

## Measurement boundary

- Complete traces: 30/30; all foreground workflows correct: True.
- Recorded 1920 background requests, 390 foreground requests, and 270 tool executions; all reached complete status.
- Background response time and TTFT start at intended open-loop arrival, including controller admission delay.
- Foreground E2E starts before the initial Brain request and includes tool subprocesses, reducer/serialization work, and request admission.
- Reservation GPU-seconds equal one-GPU trace makespan. Overlapping request durations are not summed as GPU cost.
- Native-default scheduler counters are unavailable rather than zero; its actual eager/async settings are recorded per trace.
- Method order was counterbalanced only by capacity: 512 MiB ran APC, Octopus, then native-default APC; 2 GiB ran Octopus, APC, then native-default APC. It was not interleaved per seed, and native-default always ran last, so warming or time-of-run effects can contribute to small execution-mode differences.
- The common-overlap cohort uses the same requests within each pair, selected by intended arrival no later than the minimum of the two foreground completion times. That cutoff is outcome-dependent, not a globally predeclared fixed-time cohort; whole-window background metrics remain primary alongside it.
- Real tool durations ranged from 0.023s to 0.291s (median 0.030s).

## Protocol deviations

- GPU 0 became occupied during environment preparation. GPU 3 was then confirmed idle and fixed for calibration, both pilots, and all 30 main traces; no cross-GPU timing comparison is made.
- Direct old-host to new-host authentication was unavailable. The exact Qwen revision was retrieved from the official Hugging Face revision URL; IPv4 and resumable transfers changed transport only.
- The first APC pilot was invalid because compact Commit JSON reached the 192-token branch cap. That failed trace is retained. Before either valid pilot or any main trace, request identities and model-visible tool output were shortened and the common branch cap was set to 256 for every method.
- The first calibration launch failed before measurement because the venv ninja executable was not on PATH. The existing locked dependency was used by adding the venv bin directory to PATH; the workload and environment versions did not change.
- A deterministic Future-based regression reproduced a supported idle transition where the second tool collection admitted a request after the engine check and then raised a false stall. The redundant second admit/collect calls were removed, and all 30 main traces were regenerated at the unchanged protocol and offered rates under commit acf6048fc9629519148406c4bc22048f4e7e36ea.

The invalid pilot and its original protocol are retained separately from calibration, valid pilots, and main traces. No main-matrix cell, seed, model, artificial wait, or offered rate was added or selected after observing results.
