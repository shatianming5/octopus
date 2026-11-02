# Per-seed interleaved checkpoint-retention contention supplement

This supplement contains 30 newly executed traces in 12 predeclared, deterministically shuffled seed blocks. Negative paired deltas mean Octopus was lower/faster than matched APC; positive deltas mean it was higher/slower.

## Bounded interpretation

- At 512m/low, foreground was lower in 3/3 and higher in 0/3 (median -1.121s, range [-3.097, -0.534]); whole-window background P95 was lower in 2/3 and higher in 1/3 (median -0.011s), common-overlap P95 was lower in 1/3 and higher in 2/3 (median +0.226s), and makespan was lower in 1/3 and higher in 2/3.
- At 512m/high, foreground was lower in 3/3 and higher in 0/3 (median -2.922s, range [-5.650, -2.849]); whole-window background P95 was lower in 0/3 and higher in 3/3 (median +2.263s), common-overlap P95 was lower in 0/3 and higher in 3/3 (median +2.263s), and makespan was lower in 0/3 and higher in 3/3.
- At 2g/low, foreground was lower in 1/3 and higher in 2/3 (median +0.062s, range [-0.150, +0.563]); whole-window background P95 was lower in 2/3 and higher in 1/3 (median -0.002s), common-overlap P95 was lower in 1/3 and higher in 2/3 (median +0.210s), and makespan was lower in 0/3 and higher in 3/3.
- At 2g/high, foreground was lower in 2/3 and higher in 1/3 (median -0.027s, range [-0.237, +0.091]); whole-window background P95 was lower in 2/3 and higher in 1/3 (median -0.023s), common-overlap P95 was lower in 1/3 and higher in 2/3 (median +0.009s), and makespan was lower in 2/3 and higher in 1/3.
- No cell had all three foreground traces lower while also avoiding higher values in both background P95 views across all three seeds.
- These results do not isolate an order-only effect. The grouped study reused one engine per method/capacity group, while this supplement uses a fresh process and engine, warmup, and cold prefix-cache reset for every trace. Every method in this supplement receives that same lifecycle policy; native-default APC still retains its genuine `async_scheduling=true` and `enforce_eager=false` settings.

## Paired matched traces

| KV | Load | Seed | FG APC s | FG Octopus s | FG delta s | BG P95 delta s | Common-overlap P95 delta s | Makespan delta s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512m | low | 101 | 14.712 | 13.590 | -1.121 | +0.251 | +0.226 | -0.145 |
| 512m | low | 202 | 16.548 | 13.451 | -3.097 | -0.108 | +0.403 | +0.114 |
| 512m | low | 303 | 12.786 | 12.252 | -0.534 | -0.011 | -0.044 | +0.008 |
| 512m | high | 101 | 34.934 | 32.084 | -2.849 | +3.333 | +3.333 | +3.064 |
| 512m | high | 202 | 35.498 | 29.848 | -5.650 | +2.263 | +2.263 | +2.210 |
| 512m | high | 303 | 25.185 | 22.263 | -2.922 | +1.862 | +1.902 | +0.990 |
| 2g | low | 101 | 13.582 | 13.644 | +0.062 | -0.006 | -0.255 | +0.001 |
| 2g | low | 202 | 14.699 | 15.262 | +0.563 | +0.162 | +0.210 | +0.146 |
| 2g | low | 303 | 13.078 | 12.929 | -0.150 | -0.002 | +0.292 | +0.002 |
| 2g | high | 101 | 28.721 | 28.485 | -0.237 | -0.595 | -0.595 | -0.713 |
| 2g | high | 202 | 23.190 | 23.281 | +0.091 | +0.251 | +0.250 | +0.091 |
| 2g | high | 303 | 19.403 | 19.375 | -0.027 | -0.023 | +0.009 | -0.008 |

## Cell directions and uncertainty

| KV | Load | FG median [range] s | FG -/0/+ | BG P95 median [range] s | Common-overlap median [range] s | Makespan -/0/+ |
|---|---|---:|---:|---:|---:|---:|
| 512m | low | -1.121 [-3.097, -0.534] | 3/0/0 | -0.011 [-0.108, +0.251] | +0.226 [-0.044, +0.403] | 1/0/2 |
| 512m | high | -2.922 [-5.650, -2.849] | 3/0/0 | +2.263 [+1.862, +3.333] | +2.263 [+1.902, +3.333] | 0/0/3 |
| 2g | low | +0.062 [-0.150, +0.563] | 1/0/2 | -0.002 [-0.006, +0.162] | +0.210 [-0.255, +0.292] | 0/0/3 |
| 2g | high | -0.027 [-0.237, +0.091] | 2/0/1 | -0.023 [-0.595, +0.251] | +0.009 [-0.595, +0.250] | 2/0/1 |

## Native-default APC at high load

Deltas are native-default APC minus matched synchronous/eager APC on the same arrival trace.

| KV | Seed | FG delta s | BG P95 delta s | TTFT P95 delta s | Makespan delta s |
|---|---:|---:|---:|---:|---:|
| 512m | 101 | -5.267 | -2.553 | -2.387 | -2.480 |
| 512m | 202 | -5.558 | -2.398 | -2.125 | -2.428 |
| 512m | 303 | -4.437 | -1.037 | -0.620 | -1.393 |
| 2g | 101 | -7.253 | -1.225 | -1.233 | -2.174 |
| 2g | 202 | -2.426 | -0.293 | -0.255 | -0.134 |
| 2g | 303 | -1.966 | -0.431 | -0.234 | -0.078 |

## Planned and actual block order

| Block | KV | Load | Seed | Actual method order | Start UTC | End UTC |
|---:|---|---|---:|---|---|---|
| 1 | 2g | high | 202 | apc -> apc-native-default -> octopus | 2026-09-08T13:40:24.215014+00:00 | 2026-09-08T13:42:53.204485+00:00 |
| 2 | 2g | low | 101 | octopus -> apc | 2026-09-08T13:42:53.205883+00:00 | 2026-09-08T13:45:19.736126+00:00 |
| 3 | 512m | low | 101 | apc -> octopus | 2026-09-08T13:45:19.737555+00:00 | 2026-09-08T13:47:47.531993+00:00 |
| 4 | 512m | low | 303 | apc -> octopus | 2026-09-08T13:47:47.533378+00:00 | 2026-09-08T13:51:37.320313+00:00 |
| 5 | 512m | high | 202 | octopus -> apc-native-default -> apc | 2026-09-08T13:51:37.321858+00:00 | 2026-09-08T13:54:42.252484+00:00 |
| 6 | 2g | high | 303 | octopus -> apc -> apc-native-default | 2026-09-08T13:54:42.254134+00:00 | 2026-09-08T13:57:41.080133+00:00 |
| 7 | 512m | low | 202 | octopus -> apc | 2026-09-08T13:57:41.081782+00:00 | 2026-09-08T14:00:38.078023+00:00 |
| 8 | 2g | high | 101 | apc-native-default -> octopus -> apc | 2026-09-08T14:00:38.079818+00:00 | 2026-09-08T14:02:59.854640+00:00 |
| 9 | 2g | low | 303 | octopus -> apc | 2026-09-08T14:02:59.856359+00:00 | 2026-09-08T14:06:49.758028+00:00 |
| 10 | 2g | low | 202 | apc -> octopus | 2026-09-08T14:06:49.759803+00:00 | 2026-09-08T14:09:46.516833+00:00 |
| 11 | 512m | high | 101 | apc -> octopus -> apc-native-default | 2026-09-08T14:09:46.518847+00:00 | 2026-09-08T14:12:49.947731+00:00 |
| 12 | 512m | high | 303 | apc-native-default -> apc -> octopus | 2026-09-08T14:12:49.949711+00:00 | 2026-09-08T14:15:53.716249+00:00 |

- High-load position balance per capacity: True.
- Low-load first positions across both capacities: APC 3, Octopus 3. Within each odd three-seed, two-method cell, exact balance is impossible and was not hidden by adding another seed.
- All 12 seed blocks ran contiguously: True; process overlap: False.

## Per-trace lifecycle outside task latency

| Phase | Minimum s | Median s | Maximum s |
|---|---:|---:|---:|
| Engine construction | 8.958 | 9.425 | 13.816 |
| Warmup + cold cache reset | 0.057 | 0.058 | 0.138 |
| In-trace workload | 27.435 | 43.432 | 97.055 |
| Post-trace cache reset | 0.000 | 0.000 | 0.001 |
| Engine shutdown | 1.147 | 1.196 | 1.514 |

## Mechanism evidence

| KV | Load | APC trunk recompute by seed | Octopus trunk recompute by seed | Peak active-block delta by seed |
|---|---|---|---|---|
| 512m | low | 4416, 4400, 0 | 0, 0, 0 | +19, +1, +8 |
| 512m | high | 21296, 22336, 18208 | 0, 0, 0 | +0, +0, +0 |
| 2g | low | 0, 0, 0 | 0, 0, 0 | +0, +0, +0 |
| 2g | high | 4640, 0, 0 | 0, 0, 0 | +73, +72, +126 |

- Every Octopus trace recorded four captures and four releases. Final checkpoint release preceded full background drain by 0.401s to 84.788s.

## Measurement boundary and limitations

- Complete fresh traces: 30/30; requests: 1920 background and 390 foreground; tools: 270. All requests, tools, and foreground workflows reached their required terminal/correct state.
- Planned background arrivals are open-loop and identical within each matched seed/capacity/load cell; latency starts at planned arrival.
- Foreground E2E includes the Brain, actual read-only tools, reducer and serialization work, branches, continuations, and final answer.
- The common-overlap cohort is pair-matched but uses an outcome-dependent cutoff; whole-window metrics remain visible.
- Three traffic seeds are exploratory. Low-load order is globally balanced but not exactly balanced inside an odd-three-seed cell, and the design does not fully balance carryover effects.
- Comparisons with the old grouped results mix order changes with the new uniform per-trace lifecycle reset frequency and therefore cannot support an order-only causal claim.

## Protocol deviations

- None.
