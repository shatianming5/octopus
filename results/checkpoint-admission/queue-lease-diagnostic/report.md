# Checkpoint admission: bounded real-GPU results

Status: **incomplete**. 72/81 traces: 72 controlled replay and 0 separate real-tool traces.

Each trace uses a fresh engine/process on the same GPU. The 96-request arrival-ID cohort and four intended task arrivals were frozen independently. Feasibility is background P95 TTFT <=1.05x matched APC with unchanged exact quality; one-sided moving-block bootstrap bounds are reported separately. P99 is descriptive. Three seeds do not establish a distribution-wide guarantee.

| Workload | Condition | Method | Point feasible | Bound feasible | FG faster | Paired FG delta, mean [95% interval] s |
|---|---|---|---:|---:|---:|---|
| controlled | roomy_short | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | roomy_short | cost_greedy | 3/3 | 3/3 | 0/3 | 0.065 [-0.052, 0.182] |
| controlled | roomy_short | frequency | 3/3 | 3/3 | 2/3 | -0.005 [-0.156, 0.146] |
| controlled | roomy_short | hard_pin | 3/3 | 3/3 | 2/3 | -0.043 [-0.250, 0.163] |
| controlled | roomy_short | native_default | 0/3 | 0/3 | 3/3 | -0.507 [-0.651, -0.362] |
| controlled | roomy_short | queue_aware | 3/3 | 3/3 | 1/3 | 0.088 [-0.254, 0.430] |
| controlled | roomy_short | ttl | 3/3 | 2/3 | 1/3 | 0.031 [-0.179, 0.241] |
| controlled | tight_burst | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | tight_burst | cost_greedy | 3/3 | 3/3 | 1/3 | 0.018 [-1.147, 1.183] |
| controlled | tight_burst | fixed_price | 2/3 | 2/3 | 2/3 | 0.346 [-8.217, 8.909] |
| controlled | tight_burst | frequency | 2/3 | 2/3 | 1/3 | 0.347 [-1.108, 1.801] |
| controlled | tight_burst | hard_pin | 1/3 | 1/3 | 1/3 | 0.573 [-3.798, 4.944] |
| controlled | tight_burst | native_default | 3/3 | 3/3 | 3/3 | -3.162 [-6.723, 0.399] |
| controlled | tight_burst | no_apc | 0/3 | 0/3 | 0/3 | 6.691 [4.639, 8.743] |
| controlled | tight_burst | no_eta | 1/3 | 1/3 | 1/3 | 0.270 [-6.463, 7.003] |
| controlled | tight_burst | queue_aware | 2/3 | 2/3 | 1/3 | -0.299 [-5.203, 4.604] |
| controlled | tight_burst | score | 2/3 | 2/3 | 0/3 | 0.417 [-0.390, 1.224] |
| controlled | tight_burst | ttl | 2/3 | 2/3 | 1/3 | 2.186 [-7.102, 11.474] |
| controlled | tight_low | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | tight_low | cost_greedy | 3/3 | 1/3 | 1/3 | 0.154 [-0.517, 0.824] |
| controlled | tight_low | frequency | 0/3 | 0/3 | 1/3 | 0.060 [-0.381, 0.502] |
| controlled | tight_low | hard_pin | 1/3 | 0/3 | 1/3 | 0.472 [-0.747, 1.691] |
| controlled | tight_low | queue_aware | 2/3 | 1/3 | 1/3 | 0.037 [-0.177, 0.252] |
| controlled | tight_low | ttl | 2/3 | 1/3 | 2/3 | -0.028 [-0.273, 0.218] |

## Agent-state increment beyond queue-aware

- controlled/roomy_short: agent minus queue -0.024s; agent faster in 2/3 seeds. Trace-level interval: [-0.271062210300703, 0.22395735302078393].
- controlled/tight_burst: agent minus queue 0.318s; agent faster in 2/3 seeds. Trace-level interval: [-3.6060976617232274, 4.241173383341326].
- controlled/tight_low: agent minus queue 0.116s; agent faster in 1/3 seeds. Trace-level interval: [-0.36118750691208706, 0.5941448603394723].

## Runtime and failure slices

Full-policy decision CPU totals range from 0.0260 to 0.0671 s per trace (0.046%--0.080% of trace duration). This measures Controller.decide, not all adapter scanning, event serialization, or engine-mode overhead; native-default end-to-end comparisons include those practical differences.

Per-seed ETA absolute error, observed prefix-miss fraction, decision maxima, action reasons and switching counts are in summary.json:diagnostics. The shared calibrated APC-loss proxy is deliberately coarse: observed differences across roomy, burst and real-tool slices are limitations, not silently refit test parameters. The original retention_revocations counter counts native retention-owner releases, including explicit handle close; policy DROP reasons and logical resolution events must be used to distinguish pressure, TTL and semantic retirement.


No C2 claim follows from a small mean difference alone. The full policy must also be feasible and its ETA/fixed-price/score ablations must establish a causal information increment. Per-seed quality, recompute, TTFT, queue/dispatcher timestamps, eviction counts, block-time and decision CPU time are in summary.json and request-level traces.

## Practical and correctness boundaries

Native-default APC is a practical reference, not a matched causal intervention. All of its engine overhead is included. CPU KV demotion/prefetch is not implemented. The calibrated loss probability is a coarse independent proxy, not identified counterfactual truth. CUDA event calibration measures the device timeline, including dispatch gaps, not hardware FLOPs.

The optional trunk-recompute FLOPs column is explicitly an estimate: `2 * N * recomputed_tokens`, with parameter-count N derived from the pinned Qwen config (dimensions and formula in summary.json). It omits attention's quadratic term and is not substituted for GPU seconds or task E2E.

GPU lifecycle tests include exact 33-token KEEP/warm/cold output equality and the 129-token native warm/cold numerical boundary. Octopus matches each native path there; the model/backend is not universally bitwise cache-invariant. Task correctness remains an enforced measured constraint.

The initial nine real-tool anchors accidentally used replay ETA priors. They are preserved under eta-prior-diagnostic and excluded from the primary table. The amended anchors use the already-measured independent real-tool priors with the same traffic seeds, cohort, capacity, cost curve and gate. This is a nine-case repair, not a sweep. A shared-host process appeared on the selected GPU by the initial study's cleanup; its activity during measurement was not continuously monitored. No exclusive-device or universal causal-performance guarantee is inferred.

## Exact replay model

The CPU replay optimum is restricted to its finite shared-block cost model. It is not a bound on live GPU latency or an oracle for the held-out tools. {"calibration": "synthetic shared-block model; not measured GPU parameters", "choices": [["a"], ["a"], ["b"]], "controller_choices": [["a", "b"], ["a"], ["b"]], "controller_cost": 0.7, "controller_gap": 0.09999999999999987, "cost_gpu_compute_seconds": 0.6000000000000001, "scope": "exact optimum of the fixed replay cost model; not GPU latency", "states_visited": 11, "status": "complete"}

## Incomplete units

[["real_tool", "tight_burst", 101, "apc"], ["real_tool", "tight_burst", 101, "cost_greedy"], ["real_tool", "tight_burst", 101, "queue_aware"], ["real_tool", "tight_burst", 202, "apc"], ["real_tool", "tight_burst", 202, "cost_greedy"], ["real_tool", "tight_burst", 202, "queue_aware"], ["real_tool", "tight_burst", 303, "apc"], ["real_tool", "tight_burst", 303, "cost_greedy"], ["real_tool", "tight_burst", 303, "queue_aware"]]

Repair status: **blocked**. The authorized GPU remained occupied after a bounded 1,800-second process-exit event wait (exit 75). No owned GPU process or wait remains. The same nine selectors and one new preemption lifecycle check are pending; do not mark the core objective complete.
