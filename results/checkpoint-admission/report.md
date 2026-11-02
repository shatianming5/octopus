# Checkpoint admission: bounded real-GPU results

Status: **complete**. 81/81 traces: 72 controlled replay and 9 separate real-tool traces.

Each trace uses a fresh engine/process; every matched block stays on one physical GPU. Original controlled traces use GPU3; fresh controlled queue-repair triples and repaired real-tool traces use hardware-matched GPU6. Original and repaired comparison epochs are kept separate, not pooled across devices. The 96-request arrival-ID cohort and four intended task arrivals were frozen independently. Feasibility is background P95 TTFT <=1.05x matched APC with unchanged exact quality; one-sided moving-block bootstrap bounds are reported separately. P99 is descriptive. Three seeds do not establish a distribution-wide guarantee.

## Frozen go/no-go decision

Outcome: **no-go**. The full policy fails the fixed 5% background point/bound gate at seeds [101]; the paired foreground interval does not establish an improvement.
The optional nine-trace GDS-style comparison is **not warranted and was not run**. A negative result satisfies this frozen decision loop; thresholds, waits, priors and seeds were not changed to search for a win.

The frozen 10,800-second ledger remains unchanged. The separately authorized supplement covered 6/6 exact selectors, completed 6, used 522.602 additional seconds, and recorded 8 process attempts.

2 infrastructure attempts were not accepted: one failed offline model resolution before engine initialization; one completed measurement but failed source-identity serialization in the non-Git deployment. Both are retained in the supplement ledger; neither changed a selector or entered the primary evidence.

| Workload | Condition | Method | Point feasible | Bound feasible | FG faster | Paired FG delta, mean [95% interval] s |
|---|---|---|---:|---:|---:|---|
| controlled | roomy_short | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | roomy_short | cost_greedy | 3/3 | 3/3 | 0/3 | 0.065 [-0.052, 0.182] |
| controlled | roomy_short | frequency | 3/3 | 3/3 | 2/3 | -0.005 [-0.156, 0.146] |
| controlled | roomy_short | hard_pin | 3/3 | 3/3 | 2/3 | -0.043 [-0.250, 0.163] |
| controlled | roomy_short | native_default | 0/3 | 0/3 | 3/3 | -0.507 [-0.651, -0.362] |
| controlled | roomy_short | queue_aware | 3/3 | 3/3 | 2/3 | 0.031 [-0.955, 1.017] |
| controlled | roomy_short | ttl | 3/3 | 2/3 | 1/3 | 0.031 [-0.179, 0.241] |
| controlled | tight_burst | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | tight_burst | cost_greedy | 3/3 | 3/3 | 1/3 | 0.018 [-1.147, 1.183] |
| controlled | tight_burst | fixed_price | 2/3 | 2/3 | 2/3 | 0.346 [-8.217, 8.909] |
| controlled | tight_burst | frequency | 2/3 | 2/3 | 1/3 | 0.347 [-1.108, 1.801] |
| controlled | tight_burst | hard_pin | 1/3 | 1/3 | 1/3 | 0.573 [-3.798, 4.944] |
| controlled | tight_burst | native_default | 3/3 | 3/3 | 3/3 | -3.162 [-6.723, 0.399] |
| controlled | tight_burst | no_apc | 0/3 | 0/3 | 0/3 | 6.691 [4.639, 8.743] |
| controlled | tight_burst | no_eta | 1/3 | 1/3 | 1/3 | 0.270 [-6.463, 7.003] |
| controlled | tight_burst | queue_aware | 2/3 | 1/3 | 1/3 | 1.381 [-4.795, 7.557] |
| controlled | tight_burst | score | 2/3 | 2/3 | 0/3 | 0.417 [-0.390, 1.224] |
| controlled | tight_burst | ttl | 2/3 | 2/3 | 1/3 | 2.186 [-7.102, 11.474] |
| controlled | tight_low | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| controlled | tight_low | cost_greedy | 3/3 | 1/3 | 1/3 | 0.154 [-0.517, 0.824] |
| controlled | tight_low | frequency | 0/3 | 0/3 | 1/3 | 0.060 [-0.381, 0.502] |
| controlled | tight_low | hard_pin | 1/3 | 0/3 | 1/3 | 0.472 [-0.747, 1.691] |
| controlled | tight_low | queue_aware | 1/3 | 1/3 | 1/3 | 0.003 [-0.610, 0.617] |
| controlled | tight_low | ttl | 2/3 | 1/3 | 2/3 | -0.028 [-0.273, 0.218] |
| real_tool | tight_burst | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| real_tool | tight_burst | cost_greedy | 2/3 | 2/3 | 2/3 | 0.340 [-7.758, 8.437] |
| real_tool | tight_burst | queue_aware | 2/3 | 1/3 | 1/3 | -1.307 [-8.282, 5.668] |

## Agent-state increment beyond queue-aware

- controlled/roomy_short: agent minus queue 0.428s; agent faster in 0/3 seeds. Trace-level interval: [-0.8710081080592886, 1.7277206175880224]. Full policy passes its own same-device APC uncertainty gate in 2/3 available pairs.
- controlled/tight_burst: agent minus queue -1.340s; agent faster in 2/3 seeds. Trace-level interval: [-6.446323172940434, 3.7655959351049075]. Full policy passes its own same-device APC uncertainty gate in 1/3 available pairs.
- controlled/tight_low: agent minus queue 0.095s; agent faster in 0/3 seeds. Trace-level interval: [0.00751053244738735, 0.18184620388696388]. Full policy passes its own same-device APC uncertainty gate in 1/3 available pairs.
- real_tool/tight_burst: agent minus queue 1.647s; agent faster in 1/3 seeds. Trace-level interval: [-3.1278689453742707, 6.421758418877106]. Full policy passes its own same-device APC uncertainty gate in 2/3 available pairs.

## Runtime and failure slices

Full-policy decision CPU totals range from 0.0178 to 0.0671 s per trace (0.036%--0.080% of trace duration). This measures Controller.decide, not all adapter scanning, event serialization, or engine-mode overhead; native-default end-to-end comparisons include those practical differences.

Per-seed ETA absolute error, observed prefix-miss fraction, decision maxima, action reasons and switching counts are in summary.json:diagnostics. The shared calibrated APC-loss proxy is deliberately coarse: observed differences across roomy, burst and real-tool slices are limitations, not silently refit test parameters. The original retention_revocations counter counts native retention-owner releases, including explicit handle close; policy DROP reasons and logical resolution events must be used to distinguish pressure, TTL and semantic retirement.


No C2 claim follows from a small mean difference alone. The full policy must also be feasible and its ETA/fixed-price/score ablations must establish a causal information increment. Per-seed quality, recompute, TTFT, queue/dispatcher timestamps, eviction counts, block-time and decision CPU time are in summary.json and request-level traces.

## Practical and correctness boundaries

Native-default APC is a practical reference, not a matched causal intervention. All of its engine overhead is included. CPU KV demotion/prefetch is not implemented. The calibrated loss probability is a coarse independent proxy, not identified counterfactual truth. CUDA event calibration measures the device timeline, including dispatch gaps, not hardware FLOPs.

The optional trunk-recompute FLOPs column is explicitly an estimate: `2 * N * recomputed_tokens`, with parameter-count N derived from the pinned Qwen config (dimensions and formula in summary.json). It omits attention's quadratic term and is not substituted for GPU seconds or task E2E.

GPU lifecycle tests include exact 33-token KEEP/warm/cold output equality and the 129-token native warm/cold numerical boundary. Octopus matches each native path there; the model/backend is not universally bitwise cache-invariant. Task correctness remains an enforced measured constraint.

The initial nine real-tool anchors accidentally used replay ETA priors. They are preserved under eta-prior-diagnostic and excluded from the primary table. The amended anchors use the already-measured independent real-tool priors with the same traffic seeds, cohort, capacity, cost curve and gate. This is a nine-case repair, not a sweep. A shared-host process appeared on the selected GPU by the initial study's cleanup; its activity during measurement was not continuously monitored. No exclusive-device or universal causal-performance guarantee is inferred.

The original lease-based queue comparator is also superseded and archived under queue-lease-diagnostic. The corrected comparator uses ready-waiting free-LRU preference, without leases or reservations. Its repaired controlled rows must have fresh adjacent APC/full-policy controls; C2 is not inferred by comparing a new queue run with old control timings. Native dense APC and the local frequency policy are not hybrid/Mamba retention options.

## Exact replay model

The CPU replay optimum is restricted to its finite shared-block cost model. It is not a bound on live GPU latency or an oracle for the held-out tools. {"calibration": "synthetic shared-block model; not measured GPU parameters", "choices": [["a"], ["a"], ["b"]], "controller_choices": [["a", "b"], ["a"], ["b"]], "controller_cost": 0.7, "controller_gap": 0.09999999999999987, "cost_gpu_compute_seconds": 0.6000000000000001, "scope": "exact optimum of the fixed replay cost model; not GPU latency", "states_visited": 11, "status": "complete"}
