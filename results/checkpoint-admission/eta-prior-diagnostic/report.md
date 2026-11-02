# Checkpoint admission: bounded real-GPU results

Status: **complete**. 81/81 traces: 72 controlled replay and 9 separate real-tool traces.

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
| real_tool | tight_burst | apc | 3/3 | 3/3 | 0/3 | 0.000 [0.000, 0.000] |
| real_tool | tight_burst | cost_greedy | 2/3 | 1/3 | 1/3 | -0.295 [-2.740, 2.151] |
| real_tool | tight_burst | queue_aware | 2/3 | 1/3 | 0/3 | 0.452 [-0.975, 1.879] |

## Agent-state increment beyond queue-aware

- controlled/roomy_short: agent minus queue -0.024s; agent faster in 2/3 seeds. Trace-level interval: [-0.271062210300703, 0.22395735302078393].
- controlled/tight_burst: agent minus queue 0.318s; agent faster in 2/3 seeds. Trace-level interval: [-3.6060976617232274, 4.241173383341326].
- controlled/tight_low: agent minus queue 0.116s; agent faster in 1/3 seeds. Trace-level interval: [-0.36118750691208706, 0.5941448603394723].
- real_tool/tight_burst: agent minus queue -0.747s; agent faster in 2/3 seeds. Trace-level interval: [-3.029210357417846, 1.536197388152154].

No C2 claim follows from a small mean difference alone. The full policy must also be feasible and its ETA/fixed-price/score ablations must establish a causal information increment. Per-seed quality, recompute, TTFT, queue/dispatcher timestamps, eviction counts, block-time and decision CPU time are in summary.json and request-level traces.

## Practical and correctness boundaries

Native-default APC is a practical reference, not a matched causal intervention. All of its engine overhead is included. CPU KV demotion/prefetch is not implemented. The calibrated loss probability is a coarse independent proxy, not identified counterfactual truth. CUDA event calibration measures the device timeline, including dispatch gaps, not hardware FLOPs.

GPU lifecycle tests include exact 33-token KEEP/warm/cold output equality and the 129-token native warm/cold numerical boundary. Octopus matches each native path there; the model/backend is not universally bitwise cache-invariant. Task correctness remains an enforced measured constraint.

## Exact replay model

The CPU replay optimum is restricted to its finite shared-block cost model. It is not a bound on live GPU latency or an oracle for the held-out tools. {"calibration": "synthetic shared-block model; not measured GPU parameters", "choices": [["a"], ["a"], ["b"]], "controller_choices": [["a", "b"], ["a"], ["b"]], "controller_cost": 0.7, "controller_gap": 0.09999999999999987, "cost_gpu_compute_seconds": 0.6000000000000001, "scope": "exact optimum of the fixed replay cost model; not GPU latency", "states_visited": 11, "status": "complete"}
