# R1 GPU checkpoint 02: quantitative gates pass

All six traces completed at native
`e680be9a4acd948dc45c6d6ab909c46414a08a41`, with runtime byte-identical to
library `1ad52828d3be2aa2034f72ec84e7eb45ef2ecd62`. The corrected benchmark
is packaged at `93e9dae`; it is not a further production-policy change.

| Gate / observation | Seed1601 | Seed1702 |
|---|---:|---:|
| Fixed / queue selector CPU, required <=2x | 1.8456x, pass | 1.6948x, pass |
| FG E2E versus queue, required <=0 | -1.9504% | -1.5078% |
| BG P95 versus APC, limit +5% | -1.8162% | -2.2657% |
| Baseline / fixed parent recomputed tokens | 33,888 / 0 | 33,888 / 0 |
| Fixed / APC output text and token-ID parity | 421/422 | 420/422 |

The actual GPU cost, timing-point and recompute gates pass together, unlike
[checkpoint01](../r1-gpu-01/report.md). These are two-seed R1 point results,
not a four-seed confidence result or completion of value admission,
causal/pressure validation, or two-region generalization.

## Costs and quality remain explicit

| Method / seed | Adapter (s) | Selector (s) | Prefix maintenance (s) | Deferred audit decode (s) |
|---|---:|---:|---:|---:|
| queue / 1601 | 1.1926 | 0.1241 | 0 | 0.0806 |
| fixed / 1601 | 1.3903 | 0.2290 | 0.0767 | 1.8051 |
| queue / 1702 | 1.2155 | 0.1485 | 0 | 0.1140 |
| fixed / 1702 | 1.4216 | 0.2518 | 0.0765 | 1.9090 |

The recorded controller bound sums adapter, prefix maintenance and audit
decode with intentional overlap double-counting. JSON serialization remains
unmeasured/null and is not zero or included in an all-in bound. Deferred
evidence work has not disappeared.

Official full-credit counts remain 129/422 for all six traces. Recorded
Answer F1 is 0.464514 for APC/queue and 0.464521 for fixed at seed1601;
0.464300 for APC/queue and 0.464503 for fixed at seed1702. Output parity is
not exact, and the original all-answers-correct gate remains failed.
R3 will require its separately frozen complete-cohort quality comparison.

## Reproduction and storage

[Quantitative replay](r1-gpu-02-quantitative.json) |
[complete result](r1-gpu-02-result.json) |
[compact export](r1-gpu-02-export.json.gz) |
[source](source.json) |
[raw validation](raw-archive-validation.json) |
[cost ledger](cost-ledger.json)

```bash
python3 -m benchmarks.controller_acceptance \
  --export results/controller-admission/r1-gpu-02/r1-gpu-02-export.json.gz \
  --output /tmp/r1-gpu-02-gates.json
```

Expected exit status: **0**, with cost, timing-point and recompute gates true.
This standard-library replay needs no model, native runtime, GPU or gold.
The regression suite retains both the real failing and passing checkpoints.

The same original 422 questions, intended arrivals, normal 33,198,047,232
native KV bytes, model revision, generation seed17, process hash seed1601,
and per-seed same-GPU method comparisons were preserved. The separate
424-request R2 probe was not used.

This run consumed 1,692.897638 process-seconds; all six raw hashes match the
compact export, and no owned GPU process remains. Full raw data and official
scoring persist outside Git under
`/home/ubuntu/experiments/octopus/r1-controller-gpu-02-20260912`.
The exact native source bundle remains under
`/home/ubuntu/experiments/octopus/r1-shared-prefix-followup-20260912-d3c687/evidence/candidate-source.bundle`.
These are owner-managed storage paths, not public download URLs.
