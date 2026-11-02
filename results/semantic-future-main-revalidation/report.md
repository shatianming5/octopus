# Frozen semantic-future GPU study

Status: **complete**; decision: **no-go**. Recorded traces: 16/16; independent blocks: 4/4.
Matched measured source: `ccc073837c845c9a195eff2e66b06906e14f2057` (required `ccc073837c845c9a195eff2e66b06906e14f2057`; reporting commits do not relabel runtime).

Merged-main revalidation using the SAME previously evaluated traffic seeds: not new held-out evidence; old and new runs are not pooled. One fixed workload stratum; new seeds are not new task templates. Four independent paired seed means, not 64 independent serial tasks. No pooled background SLO or averaged P95 gate; every semantic block must pass. No workload-wide or distribution-free confidence claim. Endogenous scheduling timestamps never enter future-only policy. Preference is not residency or causal savings; owner release is not cache eviction. Nested CPU timers must not be summed.

| Seed | Method | FG mean E2E s | BG P95 TTFT s | BG degradation | BG 95% upper | BG gate | Correct | BG failures |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 1101 | apc | 13.197644074913114 | 8.992058376145158 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1101 | queue_aware | 12.789317939139437 | 7.655410247873515 | -0.14864762575581225 | -0.04906584987320376 | True | 4/4 | 0 |
| 1101 | semantic_naive_pin | 12.670191288169008 | 8.895376737321673 | -0.010751891811553338 | 0.12895435265519753 | False | 4/4 | 0 |
| 1101 | semantic_future_only | 13.195413864974398 | 8.179232170641376 | -0.09039378655059804 | 0.07516077683714495 | False | 4/4 | 0 |
| 1202 | queue_aware | 15.121362834994216 | 6.0524884359479945 | -0.18061490777948175 | -0.1578026915614681 | True | 4/4 | 0 |
| 1202 | semantic_future_only | 15.099168347427621 | 6.094534732998541 | -0.1749226855884125 | -0.15246380418879657 | True | 4/4 | 0 |
| 1202 | apc | 16.27710029715672 | 7.3866225946897135 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1202 | semantic_naive_pin | 15.544883092108648 | 6.846624388651356 | -0.07310488645061752 | -0.042274296502448405 | True | 4/4 | 0 |
| 1303 | semantic_future_only | 12.623425587138627 | 4.894202583643053 | -0.10783423868923048 | -0.05183569526956988 | True | 4/4 | 0 |
| 1303 | semantic_naive_pin | 12.223259753140155 | 5.485720803618632 | -6.192335481047806e-06 | 0.053740384815577524 | False | 4/4 | 0 |
| 1303 | queue_aware | 12.589684542675968 | 4.868294816853145 | -0.11255697382063468 | -0.05324353982095864 | True | 4/4 | 0 |
| 1303 | apc | 13.140904333791696 | 5.4857547732525545 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1404 | semantic_naive_pin | 19.6627713732305 | 12.452240415375012 | 0.11173481803183893 | 0.1382469325566517 | False | 4/4 | 0 |
| 1404 | apc | 18.128049287304748 | 11.200728998863102 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1404 | semantic_future_only | 17.365059294330422 | 10.444587784648888 | -0.06750821435738363 | -0.05610264690719802 | True | 4/4 | 0 |
| 1404 | queue_aware | 17.8635229531792 | 10.499972837912505 | -0.06256344216717724 | -0.04337249979612168 | True | 4/4 | 0 |

Correct tasks: **64/64**; background requests: **1536**, failures: **0**. Lifecycle: one passing non-skipped case.

## Four independent seed means

| Contrast | Mean difference s (future minus baseline) | One-sided 95% upper s |
|---|---:|---:|
| apc | -0.615157725 | -0.037188351 |
| queue_aware | -0.020205294 | 0.416584051 |

Foreground relative improvement vs APC: **4.050841%**; target >=10%. Paired Student-t: n=4, df=3, critical=2.3533634348018264; both uppers must be <0.
Method foreground means (seconds): `apc` 15.185924498, `queue_aware` 14.590972067, `semantic_future_only` 14.570766773, `semantic_naive_pin` 15.025276377.
Failed gates: each_semantic_block_background_gate, foreground_improvement_at_least_10_percent, paired_upper_vs_queue_aware_negative.

## Identity, resources and cleanup

Physical GPU `aa7be32a-43d1-b13e-45dd-ef46033f7631`; model `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; vLLM `0.28.0`. Physical engine/workload/calibration settings match across all seeds.
Chain: `2026-09-09T14:41:34Z` to `2026-09-09T15:00:06Z`; resource wall **1112.000 s**, summed process wall **1088.000 s**, summed trace makespans **766.357145 s**. **GPU kernel time is unmeasured**, not any of these wall times.
Cleanup: no matching verification processes. Full observed device/process state is preserved in cleanup.json.

## Semantic activation and ownership

| Seed | Future evaluations | Positive evaluations | Allocations with future | Retained union block-s | Retention-only block-s |
|---|---:|---:|---:|---:|---:|
| 1101 | 480 | 109 | 21 | 0.0 | 0.0 |
| 1202 | 545 | 138 | 22 | 0.0 | 0.0 |
| 1303 | 548 | 123 | 20 | 0.0 | 0.0 |
| 1404 | 596 | 176 | 14 | 0.0 | 0.0 |

## Immutable input paths (read-only)

The reporter writes only report.md and summary.json. Exact recomputed metrics, compact per-checkpoint ownership measurements and nested CPU timers are in summary.json; full allocation/adoption identities remain in raw traces. Paths below are relative to this report directory. Published gzip traces are lossless and preferred over redundant local JSON copies.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m benchmarks.semantic_future_report --results results/semantic-future-main-revalidation --expected-source ccc073837c845c9a195eff2e66b06906e14f2057
```

- `semantic-future-gpu-protocol.json`
- `execution-plan.json`
- `lifecycle-junit.xml`
- `cleanup.json`
- `run-status.txt`
- `run.exit`
- `COMPLETE`
- `run-confirmation.sh`
- `gpu-uuid.txt`
- `traces/real_tool-tight_burst-1101-apc.json.gz`
- `traces/real_tool-tight_burst-1101-queue_aware.json.gz`
- `traces/real_tool-tight_burst-1101-semantic_naive_pin.json.gz`
- `traces/real_tool-tight_burst-1101-semantic_future_only.json.gz`
- `traces/real_tool-tight_burst-1202-queue_aware.json.gz`
- `traces/real_tool-tight_burst-1202-semantic_future_only.json.gz`
- `traces/real_tool-tight_burst-1202-apc.json.gz`
- `traces/real_tool-tight_burst-1202-semantic_naive_pin.json.gz`
- `traces/real_tool-tight_burst-1303-semantic_future_only.json.gz`
- `traces/real_tool-tight_burst-1303-semantic_naive_pin.json.gz`
- `traces/real_tool-tight_burst-1303-queue_aware.json.gz`
- `traces/real_tool-tight_burst-1303-apc.json.gz`
- `traces/real_tool-tight_burst-1404-semantic_naive_pin.json.gz`
- `traces/real_tool-tight_burst-1404-apc.json.gz`
- `traces/real_tool-tight_burst-1404-semantic_future_only.json.gz`
- `traces/real_tool-tight_burst-1404-queue_aware.json.gz`
