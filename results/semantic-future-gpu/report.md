# Frozen semantic-future GPU study

Status: **complete**; decision: **no-go**. Recorded traces: 16/16; independent blocks: 4/4.
Matched measured source: `62de966349d41d97dae596abe64d9b4bb5575509` (required `62de966349d41d97dae596abe64d9b4bb5575509`; reporting commits do not relabel runtime).

One fixed workload stratum; new seeds are not new task templates. Four independent paired seed means, not 64 independent serial tasks. No pooled background SLO or averaged P95 gate; every semantic block must pass. No workload-wide or distribution-free confidence claim. Endogenous scheduling timestamps never enter future-only policy. Preference is not residency or causal savings; owner release is not cache eviction. Nested CPU timers must not be summed.

| Seed | Method | FG mean E2E s | BG P95 TTFT s | BG degradation | BG 95% upper | BG gate | Correct | BG failures |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 1101 | apc | 12.860237414308358 | 8.314132766569108 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1101 | queue_aware | 12.873603893269319 | 8.261195059030598 | -0.006367195355764688 | 0.01749972794806043 | True | 4/4 | 0 |
| 1101 | semantic_naive_pin | 12.898349201364908 | 9.127349117235891 | 0.09781132602749665 | 0.20184599985390905 | False | 4/4 | 0 |
| 1101 | semantic_future_only | 12.981191748869605 | 7.77365425859578 | -0.0650072019713922 | -0.01807179197407982 | True | 4/4 | 0 |
| 1202 | queue_aware | 14.846357251924928 | 6.166004134589999 | -0.07615887557058043 | -0.047618696881185474 | True | 4/4 | 0 |
| 1202 | semantic_future_only | 15.369705588032957 | 6.22056051727459 | -0.0679847925138739 | -0.04431905852581974 | True | 4/4 | 0 |
| 1202 | apc | 16.01476808724692 | 6.674312250819351 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1202 | semantic_naive_pin | 15.585371214488987 | 6.876408606419239 | 0.030279727409378898 | 0.0493749952627085 | True | 4/4 | 0 |
| 1303 | semantic_future_only | 12.46273699507583 | 4.845457846705247 | -0.0879633502339432 | -0.05400231604314465 | True | 4/4 | 0 |
| 1303 | semantic_naive_pin | 11.926215620420408 | 5.070395288122553 | -0.04562448423458576 | -0.01825037168806394 | True | 4/4 | 0 |
| 1303 | queue_aware | 12.588100593944546 | 4.862484191700482 | -0.08475856522941061 | -0.05177039639708381 | True | 4/4 | 0 |
| 1303 | apc | 13.79939123033546 | 5.312788524395525 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1404 | semantic_naive_pin | 19.22816405200865 | 12.561132076353928 | 0.12940114212758358 | 0.14486314088426466 | False | 4/4 | 0 |
| 1404 | apc | 18.08170296699973 | 11.121940298989845 | 0.0 | 0.0 | True | 4/4 | 0 |
| 1404 | semantic_future_only | 18.552870204206556 | 11.226550361470387 | 0.009405738537370478 | 0.016357787557297554 | True | 4/4 | 0 |
| 1404 | queue_aware | 17.358665251289494 | 10.4296454670299 | -0.062245868378094316 | -0.04682191214032376 | True | 4/4 | 0 |

Correct tasks: **64/64**; background requests: **1536**, failures: **0**. Lifecycle: one passing non-skipped case.

## Four independent seed means

| Contrast | Mean difference s (future minus baseline) | One-sided 95% upper s |
|---|---:|---:|
| apc | -0.347398791 | 0.602884928 |
| queue_aware | 0.424944386 | 1.105997899 |

Foreground relative improvement vs APC: **2.287170%**; target >=10%. Paired Student-t: n=4, df=3, critical=2.3533634348018264; both uppers must be <0.
Method foreground means (seconds): `apc` 15.189024925, `queue_aware` 14.416681748, `semantic_future_only` 14.841626134, `semantic_naive_pin` 14.909525022.
Failed gates: foreground_improvement_at_least_10_percent, paired_upper_vs_apc_negative, paired_upper_vs_queue_aware_negative.

## Identity, resources and cleanup

Physical GPU `aa7be32a-43d1-b13e-45dd-ef46033f7631`; model `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; vLLM `0.28.0`. Physical engine/workload/calibration settings match across all seeds.
Chain: `2026-09-09T05:14:18Z` to `2026-09-09T05:32:51Z`; resource wall **1113.000 s**, summed process wall **1090.000 s**, summed trace makespans **763.961468 s**. **GPU kernel time is unmeasured**, not any of these wall times.
Cleanup: no matching verification processes. Full observed device/process state is preserved in cleanup.json.

## Semantic activation and ownership

| Seed | Future evaluations | Positive evaluations | Allocations with future | Retained union block-s | Retention-only block-s |
|---|---:|---:|---:|---:|---:|
| 1101 | 487 | 113 | 17 | 0.0 | 0.0 |
| 1202 | 557 | 136 | 19 | 0.0 | 0.0 |
| 1303 | 517 | 113 | 21 | 0.0 | 0.0 |
| 1404 | 562 | 151 | 16 | 0.0 | 0.0 |

## Immutable input paths (read-only)

The reporter writes only report.md and summary.json. Exact recomputed metrics, compact per-checkpoint ownership measurements and nested CPU timers are in summary.json; full allocation/adoption identities remain in raw traces. Paths below are relative to this report directory. Published gzip traces are lossless and preferred over redundant local JSON copies.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m benchmarks.semantic_future_report --results results/semantic-future-gpu
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
