# R1 GPU checkpoint 01: timing points pass, selector-cost gate does not

All six fixed-protocol traces completed at native source
`93450e23f4703cae816c297dd32da7b8498a26d2`, library
`54faa7187524aa9c872340f60d769a3dd90293b8`. This is a valid measured
checkpoint, not complete R1 acceptance or an R3 statistical claim.

| Gate or observation | Seed1601 | Seed1702 |
|---|---:|---:|
| Fixed / queue selector CPU, required <=2x | 3.2149x, fail | 4.4474x, fail |
| Fixed FG E2E versus queue | -1.3408% | -0.5656% |
| Fixed BG P95 versus APC, limit +5% | +2.0056% | -3.7713% |
| Queue parent recomputed tokens | 33,888 | 33,888 |
| Fixed parent recomputed tokens | 0 | 0 |
| Fixed / APC output text and token-ID parity | 422/422 | 420/422 |

The FG and BG timing point conditions both pass. The implementation-only
falsifier must not be described as a latency failure. However, real selector
cost exceeds the owner's 2x bound despite the earlier native CPU fixture's
1.64x median, so the combined R1 gate remains open. The next action is a
trace-shaped CPU cost reproduction and targeted repair, not another blind
GPU rerun or a conclusion that implementation overhead is already ruled out.

## Accounted controller costs

| Method / seed | Adapter (s) | Selector (s) | Prefix maintenance (s) | Deferred audit decode (s) |
|---|---:|---:|---:|---:|
| queue / 1601 | 1.3011 | 0.1340 | 0 | 0.0813 |
| fixed / 1601 | 1.6067 | 0.4309 | 0.0494 | 1.7901 |
| queue / 1702 | 1.2782 | 0.1705 | 0 | 0.1148 |
| fixed / 1702 | 1.9002 | 0.7581 | 0.0461 | 1.8575 |

Adapter timing includes native/provider/selector work; prefix maintenance
overlaps adapter and free-path timing. Adding adapter, maintenance and audit
decode therefore gives a deliberately conservative bound over the recorded
controller timers: fixed/queue 2.4929x and 2.7308x. It intentionally double-counts
overlap. JSON serialization CPU is **unmeasured**, not zero, and excluded from
that bound. Deferred work has not been declared eliminated.

The initial narrow adapter-plus-audit calculation is preserved in the external
original result. The [accounting correction](r1-gpu-accounting-corrected.json)
adds maintenance and explicitly separates passing timing points from the
unmet selector gate. This correction did not rerun inference or scoring.

## Controls and retained evidence

The original 422 questions, four three-question foreground papers, 410
single-question background papers, pinned Qwen revision, generation seed17,
normal 33,198,047,232 native KV bytes, and same-GPU comparisons within each
seed are unchanged. Process `PYTHONHASHSEED=1601` is common to all methods.
Official scoring remains 129/422 full-credit questions per trace; relative
timing improvement does not fix that absolute quality limitation.

This run consumed 1,695.063312 process-seconds in the new R1--R4 campaign.
All six raw hashes were validated against the compact export once, and the
supervisor confirmed zero remaining owned GPU processes.

- [Corrected result](result.json)
- [Compact six-trace export](r1-gpu-export.json.gz)
- [Raw validation record](raw-archive-validation.json)
- [Continuing protocol](../../../docs/controller-admission-protocol.md)

The quantitative gate is independently replayable without native dependencies:

```bash
python3 -m benchmarks.controller_acceptance \
  --export results/controller-admission/r1-gpu-01/r1-gpu-export.json.gz \
  --output /tmp/r1-gates.json
```

For this checkpoint the expected exit status is **1**: selector cost fails,
while timing points and recompute preservation pass. The
[recorded replay](acceptance.json) keeps those verdicts separate. Four focused
CPU regressions cover this real counterexample, the inclusive 2x boundary,
recompute loss, independent background failure, invalid cohorts/controls and
explicitly unmeasured serialization. This does not rerun inference or turn
these two seeds into a formal confidence result.

Full raw files, original/corrected results and the execution/source records
remain outside Git under
`/home/ubuntu/experiments/octopus/r1-controller-gpu-20260912`.
The exact native source bundle remains at
`/home/ubuntu/experiments/octopus/r1-controller-fastpath-20260912-d3c687/evidence/r1-native-source.bundle`.
These are owner-managed paths, not public download URLs. Historical selector
pilots and future R2 workload variants remain separate datasets.
