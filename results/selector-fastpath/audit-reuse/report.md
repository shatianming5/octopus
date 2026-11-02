# Audit-encoding reuse: completed GPU validation

**No consistent GPU end-to-end benefit was demonstrated.** All six authorized
traces completed at native source `1c5a962`, matching the audit-reuse library
change `cb73088`. Fixed is slower than fresh same-seed queue in both seeds,
and its background P95 exceeds queue by more than 5% in both. CPU fixture
improvement is real but is not a GPU speedup claim.

## Matched results

The unchanged diagnostic uses seeds 1601/1702, APC/queue/fixed, four complete
three-question foreground papers and 410 background requests. Foreground is
mean full-paper E2E including revisit waits, not mean successor response.
Positive relative changes mean slower.

| Seed | Method | FG E2E (s) | BG P95 (s) | Parent recomputed tokens | Selector CPU (s) |
|---|---|---:|---:|---:|---:|
| 1601 | APC | 241.126 | 43.020 | 33,888 | 0 |
| 1601 | queue | 241.197 | 43.569 | 33,888 | 0.199 |
| 1601 | fixed | 243.606 | 47.285 | 0 | 5.523 |
| 1702 | APC | 251.709 | 50.800 | 33,888 | 0 |
| 1702 | queue | 249.277 | 48.985 | 33,888 | 0.245 |
| 1702 | fixed | 251.362 | 52.578 | 0 | 4.852 |

| Seed | Baseline | Fixed FG change | Fixed BG P95 change |
|---|---|---:|---:|
| 1601 | APC | +1.029% | +9.914% |
| 1601 | queue | +0.999% | +8.528% |
| 1702 | APC | -0.138% | +3.499% |
| 1702 | queue | +0.837% | +7.335% |

Seed1702 improves slightly against APC and meets the 5% background point
criterion against APC alone. It does not beat the stronger same-seed queue
baseline. Neither the all-comparison foreground point criterion nor the
all-comparison 5% background point criterion passes. Two seeds do not satisfy
the original four-seed confidence protocol, and no formal acceptance is claimed.

The KV mechanism still works: fixed eliminates 33,888 parent-recomputed tokens
per seed and raises parent-adopted tokens from 34,016 to 67,904. Its measured
adapter CPU is 8.470/7.875 seconds and selector CPU is 5.523/4.852 seconds.
The CPU fixture's 33.7% reduction does not transfer directly to these real
trace timings. Comparisons with the [earlier pilot](../report.md) are
descriptive only: source, run time and scheduling epochs differ. This is not a
paired old-versus-new implementation causal experiment.

## Quality and controls

All six official scoring summaries retain 129/422 full-credit answers and zero
missing predictions. Strict QA remains failed; Answer F1 is
0.464300--0.464514. Queue and APC outputs match exactly in both seeds.
Fixed matches APC on all 422 output texts/token-ID sequences for seed1601,
but only 420/422 for seed1702. No exact output-parity claim is made for that pair.

The frozen model remains Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`, with the same input-only freeze,
generation controls, arrival schedules and 33,198,047,232 actual KV bytes.
Seed1601's three methods used `.68` GPU0 and seed1702's used GPU3, with actual
UUIDs and native settings retained in the export. Each seed's methods ran
sequentially on its same physical GPU; the seed groups ran in parallel.

The execution ledger records 6/6 exit-zero processes, no omitted trace, and
1,681.550242 process-seconds for this run. Including 6,924.792342 prior observed
process-seconds gives 8,606.342584, below the retained 10,800-second ceiling.
This is a summed process-wall cost, not elapsed campaign wall time or a
continuous GPU-utilization measurement. Both GPUs were observed at 100%
utilization at launch; that single snapshot is not campaign-wide utilization.

## Reproduction and storage

[Compact export](pilot-export.json.gz) preserves all requests, generated
outputs, timing/native counters, source/environment, raw hashes and execution.
[Recorded result](pilot-result.json) adds point comparisons, recorded official
scores, output parity, and explicitly noncausal historical comparisons.
[Source identity](source.json) and the
[exact native source bundle](source/native-1c5a962.bundle) separate this run
from native `7d7675d` and original E2 native `616ae10`.
The incremental bundle requires base
`d2a8c6550ac73a507b03790f024a132068d16b5e`.

The existing offline replay now covers both six-trace pilots independently:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m pytest -q tests/test_selector_fastpath_evidence.py
```

It reuses the unchanged full-paper and percentile functions, recomputes rows
and paired point changes, and checks exact cohort, controls, output parity,
cost and source identity. It requires no model, native runtime, GPU, network
or gold. Recorded official scores are retained, not independently regraded by
this replay.

Permanent owner-managed inputs, commands, scoring and full raw archives are
outside Git under
`/home/ubuntu/experiments/octopus/selector-audit-reuse-20260912`.
The retained `trace-data.tar.zst` is 151,879,522 bytes, SHA-256
`7438dc28fdcb847eb1d37d648ce5496edccd3b0fec9c8f394c7a89113fb1541e`.
The supervisor checked all six raw files against the compact export once;
[archive validation](raw-archive-validation.json) records their sizes/hashes.
[Source validation](source-validation.json) confirms that local and measured
remote selector/backend files match library `cb73088`, and records the retained
driver, exporter, input-only freeze, plan and commands.
Original remote raw files remain under
`/mnt/disk2/user/selector-audit-reuse-20260912/trace-data` on `.68`.
These are owner-access paths, not public download links. No further run,
budgeted expansion, tuning or quality-gate waiver follows from this result.
