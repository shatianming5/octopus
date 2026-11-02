# Selector fast path: completed two-seed GPU diagnostic

**The first selector CPU repair does not establish a foreground or Pareto win.**
All six planned traces completed. Fixed still eliminates all 33,888 parent
recomputed tokens per trace, but is slower than both fresh baselines for both
seeds. Background P95 is 8.40--8.46% higher than queue. The original strict
QASPER QA gate remains failed; this pilot records 129/422 full-credit answers
per trace, not a successful end-to-end task.

This is the already completed pilot for library change `752a8ae`, measured in
native source `7d7675d`. Later CPU changes must not be attributed these GPU
results. Historical [sixteen-trace E2](../qasper-kv/e2/report.md), source
`616ae10`, remains unchanged and is not pooled with this diagnostic.

## Matched observations

Foreground is mean complete-paper E2E across four papers with three questions
each, including revisit waits. Background P95 covers the same 410 background
requests. Percentages compare fixed with the fresh same-seed baseline; positive
means slower.

| Seed | Method | FG E2E (s) | BG P95 (s) | Parent recomputed tokens | Selector CPU (s) |
|---|---|---:|---:|---:|---:|
| 1601 | APC | 241.352 | 43.305 | 33,888 | 0 |
| 1601 | queue | 241.126 | 43.624 | 33,888 | 0.198 |
| 1601 | fixed | 243.647 | 47.317 | 0 | 5.324 |
| 1702 | APC | 251.589 | 50.787 | 33,888 | 0 |
| 1702 | queue | 249.279 | 49.011 | 33,888 | 0.272 |
| 1702 | fixed | 251.972 | 53.128 | 0 | 5.507 |

| Seed | Baseline | Fixed FG change | Fixed BG P95 change |
|---|---|---:|---:|
| 1601 | APC | +0.951% | +9.264% |
| 1601 | queue | +1.046% | +8.465% |
| 1702 | APC | +0.152% | +4.610% |
| 1702 | queue | +1.080% | +8.401% |

The native CPU fixture previously reduced fixed selection from 10.210 to
2.143 ms. In this GPU pilot, fixed adapter CPU is still 8.371/8.486 seconds,
including the 5.324/5.507 selector seconds above. These are measured diagnostic
costs, not proof that removing any one cost yields an equal GPU speedup.
Two seeds do not supply the original four-seed confidence claim. No additional
budgeted cases were run because the retained workload did not bind that budget.

## Subsequent CPU-only audit encoding repair

After recovering this completed pilot, targeted profiling found that the
`eligible_future` and `future` audit fields independently hex-encoded the same
protected hashes. Reusing those encoded tuples within each draw removes that
duplicate work without a persistent cache or reduced audit content.

The unchanged native CPU fixture's fixed median fell from 2.176 to 1.442 ms
(33.7%); per-draw encodings fell from 4,350 to 2,232, while hash reads stayed
4,578. [Before](cpu-residual/before.json) and
[after](cpu-residual/after.json) retain the timing and profiler output.
The new conversion-count regression failed in ten parameterized cases before
the fix; all 17 selector-equivalence cases passed after it, including budgeted
and B7 audit coverage. The baseline profiling command's exploratory hash bound
4,464 was too tight and exited nonzero after recording measurements; the final
run used the documented 6,810 bound. The encoding-count regression, not that
unrelated hash bound, establishes the red/green signal for this fix.

The owner subsequently authorized a separate
[six-trace GPU validation of the audit-reuse repair](audit-reuse/report.md).
It completed at native `1c5a962` / library `cb73088` with a negative result:
fixed remains slower than queue and exceeds its 5% background point limit
in both seeds. The six observations above still belong only to the first
repair and are not pooled with the later run. Neither performance nor strict
QA acceptance is established.

## Controls, quality and cost

The frozen Qwen2.5-7B-Instruct revision is
`a09a35458c702b33eeacc393d103063234e8bc28`, with unchanged input tokens,
422-question cohort, decoding controls and 33,198,047,232 native KV bytes.
The compact export retains actual engine settings. All three methods for
seed1601 used `.68` GPU0; seed1702 used `.68` GPU3, with UUIDs retained in the
execution record. Seeds could run in parallel; methods within a seed used
the same physical GPU.

All six processes exited zero, with no omitted trace. Their process-wall sum is
1,684.046 seconds; it is not campaign elapsed time or sampled GPU utilization.
The ledger separately retains 5,240.746 seconds from earlier attempts.
The recorded official quality summaries have no missing predictions and
129/422 full-credit answers each. Answer F1 ranges from 0.464300 to 0.464514.
These summaries are archived scoring evidence, not independent regrading in
the CPU replay below. They do not establish exact generated-output parity.

## Artifacts and CPU replay

- [Compact request/native export](pilot-export.json.gz): all six traces,
  execution records, generated outputs, native counters, raw sizes and hashes.
- [Recorded metrics](pilot-result.json): per-trace values, paired point
  comparisons, recorded official scoring summaries and the bounded decision.
- [Source identities](source.json) and
  [exact measured native source](source/native-7d7675d.bundle).

With the test dependency installed, replay the measurements and comparisons:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m pytest -q tests/test_selector_fastpath_evidence.py
```

This uses the existing E2 pure CPU `paper_e2e` and `percentile` functions,
checks same-seed controls and the complete cohort, reproduces every published
timing/counter row and relative comparison, and checks execution/source
identity. It needs no model, vLLM, native extension, GPU, network, or gold.
It does not apply the four-seed statistical gate to two seeds.

The incremental source bundle requires repository base
`d2a8c6550ac73a507b03790f024a132068d16b5e`. To inspect the exact native code,
fetch the bundle's `HEAD` into a separate scratch clone and detach at
`7d7675d07fdaab4a776f3f060c6b230c63478793`; do not replace the current checkout
or rerun the historical pilot to reproduce the CPU report.

The owner-managed full-artifact root is
`/home/ubuntu/experiments/octopus/selector-fastpath-20260912`.
It retains `retrieved/pilot-artifacts.tar.xz`, with recorded SHA-256
`139470ec069eb7c6ec981fa487a75116cd96c1e3bd19c5a7906af77c546bbe16`,
the original plan, driver, input-only freeze, and native source. The original
remote root is `/mnt/disk2/user/selector-fastpath-20260912` on `.68`.
These paths require owner access and are not public download URLs. Large raw
data, input corpora, model weights, environments and gold remain outside Git.
