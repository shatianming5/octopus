# B5 CPU acceptance — 2026-09-09

**S1–S4 CPU implementation is operational; generated-quality non-regression
and real GPU costs are unverified.** This is an engineering correctness gate,
not a positive inference-performance result.

Source: `feat/selective-inheritance-b5`, based on
`8d5824ed3f9dd3e850224f235cd11d2b74275da9`. No historical result was modified.

| Evidence | Observed outcome |
|---|---|
| [Full CPU suite](pytest.log), [JUnit cases](pytest.xml) | **251 passed**, 13 GPU selectors deselected; 14 upstream deprecation warnings |
| [Native scheduler preemption gate](native-preemption.log) | **7 passed**; same selectors as the suite, strengthened afterward to call native `_preempt_request` rather than manually incrementing the episode |
| [Pinned environment](environment.json) | vLLM **0.28.0**, torch **2.13.0+cu129**, CUDA hidden, CUDA uninitialized, one torch thread |
| [CPU projection output](projection.json) | **12 cases / 72 checks**; UTF-8-byte tokenizer, empty initial reducer, no model; quality and GPU cost are null |
| CPU four-method runner tests | Full/APC, Projected/APC, fixed and budgeted paths traverse native CPU allocation with explicitly mocked completions; verify orchestration/accounting, not generated quality |

The seven native selectors overlap the full suite; they are **not** seven
additional independent observations. CPU suite wall time (14.65 s) and native
selector time (6.09 s) are test durations, not model latency measurements.

## Covered contracts

- FULL preservation, exact selected token prefixes and mixed 32/33 tails,
  per-request parent frontier, and atomic batch context-budget rejection.
- Deterministic immutable capsules, raw INPUT excerpts without reducer facts,
  all selected conflict variants/fact IDs/evidence, all unresolved strings,
  mandatory constraints, exact output reserve, and ordered optional omission.
- Real native cache cold/warm/eviction behavior; parent release with accepted
  child ownership; native scheduler preemption/requeue and cold recomputation;
  capsule/peer APC hits beyond the parent frontier without incorrect reoffer.
- Public budget scheduling, replenishment and unused output reserve return,
  explicit exhaustion, required versus cancellable branches, unsubmitted
  condition cancellation, parent release, schema-valid `success=true` without
  semantic cancellation, and explicit visible-input verification.
- Identical projection/schema/sampling controls; complete branch/commit/final
  accounting; failed-cost denominators and Full-correct regression rejection.

The isolated remote snapshot was
`/mnt/disk2/user/octopus-selective-b5-4007b0b5`, reached through
`ssh -p 20068 user@198.51.100.10`. It used the existing
`/mnt/disk2/user/octopus-contention/.venv`; no packages or model were
installed. Source and needed historical JSON fixtures were copied read-only.
Artifacts were retrieved and the isolated snapshot removed; the shared
environment was untouched. No GPU inference or training process was launched.

The first broad snapshot omitted historical JSON fixtures and one legacy
`Octopus.__new__` test fixture did not initialize its reducer. The snapshot
inputs and fixture were corrected, not the historical data. The passing suite
above includes those unchanged historical report assertions. Earlier
diagnostics are retained in the session artifact
`files/selective-initial-cpu-diagnostics.log`.

The standard pre-push verifier ran **once**. Its overall exit code was 1, not
a clean pass: the unanchored `sk-` expression matched the substring in 5,372
historical `task-*` request-ID lines across 134 files. Every match was classified
from the existing scanner log; none was a non-task match or touched a staged
delivery file. `.env` and `auth.json` tracking checks passed. Two other failures
concern the host's Herdr/copilot-pane installation, not this repository.
[Classification](secret-gate.json) preserves these failures explicitly; neither
the scanner nor historical results were changed to hide them.

## Reproduce

In a checkout with the existing pinned vLLM environment and tracked fixtures:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q -m "not gpu"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_selective_native.py
PYTHONPATH=src .venv/bin/python -m examples.selective_demo
PYTHONPATH=src .venv/bin/python -m benchmarks.selective \
  --cpu --output /tmp/selective-cpu-new.json
```

The strengthened native selectors are now included in the current full suite,
so future reproduction normally needs only the first pytest command.

## Remaining acceptance gate

After explicit GPU authorization, freeze the existing local model path in the
[prepared protocol](../../benchmarks/configs/selective-gpu.json), run the four
[32/33 selected-prompt selectors](../../tests/test_gpu_selective.py), then the
12-case/four-method matrix and report commands in
[the interface/protocol document](../../docs/selective-inheritance.md).

No claim is made yet about model correctness, quality non-regression, E2E
speedup, kernel/FLOP reduction, or VRAM saving. Unique live KV and PyTorch
reserved allocation are distinct; full device/process VRAM peak is explicitly
unmeasured. Quality must pass before cost interpretation. Beating Full but
tying identical Projected/APC means projection benefit, not a unique KV effect.
