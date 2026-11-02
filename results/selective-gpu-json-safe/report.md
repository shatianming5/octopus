# Selective B5: accepted JSON-safe engineering epoch

**Quality non-regression passes on this fixed 12-case development suite.**
Measured source: da66d002ccd408872d395ee74c5375ea98090774. All 48 real workflows complete without parsing/runtime errors. This is not held-out or distribution-wide evidence.

| Method | Solved / all tasks | Mean task E2E s | Total task E2E s | Actual prefill | Actual decode | Peak unique live KV blocks |
|---|---:|---:|---:|---:|---:|---:|
| Full + APC | 5/12 | 6.876369 | 82.516430 | 61412 | 9971 | 304 |
| Projected + APC | 12/12 | 5.103252 | 61.239018 | 62114 | 8400 | 286 |
| Selective fixed | 12/12 | 5.097737 | 61.172845 | 62113 | 8397 | 286 |
| Selective budgeted | 12/12 | 5.078264 | 60.939173 | 60102 | 6523 | 278 |

## Interpretation

All projected methods solve 12/12, including every Full-correct case. Full solves 5/12 under the unchanged answer/provenance/conflict scorer; its seven semantic failures remain in the denominator and cost totals. No expected answer value is encoded in the schema or supplied to the projector/verifier.

Selective fixed uses 25.87% less task E2E than Full in this sample, but only 0.11% less than identical Projected + APC. Budgeted uses 26.15% less than Full and 0.49% less than Projected. These are one-repeat point estimates, not established stable speedups. The result supports context-projection utility on this fixture, not a unique KV-runtime performance advantage.

Budgeted starts 42 branches, skips six unsubmitted backups and aborts six already-started backups. All emitted/aborted work and preparation of unstarted candidates are included. Live KV peak is 278 versus 286 for fixed projection and 304 for Full. The KV pool remains reserved at 1 GiB; PyTorch reserved peaks are identical at 17114857472 bytes. Live-block reduction is not reserved-VRAM reduction, and full device/process VRAM peak is unmeasured.

## Measurement and total work

Task E2E includes input tokenization, Brain, projection/schema construction, capsule prefill, all branches, visible-input verification/cancellation, commit processing and final continuation. Model loading/teardown and offline scoring are outside task E2E; whole-chain wall includes setup/teardown. No token/FLOP/kernel-time conversion is used.

This epoch ran 2026-09-09T22:28:44Z to 22:39:37Z, 653 seconds resource wall. The four selected-prompt GPU correctness cases passed on a9638a5; runtime and selector source were unchanged, so they were referenced rather than rerun.

| Epoch | Source | Workflows | Resource wall s | Disposition |
|---|---|---:|---:|---|
| Initial | a9638a5 | 48 | 652 | Includes four GPU correctness cases; quality failed |
| Common output contract | 39b1ed8 | 48 | 674 | Quality failed, including 22 JSON grammar errors |
| JSON-safe acceptance | da66d00 | 48 | 653 | Quality passed |

**Cumulative: four GPU correctness cases plus 144 workflows, 1979 seconds of recorded GPU-host chain wall.** Earlier failures are preserved separately, not pooled into the accepted result. Extra epochs repaired concrete output-contract defects, not model/ratio/seed tuning. This suite was used during development, not newly held out.

## Cleanup and reproduction

GPU6 UUID is GPU-aa7be32a-43d1-b13e-45dd-ef46033f7631. cleanup/post-exit.json confirms execution PIDs are gone and the device has 3 MiB, 0% utilization and no compute apps. The earlier orchestrator-ps snapshot was taken while run.sh finished its own cleanup, not the final absence proof.

Raw records, final text/schema, protocol/source/order, environment and cleanup receipts are retained. CPU-only reproduction into a fresh path:

    PYTHONPATH=src .venv/bin/python -m benchmarks.selective --report results/selective-gpu-json-safe/records.jsonl --output /tmp/selective-json-safe-summary-new.json

Earlier epochs: results/selective-gpu and results/selective-gpu-output-contract. Native CPU regressions: results/selective-output-contract-cpu and results/selective-final-json-cpu. Their original data and scores were not overwritten.
