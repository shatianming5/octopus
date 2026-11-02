# Real Qwen semantic S1: diagnostic_no_go

**24/24 real model calls completed; all three S1 gates failed.** This is the
frozen authored semantic diagnostic, not a native KV allocation or serving result.
No model output was retried, repaired or excluded.

| Metric | Qwen | Structural slot baseline | Required | Result |
|---|---:|---:|---:|---|
| Pairwise accuracy | 50% | 50% | >= 60% | FAIL |
| Gain over baseline | 0 pp | - | >= 10 pp | FAIL |
| Recall@1 | 50% | 50% | >= 80% | FAIL |

The CPU-only scorer returned `diagnostic_no_go`, `acceptance: false`, exit 1.
All 24 response IDs and prompt digests matched; missing, failed and malformed
responses were each zero. All calls finished with `stop`. Twelve rankings tied,
six were correct and six were wrong; ties count as half credit for accuracy.

## Complete task breakdown

Each task has four crossed ID/display-order views, not four independent tasks.
Metrics average views within a task, then macro-average the six semantic tasks.

| Family / current goal | Views | Pairwise accuracy | Recall@1 |
|---|---:|---:|---:|
| Event snapshot / first observed state | 4 | 50% | 50% |
| Event snapshot / latest state | 4 | 50% | 50% |
| Configuration / null inherits | 4 | 75% | 75% |
| Configuration / null clears | 4 | 25% | 25% |
| Ordering / numeric versions | 4 | 100% | 100% |
| Ordering / opaque identifiers | 4 | 0% | 0% |

All eight event-snapshot views emitted the same fields for both candidates:
importance 0.9, reuse_likelihood 0.8, criticality true. Configuration had four
ties, two correct and two wrong rankings. Ordering preferred the numeric-version
implementation even under the opposed opaque-identifier goal. Thus the positive
result on one ordering goal is not a semantic-goal-sensitive gain across its pair.

The result does not establish that Qwen can never rank branches or that LoRA
would help. It establishes no advantage for this frozen prompt/schema/model on
these three authored families. S2, serving and training were not launched.

## Source, input and model identity

- Local implementation: `63a78303573549a0ae844a6d00a24bfbc14171c9`.
- Native integration base: `01a98ce9faf69734b4a7ab913f24efc31faee8dd`.
- The only overlaid source was `benchmarks/semantic_intent.py`, SHA256
  `7cc1d8d6f8948c234e16e3d3123c77400a0ed0ae87d0f50a01ec60278326d0ff`,
  Git blob `feb4f989e352978df08bc016a6fd8ad89d9e0312`.
- The response producer's `source_commit` records the native base, not the
  untracked module overlay; both identities above are necessary for reproduction.
- Public semantic input digest:
  `d18c75373f8b58257397bf9dcd5e0fe7da7fd0e35db0f1c8a383f071c9e2e281`.
- Qwen2.5-7B-Instruct revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Same #19 schema and frozen prompt; seed 17, 256-output-token cap per call,
  APC producer, 224 MiB KV, max length 4096, one sequence. No inference-policy tuning.

Only the public input and source overlay were copied to the GPU host. Fixture
checks, private labels and the score join remained on the CPU host. Published
`labels.json` is the CPU evaluation artifact, not a GPU producer input.

## Cost, operational failure and release

The selected idle device was `.183` GPU 2, RTX 4090 D with 49,140 MiB total,
UUID `GPU-53e85207-9c8d-91f6-ba27-992fff75dc7f`.

| Result unit | Observed amount |
|---|---:|
| Completed model calls | 24 |
| Prompt / output tokens | 15,776 / 2,376 |
| Sum of producer-call wall time | 42.016341 s |
| Successful launcher interval, including startup | approximately 76 s |
| Failed pre-inference launcher interval | approximately 86 s |
| Sum of both launcher intervals | approximately 162 s |

The first startup failed because FlashInfer could not resolve `ninja`. It produced
no prompt attempts or model responses. Adding the **existing** venv-packaged ninja
to PATH repaired the startup; no package was installed and no model prompt was
retried. Its [terminal](pre-inference-failure-01/terminal.json) and
[stderr](pre-inference-failure-01/inference.stderr) remain separate from the valid
24-call run. Neither watchdog fired. Wall intervals use second-resolution terminal
timestamps and are not integrated GPU-busy time or the full CPU preparation cost.

One captured active snapshot showed the inference PID using 15,434 MiB, with
90% SM and 98% memory activity. These are a point sample, not campaign-average
utilization. Both launchers/model identities exited; the owned screen ended and
GPU 2 returned to 4 MiB and 0% utilization. Other users' allocations were untouched.

## CPU reproduction and durable evidence

From the repository with its test environment, this uses no GPU or model:

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.semantic_intent score \
  --inputs results/model-intent/semantic-s1-real/inputs.json \
  --labels results/model-intent/semantic-s1-real/labels.json \
  --responses results/model-intent/semantic-s1-real/responses.json \
  --output /path/to/new-score.json
```

Expected exit 1 and exact [score.json](score.json) content. Raw model response
text is retained in [responses.json](responses.json), alongside public inputs,
private CPU labels, environment, source identity, successful terminal and active
snapshot. Per-attempt files, launch commands, failure status and complete artifact
inventory remain at `/home/user/octopus-semantic-s1-20260914` and in the
supervisor's retrieved `semantic-s1-real` session evidence. No output is inferred
from pane state or from successful JSON parsing alone.
