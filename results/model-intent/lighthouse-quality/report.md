# Lighthouse-10176 official quality preflight

**Terminal decision: `QUALITY_FAIL`.** The frozen Qwen+LoRA model completed the
single permitted repair episode, but produced no patch. The pinned official
SWE-bench Multimodal evaluator completed successfully and reported
`resolved=false`, so Step3 stops. No task substitution or paired policy pilot
was launched.

## Frozen setup and masking

The task was `GoogleChrome__lighthouse-10176` at base
`a5dbdd2eff4238213dcaad404d0e06d8d5d2489b`, using the official task repository
at `8d3ed32f37af9ceb8b5778b92a0088384ed057d4` and immutable image
`sha256:b8d51faee583632569625837d3f5b1a8a58f19c90f659597bd33234b145f7eae`.
New task assets totalled 1,104,822,377 bytes, below the 2 GB limit.

The model saw only the exact official problem statement and clean base code
through bounded real repository tools. The solver container did not contain or
mount `eval.sh`, `test.patch`, `gold.patch`, or `hints.md`. The final prediction
was the model-created Git diff; no authored repair or fallback was supplied.
The official evaluator injected its masked test changes only after the model
terminated.

The quality floor was frozen before model output as official `resolved=true`
over every listed FAIL_TO_PASS and PASS_TO_PASS test. This instance has one
FAIL_TO_PASS file,
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`, no PASS_TO_PASS
files, and the official injected test covers both issue-reported bugs. A
base-only masked evaluator sanity produced the expected unresolved result,
establishing evaluator functionality without becoming a model outcome.

## Actual model trajectory

The model used all 12 allowed actions. Every action was the same attempted exact
replacement in `lighthouse-core/lib/stack-collector.js`:

```text
replace "stackPacks: [{" with "stackPacks: [{ id: 'default',"
```

The visible tool correctly returned `edit old text must match exactly once` on
all 12 attempts. The model neither inspected the repository nor changed its
action in response to that observation. It reached the action limit with:

- one distinct action across 12 turns;
- zero patch bytes;
- 1,892 actual native prefill tokens and 540 decode tokens;
- 100.863 seconds of final GPU-engine scope.

Raw model JSON actions, tool errors, contexts, request metrics, native records,
empty prediction patch, and stderr are preserved under `execution/solver/`.

## Official evaluation

The pinned evaluator ran successfully in 11.355 seconds. Its Jest suite recorded
one passing test and one failing test in the official FAIL_TO_PASS file. The
failure showed the duplicate library entry remained present. The official
grade was therefore:

| Field | Result |
|---|---|
| Evaluator status | complete |
| FAIL_TO_PASS | failed |
| PASS_TO_PASS | none |
| Official resolved | false |
| Quality gate | fail |

This is a model-quality failure, not an evaluator or container failure.

## Infrastructure repairs and budget

Three pre-outcome infrastructure attempts were preserved:

1. The initial relay used `json.load(stdin)` and waited for EOF before opening
   the bidirectional tool channel. It was explicitly cancelled after diagnosis;
   no GPU model work occurred.
2. After newline framing, vLLM startup logs polluted protocol stdout. Solver
   stdout was isolated to stderr and the end-to-end SSH transport passed with
   simulated startup noise. This attempt was conservatively charged 18 GPU
   seconds.
3. FlashInfer initially selected `/usr/local/cuda` and failed JIT compilation
   before the first action. The required CUDA 12.9 environment was frozen, the
   exact failed cache directory was removed, and 11.792 GPU-engine seconds were
   charged.

The final valid episode used 100.863 GPU-engine seconds. Total charged GPU time
was 130.655 of 1,200 authorized seconds, leaving 1,069.345 seconds unused.
Infrastructure failures did not alter the task, prompt, model, evaluator,
quality floor, or action budget.

After completion, no owned tmux session, Docker container, remote process, or
GPU allocation remained. The selected GPU
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` reported 4 MiB used and 0%
utilization.

## Mechanism continuation

No paired pilot is authorized because absolute quality failed. Independently,
the actual repair loop also showed no genuine contention: at most one scheduler
request was active, native cache evictions were zero, and there was no necessary
Brain reducer/commit action. The P1-style sequential loop therefore remains
insufficient evidence for a natural multibranch contention workload.

Supplementary v4.1 native validation totals 17/17 affected cases (12 seam cases
plus five pressure/bad-binding cases). An independent current runtime and v4
mechanism review reported Standards 0, Spec 0, `VERDICT: ready`. These supplement
the existing mechanism artifacts without rewriting historical run metadata.

For the previously reported 100.262% net-oracle figure, the actual native ledger
saved 9,176 total prefill tokens versus 9,152 continuation-oracle tokens on each
seed. The additional 24 tokens came from lower measured normal-reducer prefill,
so the fraction is `9176 / 9152 = 1.0026223776`; no rerun was used.
