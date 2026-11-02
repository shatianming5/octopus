# Model x native KV functional smoke

**Terminal outcome: `functional_native_ordering_verified`.** The opt-in
`b6_model_intent` path transported one real Qwen Brain response through the public
Interface into actual native KV admission at repository commit `89cfcf2` and
native integration commit `01a98ce9`. This is a functional result, not an R2/R3
performance, quality or generalization pass.

## Validation

The native CPU gate completed with **127 passed, 1 deselected**. It covered the
public Interface, constrained native allocation and victim ordering, prefix
closure, shared prefixes, queued/held authority, hint update/consumption/reset/
release/cancellation, malformed model data and unchanged legacy policies. The
single deselection was the unchanged selector timing threshold, not a failed
functional case.

The real-model smoke made exactly three Qwen calls: two branch analyses and one
strict structured intent response. There was no retry or handwritten replacement.

| Call | Prompt tokens | Output tokens | Model time | Finish |
|---|---:|---:|---:|---|
| Migration analysis | 98 | 96 | 2.1268 s | length |
| Typography analysis | 94 | 96 | 2.0983 s | length |
| Structured intents | 709 | 120 | 3.1131 s | stop |
| **Total** | **901** | **312** | **7.3382 s** | - |

The exact model fields were:

| Candidate | Importance | Reuse likelihood | Critical | Cached before | Expected after | Actual after |
|---|---:|---:|---|---:|---:|---:|
| Migration checkpoint | 1.0 | 0.9 | true | 192 tokens | 128 tokens | 128 tokens |
| Typography checkpoint | 0.0 | 0.0 | false | 176 tokens | 0 tokens | 0 tokens |

The actual native `allocate_slots(delay_cache_blocks=True)` draw requested 247
of 255 free blocks and left eight. Twelve future-eligible blocks were present;
the budget retained eight winner blocks and reclaimed four lower-priority blocks.
The subsequent native hit check matched 128 winner tokens and zero loser tokens.
Controlled churn did not run model inference or write fabricated KV payloads.

## Execution boundary

The smoke ran once on shared `gpu.183` GPU 9 with the existing Qwen2.5-7B
snapshot. The launcher exited 0, its screen disappeared, and its 15,366 MiB model
allocation was released; the unrelated co-runner was unchanged. Original terminal
and structured artifacts remain in the supervising Copilot session and at
`/home/user/octopus-model-intent-smoke-20260914`.

This result proves only that model-produced importance, reuse likelihood and
criticality can control real native KV retention ordering through the shipped
Interface. It does not establish end-to-end speed, background non-regression,
answer quality or benefit on the prior R3 workload.

## Offline ranking preflight

The proposed S1 gate was checked against the actual frozen R2 and R3 input
cohorts before any new model or GPU run. Both cohorts contain the same four
foreground checkpoints. Each checkpoint has two definite later consumers, and
the freeze records their exact `selected_parent_tokens`.

| Checkpoint paper | Later-used prefix label | Definite-consumer parent work available to runtime |
|---|---:|---:|
| 1602.07563 | 26,131 tokens | 26,131 tokens |
| 1601.03313 | 17,889 tokens | 17,889 tokens |
| 1603.00957 | 13,930 tokens | 13,930 tokens |
| 1602.03483 | 9,993 tokens | 9,993 tokens |

The approved label is future prefill work avoided if the checkpoint survives.
A label-equivalent branch baseline induced from the same definite-consumer work
orders all **6/6 non-tied pairs correctly** in R2 and R3. It has 100% pairwise
accuracy and leaves zero possible model gain against that specific baseline.
This is an information/fairness diagnostic, not an observed branch score from
the actual `b6_semantic_value` per-block marginal allocator.

S1 therefore lacks one fair executable branch-level runtime baseline. Using the
label-equivalent baseline makes its required 10-point gain unreachable; using
history alone is a different ablation and would withhold currently known
definite dependencies from runtime. Model inference cannot decide that protocol
ambiguity, so the run stopped before model or GPU work.

The exact source files are bound by SHA256 in the machine-readable artifact;
both record native source `beb50062f7bf9603f2b4ec657072c8ff763559ee`.

This is a protocol-level blocker, not a zero-shot model failure. #19 does not expose arbitrary
background-history prefixes as model-intent checkpoints. S2/S3 and conditional
training were not run, and no GPU was claimed. Machine-readable rows and gate
state are in [`offline-ranking-preflight.json`](offline-ranking-preflight.json).

## Revised semantic S1 CPU readiness

The old QASPER preflight above remains unchanged. The new
[verifier-gated repair diagnostic](../../docs/semantic-intent-s1.md) has an
implemented fixture builder, label-free model export, collector and scorer.
CPU preparation completed 3 families / 6 opposed-goal tasks / 24 crossed ID/order
views. All 12 opposed-goal view pairs preserve identical candidate state but flip
the checker-selected next consumer. The executable structural slot baseline has
pairwise accuracy 0.5 and recall@1 0.5. These are controlled CPU input properties,
not measurements of the full native value allocator or a model.

[`semantic-s1-readiness.json`](semantic-s1-readiness.json) is generated by the
preparation command. Model results remain null. This new authored workload uses
binary verified reuse and equal abstract slots, not measured prefill tokens or
native KV capacity; it does not promote the earlier smoke into a performance win.

## Revised semantic S1 real inference

The [subsequent real Qwen run](semantic-s1-real/report.md) completed all 24 calls
on an idle `.183` GPU 2. The CPU label join returned `diagnostic_no_go`: pairwise
accuracy 50%, gain 0pp and recall@1 50%, with no missing or malformed response.
Twelve views tied, six ranked correctly and six incorrectly. Raw responses,
per-goal scores, the pre-inference startup failure and the full accounting are
preserved separately; the CPU-readiness artifact above is unchanged.
