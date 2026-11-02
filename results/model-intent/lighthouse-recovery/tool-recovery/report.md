# Frozen Generic Tool-Recovery LoRA Result

One rank-8 LoRA was trained from the original unmerged Qwen2.5-7B snapshot on
the frozen 256-case generic recovery set. It passed both genuinely unmasked
64-case held-out gates, which authorized exactly one same-task Lighthouse run.
That Lighthouse run still produced no patch and failed the original official
evaluation.

## Data and training

The deterministic seed-2301 dataset contains 256 training cases
(128 edit mismatch, 64 test failure, 64 missing path) and 64 task-level
held-out cases (32/16/16). Every case was built and validated with actual
`p1_tool_worker` execution; each training case contributes a failed-observation
to inspection transition and an inspection-evidence to corrected-action
transition. Train and held-out template, path, and prompt-hash overlap are all
zero. Generated data contains no Lighthouse or branch-value target material.

The sole scientific training run used rank 8, alpha 16, dropout 0.05,
`q_proj`/`v_proj`, bfloat16, microbatch 1, accumulation 8, two epochs, AdamW
`2e-4`, 0.05 warmup, linear schedule, and seed 2301. All 512 prompts fit without
truncation (490-796 tokens), and only target action plus end-token labels were
supervised.

- 128 optimizer steps; final epoch 2.
- 2,523,136 trainable parameters, all 112 tensors LoRA-only.
- Training loss 0.07820; first logged loss 1.14992, last 0.000153.
- Training time 249.102 seconds.
- Adapter SHA-256:
  `7819bf7aedc03be3c8d38c45c13570510e970598035aaa230c1c9e3172bfee84`.

## Genuinely unmasked held-out gate

The suite used no `FailureMemory`, action rejection, forced action/path schema,
hidden repair, retry, or replacement. All 64 cases remained in the denominator,
with two greedy model calls each under the global full action schema.

| Gate or diagnostic | Result | Required |
|---|---:|---:|
| Cases with no repeated known-failed action | 64/64 | >=58/64 |
| Correct first recovery action acquiring evidence | 64/64 | >=52/64 |
| Full two-step recovery | 64/64 | diagnostic |
| Corrected edit and fixture success | 48/48 | diagnostic |
| Invalid model steps | 0/128 | diagnostic |
| Failed model steps | 0/128 | diagnostic |
| Known-failed repeated proposals | 0 | diagnostic |

All 64 first actions and all 64 second actions exactly matched their held-out
synthetic oracle actions. The 128 prompts were unique and used 80,070 prompt
tokens and 8,253 output tokens; measured generation time was 233.671 seconds
and total suite time was 236.297 seconds. This perfect bounded synthetic result
passes the frozen gate but is not evidence of universal coding quality.

## Conditional original Lighthouse result

The gate pass triggered one merge and one original Lighthouse evaluation with
the actual trained adapter, unchanged `failure_aware_v1`, seed 17, temperature
0, bfloat16, APC, 12 slots, tools, image, and official evaluator.

- 12 model proposals, 4 distinct.
- 5 external submissions, 4 distinct; 3 succeeded and 2 failed.
- Zero repeated failed external submissions; 7 repeated proposals rejected.
- Successful relevant inspections at indices 0, 2, and 6.
- Both evidence-based edit variants still mismatched.
- No developer test action; 0 patch bytes.
- Official `FAIL_TO_PASS` failed; `resolved=false`; `QUALITY_FAIL`.

The learned adapter improved the first action from the earlier invalid
`read(start=0)` to a valid same-file read and preserved useful inspection
behavior, but it still copied numbered read output into malformed exact-edit
preconditions. The task-level limitation therefore remains above generic
observation-to-action recovery for this recipe and case.

Lighthouse native totals were 49,968 prompt tokens, 13,808 actual prefill
tokens, 1,566 decode tokens, 36,160 adopted/inherited prefix tokens, 396
capsule tokens, 8,448 checkpoint-frontier tokens, and 8,459 selected-parent
tokens, with zero preemptions, evictions, or allocation rejections. GPU-engine
time was 59.433 seconds; solver end-to-end 61.845 seconds; control 0.000150
seconds; external tools 2.176 seconds; official evaluator 9.097 seconds.

## Attempts, budget, and artifacts

Three pre-output operational defects were repaired without changing the
scientific recipe or using outcome feedback:

1. CLI dispatch failed before model load (3 seconds).
2. fixture-root paths made data hashes nondeterministic and were normalized
   before launch.
3. Transformers 5.16 required the frozen 5% warmup to be supplied as its exact
   7-step equivalent; base weights loaded, but no optimizer step or output ran
   (9 seconds).

Attempt 4 completed in 662 wall seconds. Measured training, suite, merge, and
Lighthouse GPU/model components total 559.112 seconds; a conservative
single-GPU phase wall upper bound including failed attempts is 674 seconds,
well below the 3,600-second ceiling.

Primary artifacts:

- Frozen data/recipe:
  `attempt-4/data/freeze.json`
  (SHA-256 `96b6f3253a1bf1be3f178493bdb66675bf475339843f06451c59c37c640495db`).
- Training receipt:
  `attempt-4/execution/training.json`
  (SHA-256 `a6b0ee86a851464f58c3b0c7043953bcb3ed274f2d82a229a859867a8793bcc5`).
- Raw 64-case suite:
  `attempt-4/execution/suite/suite.json`
  (SHA-256 `0d7f250e9f7bb031cb3e4c18d5d0f6052942f768db061820e61e2ac0713e94b4`).
- Lighthouse episode:
  `attempt-4/execution/lighthouse/solver/episode.json`
  (SHA-256 `cf9ca9c0e97e6e0609ac58605a917d9722faac5dc883f0ca341fccf2c76d8b62`).

The official log is retained losslessly as gzip; its decompressed 50,624,671
bytes have SHA-256
`458a474114ddc19bb0a3041714fe7510ba946dd8f408fd17a897279d91037f60`.

Paid resources are released: GPU9 is back to 4 MiB and 0% utilization with no
compute process; no Lighthouse container or tmux session remains. The 10.1 MB
adapter and 15.23 GB merged model remain at the recorded remote paths for the
parent's final same-task decision; they are not committed.
