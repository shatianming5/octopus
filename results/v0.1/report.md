# Octopus v0.1 initial results

These are the complete first-run records from the approved single-GPU protocol. The three result units are reported separately; they are not pooled as 99 homogeneous samples.

## Claim status

- **C1 lifecycle correctness:** passed: 11 state/native/GPU cases, including cold-prefill references for aligned and nonaligned forks, cancellation/release, the round barrier, and the public three-branch Brain continuation demo.
- **C2 measurable state-reuse cost:** measured below without assuming an advantage over native APC.

## Key findings

- In the matched `S=16384, N=8` point, median branch trunk recomputation was 131072 tokens for Independent, 0 for APC, and 0 for Octopus. Octopus branch time was +0.7% relative to APC.
- Native-default APC completed the same sentinel branch phase 19.5% faster than matched synchronous/eager APC, demonstrating the cost of the integration-friendly execution mode.
- Under the controlled retention pressure, APC recomputed a median 4096 trunk tokens across the two forks while Octopus recomputed 0. Octopus branch time was 8.1% lower in this narrow scenario; this is not a general throughput claim.
- Semantic tasks: apc: 12/12 solved, 12/12 valid Commit schemas; independent: 12/12 solved, 12/12 valid Commit schemas; octopus: 12/12 solved, 12/12 valid Commit schemas. Octopus median total time was +0.5% relative to APC.

## Runtime matrix

| Method | Shared tokens | Branches | Records | Median branch s | Median total s | Median branch trunk recompute | Median inherited |
|---|---:|---:|---:|---:|---:|---:|---:|
| apc | 4096 | 1 | 3/3 | 2.345 | 2.605 | 0 | 4096 |
| apc | 4096 | 4 | 3/3 | 2.427 | 2.689 | 0 | 16384 |
| apc | 4096 | 8 | 3/3 | 2.454 | 2.724 | 0 | 32768 |
| apc | 16384 | 1 | 3/3 | 2.372 | 3.641 | 0 | 16384 |
| apc | 16384 | 4 | 3/3 | 2.487 | 3.767 | 0 | 65536 |
| apc | 16384 | 8 | 3/3 | 2.516 | 3.806 | 0 | 131072 |
| independent | 4096 | 1 | 3/3 | 2.592 | 2.851 | 4096 | 0 |
| independent | 4096 | 4 | 3/3 | 3.473 | 3.729 | 16384 | 0 |
| independent | 4096 | 8 | 3/3 | 4.514 | 4.774 | 32768 | 0 |
| independent | 16384 | 1 | 3/3 | 3.683 | 4.938 | 16384 | 0 |
| independent | 16384 | 4 | 3/3 | 7.606 | 8.873 | 65536 | 0 |
| independent | 16384 | 8 | 3/3 | 13.182 | 14.461 | 131072 | 0 |
| octopus | 4096 | 1 | 3/3 | 2.335 | 2.594 | 0 | 4096 |
| octopus | 4096 | 4 | 3/3 | 2.450 | 2.713 | 0 | 16384 |
| octopus | 4096 | 8 | 3/3 | 2.468 | 2.734 | 0 | 32768 |
| octopus | 16384 | 1 | 3/3 | 2.392 | 3.670 | 0 | 16384 |
| octopus | 16384 | 4 | 3/3 | 2.520 | 3.802 | 0 | 65536 |
| octopus | 16384 | 8 | 3/3 | 2.533 | 3.821 | 0 | 131072 |

## Native-default APC sentinel

| Method | Shared tokens | Branches | Records | Median branch s | Median total s |
|---|---:|---:|---:|---:|---:|
| apc-native-default | 16384 | 8 | 3/3 | 2.026 | 3.279 |

## Checkpoint retention

| Method | Records | Median branch s | Median total s | Median branch trunk recompute | Median inherited |
|---|---:|---:|---:|---:|---:|
| apc | 3/3 | 2.490 | 3.784 | 4096 | 4096 |
| octopus | 3/3 | 2.288 | 3.590 | 0 | 8192 |

## Three-clue semantic loop

| Method | Completed tasks | Exact matches | Commit schema success | Mean evidence score | Median total s |
|---|---:|---:|---:|---:|---:|
| apc | 12/12 | 12/12 | 12/12 | 1.000 | 3.719 |
| independent | 12/12 | 12/12 | 12/12 | 1.000 | 4.726 |
| octopus | 12/12 | 12/12 | 12/12 | 1.000 | 3.737 |

## Interpretation limits

- Native APC is the primary performance baseline. The native-default APC sentinel is not mixed into the matched synchronous/eager mechanism table.
- Three timing repetitions describe initial run noise only; they do not establish statistical significance.
- Process VRAM reservation is not treated as live KV usage. The raw records separate configured KV capacity from active/pinned block counts.
- The 12 synthetic tasks validate the structured loop and preserve failures; they are not a general Agent benchmark.
- Evidence source/locator fields are fixed by the per-branch JSON schema. The evidence score measures schema-constrained locator consistency, not independent evidence discovery or grounding quality.

## Protocol changes

- The official vLLM cu129 wheel's unqualified dependencies initially resolved CUDA 13 builds that driver 565 could not load. Torch, TorchVision, TorchAudio, and TorchCodec were pinned to their official cu129 builds without changing vLLM.
- The host could not reliably reach Hugging Face during engine creation, so all runs loaded the already complete cached snapshot at revision a09a35458c702b33eeacc393d103063234e8bc28. The logical model and tokenizer identity did not change.
- The 512 MiB retention engine uses max_model_len=8192 for both APC and Octopus because vLLM rejects max_model_len=32768 with that capacity. Every retention request remains below 8192 tokens; the workload and fixed KV capacity are unchanged.
- Two pre-result administrative launches were discarded and their files removed before acceptance: one carried an incorrect expanded commit label, and one would have inherited the matched in-process setting for the native-default sentinel.
- The public API demo exposed whitespace-only truncated JSON and model-generated duplicate commit IDs. The runtime now uses native XGrammar compact JSON, assigns commit IDs itself, and explicitly hands control back to the Brain. All three semantic benchmark methods use the same compact-JSON setting; no token budget, task, model, or weight changed.
- Only the 36 semantic-task records were regenerated to include reducer, serialization, and continuation-tokenization time in completion_seconds. An equivalent idle A100 on the same host (GPU 2) was used because GPUs 0 and 1 had been occupied by other work. The 54 runtime, 6 retention, and 3 native-default sentinel records are unchanged.
