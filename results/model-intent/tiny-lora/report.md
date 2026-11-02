# Step2 outcome: learned7B route no-go

**The prescribed pipeline stops at Step2.** The single frozen Tiny LoRA run
completed, but held-out goal-flip consistency was **48/64 = 75%**, below the
required80%. Accuracy passed; the conjunction of gates did not. No serving A/B,
formal confirmation, extra seeds or result-driven rerun was launched.

| Held-out test metric | Observed | Required | Gate |
|---|---:|---:|---|
| Accuracy | 111/128 = 86.71875% | >=80% | pass |
| Recall@1 | 111/128 = 86.71875% | >=80% | pass |
| Structural baseline | 64/128 = 50% | fixed comparator | - |
| Gain over structural | +36.71875pp | >=25pp | pass |
| Opposed-goal flip consistency | 48/64 = 75% | >=80% (at least52/64) | **fail** |

All128 test requests completed exactly once. One generated JSON answer named an
unknown checkpoint ID; it is retained and counted as wrong, not retried or removed.
There were no missing or failed producer calls. Even granting the invalid answer
one additional successful flip would give only49/64, still below the gate. The
negative result is therefore not explained solely by formatting. Dev was45/64
(70.3125%) with17/32 flips (53.125%); it was diagnostic only, never used to choose
the final checkpoint or change the recipe.

## Family-held-out result

| New test operation | Correct decisions | Goal-flip pairs |
|---|---:|---:|
| Quantile interpolation | 12/16 | 4/8 |
| Sorted insertion | 16/16 | 8/8 |
| Elapsed span | 14/16 | 7/8 |
| Text units | 16/16 | 8/8 |
| Tied rank | 12/16 | 4/8 |
| Bag difference | 16/16 | 8/8 |
| Ring rotation | 9/16 | 1/8 |
| Retry schedule | 16/16 | 8/8 |

The remaining weakness is opposed-goal sensitivity, especially ring rotation and
the interpolated-quantile/competition-rank goals. A model can be correct often
without reliably changing its choice when the required semantics change. These
are eight authored held-out operations and16 semantic goals, not128 independent
tasks or natural agent trajectories. The result exceeds the earlier S1's50%
descriptively, but different suites and inference backends do not establish a
paired improvement over #21 or the1B64.0625% result.

## Actual training and execution

256 train /64 dev /128 test decisions from16/4/8 disjoint new operation families.
Base Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`; bf16 SDPA, LoRA rank8/alpha16,
q_proj/v_proj, dropout0; base weights frozen,2,523,136 trainable adapter parameters.
Only assistant checkpoint-choice tokens were supervised. Fixed2epochs,32 optimizer
steps, constant LR0.0002, AdamW, microbatch2 x accumulation8 on one GPU, seed17.
Actual encoded training length449-510, below1024; no truncation.

The final-epoch-only adapter SHA-256 is
`afbaddcb5387cbd084c58efeb596b0be0091664ebf75445b8bd038d8c63cc4d3`.
Training receipt reports93.6368seconds and average training loss0.0296676.
Train/dev/test phase walls were104/48/88seconds, total240seconds
(2026-09-15 00:31:41-00:35:41 UTC), excluding setup, transfer and artifact handling.
All64dev +128test generations ended with stop; no response retry occurred.
The one main training run was consumed; the operational sanity rerun was unused
and is not permission to retry this valid negative result.

GPU9, UUID `GPU-3e940a5a-a24d-b615-dc69-df141ced7af0`, RTX4090D;
48 utilization samples averaged74.04% SM use, maximum100%, peak framebuffer
19,378MiB. This is sampled device utilization, not a measured kernel-busy integral.
Terminal exit0; final GPU4MiB/0% with no compute processes; owned screen removed.
The experiment supervisor's exact session was subsequently accepted and closed.

## Persistent evidence and interpretation

- [freeze.json](freeze.json): pre-run recipe, splits, input identities and thresholds.
- [test-score.json](test-score.json) and [dev-score.json](dev-score.json): full CPU-private scores.
- [training receipt](execution/main-run/training.json), [terminal](execution/run/terminal.json),
  [test raw responses](execution/test-responses/responses.json), and all per-attempt files.
- `execution/source/`, `execution/deployed.sha256` and `execution/run-step2.sh`:
  exact deployed source/public-data provenance and hard-bounded runner.
- `execution/artifact-sha256.txt` inventories the full retrieved artifact set,
  including the uncommitted adapter/tokenizer; it is not a manifest limited to Git files.

The adapter/tokenizer is retained remotely at
`/home/user/octopus-tiny-lora-step2-20260915/main-run/final` and backed up under
the local session artifact directory named in [execution/worker-result.json](execution/worker-result.json).
Large checkpoint/tokenizer files and the empty Trainer directory are not copied
into Git. All raw answers, source, inputs, private labels and terminal receipts are.

Step1A still establishes100% of available native scheduled-prefill saving under
the controlled oracle fixture, not GPU latency. Step2 shows substantial semantic
choice accuracy but misses the frozen goal-flip gate. The supported conclusion is
**no-go for this fixed7B learned branch-value route**, not proof that all7B models
or larger training could never work. No end-to-end system saving or generalized
serving claim is supported, and the authorized downstream GPU route is closed.
