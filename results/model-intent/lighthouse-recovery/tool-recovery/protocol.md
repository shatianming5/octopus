# Frozen Generic Tool-Recovery LoRA Protocol

This phase trains exactly one rank-8 LoRA from the original unmerged
Qwen2.5-7B-Instruct snapshot on deterministic generic recovery trajectories.
The 256 training and 64 held-out cases are task-level disjoint and contain no
Lighthouse source, issue, patch, tests, gold data, branch-choice target, or
external teacher output.

Each training case contributes two supervised transitions built from actual
visible-repository tool execution:

1. failed edit/test/path observation to a relevant inspection;
2. successful inspection evidence to a corrected edit or discovered-file read.

The recipe is fixed before model output: rank 8, alpha 16, dropout 0.05,
`q_proj`/`v_proj`, bfloat16, one GPU, microbatch 1, accumulation 8, two epochs,
AdamW at `2e-4`, zero weight decay, 0.05 warmup ratio, linear schedule,
max-gradient norm 1, seed 2301, gradient checkpointing, maximum sequence 2048,
and prompt labels masked to `-100`.

The one 64-case held-out suite uses no `FailureMemory`, action rejection,
forced path schema, hidden repair, retry, or replacement. It permits the full
global action schema and executes two temperature-zero model actions per case
against real fixtures. Both frozen gates must pass:

- at least 58/64 cases without a repeated known-failed action;
- at least 52/64 cases whose first proposed action succeeds and acquires
  relevant evidence.

Only a dual gate pass permits one merged-adapter Lighthouse run under unchanged
`failure_aware_v1`, seed 17, temperature 0, bfloat16, APC, 12 slots, original
tools/image/evaluator, and official `resolved=true` quality criterion. Gate
failure records Lighthouse as `not_run`. No output-driven tuning or rerun is
permitted.
