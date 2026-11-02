# Base-Qwen Lighthouse Attribution Protocol

This study runs exactly one unmerged base-Qwen trajectory on
`GoogleChrome__lighthouse-10176`. It reuses the merged-PR24 Lighthouse task,
official evaluator, solver prompt, action schema, bounded tools, APC runtime,
12-action limit, generation seed 17, bfloat16 dtype, token limits, timeouts,
container limits, and official image without changing the task or revealing
evaluator-only assets to the solver.

The sole scientific treatment change is model weights and corresponding
identity metadata:

- base: Qwen2.5-7B-Instruct snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`;
- comparator: the already completed merged-LoRA episode at
  `results/model-intent/lighthouse-quality/execution/solver/episode.json`;
- no LoRA rerun, recovery training, policy change, prompt change, seed change,
  action-limit change, or worker-authored Lighthouse patch is permitted.

## Attribution interpretation frozen before model output

- Better base inspection, action diversity, or recovery under this identical
  harness supports adapter-associated policy degradation on this case only; it
  is not global proof.
- If base also fails, base plus the current harness/control remains a problem.
  The accepted exact-input audit proves journal feedback was present in the
  recorded prompts, but does not by itself prove GPU cached-KV correctness or
  global coding incapability.
- Official `resolved=true` is the quality criterion. Non-empty patches and
  behavioral proxies are descriptive evidence, not success.
- Infrastructure failure before model output may be repaired under this frozen
  protocol and budget. No outcome-driven tuning is allowed after model output.

## Accepted evidence

The parent-completed exact-input audit is copied verbatim to
`provenance/lighthouse-exact-input-audit.json` with SHA-256
`c487178f1a0e014361b9cc342d63fc85e44147a75cef2ee1a398681950749aeb`.
It is accepted once and is not recomputed here. The prior whole-text
re-encoding mismatch was an invalid proxy caused by independently tokenized
segment-boundary whitespace, not evidence of lost feedback.
