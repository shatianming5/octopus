# Lighthouse Failure-Aware Runtime Protocol

This phase executes exactly one unmerged base-Qwen trajectory on
`GoogleChrome__lighthouse-10176` with the original official evaluator. The task,
issue, image, solver tools, generation seed 17, temperature 0, bfloat16 dtype,
APC runtime, 12-proposal limit, token limits, GPU settings, timeouts, and
container limits remain identical to the accepted base attribution.

The sole treatment is `agent_policy=failure_aware_v1`:

- canonical failed actions are rejected before external tool execution;
- every rejected proposal still consumes one model call and one of 12 slots;
- actual outcomes, error types, and resource evidence persist outside the
  truncatable journal;
- after an edit mismatch, only a successful non-empty read/search of the same
  resource can permit a different edit;
- rejected controls emit stage events, not physical-tool events.

No Lighthouse path, patch, gold asset, or task-specific recovery rule is added.
No comparator/audit/LoRA rerun, prompt/seed/action-limit change, conditional
training, or output-driven tuning is permitted.

## Interpretation frozen before model output

Official `resolved=true` is the only quality pass. Recovery additionally
requires useful same-resource evidence before a corrected edit; a forced
inspection loop, empty/failed/unrelated inspection, non-empty patch, or
behavioral proxy is not quality success. Quality failure remains a valid phase
result and does not authorize a second trajectory.

The accepted predecessor is managed result
`03c65fb9-a965-4205-a9c7-37f8e48c7712`: both unmodified base and merged LoRA
submitted the same invalid edit 12 times, produced 11 repeated failed external
submissions, performed no inspection, emitted no patch, and failed the official
evaluation.
