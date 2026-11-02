# Utility v4 inline-intent mechanism result

**Decision: `SCIENTIFIC_NO_GO`.** The frozen functional smoke completed four
episodes, but the early carrier index disagreed with the normal body
`dispatch_index` in 2/4 episodes. The body itself dispatched the correct actual
follow-up in 4/4 episodes and final-answer quality was 4/4, so the failure is
specifically that the early hint was not a reliable representation of the
necessary normal output. Per the frozen stopping rule, the exact six-trace
matrix was not launched and no prompt, schema, threshold, seed, or capacity was
changed after outcomes.

## Completed units

- Remote native allocator/shared-`CaseSession` validation: 101/101 passed. The
  initial 100-pass/1-fail result was a fixture expectation error; after the
  parent corrected the expected soft-retention residue, the exact failed
  selector passed without runtime changes.
- Tokenizer freeze: 20 incremental control tokens per episode, 240 per trace,
  two carrier-envelope tokens per episode, and 501 tokens for the longest
  normal prompt plus output budget.
- XGrammar: the exact runtime tuple schema compiled as `XgrammarGrammar` for the
  frozen Qwen tokenizer before model inference.
- First smoke launch: invalid instrumentation attempt, 24 seconds. Chunked
  prompt-prefill scheduler calls were mistakenly counted as normal-output
  steps. The attempt is preserved under `instrumentation-failure-01`; the
  harness was repaired to count only output-bearing normal steps and the same
  frozen protocol was rerun.
- Valid functional smoke: four episodes, 27 runner seconds. Exact matrix:
  0/6 traces, correctly stopped by the smoke prerequisite.

## Valid smoke evidence

| Measure | Result |
|---|---:|
| Inline hint observed before normal completion | 4/4 |
| Hint applied through native model intent | 4/4 |
| Hint/body agreement | 2/4 |
| Hint matches private expected route | 2/4 |
| Normal body dispatch matches expected route | 4/4 |
| Exact final-answer quality | 4/4 |
| Choice readiness | 3 output tokens in every episode |
| Mean choice-ready time | 0.1802 s |
| Mean normal completion time | 0.5664 s |
| Mean foreground E2E | 1.3154 s |
| Background P95 | 0.3724 s |
| Policy plus adapter CPU | 0.03234 s total |

Native retention followed the early hint in every episode. When hint and body
agreed, continuation prefill was 25 tokens. When they disagreed, the body still
dispatched the correct checkpoint but that checkpoint had been mostly evicted,
so continuation prefill rose to 1593 tokens. This is the intended falsifier:
the runtime mechanism works, but this frozen normal-output carrier does not
provide a sufficiently faithful early choice.

The authored workload limitation remains: pressure was injected causally after
four output-bearing normal steps to isolate the mechanism. This result is not a
natural multi-branch contention or generalized serving claim.

## Resources and artifacts

One physical GPU,
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0`, was used for 51 total runner
wall-seconds across the invalid and valid attempts. After completion the owned
tmux session and processes were absent; the device reported 4 MiB used and 0%
utilization.

Frozen inputs, tokenizer measurements, raw per-episode traces, environment
metadata, logs, terminal records, the invalid instrumentation attempt, derived
result, and cleanup evidence are all under
`results/model-intent/utility-v4/`.
