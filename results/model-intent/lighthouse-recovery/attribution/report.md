# Base-Qwen Lighthouse Attribution

Exactly one unmerged base-Qwen episode completed under the frozen Lighthouse
protocol. The solver completed all 12 actions, the official evaluator completed
successfully, and the official task remained unresolved.

## Result

| Metric | Base Qwen | Existing merged LoRA |
|---|---:|---:|
| Official resolved | false | false |
| Actions | 12 | 12 |
| Distinct canonical actions | 1 | 1 |
| First inspection action | none | none |
| Failed actions | 12 | 12 |
| Repeated failed submissions after the first | 11 | 11 |
| Patch bytes | 0 | 0 |
| Actual native prefill tokens | 1,892 | 1,892 |
| Actual native decode tokens | 540 | 540 |
| GPU engine scope | 34.046 s | 100.863 s |

Both trajectories submitted the same invalid `edit` action 12 times and
received the same explicit `ValueError: edit old text must match exactly once`
observation 12 times. The complete canonical action sequence, observation
sequence, token costs, empty patch, and official failure outcome are identical.
The base therefore shows no inspection, diversity, or recovery improvement over
the actual merged-LoRA comparator.

This does **not** support adapter-associated policy degradation as the
differentiating cause on this case. It establishes instead that the unmerged
base plus the current harness/control reproduces the failure. The accepted
exact-input audit confirms that immediate failed-edit feedback was present in
every recorded next prompt; it does not by itself prove GPU cached-KV
correctness or a global model limitation.

## Native execution units

- Model revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`; one request at a time,
  zero evictions, zero allocation rejections, and zero preemptions.
- Request totals: 14,132 prompt tokens, 1,892 actual prefill tokens, 540 actual
  decode tokens, 12,240 adopted/inherited prefix tokens, 396 capsule tokens,
  7,392 checkpoint-frontier tokens, and 7,469 selected-parent tokens.
- Final pool: 31,790 total/free blocks, 146 warm-cached blocks, and zero active,
  request-owned, retention-only, or shared request/retention blocks.
- Consumer lifecycle records: 67 total across 11 declared, 22 ready, 11 queued,
  11 active, 11 complete, and one terminal no-live-consumer record.
- Timing: 34.046 GPU-engine seconds, 35.998 solver end-to-end seconds, 50.220
  remote-child wall seconds, 10.577 official-evaluator seconds, and 66 seconds
  total detached job wall time. One of one approved GPUs was assigned during
  the engine scope; no active-period utilization sample was manufactured.

The official log is retained losslessly as `execution/solver/official.log.gz`
(7,515,731 bytes). Its decompressed 50,624,671-byte stream has SHA-256
`7be746008ac394b907845820d4fd6bcbad604d9b1b77088eb9246c87c7127660`.

## Follow-on recovery seam

The concrete failing seam is in `benchmarks/lighthouse_quality_solver.py`
inside `repair`: model text is parsed into an action, submitted through
`tools.run(action)`, then its observation is appended to the next-turn journal.
A generic recovery integration should canonicalize and check the action at the
submission boundary after `json.loads(text)` and record failed action plus
explicit observation after `tools.run(action)`, before the next projected
request. The existing `tool_start`/`tool_complete` events carry scheduling
lifecycle but not the action/observation payload; the parent-owned
`src/octopus` memory should consume that payload without changing this frozen
benchmark policy.

## Provenance and cleanup

- Freeze SHA-256:
  `97c30d0c04fa25770e1a091cc2cabc83bc5493dbab6d188ed1b216376762fe39`.
- Base episode SHA-256:
  `02a7cf018849f319e91281a1adb4150452cdbc10e724be618d3be20162675d19`.
- Official grade SHA-256:
  `8b12d65e0661e8a7dcde86d5a9ed6769dd5dce1a7c534c96d5526fef0a9db894`.
- Accepted exact-input audit SHA-256:
  `c487178f1a0e014361b9cc342d63fc85e44147a75cef2ee1a398681950749aeb`.
- GPU9 `GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` returned to 4 MiB,
  0% utilization, with no compute process. The task-owned remote root, ignored
  task checkout, and local tmux session were removed.

Exact historical benchmark source bytes and a verified reconstruction command
are preserved in
`results/model-intent/lighthouse-recovery/source-provenance/README.md`.
