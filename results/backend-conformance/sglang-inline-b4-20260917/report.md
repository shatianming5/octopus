# B4: real GPU dynamic inline intent passed

**One frozen native selector passed on .138.** While the model's normal
producer was still active, its parsed intent crossed the worker/tokenizer
control channel and was acknowledged by the actual SGLang scheduler. The
native first victim changed from the observed LRU candidate (node 11) to
node 13. The selected prefix retained all 520 resident tokens; the unselected
prefix fell from 520 to 0. The next actually scheduled bound use consumed the
intent once, leaving no remaining intent.

| Actual evidence | Observation |
| --- | --- |
| Frozen application | `412cbab7986a7100b8c580c5216fcdf0e2b62bcf` |
| Upstream / mode | SGLang 0.5.19, Python Unified FULL LRU, native protocol 1 |
| Model | Original OpenHands-7B, separate C03 tokenizer, BF16 |
| Context / KV / prefix / pressure | 4096 / 4096 / 512 / 3072 tokens |
| Model requests | Five: two roots, normal producer, pressure, bound use |
| Normal producer | 67 prompt tokens; unmodified 381-token output |
| Bound use | 526 prompt tokens, 520 cached tokens, 8 output tokens |
| Native timing | Update sequence 3 and eviction sequence 4 while producer active; first scheduled bound use sequence 9 |
| New B4 phase | 61/1200 seconds; 1139 unused |
| Final cleanup | No selected-GPU compute app; 1 MiB used / 48539 MiB free; sibling tool survived |

The [actual conformance record](extracted/results/b4.json) preserves 413
cumulative events, request IDs, native scheduler PID/time/owner/generation
receipts, exact outputs and all four completed checks. There was no forced
dispatch choice, retry, extra planner/choice generation or output repair.
The [terminal report](terminal-report.md) and
[cleanup receipt](post-terminal-cleanup.txt) retain the bounded outcome.

This is a point-native control/eviction proof, **not** total native work,
latency, task-quality, HiCache, routing, training or Formal acceptance. Those
full-plan gates remain required. The completed 141-second foundation ledger
is separate and was not reset or reused.

All extracted run evidence is published. The original transport archive and
33 MB upstream Git bundle are retained locally but excluded from Git to avoid
duplicating the public SGLang source history. Upstream commit
`0bcd822377da7b5718e674eaf9c870d349424dd1` and transfer hashes remain in
`extracted/receipts/source-artifacts.sha256`; the native patch is versioned in
`src/octopus/sglang-intent-v0519.patch`. No model or evidence record was rewritten.
