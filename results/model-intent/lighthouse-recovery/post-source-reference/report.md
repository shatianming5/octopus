# Lighthouse post-source reference terminal report

- `scope_id`: `lighthouse-post-source-reference-709b6e11`
- `worker_id`: `a19747b4-14a1-4816-af14-19910a354b84`
- `status`: `completed`
- `terminal_outcome`: `same_canonical_redundant_edit`
- `model_requests`: `1`

The initial GPU 7 capacity check was preserved as prelaunch evidence: GPU 7
(`GPU-4cd604ce-20c4-9bb7-bed7-acc3cf542887`) was occupied by PID 1817920
using 44,232 MiB, so zero requests were made there. Under the authorized
device-only amendment, the same frozen reference was executed unchanged with
`CUDA_VISIBLE_DEVICES=9` on
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0`.

The helper completed exactly one stock-vLLM request with prefix caching
disabled. It verified 3,829 prompt tokens, the visible 190-byte source window,
and exact `schema_json` identity. The output contained 63 tokens and reproduced
both the same raw text and the same canonical redundant edit:

```json
{"action":"edit","path":"lighthouse-core/lib/stack-collector.js","old":"detector: /** @type {'js'} */ ('js'),","new":"detector: /** @type {'js'} */ ('js'),\n    id: lib.id,"}
```

Engine scope was 24.916 seconds and generation was 2.845 seconds. Owned job PID
1828249 (start ticks 77402097, SID 1828249) was start-verified, exited with code
0, and GPU 9 was released. No container was created.

For this exact source-fed decision, physical reuse is not the sole explanation.
This call cannot grade or resolve the task and does not prove global KV
correctness. The device-only amendment is explicit: a hypothetical output
difference would not have isolated physical reuse from numerical or device
effects.

Evidence:

- Redacted GPU 7 prelaunch log:
  `results/model-intent/lighthouse-recovery/post-source-reference/actual-terminal.log`
- GPU 9 terminal log:
  `results/model-intent/lighthouse-recovery/post-source-reference/actual-terminal-gpu9.log`
- Process identity:
  `results/model-intent/lighthouse-recovery/post-source-reference/run-process.json`
- Raw helper result:
  `results/model-intent/lighthouse-recovery/post-source-reference/reference-result.json`
- Structured result:
  `results/model-intent/lighthouse-recovery/post-source-reference/result.json`
- Remote GPU 9 terminal log:
  `/home/user/octopus-lighthouse-post-source-reference-v1-20260916/run-gpu9-terminal.log`
