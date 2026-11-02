# Lighthouse decoding pair

Both frozen arms completed exactly one stock vLLM model request on GPU 9. The constrained arm applied the original JSON schema string; the unconstrained arm applied `null` and omitted structured decoding. Both used the actual source-visible prompt (`3108` source bytes, `3003` input tokens) and produced one 120-token output with preserved raw model text and token IDs.

The constrained output canonically and byte-for-byte matched the frozen original. The unconstrained output also canonically and byte-for-byte matched the frozen original. Within the pair, raw text, output token IDs, and canonical action were identical. This isolates no difference for this one request pair; it is not a task-quality or performance result.

Both outputs parsed directly as JSON, so no parse error occurred. The reference helper passes the complete raw text to `json.loads(text)` without code-fence stripping; its unconstrained error path retains the exception text if parsing fails.

The pair consumed two model requests and 61 seconds of arm wall time. Engine-scope time was 27.899 seconds constrained and 26.958 seconds unconstrained; generation time was 4.495 and 4.456 seconds respectively. Immediate launch-time GPU samples were too early to establish active utilization, so no sustained utilization or active-runtime KV claim is made.

Artifacts include both raw result JSON files, output token IDs, model text, logs, wrapper, terminal status, metrics, and cleanup evidence. No official evaluation, training, download, source modification, replay, or third request was performed.
