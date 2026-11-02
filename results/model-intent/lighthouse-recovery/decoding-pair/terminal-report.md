scope_id: lighthouse-decoding-pair-e3e25abd
status: completed
actual_result: Exactly two stock vLLM requests completed, one per frozen arm. Constrained applied the original schema string; unconstrained applied null. Both saw 3108 source bytes and 3003 input tokens, preserved raw model text and 120 token IDs, and matched the original and each other in raw text and canonical action.
evidence: constrained-result.json; unconstrained-result.json; metrics.json; report.md; cleanup.json; terminal-status.json
cost: 2 model requests; 61 arm-wall seconds; 54.857 engine-scope seconds; 8.951 generation seconds
gpu: GPU-3e940a5a-a24d-b615-dc69-df141ced7af0 released; no compute process remained. Active utilization was not reliably captured.
boundary: No quality/performance, label-correctness, official-evaluation, or active-runtime-KV claim.
remote: /home/user/octopus-lighthouse-decoding-pair-v1-20260916/run
local: /home/ubuntu/repos/octopus.worktrees/tommy-dev-linux-20260915-175656-83eb/results/model-intent/lighthouse-recovery/decoding-pair
