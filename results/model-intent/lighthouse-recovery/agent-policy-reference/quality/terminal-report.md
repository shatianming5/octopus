# Software-agent 7B same-task quality scope

**Scope:** `software-agent7-same-task-quality-v1`

**Terminal status:** `CAPACITY_BLOCKED_BEFORE_MODEL_OUTPUT`

The exclusive eight-key endpoint, exact 50-action XML/V4 freeze, task-repository symlink, and nine-file source provenance were created prospectively. Local, frozen, and provenance hashes match. The freeze binds actor revision `97054e70501d79c27499518af5ce613be6278bcb`, separate tokenizer revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, the original Lighthouse task, and the required unconstrained request decoding and controller conditions.

The one-time host capacity decision found no non-interfering compatible GPU:

- GPUs 0–7 were using 44,309–45,398 MiB at 99–100% utilization.
- Preferred GPU 9 had a conflicting 14,936 MiB process and only 33,570 MiB free.
- GPU 8 was the best equivalent, but its 44,034 MiB free was 192 MiB below the 44,226 MiB required by the frozen 0.9 vLLM allocation; it also had a live 4,404 MiB process.

Launching would either fail the frozen allocation or interfere with another job. No remote source deployment, GPU process, model request, model output, action, official grade, training, download, or negative-quality retry occurred. The authorized episode remains unconsumed and ready once compatible capacity is coordinated.

**Evidence:** `capacity.json`, `freeze.json`, `execution-endpoint.json`, `terminal.json`, `cleanup.json`, and `source-provenance/`.

**Accepted acquisition pointer:** `../acquisition/terminal-report.md`

The parent end-to-end quality, multibranch, and formal objective remains open.
