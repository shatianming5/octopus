# Agent7 alternate-host quality continuation

**Scope:** `software-agent7-same-task-quality-v1`

**Terminal status:** `TRANSFER_WALL_BUDGET_EXHAUSTED`

The `.68` operational amendment found a compatible existing environment (`torch 2.13.0+cu129`, `vLLM 0.28.0+cu129`, `transformers 5.16.1`, `xgrammar 0.2.3`) and exclusive idle RTX 4090 D capacity. The verified actor and six small tokenizer/config files were therefore eligible for secure replication without a new model download.

The transfer retained the original `2026-09-16T10:23:29Z` start and `2026-09-16T11:53:29Z` deadline. The initial four-stream owner-host relay sustained only about 0.25 MiB/s, with SSH children waiting on network I/O rather than CPU or seek work. A same-scope infrastructure repair replaced it with authenticated direct `.183 -> .68` SSH using an ephemeral forwarded local agent and `rsync --append-verify`; no private key was copied and no shared credentials changed.

The repaired route was still too slow to complete the exact payload. At the hard deadline, `.68` retained `2,010,067,550` actor bytes and no tokenizer bytes, versus `15,242,760,997` required bytes. The complete destination payload could not be digest-verified, so no alternate endpoint was frozen for execution and no model job was launched.

The original quality episode remains unconsumed: zero model requests, outputs, actions, GPU seconds, grades, training runs, downloads, or negative-outcome retries. Exact owned source and destination rsync/SSH processes were stopped after deadline; retained prefixes and the original verified `.183` assets remain intact.

**Concrete blocker:** The authorized 5,400-second replication wall budget expired before the exact verified payload reached `.68`.

**Evidence:** `transport-terminal.json`, `cleanup.json`, remote `/home/user/octopus-lighthouse-software-agent7-v1-20260916/transfer/{started.json,transport-repair.json}`, and accepted acquisition `../../acquisition/terminal-report.md`.

The parent quality, multibranch, and formal completion gate remains open.
