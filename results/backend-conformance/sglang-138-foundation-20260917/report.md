# SGLang native backend foundation: passed

**The native foundation gate passed on .138 with the isolated flush correction
`af6126f8c2698e81e4e6d32b042e9317dc967c1d`.** This is actual model/GPU execution,
not only an import or subprocess fixture. It is not a full-plan, task-quality,
capacity-pressure, total-work, performance, dynamic-intent, HiCache, routing,
training or Formal pass. Those required gates remain separate and unfinished.

The dedicated environment uses SGLang 0.5.19 at upstream
`0bcd822377da7b5718e674eaf9c870d349424dd1`, Torch 2.13.0 / CUDA 13, the original
OpenHands-7B weights and separate C03 tokenizer. The selected host is
`73F3-5xA6000-221` (SSH alias .138), driver 580.173.02, GPU0 RTX A6000.
The model and dependency files were transferred through the private network;
the existing vLLM environment and cancelled Teacher were not changed or resumed.

| Actual request observation | Prompt tokens | Output tokens | Cached tokens |
| --- | ---: | ---: | ---: |
| Root | 512 | 8 | 0 |
| Concurrent branch A | 548 | 8 | 520 |
| Concurrent branch B | 549 | 9 | 520 |
| Cold restore after idle flush | 584 | 8 | 0 |
| Warm reuse | 584 | 8 | 583 |
| Final semantic continuation | 881 | 9 | 594 |

The [actual conformance record](recovery/extracted/results/recovery/conformance.json)
contains 253 cumulative stream events across 10 terminal requests. It covers
generated-tail preservation, two in-flight branches, suffix isolation, releasing
a parent with live children, bounded native idle flush, cold/warm reuse,
prefix-safe abort, structured Commit and Brain continuation. Public idle flush
is controlled eviction, not evidence of capacity-pressure eviction or total
native recomputation savings.

The initial missing-ninja PATH failure cost 39 seconds; the next run exposed
the immediate-flush adapter bug after 50 seconds. The isolated bounded-flush
fix then passed in 52 seconds: **141/1200 seconds charged, 1059 unused**, with
no ledger reset. All failures and exact source identities are retained. The
[terminal report](recovery/terminal-report.md) and
[cleanup receipt](recovery/post-terminal-cleanup.txt) confirm engine exit,
the sibling tool surviving, no selected-GPU compute process and memory restored
to 1 MiB used / 48539 MiB free. No active-memory sample is fabricated.
