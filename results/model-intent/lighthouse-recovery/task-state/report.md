# Evidence-carrying serial Brain state

**Closed: official `resolved=false`; no downstream gate is open.** The one
frozen episode produced twelve plans but no evidence selection, semantic commit,
edit, test or patch. It repeated the same successful read nine times. The
evidence-carrying path was available but was not exercised by the model.

## Interface repair

The closed regex-search episode had six nonempty tool outputs but all twelve
compiled capsules contained no facts. The serial solver neither committed
task evidence to its reducer nor selected evidence predicates for inheritance.
This establishes an unwired interface, not the cause of the model's loop.

`task_state_v1` adds model-owned decomposition and extractive citations to the
normal Brain decision. A decision contains `plan`, `evidence`, `reasoning`, then
`action`, all inside the existing 512-token output budget. There is no extra
planner, critic, branch-choice request, successful-action novelty mask, or
forced edit. Existing action-only and deliberative formats remain unchanged.

`TaskMemory` validates each quote against its indexed actual tool output before
applying an immutable semantic commit through `StateReducer`. Selected facts
then enter the next native request's capsule. Repeated citations reuse facts;
observations across file mutations retain their separate historical identities.
The latest plan and explicitly unverified interpretation survive journal
trimming, but are never promoted to verified facts. An invalid citation rejects
the entire state update and tool proposal visibly, consuming one action slot.
Failure-memory recovery and successful-verification-before-done remain intact.
Serial Brain commits are not genuine concurrent branches, and commit success
means valid extraction only, never task resolution.

The generic repair-path fixture reaches the real read/edit executor and the
existing failure controller. The [CPU replay](cpu-replay.json) uses the closed
episode's actual observations with synthetic generic plans/first-line quotes:
six grounded facts, ten nonempty subsequent capsules, and exact capsule-token
roundtrips with the pinned tokenizer. This is persistence/compilation evidence,
not model reasoning or quality evidence; no synthetic decision enters the run.

## Actual same-task outcome

The [terminal report](terminal-report.md) records 12 proposals / 4 distinct,
12 physical tools / 0 rejections, and 0 failed-action re-executions. There were
11 nonempty outputs / 3 distinct. After a file-discovery result, the model read
the same 200-line type-declaration window nine times and repeated its pre-read
rationale unchanged. The plan never advanced beyond generic duplication work;
it did not decompose both reported bugs. All evidence arrays remained empty.

Actual native work was 19,676 prefill / 1,637 decode tokens, 87.490 engine
seconds and 122 wrapper seconds. An engine-ready sample captured 100% GPU
utilization; terminal GPU and actual container cleanup were verified. There
was one simultaneous scheduler request and no eviction. This is not a
multibranch/performance result, nor evidence that usable grounded state failed:
the model never supplied a citation to populate it.
