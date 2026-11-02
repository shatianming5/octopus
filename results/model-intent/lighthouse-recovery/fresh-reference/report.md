# Exact-input fresh-vLLM reference

**Result: the valid reference reproduces the recorded failed action byte for
byte.** It used action index 4's 3,943 actual input tokens from the raw-read
episode, including the failed edit and subsequent successful inspection.
The canonical action and raw JSON text both equal the recorded repeat;
the 45 output tokens still request an edit whose `old` occurs twice.
No tool was executed and no official evaluation was run in this diagnostic.

The same merged recovery model, BF16, seed 17, temperature zero, 512-token
ceiling, xgrammar backend and whitespace setting were retained. The fresh
engine used stock `vllm.v1.core.sched.scheduler.Scheduler`, with prefix caching
disabled and no Octopus controller, checkpoint adoption or prior requests.
The schema was passed as the original serialized string, preserving property
order as well as its semantic meaning. [freeze.json](freeze.json) stores the
exact input and settings; [result.json](result.json) stores actual output tokens,
text, effective scheduler and schema. Exact execution sources are retained in
`source-provenance/`.

This argues against physical KV reuse as the **sole cause on this one input**.
It does not establish general KV correctness, identify a unique model-training
cause, or prove that every alternate agent policy must fail.

## Preserved instrumentation failure

`attempt-1/` is not a valid scientific comparison. Serializing a nested schema
with `sort_keys=True` reordered xgrammar properties, so the first instrument
used a different generation path. It emitted a different edit (also nonunique),
which cannot be attributed to the physical inference path. The original freeze,
source, output, logs and timing remain intact. A red regression test exposed
the serialization defect; the corrected implementation preserves and records
the exact schema string through disk round-trip. The valid invocation was an
instrument repair under the original protocol, not output-driven policy tuning.

Both invocations together consumed 45.765205 conservative job-wall seconds
(23.260437 invalid instrument + 22.504768 valid), below the original 300-second
aggregate cap. Valid generation took 1.219718 seconds; engine scope was
19.377558 seconds. Active GPU snapshot: 43,528 MiB, 92% utilization on the same
RTX 4090 D / UUID ending `7af0`. Both jobs exited zero; their original processes
and named tmux session were absent afterward. Terminal device state was 4 MiB,
0% utilization, with no active compute process. There is no quality or paired
performance claim.
