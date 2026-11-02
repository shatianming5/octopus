# Stage1B: direct-choice elicitation

**ELICITATION_NO_GO: proceed to the conditional Tiny LoRA stage, not serving.**
The new held-out64-decision suite produced64 valid, single-answer Qwen responses.
No answer was retried or replaced. CPU-private scoring found no missing, invalid,
duplicate or misassociated response. This is a valid negative model gate, unlike
the earlier infrastructure failures, which produced no model calls.

| Metric | Observed | Required | Gate |
|---|---:|---:|---|
| Accuracy | 41/64 =64.0625% | >=75% | Fail |
| Recall@1 | 41/64 =64.0625% | >=75% | Fail |
| Structural baseline accuracy | 32/64 =50% | Reference | -- |
| Accuracy gain | +14.0625pp | >=20pp | Fail |
| Opposed-goal flip consistency | 9/32 =28.125% | >=75% | Fail |

The denominator is8 authored families,16 semantic goals,64 ID/order views and
32 opposed-goal pairs. These are not64 independent real-agent tasks. Direct
choice improved on the structural baseline within this suite, but did not meet
any acceptance threshold and often failed to change selection with the goal.
Historical #21 used different families and a different output schema; its50%
result is not a paired estimate of the direct-choice schema's effect.

## Evidence and exact computation

- [inputs.json](inputs.json): frozen public input, semantic digest
  `5bfe226e46ed4efe65f45af63e20d99738f804e1552ab2f87e91d54e884a22fc`.
- [score.json](score.json): CPU-private full-denominator scoring and all rows.
- [responses.json](native-run/recovery/responses/responses.json): original64
  responses; sibling `attempt-00.json` through `attempt-63.json` retain each call.
- [environment.json](native-run/recovery/responses/environment.json) and
  [terminal.json](native-run/recovery/run/terminal.json): actual execution/device.
- [recovery.py](native-run/recovery/recovery.py): the exact successful producer.

Qwen2.5-7B-Instruct revisiona09a35458c702b33eeacc393d103063234e8bc28,
temperature0,seed17,max64 output tokens,one answer per decision. All64 finish
reasons were `stop`;30,992 prompt tokens and1,468 output tokens were accounted
for. Summed generation time was26.480463seconds; the successful process took
46seconds including initialization and teardown. GPU9's physical UUID was
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0`. All owned GPU/process resources
were released afterward. Native base01a98ce9 and the untracked source overlays
are recorded separately in [identity.txt](native-run/identity.txt), not presented
as a clean source checkout.

## Startup repair and retained costs

Repeated original launches hung after KV-cache setup before model calls. The
working task-local repair was to initialize the exact frozen Octopus runtime
**before importing benchmark helpers and collecting inputs**. The recovery then
used the same renderer, schema, native model request, checkpoint release and
collector. Every request hash matched the frozen prompt. The CLI now preserves
that initialization order with lazy helper imports. The narrower native reason
for this import-order sensitivity is not proven; no compiler, shared-lock or
telemetry root cause is claimed. The telemetry exception was incidental.

Four601-second zero-call timeouts and a589-second instrumented zero-call
termination are preserved in `native-run/startup-failures/` and
`native-run/instrumented-run/`:2,993seconds of failed process wall time. Seven
initialization/control probes add168seconds, including an invalid strace probe
and an invalid wrapper. With the46-second success, recorded job wall time totals
3,207seconds, excluding preparation and transfer. These are not GPU kernel-busy
seconds. The instrumented run exceeded its intended120-second stop because TERM
did not exit and KILL escalation was delayed; that execution-control failure is
retained, not hidden. Recovery used an immediate hard watchdog instead.

Only the conditional Step2 training gate is now authorized. No LoRA, six-trace
end-to-end run, formal confirmation or serving speedup is claimed here.
