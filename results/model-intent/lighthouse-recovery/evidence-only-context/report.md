# Evidence-only decision context

**Closed: original official `resolved=false`, zero patch.** This scoped treatment
separates observed external state from prior model self-reports. The preceding observation-state episode delivered
all feedback correctly, but also appended its unchanged pre-action rationale
after newer tool output on every one of eleven subsequent decisions. That is
an observed dataflow pattern, not proof that self-echo caused the loop.

`history_format=tool_observations_v1` compiles the next decision from the task,
actual action/result history, literal tool outputs, runtime-owned latest
observation capsule, persistent failure constraints and remaining action count.
It omits previous assistant messages and model plan/interpretation fields from
decision input while preserving them in the raw episode and TaskMemory audit.
The model still generates its fresh plan, evidence, reasoning and action inside
the unchanged 512-token output budget. No extra planner request, source hint,
hard action-novelty mask or forced edit is added.

This is an explicit policy counterfactual, not a claim that all agent plans
should be discarded. Root instructions, action schema, tool semantics, model,
seed and budgets are unchanged; prior history modes preserve their old bytes.
The common observation renderer retains the actual action index, errors,
rejections, output/truncation metadata and literal source.

The [CPU replay](cpu-replay.json) uses unmodified actual previous decisions and
tool outputs: old rationale re-insertion is removed on all eleven next turns,
while all eleven latest outputs remain exact in both tool history and capsules.
Maximum replay input is 2,478 tokens. Generic regressions also establish that
different unsupported model guesses yield identical next inputs when actual
tool evidence is identical, without changing the raw audit or verification gate.

The [actual terminal episode](terminal-report.md) had12physical inspections,
5distinct actions,7successful inspection repeats and0edits/tests/done/patch.
All11following inputs retained exact latest outputs in both capsules and literal
tool history, with only the pending assistant turn and no prior model-message
replay or persistent interpretation. All12raw decisions/reasoning records stayed
auditable. Thus this self-echo intervention was exercised but insufficient.

Actual work was33,796prefill/1,591decode tokens,85.876engine seconds and87.909solver
seconds; one scheduler request at a time, no eviction. Two pre-model zero-action
deployment dependency failures were preserved and repaired under the same frozen
setup, without a second model episode. No useful active GPU utilization sample
was captured; terminal compute/container release was verified. The separate
native-tool adapter tests the checkpoint's shipped function protocol, which
these custom-action prompts did not use; this closed episode will not be replayed.
