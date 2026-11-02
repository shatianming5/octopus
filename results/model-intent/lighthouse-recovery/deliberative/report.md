# Single-call deliberative same-task treatment

**Official result: `resolved=false`, zero patch bytes.** The separately frozen
runtime-policy treatment did not repair the observed real-task action-selection
failure. There were 12 genuine model decisions, 12 recorded explanations,
2 distinct tool proposals, 3 physical submissions (2 successful reads and
1 failed edit), 9 rejected repeat proposals and zero known-failed physical
resubmissions. No test action or successful edit occurred.

The only protocol delta from raw-read was `decision_format=deliberative_v1`.
Generic instructions ask for observation-grounded reasoning before a nested
tool action in the same normal inference call. The original tool schema,
including mandatory same-resource recovery inspection, is unchanged inside
the envelope. Explanations remain in the journal and raw episode. Parsing only
extracts the nested action: it never repairs, substitutes or changes tool
arguments. There is no extra planner, critic, choice call or hidden retry.

The same recovery adapter, raw reads, original issue, 12 slots, 512 output
tokens per slot, seed 17 and official evaluator were retained. Training was
not repeated or tuned. No target-specific examples, patch hints or evaluator
labels were added. Freeze and exact source snapshots preceded all outputs.

The first read covered lines 80-100. The model then tried to replace `lib.id`
with a generated fallback identifier, but that old text is nonunique in the
file. After the error it successfully reread the same range, then proposed
the identical failed edit nine more times. Its explanations acknowledged the
error yet repeatedly asserted that the old text was now known and the same
edit should proceed. This is rationalization, not verified recovery. We did
not disambiguate the edit or apply the model's intended patch manually.

All 12 journals were retained without truncation. Native totals were 35,298
prompt tokens, 9,042 actual prefill tokens, 1,944 actual decode tokens and
26,256 adopted/inherited tokens; 396 capsule tokens and 9,196 selected-parent
tokens are counted. There was at most one scheduler request, with zero
preemptions, cache evictions or allocation rejections. Changed tool histories
make these absolute costs, not a paired speedup comparison.

Engine scope was 66.473120 seconds, solver end-to-end 68.490740 seconds,
child wall 103.626773 seconds and official grading 10.954112 seconds. The
complete wrapper took 120 seconds, below this treatment's 1,200 GPU-second
ceiling. Active GPU utilization was not sampled for this invocation; actual
native request records are retained. After terminal outcome, the GPU was
4 MiB / 0%, with no compute process, original wrapper or task container.

The quality gate remains blocked. Explanations alone did not fix action
selection, and synthetic recovery success has not transferred. This negative
result does not justify another prompt sweep, new training recipe, easier task,
KV-policy tuning, multibranch run or formal claim under this frozen treatment.
