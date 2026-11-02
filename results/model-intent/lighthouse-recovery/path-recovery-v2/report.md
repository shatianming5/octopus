# Causal path recovery, same-task base model

**Official result: `resolved=false`, zero patch bytes.** The model now acquired
real source evidence but still invented edit preconditions absent from that
source. There were12 proposals /5distinct,9 physical submissions and3 rejected
known-failed edit repeats. Physical calls comprised1 root listing,4 successful
full-file reads,3 empty searches on an existing file and1 failed edit.
Successful nonempty inspection was5 calls; known-failed physical repeats were0.

The new v2 policy persists missing-path failures and restricts the next schema
and submission checks to list/search at the repository root until nonempty
successful discovery. Repeating a previously failed missing-path action
reactivates this constraint. No path or action is rewritten. V1's memory shape
and behavior remain available. The shared FailureMemory source is now included
in new freeze hashes, with its exact execution snapshot preserved here.

In this episode the model chose root listing immediately, then read all91lines
of the relevant source using the new valid1-based/count contract. No missing
path occurred, so the conditional discovery mask itself was **not activated
in this episode**; real-tool regression cases validate that transition. The
observed proactive behavior cannot be attributed exclusively to that mask,
because v2 also exposes its generic recovery instruction and memory field.

The edit attempted `id: 'React (Fast Path)'`, despite previous searches finding
no such text. The tool measured0matches. A full-file reread followed the
mismatch, but the same edit was proposed three more times and blocked. Later
explanations explicitly acknowledged absence while still requesting that edit.
No edit or test succeeded. The remaining failure is evidence-to-code grounding,
not failure to deliver or acquire the full visible source in this treatment.

All immediate feedback remained visible. Only the oldest root-list journal
entry was dropped at turn11; persistent memory and later source reads remained.
Costs:63,753 prompt /22,041 prefill /2,454 decode tokens;41,712 adopted/inherited,
396 capsule and9,592 selected-parent tokens;0preemptions. Engine87.282065s,
solver89.636908s, child112.507569s, official grading11.425066s and wrapper129s,
under1,200s. Active GPU utilization was not sampled; terminal GPU was4MiB/0%
with no compute process, original wrapper, tmux job or task container.

This is a behavior change, not a quality success, training-effect delta or
performance result. Same-task official quality must still precede multibranch
and formal validation. No manual patch, target hint or retry was injected.
