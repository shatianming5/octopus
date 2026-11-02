# Measured edit-match feedback on the same task

**Official result: `resolved=false`, zero patch bytes.** Explicit multiplicity
feedback changed the model's stated interpretation and inspection range, but
not its invalid edit. There were 12 genuine model decisions, 3 distinct
proposals, 3 physical submissions (2 successful reads and 1 failed edit),
9 rejected identical failed proposals, zero known-failed physical repeats,
zero valid edits and no test action.

Only `edit_feedback=match_count` changed from the deliberative freeze. On
precondition failure the real tool now reports the measured number of exact
occurrences, while preserving `edit_precondition_mismatch`, mandatory evidence
refresh and no write on error. It does not reveal locations, choose a match,
rewrite an action or relax uniqueness. Default legacy feedback remains intact.
Model, prompt, schema, source inspection, 12 slots, 512 output tokens, seed 17
and original evaluator are unchanged. There was no training or hidden retry.

The tool measured **2 matches** for the attempted old text. The model correctly
stated that multiplicity caused the failure and broadened its read from lines
80-100 to 70-120. It nevertheless reused exactly the same old/new/path edit
nine times. Later explanations explicitly acknowledged multiple matches while
still asserting that the same action should proceed. Recognizing an error in
text did not produce a valid recovery action. The runtime blocked every
known-failed physical resubmission; it did not manufacture a repair.

All journals remained untruncated. Actual costs: 37,185 prompt tokens, 9,489
prefill tokens, 2,351 decode tokens, 27,696 adopted/inherited tokens, 396 capsule
tokens and 9,196 selected-parent tokens. There was at most one active scheduler
request and zero preemptions, KV evictions or allocation rejections. Engine
scope was 77.359189 seconds, solver end-to-end 79.541279 seconds, child wall
102.261848 seconds and official grading 10.300699 seconds. Total wrapper wall
was 118 seconds, within the 1,200-second phase ceiling. Active utilization was
not sampled; terminal GPU was 4 MiB / 0%, without compute process, original
wrapper, named tmux session or task container. This is not a performance claim.

The real-task action-selection gate remains unmet even with measured error
detail. No further frozen treatment, multibranch or formal run is ready.
