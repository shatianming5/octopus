# Bounded inspection contract

**Official result: `resolved=false`, zero patch bytes.** The generic tool
contract defects are repaired, but the model did not acquire useful evidence
in this episode. It made 12 proposals / 4 distinct, 4 physical submissions
(2 searches and 2 directory listings), 8 rejected failed repeats and zero
known-failed physical resubmissions. Every submitted path was nonexistent.

The new read contract uses model-selected 1-based `start` and `count` in
1..200, rather than requiring the model to calculate a bounded inclusive end.
No argument is clamped or rewritten. The actual xgrammar compiler/matcher
accepted three legal and rejected five illegal argument fixtures on CPU;
54 targeted tests passed. Legacy start/end remains available. Searches now
raise an explicit missing-path error instead of a success-shaped empty result;
existing-path searches with no matches still legitimately return empty output.

The model emitted no read action here, so zero read-argument failures is not
evidence of successful reading. Actual successful nonempty inspection was zero.
It initially interpreted FileNotFoundError as absence of the searched symbol,
repeating the same failed searches. At turn10 it recognized a missing path,
but then listed another nonexistent directory and finally a URL as a local
path. This directly motivates causal missing-path discovery constraints rather
than another presentation change. No correct path or patch was supplied.

The original base model,12 slots,512 output tokens,seed17 and official
evaluator were retained. Actual costs were34,741 prompt,8,837 prefill and1,997
decode tokens;25,904 adopted/inherited,396 capsule and9,273 selected-parent
tokens. Journals were untruncated and preemptions were zero. Engine scope
70.549830s, solver73.045142s, child91.582478s, official grading9.743607s and
complete wrapper106s, below1,200s. Active utilization was not sampled for this
episode; terminal GPU was4MiB/0%, without compute process, wrapper, named tmux
session or task container. Quality and downstream research gates remain open.
