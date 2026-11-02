# Original base under the improved agent policy

**Official result: `resolved=false`, zero patch bytes.** The user explicitly
authorized this one comparison after PR25's original evidence was frozen.
It is separate from the already completed original-protocol base versus
branch-choice LoRA comparison and was not included in PR25's merged head.

Only model weight identity changed from the counted-feedback treatment:
the original unmerged Qwen2.5-7B-Instruct snapshot replaced the merged recovery
adapter. All protocol, task, prompt, schema, tool, source and evaluator settings
were equal. There was no training, new download, action rewriting or retry.

The base model made 12 proposals / 9 distinct, with 10 physical submissions /
9 distinct and 2 rejected known-failed repeats. Unlike the recovery adapter's
edit loop, it proposed only inspection: 7 physical reads and 3 searches.
However, **all 7 reads failed invalid intervals** and all searches returned
empty output. Successful evidence acquisition, edits and tests were all zero.
No known-failed exact action reached an external tool a second time.

The recovery adapter clearly changes behavior under this improved policy, but
removing it did not restore real-task success. More distinct actions do not
prove better agent control when none acquires source evidence. This is not a
training-effect generalization or evidence that an adapter is the sole culprit.

## Concrete next repair seam

The saved base trace requests start=0, oversized ranges, and 201-line inclusive
ranges. The action schema accepts arbitrary integers and does not express the
tool's 1-based / 200-line limits. The model also interprets interval failures
as truncation and uses regex-like text in a literal search.

A separate offline real-tool probe confirms that searching a nonexistent path
silently returns empty success. A read-only existence check in the unchanged
task image confirms the model's requested `lighthouse-core/lib/js-library-detector.js`
does not exist. This result was not fed back into the model or executed as an
extra episode action. It identifies a genuine observation-contract defect,
not a task-specific patch hint. Bounded inspection arguments and explicit
missing-path errors are the next surgical repair, not another trained recipe.

Native costs:43,429 prompt /17,221 actual prefill /2,198 decode tokens,
26,208 adopted/inherited tokens,396 capsule tokens and9,196 selected-parent
tokens. All journals were untruncated; no preemptions occurred. Engine scope
77.752586s, solver79.850010s, child100.325024s, official grading9.670632s,
total wrapper115s under1,200s. Active GPU snapshot was43,146MiB /59%; terminal
GPU was4MiB /0%, with no compute process, original wrapper, named tmux session
or task container. The original quality/multibranch/formal gates remain open.
