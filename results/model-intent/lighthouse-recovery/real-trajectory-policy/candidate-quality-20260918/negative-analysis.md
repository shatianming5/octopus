# M8 terminal negative-quality analysis

**Verdict:** the saved traces support a valid negative quality result under the
frozen protocol. No measurement-invalidating serving, token/text transport,
parser corruption, or identity defect was established. **B7 is unmet**; this
does not establish7B impossibility or invalidate every possible real trajectory.

All indices below are zero-based. Complete source traces/private checks remain
in the supervisor's original directories. No model, GPU, teacher, replay,
evaluation, acquisition or source modification was performed for this analysis.

## Decisive result table

| Actual candidate episode | Lighthouse | Mistune heldout |
| --- | ---: | ---: |
| Actions / model requests | 100 / 100 | 100 / 100 |
| Action kinds | 13 view,86 shell,1 message | 5 view,77 write,17 shell,1 invalid |
| Successful tool actions | 16 | 7 |
| Error/rejected actions | 83 | 93 |
| Parser errors, included above | 0 | 1 |
| Distinct actions / repeats after first | 24 / 76 | 20 / 80 |
| Structured views: calls / distinct / repeats | 13 / 9 / 4 | 5 / 4 / 1 |
| Unique nonempty successful view outputs | 8 | 3 |
| Actual reproduce / rerun requests | 0 / 0 | 0 / 0 |
| First / last successful tool index | 0 / 17 | 0 / 26 |
| Last novel structured source inspection | 15 | 2 |
| Original red / same-check green | false / false | false / false |
| Effective patch bytes | 0 | 0 |
| Terminal reason | action_limit | action_limit |
| Quality | original official false | fixed behavior false: no effective patch |

Successful native-shell inspections are counted separately from structured
views. Repeated actions include both useful repeated reads and rejected
retries; they are not all the same failure category.

## What actually happened

**Lighthouse:** actions0-10 mostly inspect stack-pack source, with four repeated
views. Action11 is an ordinary message. A grep at12 exits1 without output;
subsequent grep calls execute successfully. Action15 reads
`lighthouse-core/lib/stack-collector.js`; the last successfully executed tool
is a clipped grep at17. At18, the model requests nonexistent
`js-library-detector/library/libraries.js`.

Every remaining action19-99 is a native-shell attempt rejected **before
execution** with `path_discovery_required` (81). These include `ls`, `find`,
`grep`, and an npm attempt. There is no later structured view.
The other errors are one missing-path view and the one executed grep failure.
Therefore, the final81 outcomes are not evidence that those shell commands
ran and failed.
Its final action99 is another blocked native-shell call.

This is an important controller-specific limitation on interpretation:
native-shell discovery was not accepted as recovery inspection after the
missing path. The trace supports failure to recover/select an admissible action
under this frozen interface, not a claim of general coding incapacity. No
protocol relaxation or hypothetical new guard was introduced.

**Mistune:** actions0-2 successfully inspect the root, source directory and
`src/mistune/renderers/markdown.py`. Action3 has explanatory code fences and is
rejected by the documented “Do not use code fences” rule. Its XML call is outside
closed fences; the parser's error wording is narrower than its documented
whole-response restriction, but this is not token corruption or a reason to
repair output silently.

At4 the model attempts to create a repository-local Python test before original
red, then tries running the nonexistent file. Four distinct write proposals
are rejected with `original_red_required`;73 subsequent writes are blocked as
repeated failures. Of17 shell calls,4 `ls` calls succeed,10 execute and fail
(2 nonexistent scripts,8 syntactically invalid inline-Python payloads), and3
are repeated-failure rejections. One invalid directory view and its repeated
retry add two view errors. Total repeated-failure rejections are77:
73 writes,3 shells and1 view.
Its final action99 is another repeated write to `test_script.py`.

The model did produce Python test-like content, including assertions in some
blocked files. That content was not executed and cannot be graded as correct.
The decisive failure is that it never uses the required `reproduce`/
`rerun_reproduction` workflow and repeatedly chooses blocked legacy-style
write/run actions. The earlier data-admission `typing_extensions` issue is not
a reached failure in these actual quality actions: the observed inline failures
are syntax errors, not missing imports.

## Context, caps and no-progress accounting

| Saved request evidence | Lighthouse | Mistune |
| --- | ---: | ---: |
| Prompt tokens min / max | 1,969 / 28,646 | 1,815 / 28,672 |
| Total recorded prompt tokens | 2,619,191 | 2,184,109 |
| First history-trim index | 10 | 53 |
| Requests with trimmed old pairs | 90 | 47 |
| Maximum removed / retained old actions | 34 / 65 | 46 / 56 |
| Output tokens min / max / sum | 28 / 396 / 18,539 | 23 / 879 / 23,718 |
| Tool-output clipping events | 2 | 0 |
| Output-generation cap hits / tool timeouts | 0 / 0 | 0 / 0 |

The Lighthouse clips are the initial root view and grep17, both at16,000
characters. No stricter view-cap proposal is made. Actual trimming may influence
behavior, but it is the frozen complete-pair policy and no causal attribution
is justified here.

All200 decoded prompts contain the reproduce/rerun tool definitions, pre-red
read-only rule and current `needs_original_red` controller state. All prompts
have the correct assistant-generation suffix and stay within32,768 tokens
including the4,096 output reservation. The known private heldout-check label
marker appears in no prompt.

All200 generations finish normally with EOS151645, below the output cap.
Decoded output IDs exactly match saved generated text; saved entry text matches
generation text; raw shell-command parameter bytes match submitted shell
parameters; recorded request/generation/context records match. Each task has100
unique request IDs and one fixed sampling configuration:
temperature0, seed17, max_tokens4096, no structured-output constraint.

The no-progress rule is specifically about consecutive ordinary messages or
rejected finish requests, not all rejected tools. Lighthouse has one isolated
message (streak1 once); Mistune has no such message. Tool attempts reset the
counter, so100-action exhaustion rather than a two-message stop is consistent
with the frozen rule. This analysis does not add a semantic-progress guard.

## Identity and what the pilot did/did not establish

The actual training receipt SHA256 matches the merge receipt:
`a26c04ddad6db189c20300ac37ef7054569d31a83d261a56f8ace76a1480d5ab`.
Adapter, recipe digest and encoded-data digest match across training and merge.
Both served episode identities match candidate revision
`10f0224eeea5222849dec9f468f5ff3aef1811da07c199f50d87da6d6a541ac7`
and adapter
`ea0218b7a5bbdb6167dbf3e61156c841a79c7c5e7f0086c2187fa013b0d1b334`,
not the original baseline revision.

The records establish one completed optimizer update over four train states,
2,523,136 trainable parameters with frozen base, final adapter selection, actual
merge/verified transfer, and two completed bounded candidate episodes.
Loss1.0115886926651 is a training measurement, not proof of transferable skill.
Distinct adapter/merged-file identities and optimization provenance were checked;
no new tensor load or tensorwise weight-delta calculation was performed.

Serving is the recorded stock vLLM0.28.0+cu129 / Transformers5.16.1 /
Torch2.13.0+cu129 engine, BF16, context32768, one sequence, no prefix caching.
The original C03 tokenizer identity is retained. The different training
Transformers version is explicit provenance, not evidence of serving corruption.

Lighthouse's existing official grader completed with returncode0 and a recorded
FAILED required test file, not missing grade output. Mistune's external fixed
check returns false because there is no effective model patch; its four private
behavior tests were **not rerun on a nonexistent repair**. No official Mistune
resolution claim is made.

The historical24-request original negative is not a new matched baseline.
There is no relative-improvement calculation, zero-quality ratio, causal
attribution to the one update, representative-cohort claim, or proposal to fill
idle GPUs.

## Coverage, artifacts and closure

Machine-readable consolidated evidence:

`/home/ubuntu/.copilot/session-state/105ec35a-5b62-4a08-be72-1cd885489e9a/files/m8-negative-quality-analysis.json`

Six bounded derived count/detail/payload-integrity files are in the adjacent
`negative-analysis/` directory. The supervisor's complete original episodes,
official output, training/merge receipts and private heldout checks remain
untouched.

The episodes total208,546,110 bytes. Parsing/decoding was performed only on the
existing approved remote CPU route, with60-second wall bounds, one busy Python
process and4GiB address-space cap; no torch/model/GPU/replay/evaluator call.
The delayed Lighthouse copy, owned shell81/PID2734090, eventually exited0;
there is no outstanding transfer or CPU job to wait for. All named remote
analysis copies/scripts were removed, and the owned directory was removed.
No other worker/process or GPU experiment was stopped.

No source defect was established that invalidates these negative measurements,
and no code or policy guard was changed. This does not prove every controller
design choice ideal or rule out all possible defects beyond the reached paths.

**Terminal disposition:** preserve/accept this result and release this managed
worker. There is no new M8 training/evaluation/candidate task or hypothetical
follow-up requiring retention. B7 stays unmet and blocks dependent B8; independent
native work is separate.
