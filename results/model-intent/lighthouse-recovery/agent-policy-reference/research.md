# Same-size software-agent policy reference

**Selected candidate: `SWE-Gym/OpenHands-7B-Agent`, revision
`97054e70501d79c27499518af5ce613be6278bcb`.** This is a purpose-trained software
agent at the original 7B size, not another generic model-size escalation. No
weights have been downloaded or evaluated at this research checkpoint. The
initial guessed name without `-Agent` returned401; the public organization API
resolved the actual ungated repository. No alternate candidate was needed.

## Primary-source findings

The authors explicitly use Qwen2.5-Coder-Instruct7B/14B/32B as base models and
fine-tune OpenHands agents on491 successful real repository trajectories.
Their reported7B SWE-Bench Verified result improves from1.8% to10.6%, with
stuck-in-loop frequency decreasing from39.6% to21.0%. These are the authors'
results, not ours, and they do not guarantee Lighthouse success. The published
training trajectories average39.9 model completions. The reproduction guide
uses50-iteration runs and a separate base-model tokenizer when serving weights.
[Paper, Sections3.1-3.2/Table3](https://arxiv.org/html/2412.21139v1),
[first-party reproduction guide](https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md).

The pinned repository contains a606-byte config and four BF16 safetensors
shards, **15,231,272,470 bytes** for the five inference files. Architecture is
`Qwen2ForCausalLM`, hidden size3584,28layers/28queryheads/4KVheads, vocabulary
152064 and32768position capacity. It does not ship a tokenizer, generation
config or safetensors index. The `_4` suffix is part of the uploaded shard names;
it is not evidence of a particular training-epoch count. Reuse an explicitly
pinned Qwen2.5-Coder tokenizer rather than silently inventing model assets.
[Pinned file metadata](https://huggingface.co/api/models/SWE-Gym/OpenHands-7B-Agent/revision/97054e70501d79c27499518af5ce613be6278bcb?blobs=true),
[pinned config](https://huggingface.co/SWE-Gym/OpenHands-7B-Agent/blob/97054e70501d79c27499518af5ce613be6278bcb/config.json).

## Held-out scope

The declared SWE-Gym training environment has2438instances from11repositories.
Complete first-party repository statistics list dask/dask, facebookresearch/hydra,
conan-io/conan, getmoto/moto, pandas-dev/pandas, python/mypy, bokeh/bokeh,
iterative/dvc, Project-MONAI/MONAI, pydantic/pydantic and modin-project/modin;
their counts sum to2438 with no missing labels. **GoogleChrome/lighthouse is
absent.** The separately released SFT set has491trajectories, matching the paper.
[SWE-Gym dataset revision](https://huggingface.co/datasets/SWE-Gym/SWE-Gym/tree/bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb),
[complete repository statistics](https://datasets-server.huggingface.co/statistics?dataset=SWE-Gym%2FSWE-Gym&config=default&split=train),
[SFT dataset revision](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories/tree/4aaa5a4a4b5861f4799d2336908760c190ac3b17).

This establishes exclusion from the **declared added agent-training source**.
The release does not cryptographically bind its weights to a training-dataset
revision, and original Qwen pretraining contamination is unknown. Do not claim
stronger held-out provenance or a controlled causal SFT effect. No task patch
or assistant training solution was extracted into the research report or solver.

## Protocol and usage limitations

The released SFT system message uses text calls of the form
`<function=...><parameter=...>...</parameter></function>`, with `execute_bash`,
`finish` and `str_replace_editor`. It is **not** the Qwen native JSON
`name`/`arguments` call-body profile just evaluated. A compatible, explicitly
bounded adapter is required; merely swapping weights would not reproduce the
trained scaffold. The original OpenHands guide warns that scaffold-version
changes materially affect results.
[System-instruction source, row0 only](https://datasets-server.huggingface.co/rows?dataset=SWE-Gym%2FOpenHands-SFT-Trajectories&config=default&split=train.success.oss&offset=0&length=1).

The project code declares Apache-2.0 and both dataset cards declare MIT. The
weight repository itself has no model card or separate license file; do not
misreport an explicit weight-specific license declaration. This proposed use
is local research/reproduction of the authors' released agent, with attribution
and without redistributing weights.
[Project license](https://github.com/SWE-Gym/SWE-Gym/blob/main/LICENSE).

**Decision:** prepare one explicitly declared same-Lighthouse capability
reference using the trained action format. A50-action ceiling, selected before
model output from the documented50-iteration protocol, will replace12 only for
this new reference; retain512generated tokens/action and report the first12-turn
prefix separately. This is not a weight-only comparison or a budget sweep.
The original12-turn attribution runs and closed LoRA recipe remain unchanged.

## Directory-view contract exposed by the actual trace

The authors' [reproduction guide](https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md)
specifies their fork rather than arbitrary current OpenHands. At fork revision
`e644a2ca45c3623b27a7e6c169e3d479f0a87fbc`, the
[public editor description](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/function_calling.py#L211-L258)
says directory views list non-hidden entries up to two levels deep, and
`view_range` applies only to files. Its
[lockfile](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/poetry.lock#L5643-L5651)
pins `openhands-aci==0.1.0`. That release's
[implementation](https://github.com/OpenHands/openhands-aci/blob/0698260b8e03ff2974ba81fd97ad8585a2255297/openhands_aci/editor/editor.py#L129-L146)
rejects directory ranges and lists actual absolute paths via a depth-two,
non-hidden `find` operation.

Our command-repair episode instead obtained two directory views sliced to the
first20 entries, dominated by dotfiles. The actual root was the correct
Lighthouse checkout and both results reached the model; this was a tool
contract difference, not a missing mount or lost-feedback finding. The bounded
adapter now rejects file ranges on directories and lists non-hidden absolute
paths to two levels in Python, shallowest first, without following symlink
directories or invoking a shell. Its200-entry/6000-character limits and omission
of an empty directory's decorative root line remain explicit bounded-scaffold
differences. This source comparison does not establish that the change will
solve the task or that every remaining producer-scaffold detail is reproduced.

## Physical workspace in the trained task input

The same pinned fork's
[documented rollout script](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/scripts/rollout-swe-train-full.sh#L25-L28)
selects `CodeActAgent`. Its
[task-input construction](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/swe_bench/run_infer.py#L56-L92)
provides a real checkout path in `uploaded_files` and the original issue in
`pr_description`. Our actor input instead exposed a custom logical
`Repository: owner/repo` field; every actor episode's first guessed missing path
was the real root plus that logical identifier. Source and feedback transport
were verified, but the producer's physical-workspace input shape was absent.

The actor now receives its actual configured working directory in
`uploaded_files`, with the issue preserved verbatim in `pr_description`.
Logical repository identity remains in the frozen task and evaluator metadata,
not as an unnecessary model-visible pathname candidate. No directory is moved,
aliased or redirected, and no task-specific file or query is supplied. Other
decision formats retain their original input bytes. This is a generic input
contract correction; whether it changes model behavior still requires its own
prospective quality result, not a post hoc success claim.

## Explicit file-range contract exposed by the JSON episode

The same pinned editor's [file-view implementation](https://github.com/OpenHands/openhands-aci/blob/0698260b8e03ff2974ba81fd97ad8585a2255297/openhands_aci/editor/editor.py#L148-L190)
rejects an explicit start outside the file and an explicit finite end beyond its
line count (lines165-176); only `end=-1` requests the remainder. The completed
JSON episode instead clamped `[83,103]` against a91-line source and reported40
later beyond-EOF views as successful empty results. Actual next inputs contained
the correct extent and empty feedback, so this is not evidence of lost feedback.
The initial trace-only attribution's no-mismatch conclusion is superseded by this
new pinned-source comparison; its raw observations and report remain immutable.

The adapter now rejects explicit out-of-file ranges and returns a typed failure
with the actual extent. V4 scopes that certificate to the affected file revision,
so unchanged or unrelated edits cannot clear it, while genuine changes to that
file can make the same range valid. No path-discovery requirement, forced action
or extra model request is added. Default bounded views, `end=-1`, shell-adapter
reads, raw line endings and the existing `splitlines`-based `total_lines` convention
are preserved. In particular, this does not adopt the producer's extra terminal
empty field from `split('\n')`, nor its unlimited default view. The exact observed
boundary fails before the fix and passes after it; quality remains a separate
prospective question, not a claimed consequence of the source mismatch.

## Normal assistant messages versus format errors

A follow-up trace uses the same pinned producer revision, not newer scaffold
behavior. The [converter's no-call branch](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L740-L745)
keeps a no-function assistant message as-is. [CodeAct response conversion](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/function_calling.py#L440-L506)
turns it into `MessageAction(content=..., wait_for_response=True)`, rather than
automatically declaring a malformed function call. [SWE-bench evaluation](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/swe_bench/run_infer.py#L435-L442)
uses the first-party `codeact_user_response` callback for this path.

That [generic callback](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/utils/shared.py#L118-L165)
asks the agent to continue independently, send its answer before finishing, and
not ask for human help. After repeated user-message turns it additionally permits
giving up via finish. These are generic control responses, not task-specific
source, test, patch or answer hints. By contrast, [actual validation exceptions](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/controller/agent_controller.py#L483-L498)
add an `ErrorObservation` and return without an action. Ordinary messages and
rejected tool calls are therefore distinct producer paths. The default strict
adapter labels every no-call response `invalid_function_call`, including
normally stopped prose in earlier unconstrained XML episodes.

This is a source-backed control-flow difference, not proof that supporting
ordinary messages will solve Lighthouse or that capped/incomplete output should
be accepted. A possible bounded treatment must preserve raw text, count every
message as one proposal, create no tool success or trusted repository evidence,
retain all verification/recovery rules and use no extra model/planner request.
The same converter also [_repairs missing function closing text_](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L551-L558).
That repair is explicitly outside the authorized treatment: no appended tags,
fence repair or promotion of a length-capped generation into a completed call.

The bounded implementation is now opt-in via `--message-policy codeact_continue_v1`
with `--decision-format openhands_xml_v1`; strict XML and all constrained/JSON
comparators remain unchanged. Only nonempty, naturally stopped text with no
attempted function/parameter markup or code fence becomes `action=message`.
Its original text and actual generation metadata remain in the trace. The
response is the exact three-sentence base callback above, with its pinned source
recorded, not a tool result. We deliberately omit the repeated-user-turn give-up
extension; this bounded treatment never weakens verification or finish validity.
No initial instructions, planner prefill, tool choice or per-proposal budget changes.

Messages consume the same proposal/request slots, use control lifecycle events,
and remain ordinary turns through journal deduplication/truncation. They do not
enter tool execution, failure recovery, workspace revisions, task evidence or
the latest-observation capsule. A previous real error/verification fact survives
the message. Capped/empty/malformed/unsupported outputs still produce real errors;
a syntactically complete capped call remains executable. Quality is unproven.

## Emitted extent after character clipping

The new [actual clipping attribution](message-control-quality/clipping-metadata-attribution.md)
found that the47-line view emitted only21complete lines plus part of line22,
while `returned_lines=47` and `more_lines_after=false` described the selected
pre-clipping window. `output_truncated=true` was already correct and source was
nonempty. This is not feedback loss or proof that clipping caused recurrence;
it is a concrete distinction between the emitted and selected line extents.

The worker now finalizes view metadata after the existing character slice.
`returned_lines` counts emitted lines, including a partial last line;
`last_line_truncated` identifies that partial content when character clipping
occurs. `more_lines_after` retains the original remaining-file flag and also
accounts for selected lines absent from the emitted output. The file's true
`total_lines`, exact output text, character/line limits, input instructions and
range validity are unchanged. The same finalization applies to directory path
lines; non-view and unclipped results retain their existing shapes. The actual
JSON-worker boundary reproduced47versus22before the fix and passes afterward.

The prospective [emitted-extent run](emitted-extent-quality/terminal-report.md)
confirmed those exact fields in the next compiled input but remained official
false with no diff. The first four actions were successful views. All remaining
46outputs were genuinely length-capped attempted editor calls, not ordinary
messages: each had function/parameter markup and an incomplete `old_str`; no
call closed or supplied `new_str`. Actions5-49 repeated the exact same704-byte
partial output. The [bounded attribution](emitted-extent-quality/malformed-capped-attribution.md)
confirmed raw output/error retention and actual `length/512/512` facts in all45
available next capsules. No physical tool submission, state transition or
verification change arose from these malformed proposals.

This is the terminal attribution boundary for the frozen interface. Appending
a closing tag cannot recover a missing replacement or a truncated old string.
Treating capped attempted calls as messages would contradict the pinned producer
path. Increasing output budget, continuing generation, changing the prompt,
model or seed, or forcing a shorter edit/test would be a new intervention rather
than a demonstrated adapter repair. Exact generated token-ID persistence could
improve audit instrumentation, but the trace already retains raw text and actual
counts; it is not a behavioral quality repair and is not added here.

## Exact continuation after a clipped view

The terminal attribution correctly found no defect in rejecting the46incomplete
edits. A narrower comparison of the visible base source with the actor's own
pre-call explanation exposes an earlier actionable boundary. The actor saw only
21complete lines plus part of line22 from `stack-packs/packs/react.js`, then said
it should add a temporary `id`. The same visible file already exports `id: 'react'`
near its end, beyond the emitted window. `more_lines_after=true` says content is
missing but does not provide a valid overlap point after character clipping.
The actor instead copied the partial multi-kilobyte `reactIcon` line into
`old_str`, which cannot fit the frozen512-token response.

The bounded worker now returns `next_view_range` only for character-clipped file
views. If the final emitted line is partial, the range overlaps that line so the
next view starts with its complete source; if clipping lands on a line boundary,
it starts at the following line. The range uses the existing supported `[start,
-1]` form. It does not change output bytes, limits, prompt instructions, action
validity, sampling or request count, and it does not prescribe a task action.
Directory views omit the field because file ranges are invalid there.

The real JSON-worker seam was red before implementation because no continuation
field existed. The focused tests now cover partial lines, exact boundaries, EOF,
non-one starts, full continuation without a gap, directory omission and the exact
OpenHands next-turn suffix. This factual continuation repair is separately ready
for prospective quality evaluation; it does not assume that the actor will use it.

## Pinned producer truncation notice

The continuation episode used `[22,47]` and saw the existing React `id`, so the
range field itself worked. It later chose an invalid zero-based range and then
degenerated into prose despite complete successful index evidence remaining in
the exact prompt. Root discovery also retained every top-level entry, including
`lighthouse-core`, `stack-packs`, `package.json` and the nested Lighthouse test
tree. These checks rule out another range, journal or directory-discovery repair.

One remaining producer difference is source-pinned rather than invented wording.
`SWE-Gym/OpenHands@e644a2ca` locks `openhands-aci==0.1.0`; tag0.1.0 resolves to
`OpenHands/openhands-aci@0698260b`. Its file editor applies `maybe_truncate` to
file views, and the [fixed notice](https://github.com/OpenHands/openhands-aci/blob/0698260b8e03ff2974ba81fd97ad8585a2255297/openhands_aci/editor/prompts.py)
explicitly says only part of the file was shown and to search with `grep -n`
before retrying the tool. Directory output does not use this file-view helper.

The opt-in `view_truncation_policy=openhands_aci_v0_1_0` now records that exact
notice and pinned source on character-clipped file observations. Journal rendering
places the notice after the unchanged raw source output, matching its producer
role; provenance stays structured and is not presented as source. The treatment
does not adopt ACI's16,000-character limit, numbered output, insert/undo commands
or terminal-empty-line convention. Existing6,000-character source, exact extent,
next range, action set, prompt root, model requests and all budgets remain.
Unclipped files, directories and default protocols are unchanged.

The focused tests were red before the constants/config/renderer existed and now
pass across worker, solver, freeze/reference and strict-comparator consumers.
This is one exact upstream interface treatment, not a wording sweep or evidence
that the actor will follow the notice.

The first remote attempt exposed a deployment-specific bug before the treatment
ran: `LighthouseTools` sends `lighthouse_tool_worker.py` as inline `python -c`
source inside the product container, where a runtime import of its repository
sibling is unavailable. It stopped on the first clipped file view after four
requests, produced no observation or official grade and is invalid as quality
evidence. The notice constants now live in the self-contained worker; the
protocol re-exports them for host-side rendering/tests. The regression executes
the real source string with `python -c` from an isolated temporary repository,
and all188affected cases pass. No scientific setting or notice text changed.
