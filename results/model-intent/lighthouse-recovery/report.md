# Same-task failure attribution and recovery

**Status: real-task quality remains unmet.** Completed treatments below retain
their actual outcomes; `GoogleChrome__lighthouse-10176` remains officially
unresolved. No easier task, multibranch performance run, formal seed expansion,
or additional outcome-driven training recipe was substituted.

## Current deep-audit conclusion

The [fixed-actor failure audit](agent-policy-reference/report.md#deep-failure-audit-what-the-repeated-negatives-actually-establish)
now separates task repair from interface correctness. Fourteen valid SWE-Gym7B
episodes made 700 requests, 243 physical tool submissions and 227 successful
tool executions. Only nine edits occurred (including a self-revert pair), three
episodes had nonempty final diffs, and no developer test was even proposed.
No successful direct read reached the user-facing `js-libraries.js` audit.
The three diffs either changed collection only or rewrote unrelated React advice
strings; none established the issue's separate display requirement.

The latest notice recovery is valid and officialfalse, with no length caps:
40 normally stopped and structurally complete XML calls were rejected because
our `execute_bash` is not Bash. Thirty-eight selected `ls -R`, two selected
`find ... -not -path`. Error feedback was preserved. The producer actually
executes shell commands, so the immediate obstruction is an action-space
mismatch combined with weak actor recovery, not malformed XML or an absent
source result. The independent [producer audit](agent-policy-reference/producer-interface-audit.md)
also documents budget, history, view-format and training-domain differences.

At least 406 of 700 proposals are extra copies within each episode's longest
identical-output run. The actor weights do not learn from these retries; these
are not independent pass-at-k samples. Prior small interface fixes and large
diffs were over-interpreted as progress without a localization/edit/test baseline.
No new code or model run was added during the user-requested deep audit. The
next useful evidence is a separately authorized matched producer-shell control,
not another speculative formatting fix. Official quality and all downstream
multibranch/Formal gates remain unmet; the goal is not marked complete.

## Decision-bearing results

| Treatment | Proposals / distinct | Actual submissions | Failed raw re-executions | Rejected proposals | Patch bytes | Official resolved |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Existing branch-choice LoRA, original loop | 12 / 1 | 12 | 11 | 0 | 0 | false |
| Original base Qwen, original loop | 12 / 1 | 12 | 11 | 0 | 0 | false |
| Base Qwen, failure-aware loop | 12 / 6 | 6 | 0 | 6 | 0 | false |
| Tool-recovery LoRA, failure-aware loop | 12 / 4 | 5 | 0 | 7 | 0 | false |
| Same recovery LoRA and loop, raw read presentation | 12 / 7 | 8 | 0 | 4 | 0 | false |
| Same recovery LoRA and raw reads, single-call deliberation | 12 / 2 | 3 | 0 | 9 | 0 | false |
| Same deliberative policy, measured edit-match feedback | 12 / 3 | 3 | 0 | 9 | 0 | false |
| Original base, same improved policy | 12 / 9 | 10 | 0 | 2 | 0 | false |
| Original base, bounded inspection contract | 12 / 4 | 4 | 0 | 8 | 0 | false |
| Original base, causal path recovery v2 | 12 / 5 | 9 | 0 | 3 | 0 | false |
| Same-size coding checkpoint, JSON journal | 12 / 1 | 12 | 0 | 0 | 0 | false |
| Coding checkpoint, repository-aware root search | 12 / 1 | 12 | 0 | 0 | 0 | false |
| Coding checkpoint, native chat turns with rationale | 12 / 3 | 12 | 0 | 0 | 0 | false |
| Coding checkpoint, action-only native chat | 12 / 9 | 10 | 0 | 2 | 2211 | false |
| Same action-only policy, resource-safe Jest | 12 / 9 | 11 | 0 | 1 | 2211 | false |
| Larger same-family14B, same v2 policy | 7 / 7 | 6 | 0 | 0 | 481 | false |
| Same14B, verified-completion v3 | 12 / 7 | 7 | 0 | 5 | 481 | false |
| Same14B/v3, canonical execution identity | 12 / 6 | 5 | 0 | 7 | 481 | false |
| Same14B/canonical v3, existing deliberative contract | 12 / 7 | 12 | 0 | 0 | 0 | false |
| Same14B/deliberative, exact journal compaction | 12 / 7 | 11 | 0 | 1 | 0 | false |
| Same14B/compaction, matching-file directory discovery | 12 / 5 | 12 | 0 | 0 | 0 | false |
| Same14B/file discovery, explicit regex contract | 12 / 4 | 12 | 0 | 0 | 0 | false |
| Same14B/regex, evidence-carrying task decisions | 12 / 4 | 12 | 0 | 0 | 0 | false |
| Same14B/task state, runtime-owned observations | 12 / 3 | 12 | 0 | 0 | 0 | false |
| Same14B/runtime observations, evidence-only context | 12 / 5 | 12 | 0 | 0 | 0 | false |
| Same14B, pinned native tool-call protocol | 12 / 6 | 10 | 0 | 2 | 0 | false |

Submissions count external tool calls, excluding `done` and runtime rejections.
The repeat column counts failed raw actions that physically ran again, not
repeated proposals. Effective-argv aliases are reported separately below.

The old LoRA episode was reused, not rerun. Exactly one base comparison, one
runtime-treatment episode, and one gated recovery-LoRA Lighthouse episode were
added. Every Lighthouse treatment retained the original task, 12-proposal
limit, seed 17, and original official evaluator. Runtime or model changes are
explicit treatments, not silently changed comparison conditions.

After renewed autonomous same-task continuation, one separately frozen
[raw-read counterfactual](raw-evidence/report.md) changed only evidence
presentation, not model/prompt/schema/limits. Literal but nonunique edit
preconditions still failed. It was not a retry or retuning of the frozen LoRA
recipe; the additional treatment and its negative result are explicit here.

A valid [fresh-vLLM reference](fresh-reference/report.md) then reproduced one
raw-read failed action byte for byte with all 3,943 actual input tokens and the
original schema string, but without Octopus or prefix caching. One preceding
instrument-invalid schema-ordering output is preserved and excluded. Physical
KV reuse is not the sole explanation for that prompt; this is not a blanket
runtime-correctness claim.

The separately frozen [single-call deliberative treatment](deliberative/report.md)
added generic observation-grounded reasoning before the tool action, without
changing the model, task, limits or tool semantics. It still repeated the same
failed proposal nine times after reacquiring evidence. No valid edit or patch
resulted. More verbose reasoning was not successful recovery.

One [measured-match feedback treatment](counted-feedback/report.md) then
distinguished absent old text from nonunique old text without changing the
prompt/schema or editing semantics. The model explicitly recognized the two
matches and broadened its inspection, yet still repeated the same invalid edit
nine times. Correct verbal error recognition did not become action recovery.

After user-requested review and merge of PR25, the explicitly approved
[improved-policy base comparison](base-improved-policy/report.md) changed only
the recovery-model weight identity. More distinct actions did not help: all
reads had invalid intervals, and searches silently targeted a nonexistent file.
These were not successful evidence acquisition.

The [bounded inspection repair](bounded-inspection/report.md) aligned read
arguments with actual grammar bounds and fixed missing-path search reporting.
Its episode still guessed nonexistent paths. The subsequent
[causal path-recovery policy](path-recovery-v2/report.md) acquired a root listing
and four complete source reads, but still invented an absent edit precondition.
All changes are generic tool/runtime behavior, without task-specific answers,
manual disambiguation or replacement actions. The v2 missing-path mask is
verified by real-tool tests, but was not triggered by that actual episode.
Neither renewed base comparison is a replay of the original-protocol one.

The pinned [same-size coding checkpoint](coder-reference/report.md) also
initially stalled. Its searches were empty because the tool exhausted its
budget on ignored dependencies. [Repository-aware discovery](repository-search/report.md)
restored three actual matches for the same model query, but the model still
repeated that successful query12times. A separate
[fresh positive-input reference](positive-reference/report.md) reproduced the
same canonical action without Octopus or prefix reuse, limited to that input.

[Native chronological chat turns](chat-turns/report.md) then separated prior
model output from later literal tool evidence. This changed behavior but still
stalled on repeated reads. Removing the added rationale requirement in one
[action-only ablation](action-only-chat/report.md) yielded the first accepted
exact edit and a2211-byte diff, but not an official repair. Its developer test
hit a real Jest worker-spawn EAGAIN error in the bounded container. The original
official evaluator separately ran its two designated tests and failed one,
without that infrastructure error. These are distinct failure categories.

[Resource-safe Jest execution](serial-jest/report.md) now removes that
developer-test EAGAIN failure and returns a real code error. The model reads
the affected source, but its corrective edit has two matching old strings;
the final exact repeat is blocked. Its2211-byte patch is unchanged and official
quality remains false. No task-specific test selector was inserted. The full
suite still hits the unchanged30-second cap, with partial code feedback visible.

The separately frozen [14B capability reference](coder14-reference/quality/report.md)
kept the same task, tokenizer, tools, policy and evaluator. It made seven distinct
proposals and a481-byte redundant edit, then selected `done` after two failed
developer tests with five slots unused. Official quality remains false; the
redundant edit is not established as the cause of the official assertion failure.
This exposes a concrete completion-validity gap beyond displaying failures.
The [verified-completion treatment](verified-completion/protocol.json) therefore
freezes a generic v3 constraint, not another model-size trial: a changed workspace
or failed test forbids `done` until later successful verification. Its actual
episode prevented premature completion but still failed quality. It had zero
raw failed-action repeats yet one identical effective failed-test command repeat:
implicit and explicit `--runInBand` executed the same argv. Those counts are not
interchangeable. [Canonical execution identity](canonical-execution/protocol.json)
closes this observed alias using the tool's own normalization. Before its output,
one conditional [existing deliberative14B contract](deliberative14/protocol.json)
was also frozen. The canonical arm blocked all seven effective-repeat proposals,
physically executed only one failed test, and still failed official quality with
the same481-byte patch. That complete negative result made the deliberative arm
eligible; it then spent all12actions on inspection, produced no patch, and also
failed. No new model, rationale wording or output-token increase was introduced.

The two-arm checkpoint is closed. An additional [single-call source-fed physical reference](post-source-reference/report.md)
used the exact3829actual tokens preceding the redundant edit, including the
untruncated190-byte source window that already contains the inserted field.
StockvLLM without prefix reuse reproduced the same raw and canonical redundant
edit in24.916engine seconds, after verifying literal source visibility and exact
schema identity. That tested source-fed mistake therefore does not require
physical reuse. It is not a task-quality result or proof of globalKVcorrectness.
The equivalentGPU9substitution afterGPU7becameoccupied is explicit; zero model
requests were made on the blocked device. That reference scope is closed.

The deliberative trace repeatedly returned identical source evidence without an
intervening mutation. A separately frozen [journal-compaction treatment](journal-compaction/report.md)
therefore projected identical action/result pairs to their latest complete turn
inside read-only segments, without changing the raw audit or persistent failure
memory. CPU replay reconstructed the prior turn11 suffix token-exactly and reduced
four copies of the same source to one before ordinary budget trimming. The one
live episode changed late actions but still produced no edit or patch and remained
officially false. One known-failed proposal was rejected; no failed physical action
was re-executed. Final context accounting separated three deduplicated entries
from two ordinary budget drops, versus four budget drops in the unprojected arm.
All12raw proposals/results remain available. The projection keeps the latest
explanation, not every prior explanation; unique evidence is retained before,
not guaranteed through, the independent token-budget trimming. This closes the
compaction episode without a quality or general performance claim.

The next [directory-discovery treatment](directory-discovery/report.md) addressed
an independently reproduced tool bottleneck: the actual root search showed only
three matching files because100line hits and6,000characters were dominated by
match-heavy files. Returning matching filenames once per directory search exposed
all55files in2,191characters under unchanged caps, without ordering, path hints or
source/translation filtering. Explicit-file search still returns matching lines.
The one live100-second episode nevertheless failed official quality with no patch.
It made nine empty directory searches using `|` alternation expressions against
the still-explicitly-literal search contract, acquired only a root listing and one
source read, and encountered one unrepeated missing path. Thus the CPU coverage
repair is established, but the actual model never used a nonempty matching-file
result. The episode does not show that improved directory coverage is sufficient
for repair or that the new format caused the regex/literal mismatch. Its lower
native work reflects a different, mostly empty evidence trajectory, not a retained-
quality performance benefit. No downstream gate is opened.

A subsequent [explicit-regex contract](regex-search/report.md) addressed that
observed query/semantics mismatch without guessing or rewriting model queries.
CPU replay of the two actual alternation patterns changed empty results to58and100
matching files under the existing caps. Its one112-second live episode was also
officially false with no patch. The model instead made six empty searches using
natural-language issue phrases and five identical reads of the complete3,108-byte
source file. It used no regex operators and acquired no nonempty search evidence;
all tool calls succeeded, with zero failed repeats, no ordinary journal drops and
only two unique nonempty outputs. The remaining observed loop therefore does not
require a tool error, missing source observation, or journal-budget loss in this
episode. This is an agent-policy failure in the tested setup, not evidence that
every coding policy or physical runtime is incapable of solving the task.

After renewed autonomous continuation, a [two-request decoding diagnostic](decoding-pair/report.md)
isolated another remaining factor without modifying a task or running another
official episode. Separate fresh stock-vLLM processes received the same actual
3,003tokens, including the complete3,108-byte source, with identical sampling and
model settings. The first applied the original schema string; the second omitted
request-level structured decoding. Both produced the same120tokens and byte-exact
raw/canonical unhelpful search as the original trajectory. This specific decision
therefore does not require either prefix reuse or JSON grammar. It does not prove
global runtime correctness or rule out every structured-decoding interaction.
The pair consumed61arm-wall seconds and54.857engine-scope seconds, with no tools,
patches or official evaluations. Both outputs parsed directly; no action was
reformatted, repaired or replaced. Compute was released. The diagnostic is closed,
and official quality remains unchanged and unmet.

## Attribution and causal memory

The [offline audit and base comparison](attribution/report.md) verified all
36 separately tokenized input segments and all 11 immediate failed-action
observations. The complete preceding journal was present with no truncation;
the schema allowed inspection. Base Qwen then reproduced the old LoRA action
and observation sequence exactly. This does not support branch-choice LoRA
degradation as the differentiating cause on this case. It does not rule out
every possible harness/model interaction or establish global model incapability.

The [generic runtime](runtime/report.md) makes failed actions causally persistent
outside the truncatable journal. Known-failed exact actions cannot reach the
tool again. After edit mismatch, a successful, nonempty, same-resource read or
search is required before another edit. Every rejected proposal still consumes
one of the 12 slots and its model cost. This achieved zero external failed
resubmissions and relevant evidence reacquisition, but neither a patch nor
official correctness. Rejection is a controller invariant, not learned behavior.

## Frozen tool-recovery training

The [training and unmasked suite](tool-recovery/report.md) used one rank-8 recipe,
256 train / 64 disjoint synthetic cases, and two supervised transitions per
case. Real fixture failures, inspection evidence, and target continuations were
executed; Lighthouse issue/source/patch/test labels and branch-value data were
excluded. All base parameters stayed frozen. No model-output-driven retuning,
masked recovery score, replacement action, or retry was used in the suite.

Both frozen case-level gates passed: no known-failed repeats **64/64** (required
58), successful relevant first recovery **64/64** (required 52). Full synthetic
two-step recovery was **64/64**, including **48/48** corrected-edit fixtures;
there were **0/128** invalid or failed model steps. No base suite was authorized,
so this is a held-out capability result, not a measured training-effect delta.

The single permitted Lighthouse episode inspected successfully but still formed
mismatching exact-edit preconditions from numbered read output. Both edits
failed; seven repeated proposals were blocked; the patch was empty and official
`resolved=false`. Synthetic recovery did not transfer to this task's repair.
The conclusion is bounded to this model, recipe, tool representation and task,
not a universal coding-capacity limit or a demonstrated KV-runtime defect.

## Cost and remaining gate

Recovery training, suite, merge and Lighthouse measured components totaled
559.112 seconds; conservative phase wall time including preserved pre-output
infrastructure failures was 674 seconds, below the 3,600-second ceiling. All
owned GPU/model processes and task containers were released. Reproducible data,
raw trajectories and official logs are preserved; remote adapter/merged-weight
locations and retained non-GPU assets are listed in the phase reports.

Renewed continuation added one raw-read episode (62 seconds total wall), the
exact-input reference (45.765205 seconds including its invalid instrument),
one deliberative episode (120 seconds total wall), and one measured-match
feedback episode (118 seconds total wall). Each stayed within its
separately frozen cap. Their outputs, costs and failures are retained rather
than relabeled as successes. No GPU or task container remains active.

No treatment created concurrent requests or KV evictions. Actual Lighthouse
prefill increased from 1,892 (original base) to 10,350 (runtime) and 13,808
(recovery LoRA). There is no performance claim. Official same-task correctness
is still required before genuine multibranch, and its gates before formal
confirmation. Those downstream stages remain blocked, not passed or completed.

The current quality blocker is now the model policy's failure to convert
available code-test feedback into a valid corrective edit within the frozen
budget. Default worker selection no longer starves developer-test feedback.
The14Breference also exposed premature completion after failed verification;
v3 now makes that failure persist in completion validity. Its offline real-tool
and actual xgrammar checks pass, but its official result remains false. The
observed effective-command alias has a red/green correction and retained-trace
replay, and its live arm now has zero effective failed-command re-executions.
Both predeclared quality arms remain officially false. These are limits of the
tested setups, not a proof of universal model incapability or globalKVcorrectness.
The exact source-fed stock reference also reproduces the incorrect edit; the
remaining correctness problem is not explained solely by that decision's reuse path.
No model-size sweep or closed-recipe recovery retraining was launched.
The original end-to-end objective remains open; no downstream gate is passed.

Post-PR25 full episodes consumed115,106,129,131,96,102,91,98,96,129,101,129,114,100 and112wrapper seconds
respectively, each below its1200-second cap. The same-size model asset transfer
used2027.608315active job seconds including preserved failures and proxy
resumption, within its explicitly amended transfer-only budget. Its GPU episode
count and budget were not increased. All completed jobs released compute.

The separate14Basset acquisition verified13files/29,551,669,960bytes in
2409.619transfer seconds, below its31GB/5400-second bounds, with no GPU use.

Both search-contract episodes are closed. Their tool behavior has real-container
CPU replays and targeted regressions; their official outcomes remain false. No
additional model request, training recipe, multibranch pilot or formal seed is
queued by this checkpoint. A further same-task experiment needs a concrete
agent-policy hypothesis rather than another automatic model/budget/wording retry.
The final regex episode lacks an active utilization sample; native execution and
terminal resource release are recorded, without inventing a utilization figure.

The [task-state intervention](task-state/report.md) then wired model-produced
plans and exact observation citations into the existing reducer and selected
capsules, with no additional calls or larger output/action budget. Its122-second
episode remained officially false:12plans,0citations,0commits,0patch, and nine
identical successful reads. CPU fixtures prove the extractive path, but the
model did not exercise it. The evidence therefore does not test the efficacy
of populated semantic state. It exposes a remaining dependency: tool evidence
only becomes a capsule fact if the model voluntarily cites it.

[Runtime-owned observation persistence](observation-state/report.md) now removes
that dependency in code: every actual result is committed, and the latest
complete observation is selected for the next capsule. Actual-trace CPU replay
proves11/11prior outputs survive exactly despite empty model citations, with
unchanged model root/schema. Its single108-second live episode confirmed the
same dataflow:12actualruntimecommits,11/11exactprioroutputs in followingcapsules,
allactualcapsuletokenroundtrips. It still made12successfultools/3distinctactions,
9successfulinspectionrepeats,0edits/tests/patch, and official`resolved=false`.
The startup-admission failure was recovered only after proving no original job
or receiver remained; this did not add a model episode. Missing observation
transport/persistence alone does not explain the surviving loop. Further useful
work needs a materially justified agent-policy intervention, not another
automatic schema/wording/model/budget sweep or a task substitution. No new run
is queued, and no downstream quality/performance/formal gate has passed.

After renewed continuation, [evidence-only context](evidence-only-context/report.md)
removed prior model self-history while retaining all external observations; its
single official result remained false/zero patch. The [pinned native tool-call
adapter](native-tool-protocol/report.md) then used the checkpoint's actual tool
template rather than the custom plan/action envelope. Its one episode also
failed quality: a guessed absent replacement failed, and exact re-executions
were prevented. All actual feedback records remained exact in subsequent inputs.

The [source-verified software-agent reference](agent-policy-reference/report.md)
used an existing same-size trained checkpoint without new training. After the
original capacity/transfer blocks, private-LAN recovery verified the full15.24GB
payload in579.340491seconds. The one50-proposal episode completed with official
`resolved=false`, one failed physical view,49controller rejections and an empty
patch. The first12proposals are reported separately, without an equal-budget
claim. This is a valid negative under the limited tool adapter, not an
infrastructure-only result or an unconsumed model episode.

The actual trace exposed43escape-parser rejections of a read-only find/grep
composition. The generic adapter now handles that composition with existing
search bounds and no shell execution; path recovery names the public tools,
and filename discovery cannot substitute for content inspection. Its separate
50-proposal command-repair validation still returned official `resolved=false`
and no patch: five physical views, two successful root listings and37 final
blocked repetitions of a missing path. No find/grep call occurred, so the live
run did not exercise the repaired parser composition. The prospective input
record omitted a36-token empty compiler capsule; the original mismatch remains
preserved alongside an exact, explicitly post-hoc CPU reconstruction.

Actual root evidence and recovery guidance were present in subsequent inputs.
The remaining directory-view interface was then checked against the authors'
pinned editor: it rejects file ranges on directories and lists non-hidden
absolute paths to two levels, unlike our partial dotfile-heavy listings. That
generic correction passed145affected CPU cases, but its separate50-proposal
episode again returned official false/zero patch, with no root discovery or
successful tool. The fresh1493-token initial input matched prospectively exactly
and capsule/feedback/bounds checks passed; the corrected directory producer was
never successfully exercised.

The source-verified producer task input supplies a real uploaded workspace path,
not our custom logical repository field that seeded the repeated missing path.
The actor now receives the actual configured root and verbatim issue in that
input shape, retaining external task identity and all other decision formats.
Its23focused CPU cases passed. The separate workspace-input episode then
successfully discovered the real root, completed16successful tools and made a
legitimate621-byte product diff. Its1481-token input matched prospectively
exactly, but official quality remained false. After one missing path and four
root-only recovery rejections, it emitted the same prose-only response27times,
without developer testing or finish. No output-length cutoff explains that
collapse and no second episode was run.

The recovery gate was over-restrictive: even a view of the already-edited known
file was blocked by an unrelated missing path. V4 now allows successful bounded
inspection of existing repository resources to recover, while preserving active
failure certificates, content refresh, verification and legacy root-only modes.
The observed compaction of two distinct malformed pipeline bodies is separately
corrected by retaining their raw identity; no causal compaction claim is made.
The136-case affected gate plus the expanded2-case XML edit/missing-path/retest
loop passed. Its separate50-proposal episode still returned official false with
an empty diff: nine physical tools all succeeded, but the model reverted its
edit and emitted41prose-only responses. No missing path occurred, so the recovery
branch was unexercised; actual invalid-proposal identity handling passed.

A single stock fresh-vLLM reference on the prospectively selected dominant prose
input reproduced the exact same raw response without private checkpoint reuse.
This is narrow runtime attribution, not quality. The next explicit intervention,
`openhands_json_v1`, uses the existing schema decoder to require one same-function
JSON object, with optional reasoning inside it and unchanged semantic choices,
execution mapping, workspace, model and budgets. No output repair is used.
An installed xgrammar0.2.3 escaped-string incompatibility with `minLength` was
found by real CPU matcher validation; those failed records remain. The grammar
now checks syntax/types while the shared mapper still rejects empty command,
path and old_str values before execution. Local affected gates pass. Corrected
native CPU validation on `.134` passed all six positives and five negatives,
including escaped strings; six syntax-valid empty/whitespace cases and a pipeline
were separately rejected semantically before execution. The requested endpoint
has a matched existing runtime, idle48GB devices and the full verified15.243GB
actor/tokenizer payload after a bounded152-second private transfer. Its one
separately prospective JSON episode then produced50complete function objects
without prose-only or truncated output, but still returned official false with
an empty diff. It made45successful ranged views of one source file and no edits,
developer tests or finish. Exact1546-token initial-input equality and limits
passed; engine180.204s, official grading9.391s, cleanup complete. The active GPU
sample missed allocation and optional CPU schema replay timed out; both limits
remain explicit. JSON syntax success is not repair or a new KV attribution.
All six actor quality episodes are closed negative; no unchanged retry or
downstream run was started. Follow-up on captured inputs found correct feedback:
five source windows, then40successful empty beyond-EOF views against91lines.
A newly checked pinned producer contract rejects explicit out-of-file start/end
values instead. That real mismatch is now corrected with typed extent feedback
and file-revision-scoped failure validity; no action is forced and ordinary
bounded/default/shell reads are unchanged. The155affected cases pass, including
the actual range failure and same-hash recovery after file growth. This source
comparison supersedes only the initial trace-only no-mismatch conclusion, not
its immutable evidence. The corrected-range episode actually exercised the
typed file-scoped failure, and successful empty/beyond-EOF views fell to zero.
It nevertheless returned official false with no patch:36successful nonempty
views,27identical final-range repetitions, no edit/test/finish. All50outputs were
complete JSON and the1580-token prospective input matched. Engine111.831s,
grading9.325s, cleanup complete. All seven actor episodes are now closed negative.
The one prospectively fixed stock reference then copied all6917actual IDs from
terminal request49 and reproduced the same raw JSON and canonical view exactly
without private native reuse or APC. Schema identity is source/config-derived,
not directly captured; output token-ID equality is not claimed. It used28.143
engine seconds with an89%in-engine GPU sample; exact cleanup completed. This
isolates only that decision at that input, not a global engine/model verdict.
The next bounded discrimination, `openhands_xml_grammar_v1`, preserved the original
XML instructions while requiring one complete call through direct grammar decoding.
Earlier XML edits and JSON-only inspections motivate this distinction; they do not
establish a causal format effect. It changes no model, semantic choices, task or
budgets and adds no prefill, reasoning bound or output repair. Actual `.134`
CPU acceptance passed62positive/16negative syntax cases, seven semantic errors,
12tokenizer-mask cases and20native sampling/inline-intent regressions, without
loading weights or issuing model requests. The [separate XML grammar episode](agent-policy-reference/xml-grammar-quality/terminal-report.md)
then returned official false and no product diff: nine successful calls (seven
views, one search, one find), followed by41length-capped outputs without complete
calls. No edit/test/finish occurred. The exact1541-token prospective input matched;
engine655.868s, grading9.823s, limits and owned cleanup passed. The selected GPU
sample was64%instantaneous utilization at43,552MiB, not an average claim. Raw
outputs and earlier budgets remain immutable. The grammar's valid completed
syntax does not guarantee completing a call before the unchanged token cap.
All eight actor episodes are now closed negative. Long model reasoning without
executable calls remains a real quality blocker, with no new executor defect or
approved useful retry established. Neither multibranch nor formal may start;
the user's full objective has not been completed.

The subsequent direct trace found a concrete benchmark feedback omission: native
Brain/branch results already retain the actual finish reason and output IDs, but
the solver discarded them. Actor traces and malformed-call feedback now preserve
that reason, its measured output-token count and the unchanged token limit.
Original parser errors remain, and valid complete calls at the cap still execute.
The29affected cases pass, including12regressions that failed before the correction
and exact next-input visibility on first/continuation paths. No prompt, grammar,
reasoning bound, semantic choice or model budget changes. This factual feedback
correction is ready for a separate prospective episode; it is not official repair.

That [generation-feedback episode](agent-policy-reference/generation-feedback-quality/terminal-report.md)
completed with47complete XML calls, three malformed length-capped calls and a
legitimate5,440-byte model diff from four successful edits. Actual generation
facts reached all three next compiled inputs. Returned output-ID counts512and
backend decoded counts511are distinguished rather than labeled an engine bug.
All32physical tools succeeded, but no developer test occurred and15finish calls
were rejected for unverified changes. Official `resolved=false` still closes
this episode negative; all nine actor episodes remain unresolved. Exact1541-token
prospective input equality, unchanged grammar/model/prompt/runtime, limits and
cleanup passed; engine358.881s, grading10.407s, in-engine65%instantaneous utilization.
The corrected no-inference attribution confirms that the actual initial prompt
explicitly requires testing after changes, all15verification failures reached
their exact next-input capsule, and the documented test route remained available.
An initial instruction-asymmetry claim overlooked the imported XML constant and
is retained but superseded; it triggered no source edit or model replay. No
correction target remains at this observed boundary: the model ignored correct
verification facts and its patch failed the original official test.
Nonempty diff alone does not permit multibranch or formal; the full goal stays open.

The next opt-in XML state-grammar treatment moves the already existing runtime
verification restriction into decoding, matching the non-actor `action_schema`
prior art: exclude only currently illegal finish, then restore it after a
successful test. All other function choices remain; no test/edit/novelty is
forced. Original prompts, budgets and static comparators stay unchanged. All72
affected first/fork/retest/reference cases pass. Actual closed-state CPU acceptance
passed51non-finish positives, seven semantic errors,16syntax negatives,12positive
tokenizer cases and11blocked finish cases in both text and real tokenizer masks,
without CUDA or model requests. Failed worker admission/SSH setup and direct
bounded recovery are explicit. A separate prospective episode is ready. The prior
no-feedback-loss attribution and all nine negative episodes remain valid.

The [verification-mask episode](agent-policy-reference/verification-mask-quality/terminal-report.md)
also returned official false:11successful complete calls including two edits
and a legitimate3,494-byte diff, but39malformed length-capped outputs and no
developer test or finish. Actual verification state selected the closed grammar
for38requests after the first edit; no finish was emitted while masked, and
reopening was unexercised. Exact1541-token initial input, generation-fact feedback,
limits and owned cleanup passed; engine937.454s, grading10.941s, one100%instantaneous
in-engine sample. Earlier budgets and all ten negative actor episodes remain
immutable. Legality masking did not establish repair and justifies no automatic
masking/budget sweep.

Source68696f6 implements the pinned producer's distinct ordinary-message control
path as opt-in `message_policy=codeact_continue_v1` on unconstrained XML only.
It preserves raw naturally stopped text and sends the fixed generic base callback,
without tool submission/success, evidence/capsule refresh, failure recovery or
verification changes. Capped/partial/malformed output remains an explicit error.
Initial instructions,50/512limits and other comparators are unchanged;187affected
local cases pass. No quality effect is inferred from those interface regressions.

The [message-control episode](agent-policy-reference/message-control-quality/terminal-report.md)
returned officialfalse and0-byte diff. It emitted48complete view calls (47successful,
onefailed) and two malformed length-capped calls; no edit/test/finish or naturally
stopped ordinary message occurred. Therefore the callback was enabled but
unexercised, so this result cannot estimate its behavioral or quality effect.
Both capped errors retained actual generation facts in exact next inputs. The
fresh actual1541-token initial input matched the prospective record including its
36-token capsule. Engine244.121s and grading10.615s were bounded;955.879s remains
in this separate1200sphase ledger. One active sample was44,440MiB/100%/302.61W;
owned processes/containers are gone and GPU0 returned to0MiB/0%. All eleven
closed episodes remain negative. Original official quality still gates genuine
multibranch and Formal; the full objective remains open.

A narrower read-only check exposed selected-versus-emitted view metadata: the
6000-character output ended partway through line22, while the fields reported
47returned lines and no later lines. The truncation flag itself was correct.
View extent is now finalized after clipping, with emitted-line counts and an
explicit partial-last-line flag. Output bytes, limits, initial instructions,
range validity and sampling remain unchanged. The real JSON-worker regression
reproduced the discrepancy before the fix;165affected cases pass.

The [emitted-extent episode](agent-policy-reference/emitted-extent-quality/terminal-report.md)
exercised the repair: action3 emitted21complete lines plus partial line22 and
reported22returned, remaining lines, partial-line and truncation flags in the
exact next capsule. It still returned officialfalse and0diff. Only four views
ran; proposals4-49 were all length-capped incomplete edits. Forty-five were
byte-identical: each opened a `str_replace` for `react.js`, put the large encoded
SVG in `old_str`, then hit the512-token cap before closing `old_str`; none had
`new_str` or a closing call. These cannot become executable by appending tags.
The parser correctly rejected them, raw text was preserved, and exact error plus
length/512facts reached every available next input. No new executor, parser,
feedback or state-loss defect remains on this trace.

The actual1541-token initial input matched the prospective record. Engine
680.466s and grading11.332s stayed within caps, leaving519.534s in this separate
1200sledger. The active sample was44,442MiB/61%/265.15W; exact owned runner,
relay/GPU process and evaluator containers are absent, and GPU0 returned to
0MiB/0%. All twelve episodes are closed and officialfalse. Continuing generation,
raising output budget, changing prompt/model/seed or forcing another action
would change the frozen protocol, not repair observed infrastructure. Therefore
the original quality gate is blocked and multibranch/Formal remain prohibited.

One narrower visible-source diagnosis reopens an actionable interface boundary
before that correct rejection. The actor proposed adding a temporary React pack
`id` after receiving only the first22lines; the same base file already contains
`id: 'react'` later. Because a huge SVG line was clipped, the observation had no
exact valid continuation point, and the actor instead copied the partial SVG
into an edit that could not fit512tokens.

Character-clipped file views now include an exact `next_view_range`, overlapping
a partial line or starting after a complete boundary. It reuses existing view
semantics and changes no source output, limits, prompt, request count, validity
or sampling; directories omit the file-only range. The worker boundary and exact
OpenHands next suffix are covered by151focused passing cases. This creates one
new prospective treatment under the original quality authority. Until it passes,
the official quality gate remains unsatisfied and downstream stages stay closed.

The continuation episode used the exact range to read the unseen26lines and the
existing React id, but still returned officialfalse/0diff. It made nine views,
two correctly handled normal messages and39malformed outputs, with no edit/test/
finish. Successful index evidence and all root-level discovery anchors remained
in the actual compiled prompts, while exact errors/callbacks/generation facts
survived. This rules out new continuation, journal, message-control and root
discovery defects. Owned compute and evaluator resources were cleaned.

The pinned producer does provide one additional generic file-clipping behavior:
ACI0.1.0 appends a fixed notice instructing the agent to use `grep -n` before
retrying a partial file view. The new opt-in policy renders that exact notice
after unchanged raw source and records its source identity. It applies only to
character-clipped files; directories, unclipped observations and all default
comparators remain unchanged. Source bytes,6,000-character limit, next range,
root prompt, action set, request count, sampling and budgets do not change.
All188affected local cases pass. This exact upstream treatment is ready for one
prospective quality episode; multibranch/Formal stay gated meanwhile.

Its first remote launch is infrastructure-invalid: after four requests, the
inline product-container worker tried to import a repository sibling and failed
before returning the first clipped file observation. It produced no official
grade and is not a treatment negative. Cleanup returned GPU0 to0MiB/0% and
removed task-owned processes/containers. Notice constants now reside in the
self-contained inline worker, with host protocol re-export only. The actual
`python -c` isolated-worker regression is red before and green after the fix;
all188affected tests pass. The identical scientific treatment must be rerun.
