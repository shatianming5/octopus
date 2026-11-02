# Software-agent policy reference and state-scoped recovery

**The actor-owned reproduction tools and completion gate are implemented, but
their one authorized run failed both mechanism and original official quality.** The
fixed actor completed 100 requests without defining or executing a reproducer.
It changed a WordPress pack ID before any observed red, then produced 90 normal
messages. Its 429-byte model-authored product diff is nonempty but ineffective;
the unchanged original evaluator returned `resolved=false`. No multibranch or
Formal run was started. See the [current terminal report](actor-reproduction-quality/terminal-report.md)
and [separate mechanism/quality metrics](actor-reproduction-quality/metrics.json).

The [repository-grounded first-principles diagnosis](actor-reproduction-quality/first-principles-diagnosis.md)
now separates the initial semantic mislocalization from controller progression
and later history loss. The exact patch also breaks WordPress pack lookup.
All 100 actual inputs contain the new workflow instructions; the 85-message
identical terminal suffix performs no tools or mutations. CPU replay establishes
that early edits are allowed but a new reproducer then requires restoring the
original product. The active path uses stock vLLM without Octopus fork/KV control,
and its actor already has released SWE-Gym trajectory SFT. No new experimental
model run, training or production-policy change was made for this diagnosis.

The opt-in `actor_reproduction_v1` preserves the fixed actor, original issue,
base, 4096-output/32768-context/100-action profile and default comparison paths.
It adds confined actor-authored `reproduce(hypothesis, code)` and an immutable
`rerun_reproduction()` gate. Real Docker/stdio fixtures establish original
assertion failure, same-assertion passing after an effective product change,
mutation invalidation, timeout/error handling and artifact separation. The
[implementation validation](actor-reproduction-quality/implementation-validation.json)
does not establish real-model uptake or semantic relevance. No manual Lighthouse
fixture, correct-location hint or hidden evaluator material was given to the actor.

The new distinct phase used 1080.907 of 1200 seconds, with an active GPU sample
of 91% utilization; all owned jobs and containers were closed. The preceding
expanded phase's 679.251-used/520.749-remaining ledger is unchanged. The earlier
14 bounded episodes, matched continuation and expanded reference also remain
unresolved. The current blocker is failure of this fixed actor to select and
use relevant execution evidence under the new workflow, not missing tool
availability or a successful causal repair. See [source research](research.md)
and the original [quality protocol](quality-protocol.json).

## First-principles diagnosis of the preceding expanded reference

This earlier diagnosis reuses the completed expanded reference and its visible-source
semantic check; it runs no new model or evaluator and changes no solver policy.
The [machine-readable witness](expanded-actor-quality-recovery/first-principles-diagnosis.json)
binds the analysis to the archived episode and actual patch.

### Separate the requested behavior from its proxies

The task requires a change to observable program behavior, not a larger patch,
a successfully executed editing tool, a confident completion message or an
accepted finish action. The necessary causal path is from a public issue
requirement through the relevant runtime data flow to an observable output.

| Public requirement | Actor's substitute premise | Actual evidence |
| --- | --- | --- |
| One displayed pack despite duplicate detections | Each static pack file has a unique ID | One React definition can still be selected twice by two runtime detections |
| Experimental Fast Path entries not shown to the user | Add an object named nonUserVisibleDescriptions to the advice file | The object has no consumers; the audit output does not read it |
| Demonstrate a repair | The edit tool returned success | No developer test or reproducer was run; the public-issue fixture is unchanged |

The wrong premise is explicit in action3 reasoning, before the only edit at
action6. The patch adds an unused constant, leaving both observed behaviors
unchanged. This identifies where this trajectory first fails; it is stronger
than labeling the terminal score a generic model-capacity or formatting failure.
The public-issue fixture was created for diagnosis only, never given to the actor.

### The finish loop is a second failure, not the cause of the wrong edit

Reanalysis of actions10-99 finds one unique action, one raw output, one returned
observation and one before-state across all90proposals. There are zero physical
tool submissions and no workspace changes. All90actual prompt token sequences
are distinct: history grows, but the environment contributes no new program
behavior evidence. This is an observed no-progress cycle, not proof of cached
output replay or a mathematical fixed point of every possible future prompt.

`FailureMemory.rejection_reason` enforces the safety condition that a changed
workspace cannot finish before a successful test. The expanded solve loop appends
the rejection and asks the same fixed policy again until the action ceiling.
There is no separate progress policy in that path. A repeated "test required"
message carries the same information for a correct untested patch and an incorrect
untested patch: it does not identify a failed behavioral hypothesis. Its exact
delivery is verified, but delivery is not evidence that the actor uses it.

Thus the validity rule works while the investigation/verification policy fails.
Allowing the first finish would submit the same wrong patch. Stopping the later
retries could save work but cannot create a correct repair. Likewise, passing an
unrelated developer test would not establish the public requirements or replace
the unchanged official quality gate. No such controller change is made here.

### The workflow hypothesis tested after this diagnosis

The newly available4096tokens were never exhausted (actual maximum372). Full
history and executable tools made investigation possible, not mandatory or
correct. Fixed-weight inference is not learning from repeated failures, and
more room in the context window is not additional evidence selected by the actor.
The bottleneck on this trace is hypothesis formation and revision through actual
execution, compounded by a controller that only rejects invalid completion.
The broader research plan still lacks a working task-repair baseline; KV savings
and multibranch mechanisms cannot substitute for that prerequisite.

The user subsequently authorized the generic evidence-linked workflow:
actor-owned hypothesis and minimal reproduction, actual observed failure,
causally connected change, and rerun of the same assertions. It was implemented
and tested once as the distinct intervention reported above, without changing
the actor or budget profile or injecting the diagnostic fixture, solution paths,
a hand-written patch or hidden tests. Relevance to the public issue remains a
separate judgment, not merely a successful test invocation. The actual actor
never called either reproduction tool, so this run establishes no successful
causal repair. It does not prove intrinsic impossibility or guaranteed
solvability under every other policy.

## Deep failure audit: what the repeated negatives actually establish

The requested audit stops interface changes and new inference. Its denominator
is the 14 completed, officially graded episodes of this fixed SWE-Gym actor,
not earlier base/coder/LoRA experiments, infrastructure-invalid launches, or
single-request stock references. The [raw per-episode table](fixed-actor-failure-table.csv),
[aggregate counts](fixed-actor-failure-summary.json), and
[path/edit evidence](fixed-actor-failure-funnel.json) are computed from retained
actions, request metrics, actual patches and official grade files.

| Measured stage | Result |
| --- | ---: |
| Completed episodes / model requests | 14 / 700 |
| Official resolved=true | 0 |
| Physical tools / successful tools | 243 / 227 |
| Successful exact edits, including a self-revert pair | 9 |
| Episodes with a nonempty final patch | 3 |
| Episodes with a zero-byte final patch | 11 |
| Developer test proposals / executions | 0 / 0 |
| Successful direct reads of `audits/dobetterweb/js-libraries.js` | 0 |
| Length-capped requests | 169 |
| Proposals categorized `invalid_decision` | 352 |

### The repair process fails before official evaluation

The common failure is not an almost-correct patch rejected at the last step.
The actor does not consistently localize the issue, edit the relevant behavior,
or validate an edit. The closest episode, `workspace-input-repair`, read the
collector and matcher and changed the collector's deduplication in a 621-byte
patch. It did not address the user-visible audit or test the change. The other
two nonempty patches, 5,440 and 3,494 bytes, changed React advice strings into
`{id, description}` objects in `stack-packs/packs/react.js`; they did not change
the audit's `artifacts.Stacks` projection. Their size is not evidence of repair.
The visible base audit at lines 44-58 maps every JavaScript stack to table items,
and its base developer tests directly supply `Stacks`. Neither a pack-string
rewrite nor merely changing collection is evidence that this separate display
behavior is fixed. No gold patch or hidden test changes were needed for this check.

### The current tool scaffold blocks familiar commands

The [latest actual run](truncation-notice-quality-recovery/terminal-report.md)
is a strong counterexample to attributing everything to the 512-token cap.
It has zero length caps, but 40 normally stopped, complete and unfenced XML
`execute_bash` calls fail local command validation: 38 use `ls -R`, two use
`find ... -not -path`. These forms are not advertised in the bounded adapter.
The error is truthful and reaches all 39 available next inputs, but an actor
expecting shell behavior repeatedly fails to adapt to the custom subset.
Thirty-six consecutive raw outputs repeat the same unsupported recursive list.
`invalid_function_call` here denotes command semantics, not malformed XML.
Treating all such outcomes as formatting failures hid an important distinction.

### What has and has not been ruled out

Exact context checks retain the full successful index observation after the
older range error. Root discovery exposes all top-level entries, including
`lighthouse-core` and its test tree. Actual message callbacks preserve state,
and completed calls execute and make persistent edits. Selected fresh stock
vLLM requests reproduce both an invalid prose output and a repeated JSON view
byte-for-byte without private checkpoint adoption. These observations argue
against feedback loss or native KV reuse as the sole explanation at those
points; they are not proof that every possible runtime defect is absent.
They also do not prove the actor is intrinsically incapable under its original
scaffold. No same-actor, producer-faithful full-episode positive control has
established competence before interpreting the custom runtime experiments.

### Why the elapsed work did not become progress

Twelve of fourteen episodes contain at least 20 consecutive identical raw
outputs. Counting only the extra copies within the longest run per episode
already gives 406 of 700 proposals, a conservative recurrence count. These
greedy, differently scaffolded runs are not independent pass-at-k trials; the
actor does not learn from earlier attempts. Total engine scope is 5,063.249s
(84.4 minutes), grading 139.485s, while the first-to-last freeze interval is
11.16 hours. The rest is not continuous model training: implementation,
deployment, serial orchestration, waiting and documented infrastructure recovery
contribute wall time. Earlier coordinator decisions over-weighted interface
regressions and diff size, repeatedly promoting small fixes to full episodes
without showing a better localization/edit/test funnel. That experiment-design
failure is separate from, and compounds, the actor/scaffold mismatch.

### Independent producer comparison

The [source-verified audit](producer-interface-audit.md) traces the released
`execute_bash` through `CmdRunAction` to an actual persistent Bash session,
not a whitelist of simulated command forms. It also verifies several material
differences rather than treating the tool name/XML format as compatibility:

| Surface | Current fixed-actor episodes | Verified producer evidence |
| --- | --- | --- |
| Bash | Parsed restricted adapters | Actual shell, ordinary `ls -R` and `find` predicates |
| Response budget | 512 tokens | 4096-token fallback; exact archived 7B run cap unverified |
| Context | 16384, with 8192 journal cap | Documented 32768 context |
| Episode ceiling | 50 proposals | Dedicated test scripts specify 100 interactions |
| File view | Raw source clipped at 6000 chars | Numbered editor output with clipping; raw shell reads also exist |
| History | Prefix capsule and deduplicated recent journal | Event history reconstruction and overflow-history recovery |

The paper's fine-tuned 7B OpenHands policy achieves 10.0% on Lite (300 tasks)
and 10.6% on Verified (500), with respective reported loop rates of 31.0% and
21.0%. These are paper-level results, not a verified mapping from every byte of
the local snapshot to an archived run, and not a per-Lighthouse success estimate.
The model was trained on 491 successful software trajectories from Python tasks;
our task is a JavaScript/multimodal issue. Limited robustness and domain transfer
are plausible contributors, not proof that this fixed actor cannot solve it.
The larger verifier-selected headline score is not the 7B single-attempt score.

The 512-token restriction genuinely truncates some attempts, but it cannot
explain the latest 40 complete shell calls that failed without a length cap.
Conversely, expanding only `ls -R` would not prove that localization, correct
editing or testing will follow. The published paper/fork differ in some version
and history details; the full audit preserves those provenance uncertainties.

### Decision, not another incremental treatment

The present result is actor/scaffold interaction failure with weak localization
and recovery. It is neither evidence of intrinsic impossibility nor a reason
to lower official quality requirements. Do not infer a new adapter patch from
every failed trajectory or run more unchanged greedy episodes. One separately
authorized matched counterfactual at the first complete rejected shell command,
with actor/input/budget held fixed and producer Bash semantics in a disposable
isolated task environment, would directly test the command restriction's causal
contribution. Leaving the loop is a mechanism outcome, not `resolved=true`.
No such new experiment or solver change was made during this audit; original
quality remains unmet and multibranch/Formal remain downstream.

### Matched continuation implementation and real CPU gate

The next scoped checkpoint now implements that read-only command-space control
without another prompt-format treatment. It selects original action7, restores
only the seven mutation-free read-only prefix calls and requires their actual
observations/controller commits to match. A fresh C03 CPU check through the
existing remote-tokenizer/local-Docker relay compiled all 7,079 initial prompt
IDs exactly equal to captured context7. It made zero model requests and left
43of the original50proposal slots available; historical physical KV is not
claimed resident. Full [receipts](matched-command-control/implementation-validation.json)
and [actual CPU result](matched-command-control/cpu-context-check.json) are retained.

Actual Docker checks execute the captured `find ... -not -path ...` unchanged.
An isolation defect was caught before inference: root-only `.git` left inside
the tree made that command fail with permission denied. Private controller Git
metadata now lives outside the visible tree; native find exits0 and returns its
real dependency match, while Git history remains inaccessible. Existing root
editor/Jest/diff routes still work. Landlock, uid65534, no network and bounded
process lifetime keep the new Bash route read-only; no fabricated verification
or source claims arise from shell output. The CPU relay took18.278s, and its
exact container and temporary source deployment are gone. One prospective
matched continuation is ready; execution and official quality are still pending.

The [subsequent actual continuation](matched-command-quality/terminal-report.md)
ran to completion:43new requests after the seven replayed slots, all43physical
tools successful, no caps/format errors, and no edit/test/finish. Official quality
was false and the diff empty. The actor spent40views reading successive windows
of `proto/sample_v2_round_trip.json`; even unique valid actions did not repair
the task. Importantly, its first raw output and command differed despite all7079
initial IDs matching, before the experimental tool could provide feedback.
Fresh physical state/numerics remain uncontrolled, so this is not a clean causal
success for the Bash intervention and does not prove a native sampling/KV bug.
Engine348.983s/grade9.596s were bounded, and exact owned resources are closed.

The user has now explicitly authorized a higher token budget and one coherent
repair of the identified framework differences before further experiments.
The implementation target is a4096-output/32768-context/100-turn actor reference,
task-scoped native tools, complete paired history until the actual context limit,
producer-style file feedback, and stockvLLM without private KV/capsule projection.
It is a bundled reference configuration, not another isolated causality claim.
The actor, original task, official quality gate and bounded GPU time remain fixed.

The coherent opt-in `expanded_actor_reference_v1` implementation is now ready.
It freezes4096output/32768context/100turns, removes the8Kjournal/capsule/dedup
bottleneck, retains whole assistant/result pairs until the actual context limit,
and uses stockvLLM without prefix caching or private checkpoint adoption. Native
Bash and actor-controlled tests are confined to the writable task/scratch space;
trusted snapshots track actual shell content/type/mode changes, including failed
and timed-out writes, for V4 verification/retest and complete product diff capture.
Effective editor-view metadata declares numbered whole-file/range output at16K
characters, while shell cat remains raw. Parser errors retain real generation
facts and ordinary messages retain the fixed callback's provenance.

Actual Docker gates cover quoted commands, pipelines, workspace writes, ownership,
mode changes, native reproducers, confined Jest failure/retest, history/outside/
network denial, mutation tracking and cleanup. Existing bounded-protocol consumer
gates remain valid. The real C03/stdin-relay CPU check produced1294initial tokens
and reserved4096of32768, with zero model requests or CUDA initialization; actual
installed vLLM0.28.0 constructors/signatures and4096SamplingParams were checked
without creating an engine. [Receipts](expanded-reference-control/implementation-validation.json)
preserve the exact evidence and remaining engine-effective runtime gate.
No larger actor, extra planner, old patch/output seed, missing-tag repair or
unbounded compute was introduced. It remains a producer-oriented bounded
reference, not byte-identical original OpenHands: Bash is per-call, history is
masked, insert/undo are absent, and official quality is still unproven.

The [first expanded-profile launch](expanded-actor-quality/terminal-report.md)
failed before its first model request: the external5-second `nvidia-smi` UUID
probe timed out after engine loading. This is an infrastructure-invalid attempt
with no official score, not another model negative. Exact resources are closed.
The device gate now uses the same in-process CUDA property lookup as the existing
runtime, preserving exact frozen UUID comparison and mismatch cleanup. Five
affected regressions pass, including the original timeout reproduced before the
fix. The whole103.415s relay is conservatively charged, leaving1096.585s of the
same1200sphase. Recovery will retain4096/32768/100 and all actor/task conditions,
with operational deadlines reduced only to respect that remaining budget;
there is no token/prompt/model sweep or reset of consumed time.

The [valid expanded recovery](expanded-actor-quality-recovery/terminal-report.md)
completed100requests with the exact prospective1294-token first input. All
outputs stopped naturally; maximum output372, zero calls above512, and zero
length caps despite the4096allowance. Input grew to28592with4096reserved and
only one oldest pair dropped. The model made seven views and one edit, then
two messages claiming repair and90identical finish calls. It proposed no tests.
The488-byte diff only declares an unused `nonUserVisibleDescriptions` object.
It changes neither duplicate runtime matching nor the Fast Path audit display.

All89available next inputs contain the exact verification rejection, and the
original actual input includes the test-before-finish requirement. A separate
CPU fixture derived solely from the public issue confirms before/after equality:
duplicate React packs remain and `React (Fast Path)` stays visible. The fixture
was not model input. More available tokens and valid tool access did not create
correct task localization or test execution in this trajectory; this is not a
claim that the fixed actor is mathematically incapable under every policy.
The valid575.835s relay plus invalid103.415s uses679.251of the same1200sphase;
remaining520.749s is not authorization for another outcome-driven retry. Original
official quality remains unmet, so multibranch and Formal remain downstream.

## Verified assets

The [acquisition](acquisition/terminal-report.md) verified five inference files,
15,231,272,470 bytes, all official Git/LFS digests,339tensors and7,615,616,512BF16
parameters in3132.877seconds, within16GB/5400seconds. No inference or training
ran during acquisition. The base Qwen2.5-Coder tokenizer is selected explicitly
and independently; the released weight repository ships neither tokenizer nor
index. Model and tokenizer revisions remain distinct in runtime metadata.

## Matched format, bounded tools

`openhands_xml_v1` follows the released text-function surface with optional
reasoning before one complete call. It uses the base tokenizer's ChatML, not
Qwen's JSON tool-call body or an OpenHands CLI/service. Tool/result dialogue
uses the trained `EXECUTION RESULT of [...]` form. No closing tag or fence is
silently repaired; invalid calls are counted controller rejections with raw
output retained. The supported command subset is parsed into bounded read-only
operations and `yarn jest`, never evaluated by a shell. Editing still uses the
existing exact replacement/new-file operations. This is not a reproduction of
the original unrestricted scaffold's published scores.

## Causal failure validity

V4 fixes a real recovery dead end: a failed test must be runnable after a real
source correction. Certificates are scoped to workspace, affected file or
permanent invalid input, while exact action hashes remain unchanged. No-op edits
and inspections do not advance state. Raw post-mutation retries are reported
separately from forbidden retries against an unchanged state; no salted-hash
claim of zero raw retries is made. The audit retains history, while the prompt
contains current active constraints. The complete generic XML repair fixture
verifies same-hash test failure, real edit, successful retest and a real diff.

The explicit50-action ceiling is prospective;512generated tokens/action and
the original compute/tool/evaluator caps remain. The first12proposals will be
reported as a prefix of this50-budget episode, not an equal-budget comparison.

The original [capacity result](quality/terminal-report.md) did not consume a
model episode and is not a quality failure. No occupied GPU was taken over or
allocation silently reduced. The [operational amendment](endpoint-migration-protocol.json)
allows one known equivalent host, an existing compatible environment and a
bounded replica of already verified weights. Original blocked artifacts remain
preserved; model, task, precision and scientific limits are unchanged.

The alternate transfer ended with2,010,067,550actor bytes and no tokenizer bytes
of15,242,760,997required; whole-payload verification was impossible. Both relay
and direct-SSH repair attempts stayed under the original deadline and stopped
their exact processes. The original verified source assets remain intact. The
[user-requested source-host recheck](quality/source-host-recheck.json) also found
no exclusive compatible GPU. These historical blocked attempts did not consume
a model episode; neither a transfer failure nor occupied capacity is a
model-quality result.

The later [private-LAN recovery and actual quality result](quality/alternate-host/quality-terminal-report.md)
installed and verified15,242,760,997bytes in579.340491seconds. The one authorized
episode then consumed50native requests and2,429decoded tokens: one failed
physical view,49controller rejections, zero successful inspections, edits or
tests, and a0-byte patch. The official result is `resolved=false`. The initial
zero-output FlashInfer startup fault was infrastructure-only; the completed
episode is a valid negative under the limited adapter, not an unconsumed run.

## Command-interface repair, not a new quality result

Of the rejected proposals,43 hit `No escaped character` while requesting the
read-only `find PATH -type f -name GLOB -exec grep -l PATTERN {} \;` composition.
The previous non-POSIX split separated the escape from its semicolon before a
second decode; quoted semicolons also exposed that the composition itself was
unsupported. Raw calls and the original negative result remain unchanged.

The repaired adapter preserves quoted/escaped words and operator spelling,
decodes each word once, and compiles only this explicit find/grep form into the
existing repository-aware search with a filename filter. It never evaluates a
shell or accepts arbitrary `-exec` commands. Filename filtering precedes the
100-hit cap; the10,000-candidate,256KiB/file and output bounds remain. Root
searches retain Git ignore rules, while explicitly selected dependency
directories remain searchable as before. Missing paths are not redirected.
Repository identifiers are distinguished from filesystem paths, and model-facing
path-recovery feedback names the actual public functions without rewriting raw
controller records. Filename-only matches can recover path discovery but cannot
clear a requirement to reinspect file content after an edit mismatch.

The139affected CPU checks passed. The separately prospective
[command-repair episode](command-repair/terminal-report.md) then completed50
proposals with official `resolved=false` and a0-byte patch. It made five physical
tools: three missing-path views and two successful root views. The final37
proposals repeated a blocked view of the nonexistent nested repository path.
There were zero find/grep requests: absence of the old43 parser errors is not
evidence that the repaired composition was exercised by this model episode.

The original prospective input record also failed exact equality: it recorded
1403tokens but omitted the compiler's36-token empty prefix capsule. Actual input
was1439tokens using the correct pinned tokenizer and within bounds. The
[CPU follow-up](command-repair/follow-up-report.md) reuses the actual context
compiler and reproduces every saved first-input ID exactly, explicitly post hoc.
The original record and failed prospective comparison remain unchanged; neither
the reconstruction nor the instrumentation error authorizes a quality retry.

## Directory-view quality and physical-workspace input

The two actual root views returned only the first20 entries, dominated by
dotfiles. Saved next-model inputs contained these exact outputs, the correct
repository-root identity and public recovery guidance. The root was the actual
Lighthouse checkout, not the evaluator harness. A
[source-verified producer-contract mismatch](research.md#directory-view-contract-exposed-by-the-actual-trace)
was then identified: directory views should reject file ranges and expose
non-hidden absolute paths to two levels, not paginate a hidden first-level list.
The generic bounded implementation now does this without shell execution,
symlink traversal or path aliases. Explicit file ranges remain distinguishable
from default views, so an invalid directory range cannot poison the valid call.
The145affected CPU cases passed before the separately prospective
[directory-view episode](directory-view-repair/terminal-report.md). That episode
also returned official `resolved=false` and no patch: one physical missing-path
view, no successful tool, and no repository-root request among50proposals.
Thus it never successfully exercised the corrected directory producer. The
fresh1493-token initial record matched the actual first input exactly through
the real context compiler; feedback/capsule/bounds checks passed. GPU-engine
scope was80.621seconds and official grading9.531seconds, within the frozen caps;
owned execution resources were released and no retry occurred.

The remaining [source-verified workspace-input difference](research.md#physical-workspace-in-the-trained-task-input)
is structural: the producer passes the true filesystem checkout in
`uploaded_files`, whereas our custom user payload supplied the logical repository
identifier that the model repeatedly appended to the root. The actor input now
uses the configured physical root and verbatim issue in the producer's
`uploaded_files`/`pr_description` shape. The extra logical model-input field and
its defensive clarification are removed; external task/evaluator identity is
unchanged. No alias, relocation, task-file hint or forced action is introduced.
The23focused solver/native/dialogue cases pass, including logical-ID invariance,
verbatim issue content and unchanged non-actor input.

The [workspace-input episode](workspace-input-repair/terminal-report.md) then
consumed50proposals and returned official `resolved=false`, with a legitimate
621-byte product diff. Its1481-token initial input matched prospectively exactly.
It first viewed the real root successfully and completed17physical tools,
16successes, three finds, three grep/searches and one edit. No developer test or
finish occurred. Of29invalid decisions, two were unsupported pipelines; the
terminal27responses were identical230-token prose-only messages, stopped
normally rather than cut off by the token budget. GPU-engine scope was345.053s,
official grading9.320s; owned resources were released with no second episode.

## Resource-aware inspection recovery

After a missing-path error, the old V4 root-only recovery gate blocked four
otherwise supported inspection calls, including a search in an existing
directory and a view of the already-edited source file. The model subsequently
described that readable file as inaccessible. Generic CPU reproduction confirms
that an unrelated missing path must not prevent inspecting an existing resource.
V4 now permits bounded list/search/read/view/find inspections across the visible
repository and clears path recovery only on successful nonempty evidence.
Mutations, tests and completion remain blocked until recovery; exact active
failure certificates, edit-content refresh and post-edit verification remain
enforced. V1-V3 retain their original root-only behavior. JSON decision schemas,
system instructions and public XML feedback use the same recovery contract.

The actual trace also showed two distinct unsupported pipeline bodies compacted
together because both used the same `invalid_decision` placeholder/error.
Malformed-proposal compaction now includes raw proposal identity; identical
prose-only repeats still compact. This repairs an observed identity error, not
proof that compaction caused the collapse. Pipelines remain unsupported.
The136-case affected gate and expanded2-case XML end-to-end loop pass, including
edit, missing path, valid known-file inspection, same-hash successful retest and
finish. The [resource-recovery episode](resource-recovery-repair/terminal-report.md)
then completed50proposals with official `resolved=false` and zero final patch.
All nine physical tools succeeded: seven inspections and two edits, the second
restoring the original export. No missing path occurred, so the new recovery
branch was not exercised. The41prose-only invalid proposals had five distinct
bodies; distinct bodies remained distinct and true repeats compacted, validating
identity behavior without a quality-benefit claim. Prospective1500-token input,
feedback/capsule and limits checks passed; GPU engine340.957s, grading10.552s.

## Explicit structured function response

A [single exact-input stock reference](prose-stock-reference/terminal.json) chose
index15 before output, copied all9245captured input IDs, and used the same actor,
tokenizer, precision and decoding with stock vLLM and prefix caching off. It
reproduced the native prose response byte-for-byte in19.983engine seconds, with
no tools or official evaluation. This argues against private native KV reuse
as the sole cause at this point, not against every engine issue or for quality.

The new opt-in `openhands_json_v1` uses existing request-level JSON-schema
decoding on the initial and every subsequent request. Exactly one function name
and typed arguments are required; optional unverified reasoning stays inside
the JSON object. All existing function choices and one-request/action budget
remain, with the same parameter-to-action mapper, physical workspace and issue.
XML remains the unchanged unconstrained comparator. No generated text is
converted into synthetic XML, fenced output repaired, action forced or extra
planner/choice request added. A complete JSON object is not necessarily a valid
command/path or a correct repair; semantic checks and official grading remain.

Native CPU validation exposed an xgrammar0.2.3 incompatibility: `minLength`
rejects valid escaped quotes/newlines in string fields. The original failing
schema and matcher records are retained in `json-function-schema-cpu/`.
The decoding schema is now explicitly syntax/type-only: redundant `minLength`
clauses were removed, while empty command/path/old_str values remain rejected
by the unchanged shared semantic mapper before tool execution. Empty strings
may decode, but are not accepted as executable calls. The143-case affected gate,
13fresh-reference cases and108focused final adapter/solver/reference cases
passed. The corrected [native CPU gate](json-function-schema-cpu/syntax-only/xgrammar-result.json)
on `.134` completed all six required positives, including escaped quotes/newlines,
and rejected all five syntax/type negatives. All six empty/whitespace boundary
cases completed the syntax grammar but were rejected by the shared semantic
mapper before execution; the syntax-valid pipeline was also rejected.

The user-requested [`.134` readiness](json134-readiness/readiness-summary.json)
is complete: eight idle RTX4090D48GB devices, an existing matched runtime, and
15,242,760,997 verified actor/tokenizer bytes transferred privately in152seconds
within the16GB/3600second allowance. No inference or shared-environment change
occurred; transfer and CPU processes were cleaned up. The syntax-only schema
identity is `a97f1e16cb58364edd1f1558f4c21564893c955243a54e253c02b5726af2a4de`.

The separately prospective [JSON episode on `.134`](json-function-quality/terminal-report.md)
then consumed all50proposals and returned official `resolved=false` with a0-byte
diff. All50outputs were complete JSON function objects, with zero prose-only or
truncated responses. One zero-start range was rejected semantically; three
physical ranged-directory views failed, and one exact failed repeat was blocked.
The other45physical calls were successful ranged views of one source file.
There were no edits, developer tests or finish. The exact1546-token prospective
input matched the real first request, including the36-token empty capsule.

| Same-task50-proposal episode | Complete JSON objects | Successful physical tools | Edits / tests / finish | Final patch bytes | Official resolved |
| --- | ---: | ---: | --- | ---: | --- |
| Prior unconstrained XML, resource recovery | Not applicable | 9 | 2 / 0 / 0 | 0 | false |
| Syntax-constrained JSON on `.134` | 50 | 45 | 0 / 0 / 0 | 0 | false |

This episode removed the observed prose-only failure but did not achieve repair.
It does not independently rule out KV effects: the previous exact stock match
concerns an XML input, not this new JSON trajectory. GPU engine scope was180.204s
and official grading9.391s, within the frozen limits. The scheduled GPU sample
preceded allocation, so active utilization was not captured and is not inferred.
An optional CPU output-schema replay timed out at60seconds; its failure remains
visible and was not retried. Actual source/input/terminal/grade evidence and exact
cleanup remain preserved. No second episode or downstream run was started.

CPU attribution found that only five of the45successful views returned source:
the file had91lines, and the remaining40ranges were beyond EOF. Exact captured
next-input IDs retained the requested ranges, correct total extent and empty
feedback; no compaction occurred. Later journal trimming removed oldest turns,
not the latest feedback. Edit/test/finish choices remained available.

A subsequent [pinned producer comparison](research.md#explicit-file-range-contract-exposed-by-the-json-episode)
established a real contract mismatch: explicit out-of-file start/end values must
be errors, not clamped or successful empty views. This supersedes the initial
trace-only no-mismatch conclusion without changing its preserved evidence or
claiming that the mismatch caused the entire failure. The adapter now returns
an explicit range error and file extent; V4 failure validity follows that file's
revision. Default views, bounded remainder and shell-read semantics are unchanged.
The155affected parser/tool/state/actor cases pass, including the observed ranges
in both formats and same-hash retry after actual file growth. The separately
prospective [corrected-range episode](file-range-repair/terminal-report.md) then
exercised the boundary: `[80,100]` against91lines returned the typed error with a
file-revision-zero certificate. There were zero successful empty/beyond-EOF views.
The model changed to in-bounds ranges but still returned official false with no
patch, edit, developer test or finish. It completed36successful views with ten
distinct nonempty bodies, eventually repeating `[1,20]`27times. All50outputs were
complete JSON;11invalid ranges were rejected semantically and three physical
views failed. Deduplication reached32entries with no journal trimming.

Exact1580-token prospective input equality and frozen limits passed; engine
111.831s, grading9.325s, owned cleanup complete. The engine-loaded GPU sample
captured43,170MiB and216.81W with instantaneous0%utilization between decode steps;
it is not an average utilization estimate. The original episode remains closed.

The [prospectively selected stock reference](json-repeat-stock-reference/terminal-report.md)
then copied all6917actual input IDs from terminal request49. Fresh stock vLLM,
with APC off and no private checkpoints/forks/adoption, reproduced both the raw
JSON bytes and canonical `[1,20]` view exactly. The identical schema was derived
from the frozen source/config; the original request did not record its schema
string, so direct schema capture is not claimed. Native45versus stock46decoded
tokens use different stop accounting; source output IDs were not recorded and
token-ID equality is not claimed. The one actual request used28.143engine seconds,
with an89%in-engine sample and verified cleanup. A preserved pre-engine failure
on missing provenance hashes emitted zero model requests; its task-local fix did
not change the selected input or create a second output.

Private native reuse is not necessary for this repeated decision at this exact
JSON input. That is not a blanket engine/model verdict or an official repair.
All seven actor episodes and this diagnostic are closed, with owned execution
resources released. Original official quality remains unmet.

## Original-format syntax constraint

The next bounded discrimination preserved the actor's original XML instructions
instead of changing the response language to JSON. Earlier XML episodes made
real edits but also collapsed into prose-only output; the JSON treatment removed
that syntax failure but made no edits. This motivates a same-format syntax
constraint, not a claim that JSON caused the behavioral difference.

Opt-in `openhands_xml_grammar_v1` supplies a direct xgrammar grammar on the first
and every continuation request, including the stock-reference consumer. The
original XML prompt, optional reasoning before the call, parameter order freedom,
multiline/Unicode/quoted text, shared semantic mapper and all public functions
remain. Only complete single-call syntax is constrained; unsupported commands,
empty required strings and invalid ranges still receive real semantic errors.
There is no JSON wrapper, function/choice prefill, reasoning length restriction,
extra request or closing-tag repair. Length exhaustion remains a real failure.
The unchanged JSON schema retains its prior byte identity. The [actual CPU gate](xml-grammar-cpu/report.md)
on `.134` passed62positive cases across parameter permutations and text boundaries,
16syntax negatives, seven syntax-valid semantic errors, and12real-tokenizer mask
cases containing476tokens and EOS. Grammar identity is
`5056481db1038d428e1e61fa4e334e9bb756483d4c79f4de6bcefe4398dd6717`.
The20native sampling/unchanged inline-intent cases also passed. Both commands
stayed below60seconds with CUDA disabled, no model requests and no weight load;
the later evidence-print wrapper error is recorded separately from their actual
exit-zero results. Local135parser/solver/reference and16freeze cases passed.
The separately prospective [XML grammar episode](xml-grammar-quality/terminal-report.md)
then returned official `resolved=false` and a0-byte diff. Nine complete calls
were semantically accepted and physically successful: seven views, one search
and one find. The first12proposals included all nine successes and three invalid
decisions; this is a prefix of a50-proposal budget, not a12-budget comparison.
The remaining41outputs exhausted generation before completing a call. Independent
request accounting records41`FINISHED_LENGTH_CAPPED` outputs at511decoded tokens
and nine normal stops. There were no edits, developer tests or finish. Every raw
output and rejection remains intact; no tag, fence or function was appended.

The1541-token prospective first input matched exactly, including its36-token
empty capsule. All frozen source/input/model/tokenizer/precision/limit checks
passed. Direct grammar bytes are frozen, but per-request grammar strings were
not recorded; source/config identity is not mislabeled as direct request capture.
Engine scope was655.868s and official grading9.823s. The separate1200-second phase
ledger retains655.868sused/544.132sremaining; the earlier corrected-range ledger
remains139.974sused/1060.026sremaining. The in-engine sample recorded43,552MiB,
64%instantaneous utilization and309.26W. Owned runner, relay/GPU process and
evaluator container were released, with the selected GPU returning to0MiB.
A zero-request preparation failure from omitted `PYTHONPATH` is explicit and
was corrected without altering the scientific setup or adding an episode.

Original-format syntax constraints do not achieve repair under this unchanged
budget: they constrain completed calls, not completion before a length cap.
The new real blocker is long model reasoning without executable calls, not a
newly established executor defect. No approved useful retry is ready within the
unchanged scope; forcing actions/reasoning bounds or sweeping budgets is not a
substitute. Quality still gates multibranch and formal; the full objective is open.

## Retaining the actual generation failure cause

Tracing the newly observed length failures exposed a narrower benchmark data-loss
boundary: native `BrainResult` and `Branch` already retain `finish_reason` and
`output_token_ids`, but `repair` discarded both and supplied only the generic
parser error to the next actor request. This is not an engine stop-reason defect.
The actor trace now records actual finish reason, decoded-token count and the
unchanged output limit; malformed-call feedback carries those same producer facts.
The parser error and raw output remain unchanged. A complete, semantically valid
call at the length cap still executes; no forced shortening, reasoning limit,
function prefill, changed prompt, output repair or extra request is introduced.
The29affected solver/native-dialogue cases pass;12first/continuation regressions
failed before this correction. They check both normal and length stops, exact
facts in the actual next compiled input, real edits and same-hash successful
retesting in all three actor formats. The grammar/model/budgets are unchanged.
This prepared one distinct factual-feedback intervention, not a quality pass
or permission to replay the already closed XML episode unchanged.

The [separate generation-feedback episode](generation-feedback-quality/terminal-report.md)
then produced47complete XML calls and three malformed length-capped outputs.
All three actual failures carried the producer's `finish_reason="length"`,
`output_tokens=512` and unchanged512-token limit into their exact next compiled
capsule. These are returned output-ID counts; backend accounting reports511
decoded tokens for each capped request. They are distinct units, not a proven
off-by-one defect. No complete capped call occurred, so that case remains covered
by the CPU regression rather than claimed as exercised here.

All32physical tools succeeded:26views, one search, one find and four edits. The
model produced a legitimate5,440-byte diff in `stack-packs/packs/react.js`, but
never requested a developer test. Fifteen complete finish calls were correctly
rejected because the changed workspace had not passed verification. Official
`resolved=false`: a nonempty model diff alone does not open the quality gate.
The first12proposals contain nine successful inspections/searches, two malformed
calls and the first edit; they remain a prefix, not an equal-budget comparison.

The1541-token initial input again matched prospectively and exactly. Only the
deployed solver implementation hash changed; model, tokenizer, XML grammar,
prompt, native runtime and budgets remained fixed. Engine358.881s and grading
10.407s were within bounds, leaving841.119s of this distinct1200-second phase
ledger; previous phase accounting remains unchanged. The one in-engine sample
was65%instantaneous utilization at44,440MiB/279.25W. Owned runner, GPU/relay and
evaluator resources were released, and the selected GPU returned to0MiB.
The terminated bookkeeping/observer mistakes did not restart or alter the job.
This episode changed behavior and exercised the factual feedback, but did not
achieve official repair.

The [corrected no-inference attribution](generation-feedback-quality/finish-feedback-attribution-corrected-report.md)
found no instruction, feedback or executor mismatch at the verification boundary.
The actual first input explicitly states that finish is invalid after a change
until a test succeeds. All15rejections reached their next request's exact capsule
token sequence, and the documented, grammar-selectable `execute_bash`/`yarn jest`
route remained available. The initial attribution overlooked the imported XML
instruction and claimed an asymmetry; that report is retained but superseded by
direct decoding of the actual frozen input. No source edit or model retry was
made for that incorrect finding. The model ignored correct available verification
facts, and its nonempty diff failed official repair. No new evidence-backed
correction target or approved useful retry is established. Multibranch and formal
remain gated; the full user objective is not complete.

## Applying existing completion validity during decoding

The next explicit treatment is `openhands_xml_state_grammar_v1`. The non-actor
`action_schema` already excludes `done` when `verification_required` is true;
the static actor grammar continued offering `finish`, which the unchanged
runtime rejected15times in the last episode. The opt-in grammar now applies
that same existing completion-validity rule at decoding time. Only currently
illegal `finish` is removed. Bash inspection/testing and all editor variants
remain model-selected; no test, edit, novelty or particular function is forced.
Successful verification reopens finish. FailureMemory remains authoritative,
including its post-change and failed-test semantics; no new completion rule is
invented and the correct-feedback attribution above remains unchanged.

Original XML/static-grammar/JSON comparators keep their existing decoding. The
new format uses the exact same XML instructions and initial grammar bytes, with
no reasoning bound, budget change, output repair or additional request. First
and forked requests receive current verification state; stock references require
the actual captured boolean rather than guessing an unrestricted grammar.
The72affected cases pass, including11red-before cases and an end-to-end failed
test, real edit, successful same-hash retest and reopened finish. The new closed
grammar's [actual xgrammar/tokenizer CPU gate](verification-grammar-cpu/report.md)
passed51non-finish positives, seven semantic-error cases,16syntax negatives,
12positive tokenizer cases and11state-blocked finish cases checked both as text
and real tokenizer masks. The closed-state grammar identity is `2be9cf0f...`;
the open grammar remains `5056481d...`. CUDA stayed uninitialized and zero model
requests ran. A failed worker admission and shared-master connection are explicit;
the bounded gate ran directly after exact failed-worker absence was established.
One separate prospective episode is now ready. Neither this legality mask nor
reduced rejected proposals proves official repair.

The [verification-mask episode](verification-mask-quality/terminal-report.md)
then returned official `resolved=false` with a legitimate3,494-byte model diff.
Eleven complete XML calls all succeeded physically: seven views, one search,
one find and two edits. The other39outputs were malformed/length-capped. No
test or finish was emitted. Verification was false before requests0-11 and true
before12-49 after the first edit, so the source-selected grammar removed finish
for38requests. No finish appeared while masked; reopening was not exercised.
All non-finish functions remained available. This establishes the observed
legality boundary, not that the mask caused every subsequent length failure.

All39malformed observations retained actual generation facts, and every available
next input retained them. The1541-token prospective input matched both the actual
request and preceding unchanged initial inputs, including the36-token capsule.
Per-request grammar parameters remain source/state-derived, not directly captured.
Engine937.454s and grading10.941s were within bounds; the separate phase retains
262.546s of1200. The active sample was43,552MiB/100%instantaneous utilization at
194.77W; owned execution resources were released, with GPU0 at3MiB/0% afterward.
A missing local `transformers` dependency affected only prospective recording;
the identical recorder used the already-pinned remote tokenizer with zero model
requests, without changing scientific inputs or relaunching the episode.

The known-invalid finish exclusion worked but did not induce testing or repair.
No further masking or budget/prompt recipe is inferred from that negative result.
A subsequent [producer control-flow trace](research.md#normal-assistant-messages-versus-format-errors)
identified a distinct non-tool message path in the pinned scaffold. The opt-in
`message_policy=codeact_continue_v1` now implements this boundary on unconstrained
XML only. Naturally stopped ordinary messages retain their exact text and receive
the fixed first-party base continuation; they are not tool errors or tool results.
There is no output repair, initial prompt change, extra request, evidence commit,
workspace refresh, failure recovery or verification relaxation from a message.

The original prose-to-format-error path reproduced directly before the change.
The affected187local cases pass, including first/fork actual compiled inputs,
stopped versus capped/partial/empty/unsupported outputs, control-only events,
unchanged failure/evidence state, retained real-error capsule, real edit and
same-hash retest, and50message/request budget exhaustion without physical tools.
Existing XML grammar/native CPU gates are reused unchanged.

The separately prospective [message-control episode](message-control-quality/terminal-report.md)
returned official `resolved=false`, with a0-byte model diff. It exhausted all50
proposals:48complete XML views (47successful, onefailed) and two malformed
length-capped outputs at4and7. There were no edits, tests, finish proposals or
naturally stopped ordinary messages. The callback/control treatment was enabled
but unexercised, so no behavioral/quality effect of message handling is measured.
Its absence is not a reason to repair capped output or force messages/actions.

Both malformed errors retain actual `length/512/512` generation facts in exact
next compiled capsules; backend decoded counts511use a separate accounting unit.
Fresh prospective1541input IDs exactly match the actual request and prior initial
input hash, with36capsule tokens and no grammar. The first12prefix has nine
successful views, onefailed view and two capped errors, not an equal-budget pair.
History was bounded (maximum four dropped and42deduplicated entries); workspace
and verification state did not change. Engine244.121s, grading10.615s and runner
wall285s remain within limits, leaving955.879s of this separate1200sphase cap.
The active sample was44,440MiB/100%/302.61W. Owned runner, relay/GPU process and
evaluator container are absent; GPU0 returned to0MiB/0%, while source deployment
and pinned assets remain. All eleven closed actor episodes remain officialfalse;
neither genuine multibranch nor Formal has begun.

Read-only attribution found a stable41-action exact view loop at9-49, always
successful and nonempty. The only physical failure was an explicit `[47,94]`
range beyond the47-line file; its exact error/action remained in the next input
and failure memory. No generic parser/executor/feedback loss is established.
The narrower clipping check found that those6000characters contained only
21complete lines and a partial line22, despite `returned_lines=47` and
`more_lines_after=false`. The already-correct `output_truncated=true` signaled
clipping, but the range fields described the selected rather than emitted window.

The smallest new repair finalizes view extent after character clipping:
report the emitted line count, flag partial last-line content and retain any
remaining lines. It does not change the returned source, limits, initial prompt,
validity rules, sampling, actor or continuation policy. The real JSON worker
regression fails with47versus22before the fix;165affected cases now pass,
including mid-line/boundary/EOF/nonzero-start/directory cases and existing
recovery loops.

The separate [emitted-extent episode](emitted-extent-quality/terminal-report.md)
then validated action3's actual6000-character output as22emitted lines from a
47-line file, with `more_lines_after`, `last_line_truncated` and
`output_truncated` all true in context4. The initial1541IDs matched prospectively.
Nevertheless it was officialfalse with0diff: four successful views, followed by
46length-capped malformed edits and no physical edit/test/finish/message.

The [capped-output attribution](emitted-extent-quality/malformed-capped-attribution.md)
found zero prose-only outputs and46attempted editor calls. Every one opened a
`str_replace` for the same `react.js`; `old_str` was truncated inside the encoded
SVG and `new_str` was absent. Forty-five outputs were exact duplicates. Raw text,
parser error and native generation facts reached all available next inputs;
failure/verification state remained unchanged. No executable action can be
recovered without inventing missing content or continuing generation.

Engine680.466s, grading11.332s and runner722s were bounded;519.534s remains in
this phase. One active sample was44,442MiB/61%/265.15W. Exact task-owned runner,
relay/GPU process and evaluator containers are gone; GPU0 returned to0MiB/0%.
All twelve episodes are closed and negative. There is no remaining evidence-backed
adapter repair on this trace; arbitrary prompt/model/seed/budget or forced-action
changes are outside the frozen claim. Original quality remains blocked, so
multibranch and Formal do not start.

A subsequent visible-source check narrows one earlier decision boundary without
contradicting that terminal parser attribution. The actor's first malformed
explanation proposed adding a temporary React pack `id` after seeing only the
first22emitted lines. The already-existing `id: 'react'` is later in the same
47-line base file. The actor was told that lines remained, but not the exact
valid overlap range needed after clipping a huge SVG line; it instead attempted
to copy that partial SVG into `old_str`.

The new worker metadata reports `next_view_range`: overlap a partial final line,
or start after a complete boundary, using the existing `[start,-1]` contract.
No source bytes, output/token limits, instructions, model request, sampling or
tool validity change, and no action is forced. Directory results omit it because
ranges are file-only. The exact JSON-worker regression was red before the field;
151focused solver/protocol cases now pass, including the actor's rendered next
suffix. This is one new evidence-backed interface treatment ready for a separate
prospective episode, not an automatic retry of the previous negative.

The [view-continuation episode](view-continuation-quality/terminal-report.md)
exercised the new range: after one capped output, action5 used `[22,47]` and
successfully read the remaining26lines including `id: 'react'`. It remained
officialfalse with0diff: nine physical views, two valid messages,39malformed
decisions, no edit/test/finish. Both messages followed the upstream callback
without tool/evidence/verification side effects. All later successful index
source remained in exact contexts11-13; the claim that it was unavailable was
an actor contradiction. The root listing exposed every top-level entry and
relevant test/source anchor before clipping. No range, context or discovery loss
remains. The episode and official recovery used one model run only.

The remaining malformed behavior was one invalid `[0,22]` call, one fenced
length cap and37prose outputs; actions14-49 were identical. Exact errors and
generation facts reached all38available next inputs. Engine636.705s and grading
9.959s were bounded;563.295s remains in its separate phase. One active sample
was43,554MiB/61%/249.79W; runner/relay/GPU/evaluator resources are gone and GPU0
returned to0MiB/0%.

A final pinned-source comparison identified one producer observation treatment
not yet present: ACI0.1.0 appends a fixed file-clipping notice telling the agent
to locate lines with `grep -n` before retrying. The new opt-in policy stores its
exact text/source and renders it after unchanged raw source for clipped files
only. It keeps the6,000-character output, existing next range, initial prompt,
action set, requests, sampling and budgets; it does not import ACI's larger limit,
numbering or extra editor commands. Directories/unclipped/default comparators are
unchanged. All188affected cases pass. One prospective notice episode is ready;
no automatic prior-run replay or quality claim is made.

The first notice attempt is an explicit infrastructure-invalid result, not a
negative episode. The engine served four requests, then the first clipped file
view failed because the inline `python -c` worker imported a sibling module that
does not exist inside `/testbed`. No notice/context4/official grade was produced;
GPU and containers were cleaned. The root fix makes the inline source
self-contained and changes no treatment semantics. The real regression now
executes that source via `python -c` with an isolated cwd, failing before the fix
and passing afterward together with all188affected cases. A rerun of the same
frozen treatment is required; this does not consume a quality outcome or
authorize any protocol change.
