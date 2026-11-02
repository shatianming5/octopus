# Controller admission: verified progress and scientific blocker

Latest confirmation: [R3 completed but was not accepted](r3-confirmation/report.md).
All16 traces and6,784 official predictions are complete. Candidate FG mean
differences are+3.3145s versus APC and+4.7765s versus queue, with positive
one-sided95% upper bounds. BG P95 is+2.9735% with an upper bound+6.2241%,
above5%. Official AnswerF1 is also slightly lower; full-credit counts are
equal and no predictions are missing. Only2201 passes every reconfirmed user
condition. All owned launchers/model processes exited.

The two-seed final-runtime screen below remains `PASS_TO_R3`; it was not
rewritten after confirmation. No R4 or further runtime-only tuning was run.
Model x vLLM intent emission is a subsequent implementation direction, not
implemented code or a demonstrated performance gain in this delivery.

Prior decision: the
[final minimal runtime-only R2 screen](r2-runtime-final/report.md) is
`PASS_TO_R3` at nativebeb50062. All8 traces and424 requests per trace are
complete. Both seeds retain at least75% of fixed FG savings; BG prefill falls
13,376 tokens per seed and total prefill falls13,376/8,960 tokens versusfixed;
candidate BG P95 is96.1143/91.6632 versus APC111.4073/107.1074 seconds.
These are explicitly authorized shared-GPU measurements with the heavy live
observer bypassed, not isolated-latency evidence.

The new user-defined four-condition screen did not include the old
protection/zero-waste or quality gates. Those older negative results remain
retained below. The frozen-policy R3 with four prescribed seeds and formal
confidence/quality plus the same user-condition reconfirmation has now
completed as reported above. The immediate co-design failure branch was not
taken at the R2 screen.

Retained previous checkpoint: [actual frozen serving and observer repair on gpu.134](r2-history-serving-gpu134/report.md)
completed two10-trace matrices, each with4,240 official predictions.
The repaired native6108c2c still does not pass R2: candidate FG is4.72%/3.35%
slower than APC, BG P95 is22.60%/18.91% worse, and protection rises9.22%/1.44%
versus original semantic fixed. All prediction texts match the first matrix;
full-credit totals remain262 versus263 for APC/queue. R3/R4 remain unrun.

The observer-only repair passed26 native tests and21 exact non-CPU export
comparisons; recorded observer CPU fell85.5%. Actual candidate observer CPU
fell to17.798/16.121 seconds but remains substantial. Residual CPU diagnosis
is separate from scientific acceptance and authorizes no further GPU run.
The earlier17% FG and approximately81% BG regressions remain retained.

Under the existing original-declared-consumer whole-episode reference,
all2,118 exposed fixed body units per seed have positive demonstrated saving,
with zero no-saving or unknown body exposure against APC and queue.
The requested strict decline cannot be demonstrated below that zero floor.
This is not a claim that every interval is causally necessary; changing the
comparison or its definition is a scientific decision, not a silent gate waiver.

Prior readiness: [frozen history serving integration](r2-history-serving-integration/report.md)
is merged in library `bcd9338` (PR #16), with native source
`95481254b87a8bf9b0ffe5fc84f511bc97ebbde0`. Actual construction, committed
callbacks, live FREE observation and exports pass the focused 14-case native
gate. That gate established executable readiness, not GPU, latency, quality or complete
R2/R3/R4 acceptance.

The retained upper1702 conditional screen at `cb86ca8c` preserves89.801700%
of fixed FG savings, with BG body0 equal to history blanket,22.857206% lower
protection and0.100091% no-saving rate. The completed lower9216/seed1601
characterization preserves only15,392/24,240 =63.49835% of fixed savings,
failing the75% screen despite zero additional BG body work. That lower-boundary
failure is retained; different source schedules are not paired serving or R4.

The ten-trace upper10752 serving discovery, comprising five distinct methods,
seeds1601/1702 and9,865,003,008 KV bytes at the integrated native source,
is now complete as reported above.
The summaries below retain earlier failures.

The R1 implementation and GPU gates pass. The full R1--R4 objective does not.
Historical-hint resolution passes its first upper1601 conditional CPU screen,
preserving all33,888 FG saved tokens and zero BG body work while reducing
protection19.231884% and no-saving exposure to zero. Its unchanged upper1702
replication fails the predeclared matched-background gate: next-reuse
recomputes2,224 BG tokens versus history blanket's zero, despite retaining
96.364495% of original fixed FG savings and reducing protection19.851832%.

Relative to the matched blanket, next-reuse trades2,224 fewer FG recompute
tokens for2,224 more BG recompute tokens. Both total3,456 tokens across these
two work components; token equality is not timing or FLOP equality.
No policy tuning, lower-point outcome, serving integration or GPU followed
within that failed replication scope. Later explicitly renewed scopes are
recorded separately below. The full prospective R2-B and actual R2 latency
gates remain unmet; R3/R4 are unrun. Earlier negative evidence is retained.

| Stage | State | Decision-bearing evidence |
|---|---|---|
| R1 | Passed | Native whole-trace selector bounds 1.9401x / 1.6566x; actual GPU ratios 1.8456x / 1.6948x versus queue; preserved semantics and lower controlled overhead. |
| R1-GPU | Passed | Fixed FG versus queue -1.9504% / -1.5078%; BG P95 versus APC -1.8162% / -2.2657%; parent recompute 33,888 to zero in both seeds. |
| R2 opportunity | Measured, mixed | Four separate 424-request traces: fixed incurs 13,376 extra BG body recompute tokens in seed1601, but not seed1702. Both preserve all FG savings. |
| R2 admission | Measured; not accepted | Latest source807d46de preserves >=75% savings and reduces exposure, but fails the primary BG P95 gate. |
| R2-A | Original two-point scan; new primary point complete | First source binds at 9,216 but not 10,752 blocks. Latest source binds in 11 / 15 draws at 9,216; expansion is blocked by the negative primary screen. |
| R2-B | Measured; not passed | Common fixed-reference initial KEEP density exceeds DROP, but no-saving exposure and rates increase. All 2,118 body units/seed reconcile; no denominator shortcut. |
| R2 lifetime refinement | CPU complete; useful gain not established | Stable grants restore fixed's 24,240 / 18,272 saved FG tokens; conditional protection falls only 35.604030 block-seconds (0.005052633% pooled), with no BG recovery. |
| R2 history-matched | First CPU pair failed; second seed stopped | Selective preserves only 48.38% of fixed savings, adds 864 BG tokens versus matched blanket and increases conditional exposure 0.182226%. |
| R2 hard-pressure projection | Native gap fixed; CPU screen failed | Restores 100% of original fixed FG savings, but BG body recompute returns to 13,376 and exposure rises 3.710406% versus history blanket; actual-stream cascades are zero for both choices. |
| R2 frozen upper point | CPU complete; useful selective gain not established | 99.480642% FG savings and full BG recovery, shared with blanket; one genuine decision, only0.189531% lower protection, unchanged no-saving exposure and176 added FG tokens. |
| R2 history resolution | First conditional CPU screen passed | Source7b6a96f/native7db3e73a: full33,888 FG savings, BG body0, protection -19.231884%, no-saving exposure119,869.833364 to0; 45 explicit rejection draws and836 unique later victims. No serving or second-region result. |
| R2 resolved-history replication | Upper1702 failed matched-BG gate | Frozen7db policy: FG savings96.364495%, protection -19.851832%, no-saving -95.494532%, but BG body2,224 versus blanket0. Lower807 and serving integration stopped. |
| R2 positive KEEP continuity | Native verified; single CPU outcome failed | Nativee869df19:24 focused cases pass;92.587347% FG savings and21.551285% less protection, but944 BG tokens versusblanket0. Remaining216 positive losses are hard-projected; no serving/GPU. |
| R2 native reclamation | Single conditional CPU screen passed | Nativecb86ca8c:89.801700% FG savings, BG body0, protection -22.857206%, no-saving rate0.100091%;216 native-forced KEEP victims remain distinct from836 resolved DROP victims. Integrated at95481254; subsequent actual serving is reported below. |
| R2 history serving integration | Native readiness passed; lower conditional boundary failed | Native95481254:14 focused cases pass, with committed callbacks and live exports. Lower9216/1601 retains63.49835% of fixed savings, below75%; protection falls22.35436% versus history blanket. The prepared ten-trace upper10752 plan has subsequently completed. |
| R2 history actual serving | Complete; not accepted | All10 upper10752 traces and official scores complete at95481254. Savings100%/86.97% meet75%, but FG/BG timing and protection versus original fixed fail; full-credit count is262 versus263 for APC/queue. The dominant observer cost motivated the repair below; these negatives remain retained. |
| R2 observer-repaired serving | Complete; still not accepted | Native6108c2c:10/10 traces and scores complete; candidate FG+4.72%/+3.35% and BG P95+22.60%/+18.91% versus APC. All predictions unchanged; protection remains above original fixed. |
| R2 original-fixed reference floor | Measured | Both seeds have2,118 positive-saving exposed body units and33,888 reconciled saved tokens; zero/negative/unknown body exposure is0. Mandatory shared ancestors remain separate. No strict decline below the existing zero no-saving floor can be claimed. |
| R3 / R4 | Not performed | No frozen final policy; confirmation and held-out seeds/pressure remain untouched. |

The formal evaluator and CPU-isolated official 424-question scorer are ready
at `1d2b67e`. Scoring uses the official evaluator and binds every score to its
exact trace prediction text, rejecting stale output associations. The targeted
31-test suite and a synthetic official-API check pass; these are tooling
evidence, not measured quality or an R3/R4 result.

The first R1 GPU cost miss remains preserved in `r1-gpu-01/`; the passing
repair is in `r1-gpu-02/`. The original 422-request R1 cohort is not pooled
with the distinct 424-request recurring-BG probe.

## Earlier high-headroom blocker

The actual native replay (`4f51a26`, native `82397074`) matches all 20,248
allocation/adoption events and 1,696 native free groups from the four traces.
Keeping 75.0236% of FG recompute savings donates 529 FG keys but recovers
zero BG tokens: net change versus fixed is 8,464 more recomputed tokens.
The recorded-stream FG-only rank diagnostic needs 1,200 extra older
victims to preserve any BG body. This is not a universal impossibility
claim about other schedules, policies or pressures.

An ordinary confirmed-adoption-history union control instead recovers all
13,376 BG body tokens in seed1601 while preserving 100% of FG savings.
That result supports better reuse information/candidate coverage, not
selective admission: it keeps all eligible candidates. Calling it
value-aware admission would confound information with selection. No
rejection/exposure benefit has been demonstrated against an information-
matched all-KEEP control, and the original zero-own-benefit FG floor remains
zero rather than an invented improvement.

Admission claims from that history-union alternative would need a fair
history-aware fixed control and actual, prospectively distinguishable
competition among eligible candidates.
The current bounded input does not establish that condition. No quota,
posthoc expiration horizon, additional query, shifted arrival or held-out
result was introduced to manufacture it.

The renewed block-level check is complete at `d5a93f0`. It found no redundant
physical-copy, out-of-span-tail or post-last-adoption exposure. The small
11.198773 / 11.034512 block-second zero-next-use association is an uncovered
leading ancestor during an allocation-free idle gap, not a valid prefix-
closed DROP. Runtime measurement and prospective baseline evidence are
implemented; causal admission is not thereby proved.

The next controlled change is physical capacity: the reference has only
2,119 future copies against 28,765 free-minus-draw blocks. Its approximate
live/draw plus future working set is 9,537 blocks. Predeclared real KV
capacities of 9,216 and 10,752 blocks (7.875 / 9.1875 GiB) bracket that
boundary. This changes engine KV tensors, not a policy quota, and preserves
all 424 inputs/arrivals and the original fixed candidate universe.
Serving must establish whether the new points actually bind. The original
normal evidence and untouched held-out region remain intact.

## Persistent evidence and execution state

- `r2-native-opportunity/`: controlled native counterexample to the input model.
- `r2-serving-opportunity/`: actual four-trace outcome, source and commands.
- `r2-realized-feasibility/`: exact native replay outcome and qualified gate.
- `r2-block-marginal/`: measured physical-copy exposure and attribution limits.
- `r2-value-policy/`: tested production candidate, formula and native readiness.
- `r2-physical-pressure-01/`: complete 16-trace discovery, negative outcome and missing-exposure defect.
- `r2-current-admission-gpu/`: complete eight-trace primary screen, measured exposure and failed BG gate.
- `r2-reference-density/`: complete common-reference ranking, waste rates and interpretation limits.
- `r2-stable-admission/`: completed soft-grant refinement, exact conditional work/victim evidence and insufficient gain.
- `r2-history-matched/`: online history signal, frozen matched controls, failed first pair and forced-victim attribution limit.
- `r2-hard-pressure-prefix/`: native usable-prefix fix, exact forced-loss join, final native regressions and failed workload screen.
- `r2-frozen-upper-pressure/`: unchanged-policy upper-point pair, exact tape support and insufficient selective/waste gain.
- `../../docs/controller-admission-protocol.md`: complete unchanged success gates.

Full serving raw/compact evidence remains at
`/home/ubuntu/experiments/octopus/r2-serving-opportunity-20260912/`;
replay evidence and source bundle are at
`/home/ubuntu/experiments/octopus/r2-realized-value-20260912-d3c687/`.
The serving diagnostic consumed 1,150.644302 GPU process-seconds, with
96.7338% active utilization across its two allocated GPUs. All four traces
completed and the supervisor confirmed zero owned GPU processes.
The tested native discovery source is `3e7cf232f8aa254fd63712d9d75fa6fb486cf1be`
(library `bd7dd378`). The supervisor verified 15 matching source files,
four-arm runner/export wiring and eight exact-byte EngineArgs tests.
The CPU fixture retains 1,797 of 2,119 future copies at 9,216 blocks and all
2,119 at 10,752 blocks; actual concurrency, binding and usefulness remain
unverified. Source-relocated input freeze `0cfcac3b` preserves all 424
requests, token IDs and intended arrivals.

The supervisor is authorized to execute the prepared 16 traces at that source,
pair methods on the same physical GPU and parallelize independent groups on
available cards, with the unchanged 600+10-second watchdog. At 11:52 UTC
the supervisor confirmed all four groups running on GPUs 0--3 (native process
IDs 1560808, 1560811, 1560809 and 1560810), using 24,400 / 25,976 MiB per
lower/upper-capacity process. The terminal serving checkpoint is 16 valid
complete traces, zero failed/running traces, 4,563.827416 GPU process-seconds,
16 officially scored traces and zero owned GPU processes. Active-campaign
utilization is 95.584890%. Full artifacts are at
`/home/ubuntu/experiments/octopus/r2-physical-pressure-discovery-20260912/`.
Complete-cohort official scoring stays on the isolated CPU host after inference.

The supervisor subsequently reports all 16 serving traces complete, but the
source-relocated freeze omitted the required live FREE-exposure switch.
All original raw policy draws, free states and adoptions are retained, as are
the compact exports with explicit `not_instrumented` exposure. Missing
measurement cannot be treated as zero, a policy failure or R2-B acceptance.
No rerun is authorized: first inspect already-recorded binding, savings and
performance, while the implementation owner checked whether exact
exposure could be recovered from actual recorded lifecycle times. A native
counterexample now rules out exact recovery: identical emitted records permit
20 versus 24 block-seconds because intra-allocation membership timestamps are
missing. Recorded-end replay is a coarsened quantity, not a replacement live
measurement.
This is a real configuration defect in the attempted experiment.

The completed analysis records 603 / 976 prospective DROP node decisions at
the lower point. Fixed and value recompute the same 9,648 / 15,616 FG parent
tokens there, preserving 24,240 / 18,272 tokens versus both controls; both
recompute zero at the upper point. All 16 traces recompute all 13,376 recurring
BG body tokens. Fixed/value eviction counts and output texts match, but that
does not by itself establish identity of all victims or trajectories.
Two-seed mean value-minus-fixed BG P95 changes are +0.187764% / +0.822540%;
the per-seed directions split. These are discovery descriptions, not formal
confidence tests. The failure to demonstrate R2 benefit remains separate
from the measurement defect.

All 1,579 DROP observations in the 18 binding draws are actual victims of that
same draw, with no surviving unpreferred DROP or selected KEEP copy. The
observed rule is choosing unavoidable current victims, not demonstrating
persistent protection reduction. The dedicated helper now always captures its
required exposure at library `9101fc7` / native repair `62fe9c3b`, with a
reproduced red wrapper and three affected native green checks. Completed
source `3e7cf232`, inputs and raw files remain unchanged. No production policy
change or GPU rerun is authorized by that repair.

Changes are committed in
`worktree/tommy-dev-linux-20260912-030832-34c6`; no new PR has been opened
or merged. The active worktree and negative evidence are retained.
The latest documented pre-push check and its prior comparison are recorded in
`delivery-check.json`: 129 checks pass and one fails on the same 5,567
classified task-ID substring matches, with zero unexplained matches and no
tracked `.env` or `auth.json`. Four environment warnings remain. This is not
a globally green verifier claim.
R2 is not accepted. The configuration repair and recovery diagnosis are
complete; R3/R4 stay blocked on useful, measured admission rather than
being launched after an uninformative or failed discovery.

The owner renewed autonomous implementation at 13:29 UTC. The RUNNING-prefill
commitment variant is implemented and native-tested at `d9f96f6` / `54bc5489`,
but its two recorded lower-pressure conditional streams change zero work
records, recover no BG body tokens and show no qualifying cross-wave state.
Its native fixture is therefore not a workload acceptance result.

The current-admission refinement is now native-tested at `230424c` /
`807d46de`: include the fit-checked current request's prompt residual even
before its WAITING status changes, without forecasting other waiters.
Two conditional streams now have meaningful cross-step decisions and retain
94.785% / 78.371% of fixed savings, but add 1,264 / 3,952 FG prefill tokens
and recover zero BG body tokens. This is not a demonstrated efficiency gain.

Exactly eight new GPU traces are authorized at the primary 9,216-block point,
four fresh arms and two paired seeds, with mandatory live exposure and official
scoring. The predeclared screen requires summed exposure reduction, negative
mean paired BG P95 change versus fixed and per-seed >=75% positive matched
savings; R2-B still requires valid ranking and unnecessary-protection evidence.
The second point and formal stages wait for this strongest actual falsifier.
All completed negative evidence and the original full completion gate remain
intact. See `r2-current-admission/report.md`.

The new eight-trace campaign is now complete, validated and officially scored.
Required live exposure is present in every trace; summed exposure falls
705,079.688922 to 655,811.034380 block-seconds. BG P95 changes versus fixed
are -0.631990% / +2.747423%, failing the predeclared negative mean screen.
GPU cost is 2,254.496098 process-seconds with 97.481590% two-card active
utilization. Zero owned GPU processes remain; all 4,369,300,675 raw/sidecar
bytes are archived.

The common-reference reducer is complete at `b00964ac` / analysis `3fa39c6`.
It finds valid positive reference exposure for every body unit; first-binding
KEEP/DROP counts are 2084/34 and 2042/76, with positive KEEP and zero DROP
reference density against both controls. Prior-started reference benefit is
zero. This diagnostic association does not erase the measured increase in
actual no-saving exposure/rate or the independently failed BG screen.

The CPU-only lifetime refinement is complete: retain online KEEP soft grants
through forecast-only changes, while preserving native forced eviction,
queued precedence, semantic cleanup and fixed nonbinding behavior. The
two-stream cheap target is to remove the extra FG work and quantify genuine
protection/waste improvement, with no reference labels or producer identities
in policy. No GPU, upper-pressure or formal expansion is authorized.

The native red case reproduces forecast-only withdrawal: four online KEEP
copies become zero while the same semantic promise remains live and actual
native draw room is 14. Only genuine-binding KEEP creates a logical grant;
nonbinding fixed selection and physical recaching do not. At library `53ec5a2`
/ native `5b1d8aef`, both streams match actual fixed's parent savings and add
zero BG prefill work. Only seed1702 omits postdecision protection:
35.604030 block-seconds, all positive-commitment and zero-fixed-reference-benefit.
The pooled same-state-shadow reduction is 0.005052633%, not an actual GPU
live-exposure reduction. Exact same-prestate victim differences occur in
1 / 3 draws, without BG recovery or FG work improvement beyond fixed.

Nine new focused native cases and 25 existing changed-surface regressions pass
on the same source; no source fixes or repeated new cases were needed.
The offline conditional executions consume
44.500862 CPU seconds, including the preserved preliminary1601 execution
before adding exact victim evidence. No GPU was used. The parent accepts
the mechanism and completed evidence but not GPU readiness or useful R2 gain.
The scientific blocker is now a value signal that produces a meaningful
retention/background benefit while preserving useful work; no such candidate
has been demonstrated within the tested information and pressure settings.
This is not a universal impossibility claim. Do not rerun unchanged GPU
traces, invent another variant or mark R1--R4 complete.

## Information-matched and native pressure failures

The renewed next scope uses an existing positive diagnostic, not another
forecast parameter: the earlier history-union all-KEEP control recovered
13,376 BG body tokens without losing FG savings. It still did not establish
selective admission. First check whether its exact native-visible history
signal is available before useful decisions in the actual lower-pressure
streams; absent valid online evidence, stop.

If supported, compare one predeclared selective rule with a history-aware
blanket control sharing the same information, retaining original semantic
fixed separately. Require preserved matched savings, recovered BG work and
a useful reduction in protection/waste versus the information-matched
control, not history-only recovery or another tiny nonzero effect.
The signal is available, but the first seed1601 blanket/selective outcomes
retain only 44.818482% / 48.382838% of original fixed's FG savings.
Selective also adds 864 BG tokens versus blanket and raises conditional
protection 0.182226%. Both have zero persistent postdecision omission.
Nine native cases pass; total measured CPU cost is 81.478118 seconds.
The second seed and all GPU work were stopped without tuning.

The retained source confirms that unavoidable granted-block victims use native
LRU, not marginal value, and records 1,351 forced grant-copy evictions.
It does not identify which of the extra FG tokens those events caused.
The next bounded CPU scope first tests whether forced head reclamation loses
more usable prefix than valid tail reclamation under the same real native
shortfall. A supported fix may project soft preferences using the existing
prefix-closed value rule only at actual native shortage; forecast-only grants,
weights, history information and original baselines remain unchanged.
The native red is now repaired: at FREE15/draw14/grant4/native-room1, valid
tail reclamation retains and actually adopts 16 prefix tokens instead of zero.
The single selective1601 outcome at `f0822e4` / `94eb1baa` restores all 24,240
FG saved tokens but recomputes the entire 13,376-token recurring BG body and
adds 15,021.290989 conditional block-seconds (+3.710406%) versus blanket.
Nine new and 16 previously unrun existing changed-surface native cases pass;
outcome processing costs 23.350271 CPU seconds, with serialization untimed.

The exact 17-draw join records 2,180 hard-demoted selected copy occurrences,
zero discretionary/queued/still-preferred victims in those draws and zero
missing-ancestor cascades for both actual projection and old-LRU shadow.
It shows direct loss moving from semantic to historical paths, not a hidden
recorded-stream cascade explaining earlier failures. The native synthetic
mechanism and the workload result remain distinct.

These lower-point CPU scopes are terminal. The remaining blocker is scientific:
no tested candidate has demonstrated the required simultaneous FG preservation and
useful retention/background improvement. R3/R4 remain unrun, their held-out
inputs untouched, and the full objective must not be marked complete.

The renewed next scope changes no policy or pressure setting: first verify
support in the original16 campaign's existing upper seed1601 tape, then at
most one matched blanket/selective pair with frozen `94eb1baa` behavior at
10,752 blocks. This checks the other requested pressure point for the final
family, rather than extrapolating the tight-point failure. Use that point's
actual fixed savings denominator of 33,888 if confirmed. Missing live exposure
remains missing; only conditional endpoint integration is permitted.
No seed1702 outcome, new capacity, interpolation, variant or GPU is authorized.

That upper comparison is now complete, using unchanged `94eb1baa` policy and
readers. Exact support covers 5,056 native allocation/schedule pairs and all424
requests. Blanket/selective both recover 13,376 BG tokens; selective retains
33,712/33,888 FG savings but adds176 FG work tokens, reduces conditional
protection only0.189531%, and leaves no-saving exposure119,869.833364
block-seconds unchanged. One forecast-binding decision and276 cross-step DROP
occurrences are real; hard projection never activates. This is not an
inactive-budget failure, but useful selective/waste gain remains unestablished.
Measured CPU cost is73.750437 seconds; no unchanged native tests or GPU ran.

The frozen family has now been checked at both existing physical pressure
points, without changing formulas, inventing a third point or touching held-out
data. Those CPU scopes are terminal, no useful GPU matrix is ready,
and the full R1--R4 objective remains incomplete.

The next bounded scope checks the specific lifetime mechanism behind retained
historical-only protection after its next actual use. If source/native evidence
supports it, test resolving a speculative next-reuse hint on consumption rather
than retaining epoch-long positive history support. Preserve all observations
for the matched blanket, allow later fresh support episodes, and require native
multi-consumer/reopening coverage; no request-count, time or producer hint.
Zero-supported historical-only rejection would be an explicit value-only
change even when capacity is nonbinding. One upper1601 selective outcome is
the limit after a supported native red, with useful waste/protection reduction
and FG/BG preservation required before any further execution.

That scope is now terminal and its first conditional screen passes. The
source-supported native red had eight stale FREE preferences with no semantic
consumers at a subsequent real draw. Twelve new and19 existing native cases
pass, including multi-consumer reopening and actual-use resolution. All423
confirmed history observations remain equal to the unchanged blanket control.
At upper1601, protection falls623,286.998988 to503,417.165624 block-seconds,
while FG savings remain33,888/33,888 and BG body recompute remains0. Shared
history information accounts for the13,376-token BG recovery over original
semantic fixed; it is not a selection-specific recovery.

The prospective ledger distinguishes836 initially kept historical blocks
later resolved/reclaimed from2,118 blocks already resolved at their first
history-FREE decision. It does not call the latter an initial value ranking.
There are45 nonbinding explicit rejection draws, six same-prestate victim
differences and836 unique victims. Active post-resolution omission is
4,297.512792 block-seconds; total same-state omission is6,668.457975, distinct
from whole-trajectory protection reduction. Ranking/hard-projection draws
are both zero. Outcome processing costs31.868990 CPU seconds, with0.559418
seconds of separate serialization. The missing original live observer remains
missing; these endpoint integrals do not recover actual serving exposure.
Actual serving integration, prospective group interpretation, another
discovery condition and latency/quality gates remain open. No GPU ran.

Existing-label clarification: all4,236 matched blanket cached-use occurrences
for first history-FREE DROP units happened before those labels; KEEP units
have836 before and836 after. Per-group matched benefit numerators/densities
were not exported. Earlier semantic benefit therefore cannot establish later
DROP usefulness, and this screen does not supply a new protocol-valid initial
value-ranking result. `HistoryAdmission(scheduler, "next_reuse")` currently
wraps native `_record_scheduled` only when installed by the conditional helper;
actual serving and its historical-union live observer are still unwired.

The same owner now holds a bounded continuation: unchanged upper1702
blanket/selective replication, then, only if promising, one lower1601
selective pressure characterization and actual serving integration with a
focused native gate. New prospective support-state records must remain
distinct from original initial-label groups. No new formula, capacity,
held-out seed or GPU is authorized in that scope.

The unchanged upper1702 pair now supplies the decisive stop. Blanket/next-reuse
FG savings are30,432/32,656 of33,888, and BG body recompute is0/2,224.
Protection falls614,817.609339 to492,765.050398 block-seconds (-19.851832%);
no-saving exposure falls141,004.848534 to6,352.928834 (-95.494532%), and its
rate falls22.934419% to1.289241%. These improvements do not waive the
predeclared no-extra-BG-versus-blanket gate. Outcome processing totals121.572501
CPU seconds, plus1.473790 seconds serialization and3.885968 seconds packaging.
Measurement snapshot593c5d45 leaves the scientific7db lifecycle unchanged.
The terminal source is `0a0ff601`; complete new evidence and the pre-outcome
analysis plan are in `r2-resolved-history-followup/`. Three new measurement
tests pass; the accepted31 native lifecycle cases were not repeated.
No lower807 outcome, serving wiring, new candidate or GPU ran, and no owned
job remains. This is a real replication blocker, not completion.

| Upper1702 conditional method | FG parent recompute | BG body recompute | Whole FG prompt tokens | Whole BG prompt tokens | Whole prompt total |
|---|---:|---:|---:|---:|---:|
| Original semantic fixed | 0 | 13,376 | 34,530 | 1,997,442 | 2,031,972 |
| Information-matched history blanket | 3,456 | 0 | 37,986 | 1,984,066 | 2,022,052 |
| Frozen next-reuse | 1,232 | 2,224 | 35,762 | 1,986,290 | 2,022,052 |

All2,954 original first-eligible labels are KEEP (2,118 semantic,836 history);
there is no initial DROP ranking. Later pending-history DROP contains259
support-state episodes on190 units,139 misses and2,224 matched-blanket
reference-benefit tokens, with zero prior benefit in this group. Thus the
failed replication records actual useful future reuse among the dropped
support states, rather than proving they were all valueless.
FREE-state handoff gaps exclude30,432 semantic-benefit tokens from state
intervals, but not from the preserved full decision-onward references.
Full/prior/onward references remain separate; state-only totals must not
replace complete reference benefit or serve as a favorable density shortcut.

The next source-supported checkpoint isolates KEEP continuity. The139 BG
blocks are discretionary forecast DROP victims at attempts971/998, not resolved
hints or hard-native demotions. Two native cases reproduce the behavior for
both semantic and historical support. The tested new `e869df19` policy grants
the same soft continuity to positive nonbinding KEEP as to binding KEEP,
without changing support lifetime, physical ownership or hard reclamation.
All24 focused native cases pass; one predeclared upper1702 selective outcome
has now completed and still fails the matched-BG gate. It saves31,376/33,888
FG tokens (92.587347%), reduces conditional protection21.551285% and no-saving
exposure to2,783.940171 block-seconds, but recomputes944 BG tokens versus
blanket0. Forecast ranking draws fall to0; all216 positive selected losses
are hard-projected (59 BG,157 FG). Both arms still total2,022,052 prompt
tokens: this is a work allocation change, not measured performance acceptance.

The earlier absence of hard-demotion metadata in old forecast decisions did
not establish that all supported candidates fit actual native room; only the
small native fixtures established that property for their own states.
The observed workload now exposes the remaining hard-capacity tradeoff.
Pending-history DROP retains higher reference benefit density than KEEP
(2.137280 versus0.285500 mean), and original labels remain allKEEP.
The result and its full prospective-reference limitations are in
`r2-keep-continuity/upper1702.json`. Processing costs58.172565 CPU seconds,
plus0.791846 serialization. No further variant, GPU or serving integration
ran in this scope; no owned job remains. Full R2--R4 acceptance stays open.

The next same-prestate probe observes actual shortage of59 blocks at draw971:
added W projection takes59 historical blocks, while native tier-LRU takes59
semantic blocks on the identical queue/candidate state. Removing only that
extra next-reuse projection produces the first passing upper1702 screen:
30,432/33,888 FG tokens saved, BG body0, protection474,287.482134 versus
blanket614,817.609339 block-seconds, and no-saving exposure474.721329 versus
141,004.848534. Both methods retain identical total prompt work and all423
confirmed history observations. Seven later same-state victim differences
and836 resolved selected copies establish actual effects, not idle relabeling.
The216 forced native KEEP victims are exported separately.

Pending-history KEEP has836 later hits and13,376 reference-benefit tokens;
resolved-history DROP has no later use or benefit. Its mean lifetime-episode
reference density is0.307178494 versus0, but all2,954 original labels remain
KEEP. This is not an initial-ranking or full causal R2-B result. Original-fixed
protection and actual FG/BG latency still need comparable serving evidence.
The one conditional outcome costs57.274086 CPU seconds plus0.765148
serialization. A separate owner is now integrating the unchanged policy into
actual callbacks, controller/runner construction, live observation and exports;
no GPU is authorized by that integration scope alone.
