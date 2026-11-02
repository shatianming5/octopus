# Frozen history serving integration

**Outcome: executable readiness established; no serving-performance or R2/R3/R4 acceptance.**
Scope `octopus-r2-history-serving-integration-20260912`, managed worker
`bb698aa2-88a0-4fc3-a2d8-c4f514f0a056`. No GPU, inference, target gold,
additional worker, extra upper outcome, second lower arm or capacity was used.

## Source, configuration and callbacks

The real native source is committed at
`95481254b87a8bf9b0ffe5fc84f511bc97ebbde0`, based on the accepted scientific
rule `cb86ca8cfdd6627488b185266d3bb935850d4a2d`. It lives at
`/mnt/disk2/user/octopus-contention/r2-history-serving-integration-20260912-bb698a/native-source`
on `ssh -p 20068 user@198.51.100.10`. The native runner is included in that
commit; `native-runner.patch` preserves its change for this smaller library checkout.
The incremental `evidence/native-integration.bundle` beside that checkout
preserves the native commit relative to cb86.

`octopus.history_admission` is the runtime implementation; the old
`benchmarks.history_candidates` imports re-export it. Serving constructs
`b6_semantic_value` with `value_history_mode="next_reuse"` before scheduling,
through the explicit `candidate_parameters` record in `benchmarks.history_serving`.
The original APC, queue and semantic-fixed policies retain their behavior.
`b6_history_blanket` is a separate epoch-history mechanistic control, not
semantic fixed and not the old selective history policy.

`OctopusScheduler._record_scheduled` delivers a committed adoption directly
to the policy, independently of the optional adoption audit list. The policy
still excludes probes, duplicate initial observations and preemption episodes.
Positive initial cached adoption opens support; the actual next initial hit
or miss consumes it; no same-event renewal occurs. Reset clears history,
episodes and grants. Selection/grant/native-LRU semantics remain frozen.

## Live observation and export

The observer covers the semantic/history candidate union, including native
queued, held, free and lifetime transitions. Initial eligible labels, support
episodes, value rejections and **native-forced KEEP victims** are distinct.
Forced physical loss is not reclassified as admission DROP.

Raw/compact exports retain actual committed schedule timestamps/ranges, full
prompt logical-block correspondence, consumer/checkpoint identities and
semantic lifecycle records. They therefore allow full, prior and onward
reference association without clipping benefits to FREE intervals. This is
export readiness, not a claim that the parent's full reference evaluator or
causal R2-B analysis has passed. Requested but missing live observation fails
explicitly; no zero or conditional-endpoint replacement is supplied.

The serving loop process-CPU timer excludes deferred export and serialization.
History callback/reset maintenance is included in prefix-maintenance accounting;
selector/provider work remains in existing nested timers. Observer work has its
own CPU counter. These overlapping counters must not be summed.

## Focused native gate

`tests/test_history_serving_integration.py tests/test_r2_serving_opportunity.py`:
**14 passed, 6.94 seconds**, no skips. This uses actual `VllmBackend`
construction and actual native scheduler construction/add/schedule/free hooks;
only model execution is replaced with deterministic CPU fixture outputs.
It covers all five methods, semantic consumer adoption, reset, probes,
duplicates, preemption exclusions, native forced KEEP loss, truthful live
timestamps/exports and identical policy outcomes with observation off/audit off,
observation on/audit off, and observation on/audit on.

The initial gate had four expected metadata-comparison failures because
requested instrumentation is now explicit. The comparison was corrected to
exclude measurement metadata, preserving the same selection/work comparison.
The final gate and its earlier logs remain under the remote `evidence/` folder.

The seven-request synthetic value fixture used 5.386-5.498 ms loop CPU:
0.0562-0.0567 ms history maintenance, 0.499-0.635 ms selector work,
2.352-2.539 ms live-observer CPU, and 0.112-0.222 ms deferred live export.
Original fixed used 3.773-3.892 ms loop CPU. These are tiny CPU fixture costs,
**not actual serving latency, a campaign CPU budget pass, or a performance comparison**.

## Retained upper1601 and one lower boundary

The seven-db upper1601 evidence (`7db3e73af1aa47a5d3484f5f11c1bb0e966613fa`)
is reusable for its conditional selection/work outcome: binding, ranking,
hard projection and native-forced preferred events were all zero. The new
nonbinding continuity bookkeeping cannot change that KEEP set, and removal
of an unexecuted hard-projection branch cannot change its victims. The
committed initial-observation rule is unchanged. Grant/observer/export
records and CPU cost are not equivalent. Its old 31.868990072 s processing
time is **not** current-source timing. No extra upper outcome was run.

Exactly one lower9216/seed1601 next-reuse outcome ran on actual
`807d46de552db10e274d2910e3676b1e62bd9934`, with the identical retained
`9ee24e12c619f22ff85bbf3ce527328dc89a8d84` blanket and actual fixed denominator:

| Quantity | Conditional result |
|---|---:|
| FG savings / actual fixed savings | 15,392 / 24,240 = **63.49835%** |
| 75% savings screen | **FAIL, preserved** |
| FG parent recompute | 18,496 tokens |
| BG body recompute / difference from blanket | 0 / 0 tokens |
| Protected mass: blanket to candidate | 404,842.267447 to 314,342.353450 block-seconds |
| Protection change versus history blanket | -22.35436% |
| No-demonstrable-saving mass: blanket to candidate | 116,479.449131 to 2,224.212728 block-seconds |
| Binding / ranking draws | 15 / 0 |
| Resolved selected / native-forced KEEP copies | 836 / 1,156 |
| Hard-projection selected copies | 0 |
| Initial labels / support episodes | 2,954 / 5,908 |
| Processing CPU / separate serialization CPU | 128.195930 / 0.888680 s |

The native pool was restored. Observed history differs endogenously from the
blanket, and positive-commitment omission remains 3,593.987332 block-seconds.
This lower point is a conditional boundary failure, not an invented requirement
that every development pressure pass formal latency. Lower807 and upper3e are
different source schedules, not paired serving or R4. Reduced protection against
the history blanket alone does not establish improvement over original fixed.
Earlier e869 (944 BG tokens) and seven-db1702 (2,224 BG tokens) negatives remain
unchanged; accepted cb86 upper1702 conditional evidence is not promoted to
actual serving acceptance.

## Exact artifacts and next owner

All paths below are relative to the remote integration root:
`/mnt/disk2/user/octopus-contention/r2-history-serving-integration-20260912-bb698a/`.

- `launch-inputs/plan.json`: ten exact upper10752 launch command/selector pairs,
  five methods, seeds1601/1702, 9,865,003,008 KV bytes; no inference launched.
  Both `native_source_commit` fields name **95481254**, not historical cb86.
- `launch-inputs/five-arm-freeze.json`: discovery execution input.
  `launch-inputs/four-arm-freeze.json`: identical-candidate four-method template
  for the parent; this worker did not modify formal gates or choose R3/R4 seeds.
- `evidence/{method}-audit-{False,True}.raw.json` and `.compact.json`:
  the ten exact seven-request native fixtures; all filenames and CPU costs are
  enumerated in `integration-summary.json`.
- `evidence/lower1601-next_reuse.json`: full conditional raw outcome;
  `evidence/lower1601-next_reuse.json.write-cpu.json`: serialization cost.
- `evidence/integration-summary.json`: compact integration/upper-reuse/lower
  evidence, also persisted in this repository beside this report.
- `serving-raw/{trace_id}.raw.json` and `serving-compact/{trace_id}.json.gz`:
  planned actual-serving outputs, **not existing results**. The plan has each
  exact command and path. The supervisor must assign an authorized free GPU.

The parent owns protocol, formal-plan integration, quality/reference evaluation
and the decision to authorize actual serving. This scope leaves **zero owned
jobs and zero children**; its completion does not complete the parent R1-R4 goal.

Publication verifier: one invocation, 125 pass / 7 fail / 4 warn / 2 skip;
**not globally accepted**. All 5,567 captured credential-pattern matches were
classified as unchanged historical task identifiers, with zero changed-file
or newly staged-file hits. The other six failures concern the shared Herdr
installation and were not changed. `publication-check.json` preserves details.
