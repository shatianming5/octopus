# QASPER: completed E1 and separately authorized E2 diagnostic

**Terminal result: real native KV reuse, but failed quality and performance
claims.** E1 passed the pressure check and failed the original strict QA entry
gate. The user subsequently authorized E2 explicitly at
**2026-09-11T21:25:56+00:00** despite that known failure. E2 completed all
16 traces; it does not turn E1 into a pass or establish an end-to-end Pareto,
budget/admission, or full-Agent benefit. No further experiments are pending.

## E1: original no-go, retained as history

Both APC capacity points completed 422/422 requests near full physical VRAM.
Each had **2,118 actual target-block evictions and 33,888 full-prefix
recomputed tokens**, rather than cold-prefill or unaligned-tail counts.
Normal KV was **33,198,047,232 bytes**; stress was **28,902,293,504 bytes**.
Normal was chosen by pressure only, not latency or quality.

The unmodified official QASPER evaluator assigned full Answer F1 to
**130/422** requests at each point (**1/12 FG**, 129/410 BG), mean Answer F1
**0.4649228157982938**, missing predictions **0**. Identical outputs at both
capacities did not substitute for correctness. The original E2 entry decision
remains **no-go**. [E1 report](e1/report.md),
[original terminal snapshot](e1/terminal-summary.json), and
[original authorization and terminal decision](e1/protocol-authorization.json)
retain their historical `e2_allowed: false` state.

## E2: complete diagnostic, not original pipeline acceptance

[Separate authorization](e2/authorization.json) and [plan](e2/plan.json):
four arrival seeds (1601/1702/1803/1904), each with APC, queue-aware,
B6 semantic fixed and budgeted, using identical input tokens, same-seed
arrival schedules and **33,198,047,232 KV bytes**. Each seed's four methods
ran on one physical GPU, in rotated order: GPU0 for 1601/1904, GPU3 for
1702, GPU6 for 1803. All **6,752 request executions** completed; no failures
or timeouts occurred. Fixed is the predeclared primary, not a post-hoc winner.

| Method | Mean FG paper E2E (s) | Pooled BG response P95 (s) | Parent recompute tokens/trace | Parent adopted tokens/trace |
|---|---:|---:|---:|---:|
| APC | 247.3354751 | 49.2058898 | 33,888 | 34,016 |
| Queue-aware | 253.6576897 | 54.2509976 | 33,888 | 34,016 |
| B6 semantic fixed | 256.4574391 | 58.2275725 | 0 | 67,904 |
| B6 semantic budgeted | 256.9227312 | 58.4530122 | 0 | 67,904 |

Fixed FG was **+3.688% vs APC**, **+1.104% vs queue-aware**; the paired
one-sided upper bounds were **+10.546709 s** and **+3.398669 s** respectively,
not below zero. Fixed BG P95 was **+18.3346%**, upper **+20.3713%**;
budgeted was **+18.7927%**, upper **+21.0066%**. Both fail the inclusive
**+5%** BG gate. [Full diagnostic interpretation](e2/report.md),
[per-seed/per-question evidence and comparisons](e2/analysis/report.json),
and [aggregate summary](e2/aggregate-summary.json).

FG is each paper's first intended question arrival to final question
completion, averaged over four papers, then paired over four seeds. It
includes planned revisit offsets at **221 and 237 seconds**, not just one
model response or mean successor latency. FG uses df=3, one-sided
Student-t critical 2.3533634348018264. BG pools all four fixed seed strata,
with paired arrival-order circular blocks of 8, 1,000 resamples and RNG17.
TTFT, queue-aware BG and individual-seed tails are diagnostics, not extra
hard gates.

Fixed/budgeted eliminate parent recompute and double parent adoption **in
every seed**, but global prefill only falls from 2,052,394 to 2,018,506
tokens (about **1.65%**). Mean adapter CPU seconds were APC **1.120**,
queue **10.667**, fixed **18.529**, budgeted **18.885**. Nested selector and
preference-provider timers must not be added to adapter time. These
observations suggest control/selector overhead offsets the prefill savings;
they are not a causal overhead ablation. Budget binding was **0 in all
traces**, so no admission benefit is established.

QA full credit remains **129-130/422** per trace, mean Answer F1 about 46.4%.
Same-seed output text/token parity versus APC is **420-422/422**; all FG
outputs are identical, but a few BG outputs differ in wording/content.
There is no proven floating-point explanation for those differences.
Empty predicted evidence and nonzero official Evidence F1 do not establish
evidence extraction.

## Cost, provenance and data boundary

E2 used **4,656.746323908446 GPU-process seconds**; including prior E1 and
invalid startups (584 s), the observed total is **5,240.746323908446 s**,
below 10,800 s. Execution-event wall was **2,324.44016 s**, not the sum
across cards. Each E2 GPU was observed at **48,016/49,140 MiB, 100%**
during startup; these are snapshots, not a GPU-busy integral. Preparation,
debugging, transfer and CPU scoring are not a complete project-wall bill.
E1 includes 19 s and 18 s of failed pre-answer startups plus 547 s for the
valid two-trace sequence. All GPUs were released and archival work finished.

Measured native source is **616ae10e3f00e250e2744ef0ebd6ce02307baac6**,
preserved in the **32,366-byte** [Git bundle](source/qasper-native-616ae10.bundle)
requiring base **d2a8c6550ac73a507b03790f024a132068d16b5e**.
This delivery does not install a new native harness or promote unmeasured
candidate30c/258/17d/7c implementations.

The **1,439,029-byte** [compact export](e2/e2-export.json.gz) contains
request outputs/timing and validated native counters, but no input token
arrays, full corpus or gold. The owner completed and verified the external
**284,672,370-byte** archive of **10,322,028,866 raw bytes**:
SHA256 `979c5aa2297649e48f824d8a280e90b50d2015e77f3d547372b1b29bb668c79e`.
All 16 raw hashes matched the export in the owner's streaming validation;
publication does not repeat that 10 GB pass.

[Source provenance](source/provenance.json),
[input summary](input-summary.json), [large-artifact manifest](large-artifacts.json)
and [checksums](SHA256SUMS) distinguish tracked compact evidence from external
inputs, full raw, gold, archives, model weights and environments. Paths in
historical JSON identify the original machines, not files promised inside Git.
The E1 postprocess record's old worktree path is historical; the permanent
copy is authoritative. Full E1 payload and full E2 archive persist locally,
with E2 raw also retained on `.68`; access requires the owner's storage,
not a public download guarantee.

See [reproduction instructions](../../docs/qasper-kv-evidence.md) for offline
CPU report reconstruction, optional official regrading, exact-source fetching
and archival execution provenance. New GPU runs are not part of this delivery
and require separate authorization.
