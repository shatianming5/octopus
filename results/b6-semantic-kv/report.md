# B6 terminal result: qualified semantic-fixed go

**`b6_semantic_fixed`, 1536 MiB / 4 rps, passes the frozen formal KV gate.**
Confirmation FG intended-arrival E2E is **60.338827 s**, versus **62.478659 s**
for equalized Projected APC (-3.4249%) and **61.971169 s** for the strong B6
queue baseline (-2.6340%). Both paired one-sided 95% upper bounds are below
zero; all four BG point/upper bounds are below +5%. The 10% FG stretch is not
met. **B7 eligible = true; B7 authorized = false.**

This is a bounded result on the same twelve development templates and four
independent confirmation traffic seeds, not a held-out task benchmark or a
distribution-wide serving guarantee. **The GPU budget cap never binds**:
there is no measured budgeted-over-fixed benefit. No further experiment,
retuning, model run or B7 implementation is part of this delivery.

## Identity, completeness and quality

Measured source: `4d00650be7143c5aa4d32045c7da6d28a261fd02`.
Qwen2.5-7B-Instruct model/tokenizer revision:
`a09a35458c702b33eeacc393d103063234e8bc28`; vLLM 0.28.0;
physical GPU `GPU-c5792743-87d0-80a0-8a3e-876d31e854d9`.
BF16, length 16384, max sequences 8, batched tokens 8192, block 16,
generation seed 17, eager/synchronous APC. Frozen launch status strings
are preserved, not changed to masquerade as terminal evidence.

All **40/40 planned traces** completed: 24 discovery + 16 confirmation,
with zero retries, nonzero process exits or timeouts. Exact source/freeze,
rotated/Williams method order, engine controls, prompt/context/schema IDs,
four-branch fixed work, intended arrivals and full paired BG cohorts match.
Full shares all nonprojection controls but is not a core KV arm.
CPU post-run replay of `discover` and `confirm` equals both original reports
exactly. Final-answer and existing `validate_commit` replay agrees with all
480 stored task scores and 1,440 evidence outcomes.

| Phase / method | Traces | FG correct / planned | Required evidence / planned | BG complete / planned |
|---|---:|---:|---:|---:|
| Discovery / Projected APC | 6 | 72/72 | 216/216 | 1440/1440 |
| Discovery / B6 queue | 6 | 72/72 | 216/216 | 1440/1440 |
| Discovery / semantic fixed | 6 | 72/72 | 216/216 | 1440/1440 |
| Discovery / semantic budgeted | 6 | 72/72 | 216/216 | 1440/1440 |
| Confirmation / Projected APC | 4 | 48/48 | 144/144 | 1536/1536 |
| Confirmation / B6 queue | 4 | 48/48 | 144/144 | 1536/1536 |
| Confirmation / semantic fixed | 4 | 48/48 | 144/144 | 1536/1536 |
| Confirmation / Full APC (attribution only) | 4 | 25/48 | 125/144 | 1536/1536 |

Projected total: **432/432** tasks, **1296/1296** evidence outcomes.
All **11904/11904** BG requests finish with 2048 input tokens and 32 output
tokens. BG correctness is serving completion, not a semantic-answer oracle.
Full's 23 unsuccessful tasks are retained, not excluded or called process
failures. Every latency mean uses all twelve task slots, not only successes.

## Discovery selection and pressure

Traffic seed 1501; six predeclared points, four methods per point. The table
contains mean FG intended-arrival E2E in seconds, not pooled confirmation data.

| KV MiB / rps | Projected APC | B6 queue | Fixed | Budgeted |
|---|---:|---:|---:|---:|
| 1024 / 1 | 6.008322 | 6.124291 | 5.905477 | 6.041337 |
| 1024 / 4 | 60.567490 | 60.024235 | 58.472463 | 58.609834 |
| 1536 / 1 | 6.025249 | 5.951996 | 6.097047 | 6.061773 |
| 1536 / 4 | 61.060578 | 61.732721 | **58.175735** | 58.442216 |
| 2048 / 1 | 6.139063 | 6.153887 | 6.156812 | 6.176218 |
| 2048 / 4 | 61.823166 | 61.438471 | 59.672243 | 60.146474 |

The locked discovery-only winner maximizes the smaller eligible gain against
APC/queue: **0.04724558693850134** (4.724559%) at 1536 MiB / 4 rps.
The complete 12 candidate/point eligibility decisions and BG bounds are in
[`b6-selection.json`](b6-selection.json). Confirmation did not reselect.

| APC pressure point | Cache evictions | Native allocation rejections | Preemptions | Root recomputed tokens |
|---|---:|---:|---:|---:|
| 1024 / 1 | 15874 | 0 | 0 | 15776 |
| 1024 / 4 | 53695 | 112 | 0 | 21536 |
| 1536 / 1 | 14290 | 0 | 0 | 0 |
| 1536 / 4 | 52888 | 0 | 0 | 17952 |
| 2048 / 1 | 13704 | 0 | 0 | 0 |
| 2048 / 4 | 52303 | 0 | 0 | 17952 |

Thus pressure contrast is actual cache loss/recomputation, including native
rejections at 1024/4, not merely configured capacity or small nonzero waits.
At the selected discovery point, fixed has 1130 future-preference draws,
44032 root-adopted tokens and zero root recomputation; APC and queue each
adopt 26080 and recompute 17952 tokens.

## Formal confirmation: four paired traffic seeds

All means below are seconds per trace (twelve intended-arrival E2Es).
Full is shown separately for attribution and never enters the KV statistical gate.

| Seed | Projected APC | B6 queue | Semantic fixed | Full APC |
|---|---:|---:|---:|---:|
| 1601 | 58.830363 | 58.497433 | 58.018990 | 73.266473 |
| 1702 | 62.626725 | 62.336372 | 60.372565 | 77.594057 |
| 1803 | 61.230906 | 60.351253 | 58.873187 | 74.893251 |
| 1904 | 67.226641 | 66.699619 | 64.090567 | 78.595050 |
| Mean | 62.478659 | 61.971169 | 60.338827 | 76.087208 |
| Sample SD of seed means | 3.532337 | 3.520539 | 2.683682 | 2.445536 |

The independent unit is a traffic seed, **n=4**, not 48 independent tasks.
Paired Student-t one-sided 95% CI is `(-infinity, upper]`, df=3,
critical=2.3533634348018264; the frozen gate requires upper < 0 against **both**.

| Semantic fixed minus baseline | Mean difference (s) | One-sided 95% upper (s) | Relative mean gain | Gate |
|---|---:|---:|---:|---|
| Projected APC | -2.139831556 | -0.999431730 | 3.424900% | Pass |
| B6 queue | -1.632342162 | -0.575703314 | 2.634035% | Pass |

BG statistic: P95 intended-arrival TTFT of the full 384-request paired cohort
for **each** method/seed. Degradation is fixed/APC minus one. One-sided paired
circular moving-block bootstrap: block=8, resamples=1000, RNG seed=17.
Both point and upper must be <= +5% in every seed.

| Seed | APC P95 (s) | Fixed P95 (s) | Degradation point | One-sided 95% upper | Gate |
|---|---:|---:|---:|---:|---|
| 1601 | 19.771048 | 19.009680 | -3.850922% | -3.219437% | Pass |
| 1702 | 25.442279 | 23.675284 | -6.945115% | -3.814578% | Pass |
| 1803 | 26.033578 | 23.586338 | -9.400322% | -7.971877% | Pass |
| 1904 | 24.298827 | 22.514294 | -7.344112% | -6.179923% | Pass |

The SLO is relative, not an assertion that these absolute TTFTs are low.
Exact differences, bounds and quality gates are in
[`b6-formal-report.json`](b6-formal-report.json).

## Mechanism, budget participation and no-lease evidence

| Confirmation seed | APC / queue root adopted | APC / queue root recomputed | Fixed root adopted | Fixed root recomputed | Fixed future-preference draws |
|---|---:|---:|---:|---:|---:|
| 1601 | 22496 / 22496 | 21536 / 21536 | 44032 | 0 | 1120 |
| 1702 | 26080 / 26080 | 17952 / 17952 | 44032 | 0 | 1083 |
| 1803 | 26080 / 26080 | 17952 / 17952 | 44032 | 0 | 1086 |
| 1904 | 29680 / 29680 | 14352 / 14352 | 44032 | 0 | 1027 |

Fixed exercises **4316** future-preference draws and eliminates **71792**
root-recomputed tokens relative to either projected baseline. All selected-point
confirmation arms have zero native allocation rejections and zero preemptions;
the demonstrated pressure mechanism is cache eviction/root loss, not hard-pin
admission failure.

Across GPU discovery, budgeted makes **7896** native draws, including **5607**
with future preference, but has **0 cap-binding draws**. Fixed has 5613
future-preference draws. Neither semantic method reclaims an eligible future
key during its recorded draw. The exact headroom cap, leading-prefix selection,
key union and reclaimed-key accounting were checked on all **37678** B6
adapter draws. This validates participation/accounting, **not the binding-cap
benefit**; that distinction is demonstrated only by the previously accepted
CPU/native F16/future8/draw12 falsifier.

All **146990** physical pool samples show zero retention-only and zero shared
request/retention blocks; all traces record **0 retention block-seconds** and
every B6 draw records zero retention owners. Preference removal is not physical
release. No lease-revocation result or reserved-VRAM saving is inferred.

## Full to Projected attribution and costs

Full-to-Projected is a distinct context-projection comparison, not KV-policy gain.
Full solves 6/12, 6/12, 7/12 and 6/12 across seeds (25/48); Projected solves
12/12 in each (48/48). All Full BG cohorts finish and all nonprojection controls
match. Full latency/cost includes unsuccessful cases.

| Confirmation method | Mean FG E2E (s) | Mean actual service (s) | Mean application wait (s) | Total case-service / solved task (s) |
|---|---:|---:|---:|---:|
| Full APC | 76.087208 | 14.406454 | 61.680753 | 27.656328 |
| Projected APC | 62.478659 | 12.940414 | 49.538245 | 12.937216 |
| B6 queue | 61.971169 | 12.868676 | 49.102494 | 12.866695 |
| Semantic fixed | 60.338827 | 12.783851 | 47.554976 | 12.781904 |

Full-to-Projected mean E2E decreases **17.885463%**, alongside changed quality;
it is not an equal-quality speedup claim. Only Projected-to-fixed (-3.424900%)
is the APC KV attribution. Serial cases accumulate application wait, which is
included in E2E; fixed's actual-service reduction versus APC is about 0.157 s,
not the 2.140 s E2E reduction. The service/solved cost uses the recorded
`completion_seconds` timer; it starts slightly inside actual-start service.
Per-task nested phase timers and actual prefill/decode tokens are preserved;
neither nested timers nor tokens are converted to additive kernel time.

The child execution ledger totals **6771.940406204667 s** (1.881095 host-hours),
including startup/teardown. Enclosing campaign elapsed time is **6963 s**
(1.934167 h), also below the **10800 s** host ceiling. Maximum child ledger
wall is **206.90207425202243 s**, below the **300 s** trace ceiling.
Post-run CPU validation/compression/transfer is not another GPU experiment.
No monetary tariff or integrated GPU-busy time was recorded.
Model, reusable venv and measured campaign/source epochs are retained;
terminal publication cleanup is recorded in [`cleanup.json`](cleanup.json).

## Persistent evidence and reproduction

Small tracked evidence is source-specific to this campaign:

| Path | Content |
|---|---|
| `b6-frozen.json`, `environments.json` | Actual measured freeze and all trace environments |
| `b6-selection.json`, `b6-formal-report.json` | Original immutable discovery selection and formal decision |
| `execution.jsonl`, `validation.json` | All 40 process outcomes/costs and post-run integrity/quality replay |
| `trace-metrics.csv` | All 40 trace summaries, physical pressure, mechanism and costs |
| `foreground-tasks.csv` | All 480 task slots, quality/evidence, timing and root attribution |
| `background-cohort.csv` | All 11904 ordered BG rows with identities, arrival/timing, work and status |
| `campaign-files.json` | Sizes and SHA256 for all 159 campaign files (after the identifier replacement below) |
| `artifacts.json`, `archive-verification.json`, `SHA256SUMS` | Durable release location and archive/upload integrity |

The **284192764-byte** Zstandard tar archive exceeds normal Git blob limits and
is attached to a release of this repository:
[B6 measured-source release](https://github.com/shatianming5/octopus/releases/tag/b6-4d00650-evidence) /
[archive](https://github.com/shatianming5/octopus/releases/download/b6-4d00650-evidence/b6-terminal-evidence.tar.zst).
It contains all **159** campaign files (14730562704 bytes), including
all 40 complete raw JSONs and logs, both phase plans/selectors, ledger, source
tar/object pack, freeze, selection, formal report, original runner and terminal
cleanup; plus the CPU-only post-run validation helper (160 regular files total).
Every decompressed member's size and SHA256 matches `campaign-files.json`.
On 2026-11-01 the user and path names inside 50 of these files (49 campaign files and
the helper) were
replaced with the placeholders used in the rest of this repository (for example
`/mnt/disk2/user`), so sizes and hashes are those of the edited files; nothing
else changed. The original raw campaign remains on the measured host.

Archive SHA256:
`6b2b1d4e28e5a979dee03ec24252fdf8aedc97a468fb1ad18d5cddfcfe596381`.

```bash
# Requires gh, tar and zstd; no GPU or model.
gh release download b6-4d00650-evidence --repo shatianming5/octopus \
  --pattern b6-terminal-evidence.tar.zst --pattern SHA256SUMS
sha256sum -c SHA256SUMS
tar -I 'zstd --long=27 -d' -xf b6-terminal-evidence.tar.zst

# At the measured source with its existing environment (CPU-only):
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=.:src
CAMPAIGN=/path/to/octopus-b6-campaign-4d00650-af58fecc
python -m benchmarks.b6 report --phase discovery --directory "$CAMPAIGN" \
  --output /new/path/discovery-replay.json
python -m benchmarks.b6 report --phase confirmation --directory "$CAMPAIGN" \
  --selection "$CAMPAIGN/b6-selection.json" --output /new/path/formal-replay.json
```

Output paths must be new; preserve the immutable originals. This procedure
recomputes reports, never launches traces. The unchanged implementation's
152-distinct-test CPU acceptance is retained; no generic re-review or rerun
of those tests was needed for this artifact/documentation delivery.
The documented pre-push verifier was run once and exits 1 on the unchanged
historical task-ID substring false positives: 5564 matches on 5504 lines in
137 files, all classified, zero unexplained key matches, no tracked `.env` or
`auth.json`. This is **not a globally green verifier claim**; see
[`delivery-validation.json`](delivery-validation.json).
