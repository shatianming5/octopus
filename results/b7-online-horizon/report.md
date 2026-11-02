# B7 GPU campaign terminal report

## Conclusion

**Terminal decision: `bounded_no_go`.** Discovery completed all 16 frozen valid
selectors on one physical GPU. B7 exercised a meaningful causal
ranking-to-adoption mechanism at both binding 1024 MiB points and passed quality
and background fairness, but it was slower than at least one B6 baseline at
every point. No point was eligible, so confirmation was correctly not planned or
executed.

- Source: clean `e8ad69f76dee9f08c571b420e5fd1ec27a11f25a`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Valid epoch host/device: `.183` GPU2,
  `GPU-53e85207-9c8d-91f6-ba27-992fff75dc7f`
- Active-run utilization snapshot: `100%` compute, `18,150 / 49,140 MiB`
  physical memory allocated
- Valid result units: discovery `16 completed / 0 running / 0 failed`;
  confirmation `0 / 0 / 0`
- Actual process attempts: `20` total = `16` valid scientific selectors +
  `4` preserved invalid infrastructure starts (`1` on `.68`, `3` on `.183`)
- Child-ledger wall: `2987.870483` valid + `97.906509` failed initialization =
  `3085.776992` seconds; not all-inclusive host wall
- Conservative outer-driver accounting: at most `3621` seconds, leaving at
  least `7179` seconds under the `10800`-second cap; see definitions below
- Ready authorized work: none; `bounded_no_go` is terminal under the frozen
  decision rule
- Current blocker: none

## Acceptance and failure boundary

All 16 arms passed quality: `192/192` foreground task slots were correct.
All `6144/6144` planned background requests completed, with no failed or
censored requests. The reporter validated matched controls and the full
denominator.

The frozen treatment deliberately set total native KV cache capacity
(`kv_cache_memory_bytes`) to 1024 or 1536 MiB. This is distinct from both the
49,140 MiB physical VRAM and the demand-derived per-draw future-prefix preference
cap. The 18,150 MiB sample includes model/runtime allocations as well as KV; it
does not show that the remaining physical memory was technically unavailable
or necessary. Increasing the KV budget would change the frozen pressure point;
colocated jobs would alter matched controls. GPU compute reached 100% in an
active-run snapshot despite memory not being full, not a measured 100% average
over the campaign.

Every online background point and one-sided bootstrap upper bound was below the
5% APC-relative limit:

| Capacity | Concurrency | BG point | BG upper | Mechanism | Minimum FG gain |
|---:|---:|---:|---:|:---:|---:|
| 1024 MiB | 2 | -5.770% | -4.800% | pass | -0.894% |
| 1024 MiB | 4 | -5.435% | -4.482% | pass | -1.776% |
| 1536 MiB | 2 | -5.727% | -2.675% | absent | -1.295% |
| 1536 MiB | 4 | -6.513% | -5.473% | absent | -3.056% |

At 1024 MiB, the cap bound, causal ordering changed the preferred set, and
subsequent native adoption replayed:

- concurrency 2: 7 decision-bearing adoption events, 848 draw-linked block
  tokens
- concurrency 4: 7 decision-bearing adoption events, 1504 draw-linked block
  tokens

These are draw-linked attribution units and may repeat across decisions; they
are not claimed as unique recompute savings. At 1536 MiB, there was no binding
or ranking/adoption opportunity. Because meaningful valid mechanism was observed
at 1024 MiB, the terminal result is `bounded_no_go`, not `not_decided`.

B7 improved foreground mean E2E versus APC at the two mechanism-bearing points,
but failed the required all-three-baseline gate:

| KV MiB / concurrency | APC | B6 fixed | B6 budgeted | B7 online |
|---|---:|---:|---:|---:|
| 1024 / 2 | 53.7603 s | 51.2593 s | 50.9110 s | 51.3662 s |
| 1024 / 4 | 41.7603 s | 38.9085 s | 38.8386 s | 39.5284 s |
| 1536 / 2 | 53.8067 s | 49.8391 s | 50.0220 s | 50.4848 s |
| 1536 / 4 | 42.4450 s | 38.4796 s | 40.1578 s | 39.6555 s |

Each cell is one trace's mean over all 12 foreground slots, not just successes.
Each method contributes 4/4 valid traces, 48/48 correct foreground tasks and
1536/1536 completed background requests. The four points share discovery seed
2501; this table is not a four-independent-seed confirmation test.

## Operational amendment and limitations

No valid trace existed on `.68`. Its 12.247993-second resource-collision
startup is preserved separately. The approved replacement epoch on `.183`
retained the same source, model, methods, points, seeds, metrics, thresholds,
and one GPU UUID.

Three `.183` task-local instrumentation starts were also preserved before valid
execution: missing relocated `ninja` PATH (`38.438711s`), CUDA-JIT compiler
mismatch (`25.160577s`), and stale CUDA-toolchain cache generation
(`22.059227s`). Each failed before a valid workload trace, was diagnosed from
preserved evidence, and received a distinct environment-path/toolchain repair;
none was treated as a scientific selector, candidate result, tuning run, or
blind retry. Operationally, these were additional process attempts outside the
original no-retry execution envelope; the approved amendment and this report
disclose them rather than representing the campaign as having only the `.68`
failed start.

## Wall-cost definitions and budget

The supervisor clarified three different clocks; they must not be interchanged:

| Accounting unit | Duration | Coverage |
|---|---:|---|
| Valid child ledger | 2987.870483 s | Sum of parent-measured subprocess wall for 16 valid selectors, including process startup/teardown |
| Failed-initialization child ledger | 97.906509 s | Four failed starts: 12.247993 s on `.68`, plus 85.658515 s across the three `.183` repairs |
| Combined child ledger | 3085.776992 s | All 20 process attempts; excludes enclosing driver work and gaps, so not an all-inclusive host-wall total |
| Valid outer driver | approximately 3517 s | Recorded UTC interval 2026-09-10 15:01:18 to 15:59:55 |
| Four failed outer drivers | approximately 99 s | Whole-second intervals of 12 + 39 + 26 + 22 s |
| Combined outer-driver intervals | approximately 3616 s | Sum of the five separately recorded execution epochs, not continuous calendar time across host migration |
| Conservative outer accounting | **at most 3621 s** | Add one second per epoch (5 s total) for whole-second timestamp truncation |
| Remaining against the 10800 s cap | **at least 7179 s** | Conservative accounting margin, not authorization to run more work |

`execute()` uses a monotonic `campaign_start` timer for the current phase's
subprocesses, startup, inter-trace validation and gaps; each child also has a
300-second limit. That phase timer excludes freeze/plan before `execute()` and
reporting after it. Its exact elapsed value was **not persisted**, so there is
no exact measured overall hard-timer value to publish. The enclosing driver
intervals provide a separate, conservative accounting bound showing this run
was below the cap. Neither 3085.776992 s as all-inclusive host wall nor
7714.223008 s as conservative remaining budget is justified.

Infrastructure setup/migration time between driver epochs and later
CPU-only replay/transfer/publication are outside this summed execution-epoch
accounting; this is not a claim about total project elapsed time or billed GPU
occupancy. No monetary tariff or integrated GPU-busy time was recorded. The
100% compute figure above is a snapshot, not a cost integral.

## Runtime and scientific limits

The final runtime remained Python 3.12.14, Torch `2.13.0+cu129`, vLLM `0.28.0`,
CUDA runtime 12.9, and the frozen model. FlashInfer's path-keyed SM89 extension
was rebuilt from the installed frozen sources using the host's existing CUDA
12.2 compiler and the source host's GCC 11.4 toolchain. No package, model,
kernel source, protocol, threshold, point, or seed changed.

The 8,840,254,804-byte (about 8.3 GiB) raw matrix remains on `.183` at
`/mnt/ssd1/user/octopus-b7-campaign-183`. Compact publication artifacts,
per-raw SHA-256 hashes, sizes, selectors, logs, and reports were retrieved
locally. The 16 valid full raw JSON files were not duplicated into the checkout;
the small invalid-start raw records are retained with their failure epochs.
This is discovery evidence from the twelve development templates and seed 2501,
not independent confirmation or held-out coding-task generalization. B6's
historical evidence and the measured B7 source/configuration are unchanged.
The separate [P1 document](../../docs/p1-real-agent-protocol.md) states unexecuted,
unfrozen real-patch evaluation prerequisites, not SWE-bench results.

## Evidence

| Path | Scope and interpretation |
|---|---|
| [`terminal-summary.json`](terminal-summary.json), this report | Current accepted campaign decision, complete denominators and cross-epoch accounting |
| [`valid-183/b7-selection.json`](valid-183/b7-selection.json), [`b7-selection-replay.json`](valid-183/b7-selection-replay.json) | Authoritative valid discovery report and owner's byte-matching replay |
| [`valid-183/terminal.json`](valid-183/terminal.json) | Original valid driver record; named confirmation output paths are planned placeholders, not executed artifacts |
| [`valid-183/b7-frozen.json`](valid-183/b7-frozen.json), [`discovery/plan.json`](valid-183/discovery/plan.json), `discovery/*.selector.json` | Frozen source/protocol and all 16 valid selectors in their executed order |
| [`valid-183/execution.jsonl`](valid-183/execution.jsonl), `valid-183/discovery/*.log` | All 16 valid child commands, exits, parent-measured wall times and process logs |
| [`valid-183/raw-inventory.json`](valid-183/raw-inventory.json), [`RAW_SHA256SUMS`](valid-183/RAW_SHA256SUMS) | Per-raw size/hash inventory; raw paths are relative to the remote campaign root |
| Root `b7-selection*.json`, `terminal.json`, `execution.jsonl`, `discovery/` | Preserved `.68` failed-start epoch, whose `incomplete` decision is not the terminal scientific result |
| [`failed-epochs/`](failed-epochs/) | Three separately preserved `.183` infrastructure failures; not pooled into valid metrics |
| [`host-migration-amendment.json`](host-migration-amendment.json) | Host replacement, additional failed starts and explicit accounting definitions |
| [`process-cleanup-final.json`](process-cleanup-final.json) | Supervisor's terminal cleanup evidence |
| [`SHA256SUMS`](SHA256SUMS) | Compact publication bundle checksums, relative to repository root; excludes itself |

Reporter replay byte-matched the authoritative discovery report. The campaign
driver and all synchronous children exited; no B7 process or temporary transfer
key remains, and no unrelated process was signaled.

## Raw access and CPU-only reproduction

The raw matrix is retained on the owner's SSH host, **not a public GitHub
download**. Access requires permission and an existing authorized SSH identity
for `user@198.51.100.10`, port `20183`; no credential or temporary transfer key
is included. Without that access, readers can inspect the committed selection,
full selector inventory, logs, failure records and hashes but cannot replay the
raw matrix. Request access from the repository/campaign owner. Allow at least
8.3 GiB for raw files plus Python's in-memory JSON overhead.

From the repository root, with ordinary Python 3 (no model, CUDA or vLLM needed
for the reporter), use a new temporary destination so original evidence is not
overwritten:

```bash
REPO="$PWD"
REPLAY="$(mktemp -d /tmp/octopus-b7-replay.XXXXXX)"
cp -a results/b7-online-horizon/valid-183/. "$REPLAY/"
scp -P 20183 \
  'user@198.51.100.10:/mnt/ssd1/user/octopus-b7-campaign-183/discovery/*.raw.json' \
  "$REPLAY/discovery/"
(cd "$REPLAY" && sha256sum -c RAW_SHA256SUMS)
git worktree add --detach "$REPLAY/source" e8ad69f76dee9f08c571b420e5fd1ec27a11f25a
(
  cd "$REPLAY/source"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python3 -m benchmarks.b7 report \
    --phase discovery --directory "$REPLAY" \
    --output "$REPLAY/b7-selection-local-replay.json"
)
cmp "$REPLAY/b7-selection-local-replay.json" \
  "$REPO/results/b7-online-horizon/valid-183/b7-selection.json"
git worktree remove "$REPLAY/source"
```

`RAW_SHA256SUMS` covers the 16 valid raw files only. The compact bundle can be
checked separately from the repository root with
`sha256sum -c results/b7-online-horizon/SHA256SUMS`. Never run the reporter against
this result directory's root and mistake the preserved `.68` incomplete epoch
for the accepted valid matrix. These commands reproduce the report, not the
GPU experiment; there is no remaining authorized B7 execution.

## Publication acceptance

The runtime owner accepted scientific correctness and exact reporter replay
before this publication. The
[publication-only consistency check](publication-validation.json) covers compact
selector/log/hash inventory, preservation of retrieved evidence, accepted report
denominators and terminal metadata, accounting definitions, local document links,
and unchanged B6/source/configuration. It does not rerun a model, GPU workload,
reporter or raw-metric validation.

The [documented pre-push verifier](secret-gate.json) ran once and exited **1**:
126 checks passed, one failed, four warned and two were skipped. The sole failure
is the unchanged historical long-key-pattern false positives: 5,564 matches on
5,504 lines in 137 files, all task-ID substrings, zero unexplained matches and
zero staged publication files matching. `.env` and `auth.json` are not tracked.
This is explicitly **not a globally green verifier** claim.
