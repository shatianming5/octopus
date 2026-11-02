# P1 real-issue pilot terminal report

## Conclusion

**Terminal status: `pilot_complete`; both methods resolved 0/4 issues.**
Projected APC and B6 semantic fixed completed all four solver episodes on
official SWE-bench Verified issues, but neither produced a resolved patch. There is therefore no
quality winner, and reserved-device cost per solved issue is undefined (`null`)
for both methods.

- Valid source: `c37222e3a3a37d2fcf052b2bcfdc162daf4b1263`
- Dataset: `SWE-bench/SWE-bench_Verified` revision
  `78f471bf655a3137b2e8a75af1501690ec009ec3`
- Harness: `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`
- Device: `.68` GPU0,
  `GPU-c5792743-87d0-80a0-8a3e-876d31e854d9`
- Valid episodes: `8 completed / 0 running / 0 failed`
- Actual solver process attempts: 11 = 8 valid episodes + 3 preserved
  pre-action infrastructure epochs
- Official grading: Projected APC `0/4`; B6 fixed `0/4`

## Solver and grading outcomes

Each method produced three empty patches and the same 795-character patch for
`sympy__sympy-12096`. The official harness applied that patch successfully but
the task remained unresolved. The other three tasks were recorded as empty
predictions, not silently dropped:

| Issue | Projected APC | B6 fixed |
|---|---|---|
| `django__django-10097` | empty | empty |
| `django__django-10554` | empty | empty |
| `sympy__sympy-11618` | empty | empty |
| `sympy__sympy-12096` | applied, unresolved | applied, unresolved |

All eight episodes reached the frozen 10-action limit. Matched action payloads
and final patches were identical between methods for every issue. Grading used
distinct source/method run IDs, covered all four planned issues per method, and
reported no evaluator errors or infrastructure failures.

These are **eight terminal prediction grades, not eight task-test executions**.
Only the nonempty `sympy__sympy-12096` prediction ran the official task tests,
once per method. The six empty predictions received terminal quality-failure
outcomes without running their task tests; they remain in the 0/4 denominator
for each arm.

The empty patches were valid model/tool outcomes, not plumbing failures:

| Issue | Shared per-method model/tool outcome |
|---|---|
| `django__django-10097` | Ten consecutive edit attempts; every edit failed `old text must match exactly once`; no visible tests, empty patch at action limit |
| `django__django-10554` | Ten consecutive edit attempts; every edit failed `old text must match exactly once`; no visible tests, empty patch at action limit |
| `sympy__sympy-11618` | Ten consecutive edit attempts; every edit failed `old text must match exactly once`; no visible tests, empty patch at action limit |
| `sympy__sympy-12096` | Three successful edits, two writes rejected because target files already existed, five visible test commands with exit code 1; nonempty patch at action limit |

Visible developer-test commands are solver actions and their time is included
in solver cost. They are distinct from the later official task-test execution.
All valid episodes have no infrastructure status/errors. These outcomes do
not establish that the workflow is useful for issue resolution.

The evaluator-only gold sanity passed before generation:
`django__django-10097` resolved with 438/438 FAIL_TO_PASS and 1427/1427
PASS_TO_PASS tests. It was an instrument check only and did not enter prompts,
solver state, model quality, or GPU cost.

## Runtime and mechanism

| Metric | Projected APC | B6 semantic fixed |
|---|---:|---:|
| Mean solver E2E | 45.3717 s | 43.2315 s |
| Reserved-device child wall | 204.8433 s | 197.3518 s |
| Recorded engine scope | 142.0563 s | 134.6276 s |
| Tool wall | 17.6063 s | 17.5393 s |
| Prefill tokens | 7,759 | 7,761 |
| Decode tokens | 3,094 | 3,094 |
| Adopted-prefix tokens | 39,584 | 39,584 |
| Resolved issues | 0 | 0 |
| Device seconds per solved | `null` | `null` |

B6 fixed had 233 native policy draw records. Zero retained block-seconds is
expected for this no-lease policy; it is not evidence that the policy was
inactive. There were no cache evictions or allocation rejections, adopted-prefix
totals were identical, and no actions/patches differed. This serial real-task
pilot therefore did not demonstrate a retention benefit. B6 fixed's mean E2E
was about 4.7% lower, but with zero solved issues, four tasks and no
retention-pressure contrast, this is descriptive timing only, not evidence of a
causal KV-policy advantage, quality noninferiority or general performance gain.

Independent post-publication review also found an input-fairness limitation:
the historical journal serializer exposed observation dictionary key order to
the tokenizer. Semantically equal observations for `django__django-10097`
produced different tokens in follow-up contexts 1--9 (zero-based), or **9 of
40 paired action contexts**. The other three issues' paired prompts match.
Actions and final patches remain identical and the official 0/4 outcomes are
unchanged, but the historical full prompts were not all token-identical.
Neither the two-token prefill difference nor the timing difference can be
attributed to retention. The future serializer is canonicalized; all measured
raw prompts, source identities, predictions, metrics and clocks remain
immutable, without a normalization or GPU rerun.

An active-run snapshot observed 43,058/49,140 MiB physical memory and 62%
device utilization. Normal vLLM profiling allocated 29,147,267,072 bytes
(about 27.15 GiB) of KV capacity under `gpu_memory_utilization=0.90`; no explicit
KV byte cap was used. This observation is neither a peak/average utilization
measurement nor a GPU-busy integral.

## Accounting

The valid solve parent interval was 403.669382 seconds; its child ledger was
402.195056 seconds. Three pre-action infrastructure epochs consumed another
106.583725 parent seconds:

1. `.183`, source `aa20aea`: Docker reported container removal already in
   progress and the wrapper misclassified successful cleanup.
2. `.68`, source `635c3e6`: the SSH endpoint omitted the retained venv's
   executable path, so `ninja` was unavailable.
3. `.68`, source `d944a0b`: vLLM reported the canonical GPU UUID without the
   administrative `GPU-` prefix.

Each failure occurred before a valid model action, was preserved with its
original source/device/selector identity, and received a distinct
infrastructure-only repair. No task, issue, prompt, model, method, seed,
generation limit, memory policy, or grading threshold changed.

The parent clock is the persisted monotonic `solve` interval, including child
startup, SSH/relay and local tool waits, teardown and inter-episode bookkeeping.
The child ledger covers subprocess intervals only. Summing the four separate
parent epochs charges all eleven attempts, but is not continuous calendar time
across capacity waits/migration, a GPU-busy integral or a monetary bill.
The eight valid child intervals alone sum to 402.195056 s; they must not be
presented as the whole campaign cost.

- Cumulative GPU-host parent wall: **510.253107 seconds**
- Remaining from the 10,800-second ceiling: **10,289.746893 seconds**
- Official grading wall: **77.536597 seconds**, excluded from solver/GPU cost
- Accepted gold sanity wall: **539.72 seconds**, CPU instrumentation only
- Measured CPU preparation/grading floor: **2,541.04 seconds**

Other recorded CPU preparation consists of obsolete evaluator sanity attempts
of 499.28 s and 513.05 s, four full CPU-gate durations of
10.25 + 9.30 + 7.45 + 7.45 s, and image-build progress of approximately 877 s.
Those instrument/preparation runs are not P1 model-quality outcomes.

An aggregate clock for all downloads, image pulls/derivations, short tests and
repair commands was not persisted. Thus 2,541.04 seconds is only a reported
measured floor, not a complete CPU bill. Compliance with the cumulative
7,200-second CPU preparation/grading envelope is **UNVERIFIED**, and **no
remaining CPU budget can be inferred**. All official grading is terminal.
Unused GPU-budget arithmetic does not authorize additional experiments.

## Preparation and source history

The early `c104f840` metadata came from the legacy mirror, not the authoritative
`SWE-bench/SWE-bench_Verified` alias. It was withdrawn before model execution.
Canonical revision `78f471bf655a3137b2e8a75af1501690ec009ec3` has parquet SHA-256
`030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25`.
The selected four IDs, base commits and visible issues did not change in
response to model or scorer outcomes.

[`prelaunch-identity-confirmation.json`](prelaunch-identity-confirmation.json)
confirms zero GPU episodes on the obsolete revision.
[`prelaunch-dataset-correction.json`](prelaunch-dataset-correction.json) is the
original preparation snapshot: its pending/status fields describe that earlier
time, not this terminal campaign. Earlier implementation/device identities in
that snapshot are historical. The authoritative valid identity is
[`p1-frozen.json`](p1-frozen.json), source `c37222e`; no prior source was
retroactively relabelled.

Final source/test evidence, as confirmed by the execution owner without a
publication rerun:

| Source | Repair | Full CPU gate |
|---|---|---|
| `635c3e6a17117483bcd49e6162b1123f8d72f82b` | Observed Docker cleanup race | 44 passed in 9.30 s |
| `d944a0b213c3934341d95dddac57856496f111d2` | Retained venv executable PATH | 45 passed in 7.45 s |
| `c37222e3a3a37d2fcf052b2bcfdc162daf4b1263` | Administrative GPU-UUID prefix normalization | **46 passed in 7.45 s** |

These gates used CUDA hidden and emitted 14 upstream deprecation warnings
each; the final two ran on `.68`. All eight valid episode records have exactly
the final full source hash above. The failed epochs keep their `aa20aea`,
`635c3e6` and `d944a0b` identities.

Official task images include an environment setup commit above the issue base.
The supervisor derived immutable solver-only images with exact clean base HEAD
and the same installed environment, without adding gold/test patches. Both
methods used the same per-issue derived ID. Official grading retained the
official images/tags. The dataset, task-environment and image identities are
preserved in [`preparation/`](preparation/). `.183` capacity waits were superseded
by the coordinator's already-provisioned `.68` GPU0 assignment; all valid
episodes ran on that one device. B6/B7 historical results are unchanged and not
pooled with P1.

## Evidence and cleanup

| Path | Public evidence |
|---|---|
| [`terminal-summary.json`](terminal-summary.json), [`report.json`](report.json) | Terminal synthesis and original machine-readable pilot report |
| [`p1-frozen.json`](p1-frozen.json), [`solver-manifest.json`](solver-manifest.json), [`execution-endpoint.json`](execution-endpoint.json), [`p1-plan.json`](p1-plan.json) | Canonical sanitized inputs, local image IDs, actual source/endpoint and eight selectors |
| [`solve/`](solve/) | All eight full solver records, exact contexts/actions/tool observations, native metrics, selectors, process logs, predictions and execution clocks |
| [`grade-projected_apc/`](grade-projected_apc/), [`grade-b6_semantic_fixed/`](grade-b6_semantic_fixed/) | Original official aggregate/per-task outcome reports, generated patch, invocations and compact driver logs |
| [`gold-sanity-summary.json`](gold-sanity-summary.json) | Instrument-only pass summary, never a model outcome |
| [`preparation/`](preparation/) and the two prelaunch records above | Canonical dataset/image provenance and explicitly superseded preparation history |
| [`prior-infrastructure-epochs/`](prior-infrastructure-epochs/) | Three original failed records and parent/child summaries, excluded from valid scientific metrics but charged in execution cost |
| [`cleanup.json`](cleanup.json) | Supervisor's terminal cleanup evidence |
| [`SHA256SUMS`](SHA256SUMS) | Checksums for every public bundle file except itself; paths are relative to repository root |

No P1 relay process or task container remains. No unrelated process was
signaled, no temporary credential remains, and evaluator-only gold records were
not copied into this publication bundle.

## Publication and CPU-only reproduction

All eight raw solver records fit in this compact tracked bundle; no additional
remote raw download is required to reconstruct the report. Gold/test-patch
inputs, evaluator execution scripts (`eval.sh`), verbose evaluator logs/test
output and evaluator run metadata remain private. Publication omitted those
execution internals without reading their contents; original copies remain
outside the repository. Official outcome summaries and generated predictions
are retained. This separation does not censor unsuccessful issue results.

Images and model weights are not in Git. The image inventory identifies the
retained local official/derived images, not a promised public image release.
Exact-image access requires the campaign owner's retained environment/export;
independent rebuilds use the pinned official dataset/harness and can have
different image IDs. Do not silently substitute rebuilt IDs into the recorded
freeze. New model runs require separate authorization.

From the publication checkout, the following reconstructs only the CPU report
with the measured source and published records. It does not invoke a model,
Docker or official grading:

```bash
EVIDENCE="$PWD/results/p1-real-agent"
sha256sum -c "$EVIDENCE/SHA256SUMS"
REPLAY="$(mktemp -d /tmp/octopus-p1-report.XXXXXX)"
git worktree add --detach "$REPLAY/source" c37222e3a3a37d2fcf052b2bcfdc162daf4b1263
(
  cd "$REPLAY/source"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python3 -m benchmarks.p1 report \
    --frozen "$EVIDENCE/p1-frozen.json" --directory "$EVIDENCE/solve" \
    --grades "$EVIDENCE/grades.json" --output "$REPLAY/report.json"
)
cmp "$REPLAY/report.json" "$EVIDENCE/report.json"
git worktree remove "$REPLAY/source"
```

Independent official grading uses the published per-method predictions and the
pinned evaluator-only dataset/harness in a permitted Docker environment, with a
new source/method run ID. Original invocation files record the exact commands
and `logs/evaluation/<run_id>/results.json` paths. The protocol document
describes the CLI contract; the original private evaluator JSON is not published.

## Publication acceptance

The supervisor owns and accepted the experiment/grading outcome. The
[publication-only consistency check](publication-validation.json) covers source,
endpoint and canonical input identities; complete selector/prediction/grade
denominators; accepted compact accounting; preservation of retrieved public
evidence; evaluator-only exclusions; and local document links. It does not
repeat raw scientific validation, the reporter, model runs or grading. Original
machine records, process logs and generated patches retain their exact bytes.

The bounded terminal clarification additionally checks the 46-test final-source
evidence, eight-grade/two-test-execution denominators, per-issue model/tool
failure reasons and the absence of any remaining-CPU-budget inference. It uses
the supervisor's derived clarification, not a second scientific evaluation.

The clarification delivery's [documented pre-push verifier](secret-gate.json)
ran once and exited **1**:
126 passed, one failed, four warnings and two skipped. Its sole failure has
5,565 classified long-key-pattern matches on 5,505 lines in 138 files:
5,564 unchanged historical task-ID substrings plus one new false positive
from an existing preparation filename in `SHA256SUMS`. Thus **one staged
checksum file matched**, not zero. No match is an unexplained credential;
`.env` and `auth.json` are not tracked. This is **not a globally green verifier**
claim.
