# Lighthouse Failure-Aware Runtime Result

Exactly one base-Qwen `failure_aware_v1` episode completed under the frozen
same-task protocol. The generic action-validity memory eliminated repeated
failed external submissions and changed the trajectory, but the model produced
no valid edit or patch and the original official evaluator remained unresolved.

## Quality result

- Terminal: complete, exit 0, `QUALITY_FAIL`.
- Official evaluator: complete, return code 0, `resolved=false`.
- `FAIL_TO_PASS`: `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`
  failed; `PASS_TO_PASS` is empty.
- Patch: 0 bytes across 0 files.
- Developer test actions: 0.

Only official `resolved=true` is quality success, so this phase is a quality
failure even though the control behavior improved.

## Proposals, submissions, and recovery

The model made 12 proposals comprising 6 distinct canonical actions. Six were
submitted to the external repository tool and all 6 submitted actions were
distinct; 3 succeeded and 3 failed. Six repeated failed proposals were rejected
before external execution, leaving **0 repeated failed external submissions**
versus 11 in the accepted base/comparator trajectories.

| Index | Proposal | External result |
|---:|---|---|
| 0 | invalid same-file `read(start=0)` | submitted; failed |
| 1 | same-file search for `id` | submitted; successful non-empty evidence |
| 2 | first edit variant | submitted; precondition mismatch |
| 3-6 | repeated invalid read | rejected before external execution |
| 7 | same-file search for `id: lib.id` | submitted; successful post-mismatch evidence |
| 8 | revised edit including the comma | submitted; precondition mismatch |
| 9 | repeated invalid read | rejected before external execution |
| 10 | valid same-file read from line 57 | submitted; successful non-empty evidence |
| 11 | repeated failed revised edit | rejected before external execution |

The first inspection proposal was index 0 but failed. The first successful
relevant inspection was index 1. After the edit mismatch at index 2, index 7
provided successful non-empty evidence from the affected resource before the
changed edit at index 8. This is genuine action-level recovery from blind
resubmission, not merely a forced inspection loop, but the corrected edit still
failed and no task repair resulted.

## Native and timing units

- 12 model calls; 29,374 prompt tokens; 10,350 actual prefill tokens; 428
  actual decode tokens.
- 19,024 adopted and inherited prefix tokens; 396 capsule tokens; 8,448
  checkpoint-frontier tokens; 8,459 selected-parent tokens.
- Zero preemptions, cache evictions, or allocation rejections; maximum one
  simultaneous scheduler request.
- Final pool: 31,790 total/free blocks, 668 warm-cached and 31,122 cold-free
  blocks, with zero active or retained ownership.
- Timing: 29.828 GPU-engine seconds, 32.181 solver end-to-end seconds, 46.580
  remote-child seconds, 46.634 parent solve seconds, 0.000107 control seconds,
  2.417 external-tool seconds, 10.530 official-evaluator seconds, and 62 seconds
  total detached job wall time.

Per-call prompt/prefill/decode and inheritance units are preserved in
`metrics.json`; every rejected proposal consumed its own real model call and
one of the 12 slots. One of one approved GPUs was assigned during engine scope;
no active-period utilization percentage was inferred.

## Interpretation

`failure_aware_v1` achieved its narrow operational objective on this case:
canonical failed actions were not re-executed externally, persistent failure
state survived across turns, and the model eventually gathered relevant
same-resource evidence and changed its edit. It did not achieve the task-level
objective: both edit variants mismatched, no developer test ran, the patch was
empty, and official quality failed. This result supports continuing only
through the parent-selected same-task conditional route; it does not authorize
a second trajectory or policy tuning in this scope.

## Identity, artifacts, and cleanup

- Base model revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- GPU: index 9, `GPU-3e940a5a-a24d-b615-dc69-df141ced7af0`.
- Wrapper PID 1485222; start `2026-09-15T18:26:34Z`.
- Freeze SHA-256:
  `b77590461c43f21ca177b42bf81b4f84343d843f2c35c13aec6c7992992cf8bc`.
- Episode SHA-256:
  `f8ee184771df752c31511d364d8e1a43b6f7dec6e8d51a6cf5bff094005fef31`.
- `src/octopus/tool_memory.py` SHA-256:
  `cce65d8756196706ad65459f6b149a208258792edaa7c9129470ea9b3d19dd0d`.
- Solver SHA-256:
  `f2bbeb4eb6a908da4fbfd91a52c0f55554973b5f8ebf4c5ff3cfa0d765afd98a`.

The official log is retained losslessly at
`execution/solver/official.log.gz` (7,515,741 bytes; stored SHA-256
`19c52777fa9c524d9e137ac61306c53f8dc4d984e4d22bbd3b131fcf4e439fc1`;
decompressed 50,624,671-byte SHA-256
`c3cad8e921028381a084bf5fba0a8ed4a7119a723b5ca26077b77d4833538e3d`).

Paid resources are released: GPU9 returned to 4 MiB and 0% utilization with no
compute process; no Lighthouse container or tmux session remains. The reusable,
non-GPU overlay and ignored task checkout remain for a concrete parent-selected
conditional next stage:

- `/home/user/octopus-lighthouse-runtime-recovery-v1-20260915`
- `results/model-intent/lighthouse-recovery/runtime/assets/task-repo`

The exact benchmark source bytes pinned by this freeze are preserved and
hash-verified through
`results/model-intent/lighthouse-recovery/source-provenance/README.md`.
