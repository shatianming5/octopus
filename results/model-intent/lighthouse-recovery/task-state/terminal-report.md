# lighthouse-task-state-v1 terminal result

- **scope_id:** `lighthouse-task-state-v1`
- **status:** `QUALITY_FAIL` (terminal, no rerun)
- **parent objective:** not complete; this report closes only the single task-state quality episode
- **quality gate:** original official evaluator returned `resolved=false`; `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` remained `FAILED`
- **episode:** solver `complete`, action limit reached after 12/12 actions, no patch, no test action, no `done` action

## Frozen execution and provenance

- Freeze SHA-256: `dd4ee292fdcead625777453becea408391fb01c4eb70c5401e095d07248e63f8`
- The protocol matched `regex-search` except for the predeclared decision-format delta `deliberative_v1 -> task_state_v1`.
- Limits remained 1170 GPU budget seconds, 1070 episode timeout seconds, 1800 CPU evaluator timeout seconds, and one model episode.
- All seven frozen implementation hashes matched both the local source and the fresh deployment, including `src/octopus/task_memory.py`.
- The engine initialized the pinned Qwen2.5-Coder-14B revision on GPU UUID `GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` with BF16, seed 17, 16384 maximum model length, and APC.

## Actual result and trace metrics

- Actions: raw 12, distinct canonical 4, physically submitted 12, physical distinct 4, rejected 0.
- The physical sequence was one list, two distinct searches, and nine identical successful reads of `types/lhr.d.ts`; successful inspection repeats: 8.
- Unsuccessful physical actions: 0; unsuccessful raw repeats: 0; unsuccessful canonical repeats: 0.
- Observation outputs: 11 nonempty, 3 unique.
- Plans: 12 nonempty. Evidence selections, semantic commits, and grounded facts: 0 each. `invalid_task_evidence` rejections: 0.
- Because the model cited no quote and produced no semantic commit, there was no committed excerpt to verify in a later decoded context. The evidence-carrying part of the intervention was not exercised by model behavior.
- Journal compaction deduplicated 28 entries and dropped 0 entries; maximum actual prompt context was 4608 tokens.
- Native requests: 12; actual prefill tokens 19,676; actual decode tokens 1,637; maximum simultaneous scheduler requests 1; cache evictions 0; allocation rejections 0.
- GPU engine scope: 87.490 seconds; solver end-to-end: 89.703 seconds; remote parent solve elapsed: 107.224 seconds; official grading wall: 10.190 seconds; wrapper wall: 122 seconds.
- Patch bytes: 0. Tests executed by the solver: 0. Model `done`: false.

## Capacity and cleanup

- At engine-ready observation, GPU 9 was active at 29,772 MiB and 100% utilization.
- The fresh deployment `/home/user/octopus-lighthouse-task-state-v1-20260916` was removed after terminal artifact retrieval; no owned remote process remained.
- Actual owned container names were `octopus-lighthouse-bf643167acdc4f9b8c18e0452d1dd06b` and `octopus-lighthouse-eval-ffa5201c50ca463ea35a24e50e99f44e`; both were destroyed and zero matching containers remain.
- Final GPU 9 observation was 2 MiB and 0% utilization.

## Artifacts

- `results/model-intent/lighthouse-recovery/task-state/protocol.json`
- `results/model-intent/lighthouse-recovery/task-state/freeze.json`
- `results/model-intent/lighthouse-recovery/task-state/deployed-source-sha256.txt`
- `results/model-intent/lighthouse-recovery/task-state/execution/solver/episode.json`
- `results/model-intent/lighthouse-recovery/task-state/execution/solver/grade.json`
- `results/model-intent/lighthouse-recovery/task-state/execution/solver/result.json`
- `results/model-intent/lighthouse-recovery/task-state/execution/solver/official.log.gz`
- `results/model-intent/lighthouse-recovery/task-state/execution/terminal.json`

The terminal negative result supports only this scoped conclusion: under the frozen model and budgets, the model emitted plans but elected not to cite any observed evidence, repeated a successful read eight times, made no edit, ran no test, and failed the original official evaluator. It does not establish generalized model incapability and does not open or complete the parent's multibranch or formal gates.
