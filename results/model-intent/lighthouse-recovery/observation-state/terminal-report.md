# lighthouse-observation-state-v1 terminal result

- **scope_id:** `lighthouse-observation-state-v1`
- **status:** `QUALITY_FAIL` (terminal, no rerun)
- **parent objective:** not complete; this report closes only the single observation-state episode
- **quality gate:** the original official evaluator returned `resolved=false`; `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` remained `FAILED`
- **episode:** solver `complete`, action limit reached after 12/12 actions, no patch, no test action, and no `done` action

## Frozen execution and evidence transport

- Freeze SHA-256: `86bbddb1b3fc3bfbb8c8a1cba088d1ca24b82ed8fae460238394ff1ae27b5013`.
- The sole frozen protocol delta from task-state was `observation_memory: latest_v1`; all other protocol fields matched.
- The initial prompt token IDs were byte-for-byte identical to task-state, preserving the root instructions and decision schema.
- All seven implementation hashes matched local source, fresh deployment, freeze, and retained source provenance.
- The engine initialized the pinned Qwen2.5-Coder-14B revision on GPU UUID `GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` with BF16, seed 17, 16384 maximum model length, APC, and the frozen budgets.
- Every actual outcome produced an observation commit: 12/12 commits and 12 reducer-history facts. Each next capsule selected only the latest complete observation.
- Using the pinned tokenizer, all 12 capsule token segments decoded and re-encoded exactly. For every next turn, the capsule contained the exact preceding delivered output: 11/11.

## Actual result and trace metrics

- Actions: raw 12, distinct canonical 3, physically submitted 12, physical distinct 3, rejected 0.
- Sequence: one list, one read, and ten searches; nine successful inspection repeats.
- Known-failed physical repeats: raw 0, canonical 0. All physical actions succeeded.
- Observation outputs: 12 nonempty, 3 unique.
- Model citations and semantic commits: 0 each; runtime observation commits: 12. Invalid evidence rejections: 0.
- Plans: 12 nonempty. World-state observation facts: 12. Journal compaction deduplicated 36 entries and dropped 0.
- Maximum capsule was 423 tokens; maximum prompt was 3648 tokens.
- Native requests: 12; actual prefill tokens 23,742; actual decode tokens 1,418; maximum simultaneous scheduler requests 1; cache evictions 0; allocation rejections 0.
- GPU engine scope: 75.745 seconds; solver end-to-end: 78.097 seconds; remote child: 93.562 seconds; parent solve: 93.622 seconds; official grading: 9.495 seconds; wrapper wall: 108 seconds.
- Engine-resident sample: 43,340 MiB on GPU 9. Patch bytes: 0. Tests executed by the solver: 0. Model `done`: false.

## Cleanup and artifacts

- Actual owned containers were `octopus-lighthouse-59a111f0e098475894430d4f02572144` and `octopus-lighthouse-eval-56a4104d950a424fb47f5cafdb8490d7`; both were destroyed and are absent.
- The fresh deployment `/home/user/octopus-lighthouse-observation-state-v1-20260916` was removed; no owned remote process remains.
- Final GPU 9 observation was 2 MiB and 0% utilization.
- Terminal artifacts are under `results/model-intent/lighthouse-recovery/observation-state/`, including `freeze.json`, `deployed-source-sha256.txt`, `execution/solver/episode.json`, `execution/solver/grade.json`, `execution/solver/result.json`, `execution/solver/official.log.gz`, `metrics.json`, `capsule-validation.txt`, `cleanup.json`, and `source-provenance/`.

The runtime-owned observation path was exercised and validated exactly, despite 12 empty model citation arrays. The frozen model still repeated successful inspection, made no edit, ran no test, and failed the original official evaluator. This closes only the observation-state episode and does not complete or authorize the parent multibranch or formal objective.
