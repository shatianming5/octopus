# Lighthouse evidence-only-context terminal result

- Scope: `lighthouse-evidence-only-context-v1`
- Status: terminal, cleaned up
- Official result: `QUALITY_FAIL` (`resolved=false`)
- Episode: 1 actual model episode, 12/12 actions, 0 rejected, 0 failed physical actions, 5 distinct canonical actions
- Behavior: 12 physical inspections, 7 successful inspection repeats, 0 edits, 0 tests, 0 `done`, 0 patch bytes
- Outputs: 4/12 physical outputs nonempty, 5 unique outputs
- Projection validation: 12/12 capsule token round-trips; 11/11 exact latest outputs in the next capsule and literal tool history; one pending assistant turn per input; prior raw model text and `model_interpretation_unverified` absent from actual decision inputs
- Audit: 12/12 raw model decisions and reasoning records retained; 12 runtime observation commits; final model plan and unverified interpretation retained only in audit; 0 model citations and 0 semantic commits
- Contract: frozen/local hashes 7/7; deployed executable hashes 6/6 plus frozen wrapper hash; sole protocol delta `history_format: chat_turns_v1 -> tool_observations_v1`; root instructions and decision schema equal
- Runtime: 12 native requests, 33,796 actual prefill tokens, 1,591 actual decode tokens, max scheduler concurrency 1, 0 cache evictions, 0 allocation rejections, 0 preemptions, 85.876 GPU-engine-scope seconds, 87.909 solver seconds
- Context/journal: max 5,298 input tokens, 25 deduplicated entries, 0 dropped entries
- Source visibility: `lighthouse-core/lib/stack-packs.js` was successfully read
- Official evaluator: return code 0; `lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` failed
- Infrastructure: two pre-model, zero-action dependency failures were preserved under `infrastructure-attempt-1/` and `infrastructure-attempt-2/`; the same frozen setup was repaired without consuming another model episode
- Utilization evidence: no useful active `nvidia-smi` sample was captured; actual engine request/token/timing records are preserved in the episode artifact
- Cleanup: wrapper exit 0; local Lighthouse container namespace empty; owned GPU released at 2 MiB/0%; remote deployment removed; source provenance retained
- Interpretation: excluding model self-history while retaining external evidence was insufficient for this task. No retry or downstream gate was opened.

Artifacts:

- `results/model-intent/lighthouse-recovery/evidence-only-context/freeze.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/execution/solver/episode.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/execution/solver/grade.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/execution/solver/result.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/execution/solver/official.log.gz`
- `results/model-intent/lighthouse-recovery/evidence-only-context/metrics.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/capsule-validation.txt`
- `results/model-intent/lighthouse-recovery/evidence-only-context/cleanup.json`
- `results/model-intent/lighthouse-recovery/evidence-only-context/source-provenance/`
