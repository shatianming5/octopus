# Native Qwen tool protocol Lighthouse terminal report

**Scope:** `lighthouse-native-tool-protocol-v1`  
**Status:** terminal, original official quality failed; no quality retry performed  
**Result:** `QUALITY_FAIL` (`official_resolved=false`)

The single authorized episode completed through the unchanged official Lighthouse evaluator. It produced 12 native JSON call bodies and 12 constrained inference proposals, all parsed losslessly to canonical actions with zero parse errors. Ten actions reached the repository tool, two known-failed edit re-executions were rejected before tool submission, and the episode stopped at the 12-action limit with a zero-byte patch. The official `js-libraries-test.js` FAIL_TO_PASS test remained `FAILED`.

The actual model input used the pinned tokenizer template SHA-256 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, contained the shipped `done/edit/list/read/search/test/write` tool registry, ended at the native `<tool_call>` pending-call prefix, and used tool-role history on later turns. Closing/history XML was template-supplied rather than sampled output. Plan, citation, and custom reasoning fields are N/A for this native call-body format.

Runtime evidence: 12 raw bodies, 6 distinct raw bodies, 6 distinct canonical actions, 10 physical submissions, 2 rejected known-failed submissions, 1 physical edit failure, 9 successful actions, 4 successful inspection repeats with 0 output novelty, 12 serial observation commits, 0 cache evictions, 0 allocation rejections, and maximum scheduler concurrency 1. The 12 unique serial branch labels are state-commit identifiers, not concurrent branches. GPU engine scope was 48.467 seconds; the active sample was 91% utilization and 44,300 MiB on GPU 9. Total runtime prefill/decode was 59,044/348 tokens; prompt range was 1,623-13,097 tokens.

CPU replay remains protocol-only evidence: 12 canonical round trips, 11 exact next capsules, root 1,491 tokens, maximum input 5,774 tokens, and zero model requests. Compared with observation-state, only `decision_format` changes. Compared with evidence-only-context, both `decision_format` and `history_format` change; no equal-input claim is made.

Three preserved infrastructure attempts consumed zero model requests: an incorrect local asset symlink, an unavailable local `python` alias after completed source copying, and an omitted SSH username that failed authentication before remote execution. The repaired episode then ran exactly once; negative quality was not retried.

Cleanup is complete: owned remote PID `1921407` is absent, GPU 9 is at 2 MiB and 0% utilization with no compute process, no container from the frozen official image remains, and temporary prompt-validation files were removed.

Primary artifacts:

- `results/model-intent/lighthouse-recovery/native-tool-protocol/metrics.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/artifact-validation.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/cleanup.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/wrapper-terminal.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/execution/solver/episode.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/execution/solver/grade.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/execution/solver/result.json`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/execution/solver/official.log.gz`
- `results/model-intent/lighthouse-recovery/native-tool-protocol/source-provenance/`

The immediate scope is terminal. The parent end-to-end quality, multibranch, and formal-gate objective remains incomplete.
