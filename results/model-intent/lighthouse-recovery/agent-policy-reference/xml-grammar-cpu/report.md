# XML grammar CPU compatibility report

- `scope_id`: `actor-xml-grammar-cpu-v1`
- `worker_id`: `8045f46a-fac4-4b03-9d55-72b3554dff75`
- `status`: `completed`
- `result`: `passed`
- `host`: `73F3-8x4090-134` (`198.51.100.10:20134`)
- `remote_deployment`: `/home/user/.octopus-tasks/xml-grammar-cpu-8045f46a-fac4-4b03-9d55-72b3554dff75`
- `runtime`: Python 3.12.14, xgrammar 0.2.3, Transformers 5.16.1, vLLM 0.28.0+cu129, Torch 2.13.0+cu129
- `limits`: `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, 60-second deadline per command

## Actual gates

1. `PYTHONPATH=src python -m benchmarks.openhands_xml_grammar_check --tokenizer /home/user/octopus-qwen-coder-reference-v1-20260915/model/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242 --output xml-grammar-check.json`
   - Command exit: 0
   - Started: 2026-09-16T18:18:15Z
   - Ended: 2026-09-16T18:18:22Z
   - Wall time: 6.49 seconds
   - Result: 62 grammar-positive cases, 16 grammar-negative cases, 7 semantic-error cases, 12 real tokenizer-mask cases, 476 accepted tokens
   - EOS acceptance included by the checker
   - Grammar SHA-256: `5056481db1038d428e1e61fa4e334e9bb756483d4c79f4de6bcefe4398dd6717`
   - JSON schema SHA-256: `a97f1e16cb58364edd1f1558f4c21564893c955243a54e253c02b5726af2a4de`
   - CUDA initialized: false
   - Model requests: 0
2. `PYTHONPATH=src python -m pytest -q tests/test_structured_decoding_native.py tests/test_inline_intent_native.py`
   - Command exit: 0
   - Started: 2026-09-16T18:18:31Z
   - Ended: 2026-09-16T18:18:45Z
   - Wall time: 13.17 seconds
   - Result: 20 passed, 14 Torch deprecation warnings, 10.79 seconds pytest time

The XML checker command completed and persisted exit 0 before the outer evidence-print wrapper returned exit 2 because Bash interpreted a later `printf "-- stderr --"` format as an option. The persisted command metadata and result are unaffected; the wrapper anomaly is retained in the parent worker transcript.

## Evidence and cleanup

- Full stdout, stderr, command metadata, result JSON, and deployment manifest are retained beside this report.
- Deployment manifest SHA-256: `98bf3c5af3e34505bba53e9fbb59d37d106fa4dc4eceb291accdc505b14eb40c`
- No task-owned process remains alive.
- No GPU work, model inference, model request, weight load, package install, shared upgrade, or source edit was performed.
- The private mode-700 deployment is retained only for concrete same-source follow-up.
