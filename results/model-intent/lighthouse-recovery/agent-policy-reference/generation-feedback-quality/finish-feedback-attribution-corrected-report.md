# Corrected finish-feedback attribution

**Outcome:** No instruction loss, feedback loss, producer mismatch, or executor defect was observed at the finish-verification boundary. The prior managed result `27f363cd-ecbc-4f21-a6a1-df690487fa6f` is retained but superseded because it missed explicit wording already present in the imported original-XML instructions.

`benchmarks/openhands_tool_protocol.py:71` states verbatim: “After a change or failed test, finish is invalid until a test succeeds. Timeouts do not verify a fix.” Actor-format `solver_instructions()` imports and returns that constant. CPU decoding of the actual frozen 1,541-token first prompt with pinned tokenizer C03 confirmed the exact sentence is present.

All 15 finish proposals were rejected consistently with `error_type="verification_required"` and “A changed workspace or failed test needs a successful test before done.” Every rejection had a next request containing the exact fact in its captured capsule token sequence. Four successful edits set `verification_required=true`; no test action occurred; the unchanged `execute_bash`/`yarn jest` route remained documented, grammar-selectable, and mapped to the semantic `test` action. The final episode action was a successful view, not a finish, so its lack of a next request is unrelated.

**Attribution:** All 15 rejected finish calls reflect the model ignoring correct available instructions and feedback. There is no source correction target at this failure boundary. This remains a narrow trace attribution, not a global correctness verdict. The 512 producer-returned output IDs and 511 backend decode count remain distinct accounting units, with no inferred off-by-one defect.

**Evidence:** `finish-feedback-attribution.json`; `execution/solver/episode.json`; `initial-input-record.json`; `benchmarks/openhands_tool_protocol.py:22-40,67-72,299-302`; `benchmarks/lighthouse_quality_solver.py:237-244,379-461,694-699`; `src/octopus/tool_memory.py:101-102,111-148,192-196`.
