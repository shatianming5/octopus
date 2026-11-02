# Finish-feedback attribution

**Corrected attribution:** No instruction loss, finish-feedback loss, producer mismatch, or executor defect was observed at this boundary. The prior report incorrectly treated the non-actor branch’s appended verification wording as the only explicit statement. In fact, `benchmarks/openhands_tool_protocol.py:71`, imported for actor formats by `benchmarks/lighthouse_quality_solver.py:240-244`, already states verbatim: “After a change or failed test, finish is invalid until a test succeeds. Timeouts do not verify a fix.”

The actual frozen 1,541-token initial prompt was decoded with pinned tokenizer revision `c03e6d358207e414f1eca0bb1891e29f1db0e242` and contains that exact sentence. The original managed report record `27f363cd-ecbc-4f21-a6a1-df690487fa6f` remains retained and is superseded by this correction.

The first rejected finish was action 13 at workspace revision 2; the terminal rejected finish was action 45 at revision 4. Both, and all 13 between them, had `error_type="verification_required"` and the exact message: “A changed workspace or failed test needs a successful test before done.” Verification was true before and after every rejection. Four successful edits correctly set the requirement; no test action occurred, and only a successful test can clear it in `src/octopus/tool_memory.py:192-196`.

All 15 finish rejections had a subsequent model request. Each exact error/type fact appears in that next request’s captured `capsule_json`; its 189-token `capsule_token_ids` sequence occurs exactly once contiguously in the corresponding `prompt_token_ids`. Thus, after action 13, the true precondition was not hidden or compacted away. The final episode action was a successful `view` at index 49 and had no next request; no terminal finish feedback was lost.

The execution route remained available and unchanged: the frozen grammar contains `execute_bash` and `finish`; `benchmarks/openhands_tool_protocol.py:22-35` publicly lists `yarn jest [ARGS]`, and lines 299-302 map a repository-root `yarn jest` command to the semantic `test` action. No observation claims testing is impossible. All 15 rejected finish calls, including the first, therefore reflect the model ignoring correct available instructions and feedback, not a producer/executor failure.

This is only the observed failure-boundary attribution, not a global correctness verdict. The 512 producer-returned output IDs and 511 backend decode count for malformed capped generations remain distinct accounting units; no off-by-one bug is inferred.

**Evidence:** `execution/solver/episode.json`; `initial-input-record.json`; `benchmarks/lighthouse_quality_solver.py:237-244,379-461,694-699`; `src/octopus/tool_memory.py:101-102,111-148,192-196`; `benchmarks/openhands_tool_protocol.py:22-40,67-72,299-302`.
