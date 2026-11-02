# Native final-JSON safety correction

The actual pinned XGrammar accepted two concatenated JSON roots and an invalid
JSON escape under the output-contract answer pattern, while Python's strict
JSON parser rejected them. The new native regressions were **2 failed / 6 passed**
before the fix. No model or GPU was used.

The pattern now permits only generic ASCII identifier characters for key/mode
and decimal port digits, not expected values. Strict JSON parsing is retained.
Final raw text/schema and commits are persisted before parsing; malformed model
output is still a failed task, never truncated to its first JSON object.

The corrected native grammar plus benchmark/failed-output persistence selectors
are **20 passed**, including all four backend-method paths. Runtime, projector,
policy, visible inputs, verifier, task scoring and budgets are unchanged.
`red.xml` and `green.xml` preserve the actual native CPU results. GPU acceptance
of this exact correction is not claimed by these unit tests.
