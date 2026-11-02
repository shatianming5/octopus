# Selective common output-contract CPU repair

Scope: application task instructions and structured-output shape only. Runtime,
context compiler, branch policy, visible case data, online verifier, budgets,
offline task scoring and aggregate quality gate remain unchanged.

A native XGrammar CPU regression on the first implementation observed **5 failed,
1 passed**: the grammar accepted a missing legacy fact, duplicate current-source
facts and a malformed final-answer string; code/log tasks also received unrelated
DOCS output requirements. Positive payloads passed, and wrong fact values were
accepted structurally but rejected semantically. The first harness probe used
mismatched JSON separators; it was corrected before counting these five failures.

After the repair, the six output-contract selectors plus the eight existing
selective benchmark selectors passed: **14 passed**, no model/GPU execution.
The native compiler used one thread, byte vocabulary and explicit compact JSON
separators. `red.xml` and `green.xml` preserve actual results.

DOCS schemas require exactly two ordered source slots. Values/excerpts are not
filled from hidden labels. Final grammar constrains only the already-required
three-part answer format. All methods share the same change. The original
48-record quality replay still fails its original gate; no evaluator relaxation
or retroactive success is permitted. GPU acceptance of the corrected code remains
pending, and the previous four runtime correctness cases need not be repeated
because their runtime/test source is unchanged.
