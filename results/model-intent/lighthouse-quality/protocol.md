# Lighthouse-10176 absolute quality preflight

This is a single-method quality gate, not a cache-policy pilot. The frozen
Qwen2.5-7B-Instruct plus existing LoRA merge receives the exact official problem
statement and a clean base checkout at
`a5dbdd2eff4238213dcaad404d0e06d8d5d2489b`. It must inspect and modify the
repository through bounded real `list`, `search`, `read`, `edit`, `write`, and
`yarn jest` tools. The prediction is the model-created `git diff`; no
worker-authored fix, gold patch, or handcrafted fallback is permitted.

The official task checkout is pinned at
`8d3ed32f37af9ceb8b5778b92a0088384ed057d4`, and the immutable official image is
`sha256:b8d51faee583632569625837d3f5b1a8a58f19c90f659597bd33234b145f7eae`.
The solver container contains only the clean base repository. `eval.sh`,
`test.patch`, `gold.patch`, and `hints.md` remain outside that container and are
used only after the model terminates. The two problem images are not converted
into descriptions or otherwise leaked; this frozen baseline is text plus base
code because the existing model is text-only.

The official denominator is one `FAIL_TO_PASS` file,
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`, and no
`PASS_TO_PASS` files. The official injected tests cover both issue-reported
bugs. The absolute quality floor is official `resolved=true`: every listed
FAIL_TO_PASS and PASS_TO_PASS test must pass under the pinned task evaluator.
The base-only masked evaluator sanity is expected unresolved and is
instrumentation evidence, not a model outcome.

The model gets at most 12 actions, 512 generated tokens per action, a 16,384
token context, 8,192 journal tokens, and 30 seconds per tool call. The remote
GPU episode has an 1,100-second hard timeout within a 1,200-second GPU budget.
The official evaluator has an 1,800-second CPU timeout and runs with one
container, four CPUs, 8 GiB memory, and no network. New task assets total
1,104,822,377 bytes, below the 2 GB cap.

Quality failure terminates Step3 with no task substitution. Infrastructure
failure requires repair under this unchanged setup. Even if quality passes, a
paired pilot remains unauthorized unless the actual workload also exhibits a
necessary Brain parent/reducer action and measured natural competing KV-prefix
contention. The sequential P1-style action loop is not itself such evidence.
