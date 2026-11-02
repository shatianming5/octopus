# Qwen2.5-Coder-14B Lighthouse quality result

**Official result: `resolved=false` (`QUALITY_FAIL`).** The one authorized,
frozen `Qwen/Qwen2.5-Coder-14B-Instruct` episode completed normally, produced a
481-byte patch, and failed the original `FAIL_TO_PASS` test
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`. The official
evaluator returned zero, completed in 11.673 seconds, and recorded no
`PASS_TO_PASS` tests.

The model made 7 proposals / 7 distinct. Six were physically submitted with
zero runtime rejections, zero repeated physical actions, and zero known-failed
physical repeats: one list, one search, one read, one successful edit, and two
developer-test runs. After both developer tests failed, it selected `done` with
5 of the 12 action slots remaining. The edit redundantly duplicated
`id: lib.id` inside `lighthouse-core/lib/stack-collector.js`, and the task
remained unresolved. The retained result establishes neither that this
semantic no-op caused the official assertion failure nor the assertion's
underlying cause.

The first requested `yarn jest` executed as
`yarn jest --runInBand`; it made real progress through multiple passing suites
but hit the tool timeout (`exit_code=124`). The second requested
`yarn jest --offline` executed as `yarn jest --runInBand --offline` and failed
immediately because Jest does not recognize `--offline` (`exit_code=1`). These
are actual tool observations, not inferred test outcomes.

The solver completed 7 model requests with 6,756 actual prefill tokens and 167
decode tokens. There were zero preemptions, allocation rejections, or cache
evictions; maximum simultaneous scheduler requests was one. GPU engine scope
was 62.319 seconds of the 1,170-second execution budget on one RTX 4090 D in
BF16. End-to-end solver time was 64.995 seconds and wrapper terminal time was
96 seconds.

All owned compute resources were released. GPU
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` returned to 4 MiB and 0%
utilization with no compute process, and no owned model process remained.
Exact UUID-generated solver/evaluator container names
were not captured, and the wrapper signal is not treated as a container
identity. A read-only check at `2026-09-16T03:35:42Z` found the complete
`octopus-lighthouse-` container namespace empty. The verified model assets were
retained.

Raw evidence is preserved under `execution/`: `solver/episode.json`,
`solver/prediction.json`, `solver/grade.json`, the lossless
`solver/official.log.gz`, solver logs, start and terminal receipts. Frozen
execution source is retained under `source-provenance/`. The accepted immutable
asset receipt remains at `../download-terminal.json`.

Evidence SHA-256 values:

- Freeze: `68acdadc74924d3c7abbb82b6f8e7933eb5fafcb4f36d8c1ec97beb6369ff6d3`
- Episode: `3a6f86da8fd4fa95869cf6551534f34b318df2cb494f908484c405af445c0c93`
- Grade: `d3635c8ee49ae4544724084f9bf211db28c8ebe4eafe3dc7f2246411f64acd6c`
- Prediction: `72156ef58251b31bfe81c8c58ddcf5cc012967184041d6b0353568a2420a2b70`
- Compressed official log:
  `566aca882d8a1295118b50e7c70659256d66a81807db6d654b0c08143c02c26d`
- Decompressed official log:
  `b40d418faa76b58ad4c01c9d62be15528f47077671938b0b61acd05229703242`

The official failure ends this episode. It does not authorize a changed setup,
another model episode, or multibranch continuation.
