# Deliberative 14B conditional arm

**Official result: `resolved=false` (`QUALITY_FAIL`).** Arm B ran only after
arm A completed normally with official failure. The deliberative arm used all
12 action slots for inspection, produced no edit and no patch, and never ran a
developer test.

It made 12 proposals / 7 distinct raw proposals, for 5 raw proposed-action
repeats. All 12 were physically submitted: one list, five searches, and six
reads. One search failed because its requested path was absent; the other
inspections succeeded. Because no test was executed, there were zero effective
failed-test command repeats and zero repeated physical failed-test executions.

The completion invariant was not activated: there was no edit/write and no
failed test, so final `verification_required=false`. No `done` action was
proposed or accepted; the episode stopped at the action limit. The original
official test remained failed. Retained evidence does not establish the
underlying cause of the official assertion failure.

GPU engine scope was 92.860 seconds on GPU7
`GPU-4cd604ce-20c4-9bb7-bed7-acc3cf542887`. The arm released its model process
and the selected GPU returned to 2 MiB and 0% utilization. Exact UUID-generated
solver/evaluator container names were not captured; the wrapper signal is not
a container identity. A read-only check at `2026-09-16T03:35:42Z` found the
complete `octopus-lighthouse-` container namespace empty.

Evidence SHA-256:

- Freeze: `dc1b6c304cd44af5001fdd18e8d7601a7edbea34e9281c4de7d532a175aabe5b`
- Episode: `b7c7fee5f8f45a091ad57b39767d077f96aabc722c4bc943a83a67bac2f73d4e`
- Grade: `be7236291fcd92cd468d67907607c06b5b26d5e28a6c502eb839f090d62d5b3d`
- Prediction: `bff766023d9c42e2fd4178adcea5323ac62800e1478ff48ecab936edfe2372b0`
- Compressed official log:
  `a4b7fdfaaad952addc0dfa2fb14dc37833bc1d2bd8de17568c438f051c52bab9`
- Decompressed official log:
  `e2e251d7b69c2d73afcf36135afa367424c02fdb17c54841bb11d9d81cffbf81`
