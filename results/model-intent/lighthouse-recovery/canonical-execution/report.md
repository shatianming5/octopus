# Canonical execution identity arm

**Official result: `resolved=false` (`QUALITY_FAIL`).** The canonical arm
completed normally, so the predeclared deliberative arm became eligible. It
made 12 proposals / 6 distinct raw proposals, with 6 raw proposed-action
repeats. Five actions were physically submitted and seven were rejected.

The only physical test was proposed as `yarn jest`, normalized to
`yarn jest --runInBand`, and timed out after reporting real passing-suite
progress. Its `proposal_hash` differs from its effective `action_hash`. The
next raw proposal explicitly requested `yarn jest --runInBand`; although this
was not an exact raw repeat, it had the same effective failed `action_hash` and
was rejected before execution. Six later exact raw repeats were also rejected.
Thus this arm had 7 effective failed-test repeat proposals, but zero repeated
physical executions of a failed effective command.

The completion invariant held: the edit and failed test left
`verification_required=true`; no visible test succeeded and no `done` action
was proposed or accepted. The run stopped at the 12-action limit with a
481-byte patch. The original official test remained failed. Retained evidence
does not establish the underlying cause of the official assertion failure.

GPU engine scope was 66.967 seconds on GPU7
`GPU-4cd604ce-20c4-9bb7-bed7-acc3cf542887`. The arm released its model process
and the selected GPU returned to 2 MiB and 0% utilization. Exact UUID-generated
solver/evaluator container names were not captured; the wrapper signal is not
a container identity. A read-only check at `2026-09-16T03:35:42Z` found the
complete `octopus-lighthouse-` container namespace empty.

Evidence SHA-256:

- Freeze: `0a5e903c70655347fdc0a2d7282f8fd6dedd6e3623c5e96bb5bb8050f54d4382`
- Episode: `4bfcda84b1a9b25672d9f02ab2df1624d68601fb93b197921d7fa0d99332f53e`
- Grade: `78ffa00feca4c002d800dea435161abe6b9774a26648386691094f87b18435e0`
- Prediction: `72156ef58251b31bfe81c8c58ddcf5cc012967184041d6b0353568a2420a2b70`
- Compressed official log:
  `567fa483eb48aaa7057a49e56bf7480b23101dbcfd9016823530efc72375b3d4`
- Decompressed official log:
  `4a62eeab1f3b3bfe6e77e9b2d5e13fdddc9743c6b6faaa9935b0387f68c71617`
