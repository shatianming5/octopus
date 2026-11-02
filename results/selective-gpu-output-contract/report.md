# Selective output-contract epoch

Source: `39b1ed8b2936a2db869b6ee42929d1548da6ec39`.
**48/48 workflows recorded; quality acceptance failed.** This epoch is not a
quality-preserving speedup result and does not overwrite the first epoch.

| Method | Complete final JSON | Solved | Failed final JSON | Total task E2E s |
|---|---:|---:|---:|---:|
| Full + APC | 8/12 | 6/12 | 4 | 87.397209 |
| Projected + APC | 5/12 | 5/12 | 7 | 66.907066 |
| Selective fixed | 5/12 | 5/12 | 7 | 66.985837 |
| Selective budgeted | 8/12 | 8/12 | 4 | 64.350924 |

The 22 failures report `JSONDecodeError: Extra data`. Full additionally has two
code-evidence failures among parsed responses. The original raw records,
summary, source/protocol, execution and cleanup receipts remain unchanged.

Native CPU grammar replay confirmed that the newly introduced answer regex
could accept two JSON roots and invalid backslash escapes. This is an output
contract implementation defect, not a reason to weaken `json.loads`, drop failed
cases or attribute a quality loss to KV reuse. The next correction restricts
only syntax to the benchmark identifier domain and retains actual values as
model outputs. Corrected-source acceptance requires fresh matched evidence.

No extra GPU correctness run was performed: the four cases passed in the first
epoch and their runtime/selector source was unchanged. GPU6 resource wall was
674 seconds; kernel time was unmeasured, and execution/transfer cleanup ended.
