# Initial Selective B5 GPU acceptance

**48/48 workflows completed, but quality non-regression failed.** Runtime source:
`a9638a585a6de38900945d3d88ac03d10b3af513`. This result is preserved unchanged.

| Method | Final answer correct | Full task/evidence solved | Total task E2E s |
|---|---:|---:|---:|
| Full + APC | 12/12 | 4/12 | 62.087847 |
| Projected + APC | 12/12 | 5/12 | 54.303586 |
| Selective fixed | 12/12 | 5/12 | 54.260374 |
| Selective budgeted | 11/12 | 4/12 | 54.569403 |

All four Full-correct cases (task-01/02/05/11) regress in each projected method.
Thus lower raw E2E values do not establish quality-preserving cost improvement.
Full code evidence fails in eight cases; projected DOCS evidence fails in seven
cases per method. All 36 projected DOCS capsules contain both current/legacy
observations and their constraint. In task-01 the model emits only the current
fact; in task-00 Full quotes DOCS text as code evidence; budgeted task-08 puts a
Python-dictionary-shaped string in the final answer field.

The native selected-prompt correctness gate passed **4/4**, warm/cold outputs
matched. No stale-KV or inheritance correctness failure was observed. The GPU
chain used GPU6, took 652 seconds wall including setup, and terminated with no
owned process; kernel time was not measured. Raw records, original summary,
protocol/source, JUnit, logs and cleanup receipts remain in this directory.

The shared output contract was subsequently repaired uniformly, without changing
visible values, expected answers, semantic verification or scoring. This first
epoch is not relabeled as success. The corrective source requires a fresh matched
matrix before any quality/cost acceptance claim.
