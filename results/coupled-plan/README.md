# Coupled-SWE plan: pre-build checks (Q1–Q6)

Reports for the six questions answered before building the coupled datasets (2026-10-20). Each report was written by a
worker, checked by an independent reviewer, and carries a "审查更正" (review corrections) section at the top; the
corrected statements there override the worker text below them.

| file | question | one-line result |
|---|---|---|
| q1.md | supply of same-file 2–4 task bundles | 420/80/76 capped with the collide chain (same function/class or ≤20 lines), but 48.5% of those bundles are linked only by "same class"; under the rule the user chose (≥1 pair ≤20 lines) the lower bound on the old plans is 353/54/31, so Val and Test fell short of 60 before more eval repos were added; the 91% acceptance sample is not reproducible as written |
| q2.md | is the KV mount faithful on long mounts | no decay with length: ≥2k mounts push the reader in the text direction (cosine 0.38 / 0.72); lagging mounts carry ~25–33% of the text effect; "grows with length" is collinear with source-ahead positions. That e14's long-mount harm is irrelevant content comes from e14's usage stats (8% / 10% of read pairs), not from Q2 |
| q3.md | can sibling reads prevent same-file merge conflicts | only in e14's async setting (main edits while branches run) and with all shared-file conflicts as the denominator: 74% hit main's own post-fork edits, sibling reads touch ≤22.3%; with sibling-only conflicts as the denominator the answer is uncertain (replay drops ~0–55%); main-direction reads were not evaluated |
| q4.md | CooperBench feasibility | 9 Python libs = 24 PRs / 165 features / 560 pairs; images fetchable; combined.patch is the intended both-pass oracle (checked end to end only on click task2800 f1/f2 at the time) |
| q56.md | e14 read-control re-score; local vs whole-episode advantage | after symmetric re-scoring content reads (−0.061) ≈ placebo (−0.057), both CIs include 0, ~0.03 residual scoring bias; the plan's (iii) filter leaves 6 pairs; own-task correlation A_branch +0.45 vs A_ep +0.42 (corr(A_ep, y) is constructed) |

Paths inside the reports point to the session scratchpad and to node-94 / node-136 / node-134 work directories of that day.
