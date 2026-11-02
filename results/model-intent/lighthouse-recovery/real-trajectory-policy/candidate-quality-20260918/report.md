# Final-step1 candidate: both quality gates failed

The actual merged candidate was evaluated once on original Lighthouse and once
on the independent admitted Mistune task. **Neither episode produced a relevant
original red, matching green or effective model patch. B7 has not passed.**

| Actual candidate episode | Model requests/actions | Stop | Patch bytes | Original red / same green | Grade |
| --- | ---: | --- | ---: | --- | --- |
| Lighthouse | 100 / 100 | action_limit | 0 | false / false | Original official resolved=false |
| Mistune | 100 / 100 | action_limit | 0 | false / false | Fixed behavioral gate=false: no effective patch |

Lighthouse episode wall was1055.333seconds, with915.854seconds solver time;
Mistune wall was991.103seconds, with903.100seconds solver time. Official grading
took16.053seconds and the fixed heldout grade1.379seconds. GPU peak memory was
not emitted by the frozen episode receipts and remains unknown. All owned
remote processes and both GPU allocations were released.

The candidate is
`10f0224eeea5222849dec9f468f5ff3aef1811da07c199f50d87da6d6a541ac7`,
not the original reference. Its only adapter came from the four-state, one-pass,
one-update engineering pilot. CPU merge took24.078seconds; direct server-to-server
transfer took148.023seconds for15,231,302,414bytes, and destination file hashes
matched. The .134 endpoint is publicly addressed; this is not a claim that the
serving host has only private interfaces.

The reached pair-builder path error was fixed in `e3e1625` before either GPU
episode, preserving the original baseline path and strict identity guard. Those
two pre-GPU failures cost no model/GPU execution. Controller-v2, C03, BF16,
temperature0/seed17,32k context,4096 output limit and100-action limit were retained;
the stock vLLM actor still had prefix caching disabled. No teacher checks or
grading payload entered the heldout actor inputs. Mistune's four behavioral
checks are not official SWE-smith resolution; its original source label is not
rewritten.

The historical original-model negative was not rerun and is not a fresh matched
comparison. No relative quality improvement, causal training effect, multi-seed
confidence interval or generalization claim follows from this tiny prototype.
Low training loss and a valid merged checkpoint do not satisfy task quality.

Small exact receipts, pair configurations, assessments and the two empty model
patches are retained here. `artifact-manifest.json` locates the complete unchanged
raw histories and official log outside Git, recording sizes and hash provenance.
No dataset corpus, adapter tensors or merged model weights are committed.

[Completed bounded analysis](negative-analysis.md) found no measurement-invalidating
implementation defect: all200 prompts retained workflow/current-state instructions,
and saved tokens, text and shell parameters matched. Lighthouse entered a
controller-specific path-recovery lockout after view18:81 later shell attempts
were rejected before execution, not executed failures. Mistune repeatedly tried
pre-red repository writes;73 repeated writes were rejected. Neither actor called
reproduce or rerun_reproduction. This is a valid negative under the frozen
protocol, not a universal correctness or model-impossibility claim.
No additional training, checkpoint selection or unchanged quality rerun is
authorized by these results. Joint/Formal quality/performance acceptance remains
blocked; independent required native mechanism work can continue.
