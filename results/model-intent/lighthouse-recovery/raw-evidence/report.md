# Same-task read-evidence counterfactual

**Status: completed, official `resolved=false`, zero patch bytes.** The one
representation-only counterfactual did not repair Lighthouse. The prior
training recipe and all its outcomes remain unchanged; no training or
outcome-driven episode retry was added.

## Observed transfer failure

The existing recovery-LoRA Lighthouse episode's actual submitted edits at
indices 1 and 5 share the same seven-line `old`. Both failed the real tool's
exact-match precondition. Only model-visible source was used for diagnosis: the
first successful read covers all 91 lines without truncation. No gold patch,
evaluator test contents, new task, or teacher action was used.

Replaying the captured first edit on a disposable fixture reconstructed from
that read reproduces `ValueError: edit old text must match exactly once`, with
no file change. The fixture adds a terminal newline outside the edit span;
this is not a claim to reconstruct an otherwise unrecorded final file byte.

Three hypotheses were checked before changing inference inputs:

- Presentation annotations alone: falsified as the complete explanation.
  Original `old` matches 0 times; removing line-number prefixes still matches 0.
- Content/whitespace copying: the nearest already-visible source span is
  semantically identical, but the model also omitted one blank line. Removing
  annotations and restoring that blank line yields exactly one match. This is
  a diagnostic calculation only, never a production action transformation.
- Short synthetic recovery coverage: supervised edit targets cover three-line
  replacements and held-out edits cover four lines, versus this seven-line
  precondition. That limits transfer evidence, but does not by itself establish
  a causal sequence-length explanation or justify retuning the frozen recipe.

## One isolated representation treatment

`read_output_format=raw` returns the selected original text with line endings
and blank lines preserved instead of adding per-line presentation numbers.
The default remains numbered. The prompt, trained adapter, all other tools,
failure memory, JSON schema, 12-action budget, seed and official evaluator are
unchanged. In particular, malformed model edits still fail: there is no fuzzy
matching, annotation stripping, whitespace repair, forced edit, or hidden hint.

The direct raw-read/exact-edit regression first failed on the old renderer;
after implementation all 33 targeted tests passed, including unchanged numbered
behavior, empty-evidence behavior and rejection of whitespace-mismatched edits.
This validates the representation contract, not whether the model will copy
accurately or solve Lighthouse.

## Actual one-episode outcome

The freeze matched the reference model, prompt/solver code, schema, failure
memory, limits, seed and official evaluator exactly. Its only protocol delta
was `read_output_format=raw`; the execution source directory changed only to
isolate the deployment. The full episode and original official grade are in
`execution/solver/`; compact verified totals are in `metrics.json`.

- 12 proposals, 7 distinct; 8 actual tool submissions.
- 4 successful reads and one successful search; 3 actual edit mismatches.
- Zero repeated known-failed external actions; 4 repeated proposals rejected.
- Zero successful edits, zero patch bytes, no developer test action.
- Original official evaluator completed with return code 0; its FAIL_TO_PASS
  failed and `resolved=false`.

All three submitted edits now used literal `old="id: lib.id,"`, which occurs
twice in the already-visible source. The model varied `new` instead of making
`old` unique. Thus line-number contamination is not the sole failure cause;
removing presentation numbers alone did not produce valid editing behavior.
No ambiguous edit was auto-disambiguated or silently applied.

Native work: 12 calls, 49,349 prompt, 13,349 actual prefill, 457 decode and
36,000 adopted prefix tokens; no preemptions, evictions or allocation failures.
GPU engine scope was 30.294 seconds and total job wall time was 62 seconds,
within the new 1,200-second ceiling. The initial root request has no parent
frontier count; the single nullable field is exposed in the metrics rather
than treated as a measured quantity. No paired speed claim is made.

The original job exited and its named tmux session and task container were
gone; GPU9 returned to 4 MiB/0%. Gzip integrity and artifact hashes passed.
The current Copilot coordinator supervised this continuous run directly after
two Herdr worker startup timeouts, both before any worker tool call or GPU job.
The same-task quality gate remains unmet; multibranch and formal stay blocked.
