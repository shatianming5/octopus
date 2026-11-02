# Fresh inference after positive search evidence

**Result: the canonical tool action is still the same search.** A fresh stock
vLLM engine received action1's exact1444input tokens after the first successful
search returned three references. The nested decision schema string, model,
precision and direct sampling settings were preserved; prefix caching and
Octopus were absent. No tool or official evaluation was run in this probe.

The fresh explanation differed, but the tool action still exactly matched the
captured repeated search. It did not acknowledge the concrete source locations.
This argues against physical reuse as the sole explanation **for this input**;
it does not prove global KV correctness or rule out all numerical differences.

The comparator now parses nested decisions correctly, rather than comparing
reasoning text as if it were a tool action. Its pre-output source snapshots,
exact prompt tokens and schema, output156tokens, raw text and scheduler
identity are preserved. A targeted regression covers this comparison shape.

Stock scheduler and disabled prefix caching were checked. Generation took
2.811297seconds and engine scope24.266599seconds; the one job was below its
300-second cap. Active GPU snapshot was43,144MiB/92%; terminal GPU was4MiB/0%
with no compute process. This is attribution evidence, not task quality.

An additional CPU-only capability probe found that installed xgrammar0.2.3
does not enforce the tested allOf/not exact-action exclusion. Such a schema
must not be advertised as preventing repeats without a real acceptance check.
