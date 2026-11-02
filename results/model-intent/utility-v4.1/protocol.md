# Utility v4.1 carrier-free architecture amendment

V4.1 consumes the genuine first field of the necessary normal action output.
It does not add a score, classifier, private label, duplicated choice, extra
normal-only turn, or separate inference request.

The normal schema and task remain the v4.0 bare object:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["dispatch_index"],
  "properties": {
    "dispatch_index": {"type": "integer", "enum": [0, 1]}
  }
}
```

`dispatch_index` remains the first and only property. The normal prompt changes
only by removing the v4.0 instruction that asked a redundant carrier index to
match the body. Runtime receives the same `inline_intent_checkpoints` argument,
streams the constrained `dispatch_index` digit when it appears at the end of a
cumulative output chunk, and applies that existing action value before the
closing brace. `BrainResult.text` and `output_token_ids` are the actual bare
object; `inline_intent.body` is the same fully parsed object.

V4.0 failures and smoke artifacts remain untouched. Its four episodes are seen
integration diagnostics and are not used to change semantic prompts, data,
thresholds, weights, capacity, or expected labels. V4.1 first reruns that same
seen four-episode cohort strictly as an integration diagnostic for the new
bare-object seam; it is not a held-out result or tuning surface. Only if that
diagnostic is functional does v4.1 run the exact v4.0 held-out cohort hashes,
seeds 8101/8202, method order, candidate prefixes, pressure, arrivals,
accounting, quality evaluator, and every scientific gate. There is no
normal-only companion. The six held-out traces decide.

The only output-format-dependent amendment is the pressure output-bearing step.
It is frozen from the Qwen tokenizer before v4.1 inference by finding the first
token boundary whose decoded cumulative bare object ends with the constrained
0/1 digit, and requiring that boundary to precede the closing brace for both
compact and standard JSON serialization. Marginal prompt guidance and carrier
envelope cost are exactly zero by construction.

The remaining authorized GPU budget is 1749 seconds on one physical GPU. A
valid per-seed gate failure is terminal `SCIENTIFIC_NO_GO`; instrumentation or
runtime failures require repair and rerun of the unchanged v4.1 protocol.
