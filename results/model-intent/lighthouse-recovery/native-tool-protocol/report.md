# Pinned Qwen native tool-call adapter

**Closed: original official `resolved=false`, zero patch.** The shipped Qwen tokenizer template explicitly defines
function signatures in `<tools>`, assistant calls using `name`/`arguments` inside
`<tool_call>`, and tool-role results. Earlier custom JSON action/plan envelopes
did not use that complete protocol. Grammar-free references retained the same
custom prompt and are not this counterfactual.

`qwen_tool_call_v1` uses the actual backend tokenizer's `apply_chat_template`
for function definitions and conversation history. It prefills the native
call-opening tag, then generates one constrained JSON call body. Parsing maps
that body losslessly to the existing repository action; the original action
schema still defines tool arguments and failure/path/verification constraints.
Native calls do not require custom model plan/citation/reasoning fields, while
runtime-owned tool observations remain persistent. Raw output is saved before
parsing, so even a budget-truncated body remains auditable.

This is **native call-body completion**, not unrestricted multi-call chat.
The opening tag belongs to counted input; closing/history framing is supplied
by the tokenizer from the parsed call and is not claimed as generated output.
There is still one model request per action, twelve proposals and at most512
generated tokens per request, without source/patch hints or forced edits.

The [CPU replay](cpu-replay.json) matches the shipped template fixture exactly,
roundtrips12actual prior canonical actions and11next observations, with1491root
tokens and maximum5774input tokens. These checks do not claim task quality.

The [single actual episode](terminal-report.md) produced12valid native call
bodies/6distinct canonical actions. Ten tools ran; one edit failed with zero
old-text matches, and two known-failed re-executions were blocked. After an
empty search, the model tried an unsupported literal replacement in an
inspected file. There was no patch, test or done. The native template/hash,
registry and pending-call prefix were verified in actual inputs. A separate
[live capsule check](live-capsule-validation.json) confirms all12token roundtrips
and11/11exact preceding action/result/output records, including failures and
controller rejections, rather than substituting CPU replay for live evidence.

Work was59,044prefill/348decode tokens,48.467engine seconds and81wrapper seconds.
The active GPU sample was91%; compute/containers were released. Three preserved
pre-model infrastructure attempts made zero model requests. No quality retry ran.
