# Resource-safe developer tests, same Lighthouse

**Official quality remains false.** Default serial Jest execution removes the
observed worker-spawn failure and returns real code feedback. The model reacts
with a source read, but does not repair its patch within the frozen12actions.
Multibranch and formal confirmation remain blocked.

| Measured outcome | Action-only native reference | Serial Jest |
| --- | ---: | ---: |
| Proposals / distinct | 12 / 9 | 12 / 9 |
| Actual submissions / rejected proposals | 10 / 2 | 11 / 1 |
| Known-failed external repeats | 0 | 0 |
| Nonempty inspections / unique results | 6 / 5 | 7 / 6 |
| Accepted edits | 1 | 1 |
| Patch bytes | 2211 | 2211 |
| Developer-test EAGAIN | yes | no |
| Developer-test code error returned | no | yes |
| Official resolved | false | false |
| Wrapper wall seconds | 91 | 98 |
| Native prefill tokens | 13346 | 23744 |

The patch is byte-identical to the reference:0% patch-size change and no quality
improvement. Submissions increase10%, wrapper time increases7.7%, and native
prefill increases77.9%; none is a performance benefit. These are single-seed,
single-method observations, not paired confidence intervals.

The first eight requested actions are identical, through the developer test
`["yarn", "jest"]`. The tool transparently executes
`["yarn", "jest", "--runInBand"]` and records `effective_args`. Explicit worker
options and test selectors are preserved. Both the actual episode and a
CPU-only replay of its predecessor's model edit/test return
`ReferenceError: Cannot access 'str_' before initialization`, not EAGAIN.
The complete suite still exceeds the unchanged30-second tool limit; partial
code feedback is retained and the timeout is explicitly reported, not passed.

After that observation the model reads the modified file, proposes an edit
whose old text occurs twice, reads again after the measured mismatch, and
proposes that same failed edit. Failure memory blocks the final submission.
The resource fix therefore changes recovery behavior without establishing
coding correctness. The surviving blocker is this policy's failure to form a
valid corrective edit under the frozen budget, not absent test feedback.
There is no evidence-backed infrastructure fix or further frozen-recipe
training currently ready; no additional model/prompt trial is launched here.

The original official evaluator separately reports one failed suite with
one failed and one passed test, without EAGAIN or worker-retry errors. Its
`resolved=false` is a genuine quality result, not an infrastructure exclusion.

The [protocol](protocol.json), [metrics](metrics.json), [tool replay](tool-replay.json)
and exact frozen source snapshots preserve this treatment. An initial wrapper
attempt failed before inference because an optional tmux start notification
required a running server. That zero-episode failure is retained under
`startup-failure/`; the signal is now best-effort like the existing terminal
signal, without starting a server. The single model episode completed in98
wrapper seconds (66.259engine-scope seconds), below1200. The task container
was removed and GPU9 returned to4MiB/0%. There were no concurrent requests or
KV evictions, and no downstream performance gate was exercised.
