# Matched command-space continuation: terminal result

**Outcome: official `resolved=false`, zero-byte model diff.** The current Copilot
session directly supervised the single approved continuation after a failed
worker admission that started no job. Published implementation was `11f1584`.
Seven historical read-only actions were replayed with matching actual observations
and controller commits; 43 new model requests consumed the remaining original
50 action slots. There was no new seed/model/prompt/token-budget change.

## Matching and causal limit

All 7,079 actual initial input token IDs match captured context7. However, the
first new output differs before receiving any experimental tool feedback:
the baseline proposed `find /testbed -type f -name "react.js" -not -path
"/testbed/stack-packs/packs/react.js"`, whereas the fresh continuation proposed
`ls /testbed/stack-packs/packs/`. Neither command nor raw text is equal.
Therefore the later trajectory cannot be attributed solely to the command-space
intervention. Historical physical KV was not restored. Source inspection finds
temperature0, seed17, max512 and ignore-EOSfalse in both first-Brain and projected
request definitions; actual sampling objects were not captured in the old trace.
No sampling or KV defect is proved, and no numerical explanation is fabricated.

## Actual subsequent behavior

All 43 new tool calls executed successfully: two native read-only Bash commands
and 41 views. There were no format errors, length caps, edits, tests or finishes.
All 43 raw outputs and canonical actions were distinct. The actor inspected
`amp.js`, found JSON files, then spent 40 views advancing through the unrelated
`proto/sample_v2_round_trip.json`, ending at lines2321-2380. Thus executable
commands and absence of exact repetition still did not establish localization
or repair. This continuation is not added to the independent fresh-episode
denominator in the earlier14-episode audit.

## Budget, sampling and cleanup

Engine scope348.983s, official grading9.596s, total runner393.363s;851.017s
remains in this distinct1200sphase. Native counters contain only the43new
requests:357450prefill and9268decode tokens. The one event-triggered external
sample was after engine loading, at42946MiB/0%/75.57W; it is not represented as
an active-decode sample. The exact owned tool container is absent, the runner
terminated, and GPU0 returned to0MiB/0% with no compute process. The remote
source deployment is retained; the temporary local transfer archive was removed.
