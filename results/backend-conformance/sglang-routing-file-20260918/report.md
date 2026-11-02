# Native two-engine routing: functional pass, not joint acceptance

The fourth bounded attempt passed the native B6 **functional** selector on two
distinct real A6000 engines on the same host. All four attempts and failures
remain below: **877.665/1200 seconds cumulative**, with exact owned processes,
ports and GPU allocations released. Unused budget is closed, not a new run.

| Attempt | Actual outcome | Phase seconds |
| --- | --- | ---: |
| 1 | Native HiCache hash lookup could not find Ninja because the absolute Python path did not populate PATH. First request was acknowledged with unknown outcome; not blindly retried. | 215.863 |
| 2 | Cross-worker L3 restore reached512tokens and29,360,128 actual H2D bytes, but cold B's first native setup outlasted A's whole generation, so no output overlap. | 371.189 |
| 3 | Accounted cold wave followed by one same-key32-token wave proved real cross-worker output overlap; registry removal/rebind succeeded. Final reducer assertion failed on absent routing fact. Actual final branch text was not persisted, so its content is unknown. | 142.831 |
| 4 | All functional checks passed, including exact returned structured Commit, unchanged reducer state transition and continuation. | 147.782 |

The normal cold four-branch wave is retained as first-use/L3 evidence. Exactly
one further four-branch wave uses the same semantic keys and32-token sampling;
overlap means intersecting **first-output to terminal** intervals, not pending
or acknowledged requests. All15 attempts in the final selector are accounted.
This conditioning is not a latency-benefit comparison or extra choice generation.

The selector now constrains the already-public synthetic Commit it expects,
preserves raw final branch/reducer evidence before assertions, and requires a
successful exact Commit before continuation. It does not insert a fact into the
reducer or relax parsing. This is transport/structured conformance, not evidence
that a model repaired a task or inferred hidden information.

Final checks cover exact token transport, two real engines, stable affinity,
actual origin reassignment, native registry removal, canonical state rebind,
precise remote abort and unaffected sibling ownership, cached prefixes and
genuine output overlap, structured Commit and continuation. Root L3 readiness
was512tokens; measured branch cached counts were528 each and rebind cached521.

Source is exact `aa7ef2f7b14882b96b24312fcada134eb951e873` plus the two isolated
selector patches in `source/`. Backend/native code was unchanged. The existing
SGLang0.5.19/M6 patch, original OpenHands7B/C03/BF16 and file-L3 layout were used;
native ModelGateway0.3.2 ran manual/min_load with retries disabled. Corrected
PATH retains the existing task-owned Ninja/native hash build environment.
Independent CPU evidence was3 real-gateway fixture checks and2 actual XGrammar/
unchanged-reducer checks; model/DMA fixtures were not counted as GPU evidence.

Each `attemptN/report.json` records its actual outcome and cleanup; raw selector
results, worker identities and logs are preserved beside it. The final result
SHA256 is `19785a05affb6b9d83ca6c2fafc13626e170a68cdc04d3b58eb0d18ce704789f`.
No KV payloads or model weights are copied into this evidence archive.

**Not passed here:** cross-compute-host/distributed Mooncake or provider fault,
routed dynamic M5, native total-work/physical network accounting, task quality,
latency benefit, fairness or joint B8/Formal. Physical metrics absent from these
receipts remain unknown, not zero. The failed trained-candidate B7 quality gate
is independent of this native functional result.
