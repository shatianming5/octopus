# Actor reproduction quality: terminal result

**Mechanism FAIL; official resolved=false.** The one authorized
`actor_reproduction_v1` episode completed 100 model requests and stopped at the
action limit. The actual first input exactly matched the prospective 1,772-token
C03 record. The fixed actor, tokenizer, BF16 stock-vLLM profile, seed 17,
original issue/base/image, 4K output limit, 32K context and unchanged official
grader were preserved.

## Independent mechanism judgment

The actor created zero reproduction definitions and executed zero original or
patched assertions. It first viewed four paths, then changed
`stack-packs/packs/wordpress.js` from `id: 'wordpress'` to
`id: 'wordpress-react'` at action 4 before any red observation. Its finish at
action 7 was correctly rejected with the same-assertions red/green requirement;
four more views followed at actions 10-13. There were 90 normal messages in
total. One response appeared at action 9 and then throughout actions 15-99:
86 occurrences overall, including an 85-message identical terminal suffix.
Neither `reproduce` nor `rerun_reproduction` was called.

This is not causal evidence for either public behavior. The edit treats already
distinct static IDs (`wordpress` and `react`) as duplicates, does not exercise
runtime duplicate stack-pack collection, and does not hide `React (Fast Path)`
from the user-visible libraries audit. Coverage is therefore none of the two
reported behaviors, not partial success.

The subsequent [first-principles diagnosis](first-principles-diagnosis.md)
replayed the real controller and decoded all actual inputs without inference.
It distinguishes completion gating from pre-edit sequencing and progress,
confirms the later loss of source/rejection history, and shows that the exact
patch additionally removes the WordPress pack for a WordPress detection.
The public diagnostic fixture was not actor input or an actor-owned reproducer.

## Original official quality

The model-authored 429-byte production diff is nonempty and syntactically
legitimate, but ineffective for the issue. The unchanged original grader
completed normally and
`lighthouse-core/test/audits/dobetterweb/js-libraries-test.js` failed, so
`resolved=false`. This negative result does not authorize multibranch or another
model episode.

## Accounting, execution and cleanup

The relay used 1,080.907 of the new 1,200-second phase, leaving 119.093 seconds;
the historical 679.251/520.749-second negative ledger is unchanged. All 100
outputs stopped naturally, totaling 22,962 output tokens (maximum 443). The
largest input was 28,670 tokens, or 32,766 with the reserved 4,096 output tokens.
An event-triggered active sample recorded GPU0 at 44,978 MiB, 91% utilization
and 323.73 W for engine PID 42887.

The completed episode was preserved before a driver-only validation mismatch:
the row contains `actor_reproduction` evidence but no redundant top-level
`workflow` key. That validation was corrected with zero additional model
requests and no rerun. Runner PID 2372955, engine PID 42887, the input-check and
episode containers, and all GPU0 compute applications are gone; GPU0 returned to
0 MiB and 0% utilization.
