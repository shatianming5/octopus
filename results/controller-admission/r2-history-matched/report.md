# Information-matched history: first pair fails, remaining runs stopped

The CPU prototype at library `a648156dd611c7c5eb707fecdbedff5e9cc3f604` /
native `9ee24e12c619f22ff85bbf3ce527328dc89a8d84` fails its first matched
seed1601 screen. History restores BG reuse but costs too much FG reuse.
Selective history preserves only 48.382838% of actual original fixed's
positive FG savings, adds 864 BG tokens versus the history-aware blanket
control, and increases conditional protection 0.182226%.
Seed1702 outcomes and GPU runs were not executed. The rule was not retuned.

## Online signal and frozen comparison

The exact earlier `82397074` history predicate is positive initial cached
adoption after native `_record_scheduled`. Probes, completed-prefill inference
and future requests do not qualify. Both actual `807d46de` streams observe
the warmup's 13,392-token adoption and subsequently expose 836 native-FREE
body copies before another allocation. This establishes decision-time
availability, not beneficial selection.

Both controls receive the same information model: original semantic candidates
union currently cached leading prefixes learned from confirmed adoption.
Each learns from its own execution, never the other arm's outcomes.
Original semantic fixed remains a separately labeled reference. No historical
`ConsumerIntent`, owner or lease is invented.

Before outcomes, `policy-plan.json` freezes the existing analytic
`W_D(t) = D*t + t*(t+1)/2`, weighted per logical block by the maximum of
distinct remaining definite consumers and distinct initial cached adopters.
Overlapping historical adopter identities are unioned; the counts are not
summed. These are observed-repeat/declared-use proxy units, not calibrated
future probabilities or measured FLOPs. Current-plus-RUNNING capacity,
prefix-closed marginal ranking and stable soft grants remain unchanged.
History evidence/grants clear at cache reset, without a fitted expiry horizon.

## First matched pair

The original fixed column is actual serving evidence. The two history
columns are conditional native executions on the same recorded value807
allocation opportunities, output tokens and release times.

| Metric, seed1601 | Original semantic fixed | History blanket | History selective |
|---|---:|---:|---:|
| FG parent recompute | 9,648 | 23,024 | 22,160 |
| FG savings versus APC/queue | 24,240 | 10,864 | 11,728 |
| Fraction of original fixed savings | 100% | 44.818482% | 48.382838% |
| BG prompt work | 1,997,442 | 1,984,066 | 1,984,930 |
| Recurring BG body recompute | 13,376 | 0 | 864 |
| FG parent + recurring BG recompute | 23,024 | 23,024 | 23,024 |
| Conditional protected block-seconds | Not this measurement | 404,842.267447 | 405,579.994361 |
| Native forced preferred-copy evictions | Not this comparison | 1,439 | 1,351 |

Both history controls fail the >=75% FG-savings condition. Selective also
fails no-added-BG versus the matched blanket's zero floor and lower protection:
it adds 737.726914 block-seconds. Conserved token sums do not imply equal
FLOPs, context positions, latency or GPU performance.

Both controls have zero postdecision omission relative to their own
same-information, same-state blanket shadow and zero cross-step unpreferred
occurrences. Selective differs from that same-state shadow's native victims
in three draws, with 142 selective-only selected-key occurrences.
Its new/reused grant-copy occurrences are 2,866 / 1,366,168; forecast excess
appears in 16 draws, peaking at 851 blocks.

The post-run no-demonstrable-saving association falls from 116,479.449131 to
110,990.683729 block-seconds, but another-copy ambiguity remains substantial
(98,085.961789 / 95,336.801850 block-seconds). This is not a causal waste
reduction and does not override higher total protection or failed work gates.

## Attribution and native selection boundary

The extra 13,376 / 12,512 FG tokens cannot be partitioned exactly between
discretionary DROP and forced eviction using the retained aggregates.
They lack the per-draw selected-key/grant ledger joined to each subsequently
lost logical FG prefix. The 1,351 forced grant-copy evictions alone do not
prove which later tokens they caused.

The source does establish a narrower mechanism. `history_candidates.py`
retains every currently valid grant via `kept = granted | new`.
When ordinary free supply cannot satisfy a native draw, QueueRetention takes
future-preferred blocks in native LRU order, followed by queued-preferred
blocks; that fallback does not rank marginal W. Source locations at the tested
revision are `src/octopus/queue_retention.py:48-64,192-207` and
`benchmarks/history_candidates.py:190-212`. This is not yet proof of
avoidable prefix loss on a particular request.

The next separately authorized CPU scope must first demonstrate a native
counterexample where actual forced head eviction loses more usable prefix
than valid tail eviction satisfying the same draw. Any fix is limited to
current unavoidable native shortfall, not another forecast or value formula.
The failed history screen above remains immutable.

## Validation, cost and evidence

Nine focused native cases pass, covering actual initial adoption, held-to-FREE
visibility, overlapping observations/physical charge, matched information,
no fake intent, reset, empty-history semantic parity, stable grants, exact
conditional shadow measurement, duplicate-copy ambiguity and elided-chunk
adoption capture. The prior 34 cases were not repeated. There were no
measurement failures or post-outcome rule/group changes.

The one availability check costs 5.854803 CPU seconds; blanket and selective
outcomes cost 32.943668 / 42.679647, totaling 81.478118 measured CPU seconds.
Each control ran once. Serialization has separate sidecars; these offline
costs are not serving overhead. No second-seed outcome or GPU run was used.

Full local root:
`/home/ubuntu/experiments/octopus/r2-history-matched-20260912-d3c687/`.
The matching remote root is
`/mnt/disk2/user/octopus-contention/r2-history-matched-20260912-d3c687/`.
It contains `result.json`, `policy-plan.json`, `evidence/online-signal.json`,
`evidence/blanket-1601.json`, `evidence/selective-1601.json`, native test logs,
serialization sidecars and `evidence/history-matched-source.bundle`.

R2 is not accepted. No final policy, R3 or R4 result follows from this scope.
