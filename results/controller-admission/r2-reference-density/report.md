# R2-B diagnostic: initial ranking works, unnecessary protection increases

The missing common-reference reducer is implemented at library
`b00964acb5882ef70e10ac3270765b2d8c02956e` / analysis snapshot
`3fa39c6af7b11cf5ac31b1e7e0b3da180a876867`. It evaluates the eight actual
`807d46de` traces without changing serving source, policy, gold, inputs or
held-out data.

**R2-B does not pass its unnecessary-protection requirement.** Initial
KEEP groups have larger common-fixed-reference benefit density than DROP
groups, but actual no-saving exposure increases 17.4200% and its pooled
exposure-weighted rate rises from 18.1291% to 22.8864%.
The separate measured BG P95 failure of +1.057716% remains unchanged.

## Fixed diagnostic rule and coverage

The unit is one logical full semantic body block in its original FG
checkpoint/consumer episode. Its first eligible binding KEEP/DROP label is
immutable; repeated observations do not repeat its reference benefit or
exposure. Later flips remain separate. Both groups use the same whole-episode
fixed exposure and actual fixed-versus-queue/APC incremental adoption or
recompute benefit.

All 2,118 body units per seed have valid positive fixed-reference exposure;
unmatched coverage is zero. Shared ancestors, native-covered/never-binding
strata and parent mix remain separately visible. All positive reference
benefit belongs to consumer IDs remaining at the first label; the
prior-started reference-benefit partition is zero. Timing classification uses
matched consumer events in the VALUE run's own clock, never absolute clocks
from different runs.

The grouping rule was frozen before group reduction, after aggregate discovery
outcomes were already known. These are diagnostic associations, not
independent confirmation or a causal treatment effect of DROP. In particular,
all DROP units in both seeds come from one producer; blocks are not independent
experimental replicates.

## Common-reference ranking

Both APC and queue references give the same results:

| Seed | KEEP / DROP body units | Mean per-unit reference density, KEEP / DROP | Pooled reference density, KEEP / DROP |
|---:|---:|---:|---:|
| 1601 | 2084 / 34 | 0.05360434 / 0 | 0.06316121 / 0 |
| 1702 | 2042 / 76 | 0.04112242 / 0 | 0.05862177 / 0 |

Density units are saved parent tokens per fixed-reference physical
block-second. DROP's reference denominator is valid and positive; its
reference benefit is zero. Its **actual policy-protected** benefit/exposure
ratio remains undefined, as required. Neither that null nor the earlier
arm-level density comparison is substituted for this ranking.

## Actual unnecessary protection

These rates count actual positive future-only exposure with no demonstrable
incremental saving against the controls, including exposure accumulated by
keys later dropped and lost. They are not a claim that every interval with
later benefit was necessary.

| Seed | Fixed exposure-weighted no-saving rate | Value exposure-weighted no-saving rate |
|---:|---:|---:|
| 1601 | 15.3244% | 16.8433% |
| 1702 | 21.5658% | 30.9517% |
| Pooled | 18.1291% | 22.8864% |

Pooled no-saving exposure increases from 127,824.2624 to 150,091.2890
block-seconds, despite lower total exposure. Thus reducing total preference
exposure did not reduce wasted protection: the current rule also loses
otherwise beneficial FG reuse. This is a measured negative result, not an
undefined-denominator artifact.

## Validation and artifacts

Ten focused reducer tests and one actual-export-status regression pass.
A real schema-status mismatch found before grouping was preserved and fixed;
no group or time selection was used to repair it. Offline processing takes
13.8420 seconds with a separate write sidecar; serving costs are unchanged.

Full frozen plan, result, per-unit evidence, tests and analysis source bundle:
`/home/ubuntu/experiments/octopus/r2-reference-density-20260912-d3c687/`.
The decision-bearing detailed result is `evidence/reference-density.json`.
The actual GPU report is `../r2-current-admission-gpu/report.md`.

## Next bounded lifetime hypothesis

Initial online binding DROP labels reject zero-reference-benefit units,
whereas subsequent behavior loses additional useful FG reuse. Test one
CPU-only lifetime distinction: an online binding KEEP becomes a non-revoking
soft admission grant while its semantic promise remains live, rather than
being withdrawn solely by later forecast changes. Native mandatory eviction
and queued priority still apply; this must never create owners or leases.

Keep W, candidate information and commitment arithmetic unchanged, preserve
fixed nonbinding selection, and explicitly report any grandfathered forecast
excess. Do not hardcode the measured unit counts, producer or reference labels.
The cheap two-stream target is elimination of the observed extra FG work
relative to fixed, plus a quantified useful protection/waste distinction,
not idle-only relabeling. This is a justification screen before considering
another GPU run, not a replacement of the user's 75% acceptance condition.
