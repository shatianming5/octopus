# e14-read: RL with the read_thread interface (Qwen3.5-4B task bundles)

Bottom line: held-out y rose from 0.283 to 0.334 (s40; M1 +0.050, bundle-clustered 95% CI [−0.002, +0.102], not
significant; s40 picked as the best checkpoint), but that gain came from more efficient linear/closing behaviour. It did
not come from forking or reading. Fork value (M2) is ≤ 0 at every late checkpoint. When the read-interface bugs were fixed
without retraining (PR #154), valid reads went from 18% to 47%, but outcomes did not move. In 375 read-vs-control pairs,
reading shows no value (−0.027±0.044). Large mounts with real content are associated with lower pass rates (post-hoc
split).

Caveat (found 10-20, PR #191): sections 3 and 4 (every held-out row, the s40 re-evals, their summaries here) ran
dev_eval straight to SGLang's `/v1/chat/completions`, which ignored read_thread's `mount`: each "Mounted ..." result was
answered without the mounted tokens (174 of 174 paired mounts of ≥ 1,000 tokens grew the prompt by < 0.5·M in the
first eval, 46 of 48 in the afef947 re-eval, none by ≥ 0.9·M). Mounts happen only in forked episodes, from the first
mount on. Episodes with a dropped mount: base 27 of 120, s10 26, s20 26, s30 34, s40 19, afef947 re-eval 47.
- Still valid: unforked y (0.299 → 0.349), the fork rate and the first-fork decision, the unforked part of the "gain
  from linear/closing efficiency" attribution, section 4's valid-read share (18% → 47%), the training analyses
  (sections 2 and 5: AReaL proxy, `engine: mount ignored` 0) and A0 (real mounts).
- Biased by a bounded share of episodes: y and M1 (s40's M1 +0.050 already touches 0).
- Invalid: M2 (every dropped mount is in the fork arm: ~57% of base's forked episodes, ~30% at s40), M4's wall time (no
  real mount cost), section 4's "outcomes did not improve", and the summary's "did not come from ... reading" and
  "outcomes did not move".

Files here (all `summary.py` runs are `scripts/eval/summary.py`):
- `eval-summary.txt`: held-out eval, `summary.py bundle-e14-read/merged --base bundle-e14-read/merged/base/step0.jsonl
  --late 20,30,40`.
- `eval-s40-afef947-summary.txt`: s40 re-eval under the read fix against base, `summary.py
  bundle-e14-s40-afef947/merged --base bundle-e14-read/merged/base/step0.jsonl`.
- `eval-s40-afef947-vs-7c27967-summary.txt`: the same re-eval paired against the old s40, `summary.py
  bundle-e14-s40-afef947/merged --base bundle-e14-read/merged/results/step40.jsonl`.
- `train-steps.csv`: per-step stats parsed from the AReaL `main.log` tables.

## 1. Run

| item | value |
|---|---|
| trial | `bundle-e14-read-10191636`, 40 steps, 2026-10-19 16:38 → 10-20 04:25 |
| code | `7c27967` (#152 informed reads, #153 append-only read_thread); cursor SGLang build `cursor-564f6b1` (#148 read control) |
| model | Qwen3.5-4B LoRA r64, thinking off |
| batch | G=8 x 4 bundles/step (32 episodes) |
| flags | `OCTOPUS_CALLS_PER_TASK=25`, `OCTOPUS_READ_CONTROL=1`, read tool on |
| hosts | node-94 (3 SGLang + 5 FSDP) + node-136 (5 SGLang) |
| health | 0 crashes / OOM / engine read errors; watchdog stopped at step 40 and released GPUs |
| prefix cache | cursor hit 98.65% token-weighted over 8 servers (296k prefill batches; 10.07B cached / 138M new tokens) |

## 2. Training metrics

Means over 10-step windows, from the AReaL `main.log` stats tables. Each table belongs to the "Train step N done" line
just above it. Step 1 is excluded from the first window because it has no off-policy samples yet (staleness 0).
Per-step values are in `train-steps.csv`.

| metric | s2-10 | s11-20 | s21-30 | s31-40 |
|---|---|---|---|---|
| reward | −0.283 | −0.188 | −0.237 | −0.109 |
| entropy | 0.383 | 0.356 | 0.345 | 0.363 |
| turns/ep | 120 | 108 | 108 | 99 |
| grad_norm | 0.010 | 0.013 | 0.012 | 0.013 |
| zero-adv share | 5.8% | 8.4% | 5.2% | 7.9% |

Throughout the run: clip ratio was 0, staleness was about 1.9 versions (1.8-2.0 per window), and context overflow was 0.
Over the training episodes (1,340 scored), the fork rate was 42%. Training M2 was −0.019 ± 0.022 (95% CI; SE 0.011; n=534).

## 3. Held-out eval

Setup: tasks-eval-v3, 60 bundles x k=2, `--control` (main-first control), read tool on, code 7c27967. M1 is paired against
base. M4 has three columns: the median fork/control wall ratio, the share of pairs where the fork is faster, and y per hour.

| ckpt | y | fork | M1 | M2 main-first | M4 wall (fork faster) | y/h |
|---|---|---|---|---|---|---|
| base | 0.283 | 39% | — | +0.014±0.080 n=45 | 0.42 (87%) | 0.95 |
| s10 | 0.258 | 40% | −0.025±0.052 | −0.004±0.072 n=44 | 0.69 (84%) | 0.87 |
| s20 | 0.260 | 43% | −0.023±0.053 | −0.070±0.083 n=44 | 0.55 (86%) | 1.05 |
| s30 | 0.324 | 52% | +0.041±0.049 | −0.014±0.059 n=60 | 0.63 (82%) | 1.49 |
| s40 | 0.334 (n=119) | 53% | +0.050±0.049 n=119 | −0.049±0.067 n=56 | 0.68 (80%) | 1.51 |
| pooled late M2 (s20/30/40) | | | | −0.042±0.039 n=160 | | |

The s40 row has n=119 because one row ended in a VerifierTimeoutError.

Caveats:
- **CIs.** The CIs that `summary.py` prints are naive episode-level intervals (1.96·sd/√n).
- **M1.**
  - Bundle-clustered bootstrap 95% CI for M1 at s40: [−0.002, +0.102] (60 bundles resampled, 10,000 resamples, seed 0,
    percentile interval).
  - At the +0.05 threshold, the M1 gain is not significant, and the +0.08 target is missed.
  - s40 was picked as the checkpoint with the highest M1, so its M1 is biased upward.
  - Counting the timeout row as 0 gives M1 = +0.048.
- **M2.**
  - M2 is ≤ 0 at every late step: forked episodes lost to their control 72 times and won 42 times.
  - The naive pooled CI (−0.042±0.039) excludes 0, but the bundle-clustered CI [−0.084, +0.004] does not: no evidence
    forking adds value, and no robust evidence it hurts.
- **M4.**
  - M4 passes only as a critical-path measure. At s40, forked trees use 21% more total completion tokens than control.
  - The y/h gain decomposes into 1.18x higher y times 1.35x shorter episodes. Unforked episodes also got shorter.
- **Source of the y gain.** It comes from linear/closing efficiency. This is the same pattern as e13.
  - Episodes ending in "done" rose from 40 to 69.
  - Episodes ending on the call budget fell from 72 to 40.
  - Unforked y rose from 0.299 to 0.349.
  - Control improved more than fork did.
- **Asymmetric comparison.** The control has no read_thread tool. M2 therefore compares a fork with reads against a linear
  episode without reads.

## 4. Read interface fix without retraining

The s40 adapter was re-evaluated under code afef947, the #156 merge commit. Its tree contains the #154/#155 read-range
fix.
Bundles, k and control are the same as in section 3. The `patches/` directory is identical between the two runs. Only
`grok_harbor_agent.py` and `harbor_runtime.py` differ.

| metric | 7c27967 | afef947 |
|---|---|---|
| valid read share | 18% (43/243) | 47% (124/266) |
| episodes with a valid mount | 16% | 39% |
| malformed reads | 49 | 2 |
| not-readable reads | 100 | 43 |
| y | 0.334 | 0.345 (paired +0.014±0.042 n=119, n.s.; `eval-s40-afef947-vs-7c27967-summary.txt`) |
| M2 main-first | −0.049±0.067 n=56 | −0.009±0.063 n=57 (n.s.) |
| M4 wall (fork faster) | 0.68 (80%) | 0.90 (56%) |
| y/h | 1.51 | 1.56 |

Conclusion: the mechanism is fixed, but outcomes did not improve. For reference, M1 at afef947 against base is
+0.061±0.048 n=120. This comparison is not paired on the same runtime, because base ran on 7c27967.

## 5. Why reading shows no value

Data: 1,340 training episodes and 375 read-control pairs (a branch that read, against its same-state no-mount
continuation). The analysis scripts are on node-94 under `octopus-sft/why-e14/`: `rcpairs.py`, `rcvalues.py`,
`samefile.py`, `readcmp.py`, `read_audit_new.py`.

**Headline.**
- Read − control y: −0.027±0.044 (n=375). The bundle-clustered 95% CI is [−0.068, +0.014], which rules out gains above
  +0.014.
- Bundle-clustered CIs in this section resample bundles (5,000 resamples, seed 0, percentile interval).
- Pair outcomes: both fail 203, both pass 100, reader only 31, control only 41.
- 339 of the 375 pairs are branch→main reads.

**Mount content.**
- 61% of controlled mounts carry no content: only checkpoint, fork and join steps. For example, a branch that reads main
  right after the fork sees only the "Before forking tasks" checkpoint.
- Reads of mounts without content act as a placebo: +0.004±0.058 (n=230).
- Reads of mounts with content: −0.076±0.066 (n=145; bundle-clustered [−0.143, −0.007]).
- By mount size (tokens; post-hoc split, so read the trend, not the per-bin significance):

| mount size | read − control |
|---|---|
| <500 | +0.030 (n=169) |
| 500-2k | −0.029 (n=68) |
| 2k-5k | −0.065 (n=77) |
| >5k | −0.131±0.106 (n=61; bundle-clustered [−0.224, −0.044]) |

**Reader behaviour.**
- Readers with content close their tasks less often. Here "closed" means the branch ended in commit or abort, over the 136
  content pairs where neither the reader nor the control forked afterwards. Only the control closed in 28 pairs and only
  the reader in 10 (sign test p=0.005). With commit alone, the counts are 25 against 11 (p=0.029).
- Readers with content also read more afterwards: 1.05 further reads against 0.56.
- Little of the mounted material is useful or used:
  - The source edited or committed a file within the mount in 11% / 2% of cases.
  - Mounted files reach the reader's patch in 8% of pairs.
  - The reader uses mounted files after the read in 10% of pairs.
  - Reader and source work on the same file in 3% of pairs.

**Credit and control signal.**
- The logged read advantage is −0.178 and is negative in every window.
- Only 13% of 2,822 read calls were credited, and 78% were refused. Even so, reads per episode rose from 1.2 to 2.4.
- main→branch valid reads rose from 6% to 24%, and these reads have no control.

**Ruled out.**
- Budget exhaustion: at a read, a median of 61k tokens and 98 calls remain.
- Mount latency.
- Drift into source files.

**Coupled forks.**
- In 63% of forks with at least 2 patches, two tasks patch the same file.
- The conflict rate is 20.9% when tasks share a file and 12.6% otherwise.
- The read control in coupled forks is −0.009±0.057 (n=223), so even here reading shows no value.

## 6. Open questions answered this round

| Q | answer |
|---|---|
| Q1 thinking-on per-turn truncation | Setup: Qwen3.5-4B base, 40 OpenResearcher validation questions, node-138. Per-turn completion tokens: p50 about 165, p90 about 380, p99 565-740, max 1100. A 1024 cap truncates 0.4% of turns (2/505), but each truncation kills its episode (5% of episodes). Caps of 1536 and 8192 truncate nothing. **Use 1536** (2048 for margin). Thinking on uses about 13 tool calls per episode against about 8 with thinking off, and generates 2-3x as much. |
| Q2 rewound tokens | They are trained: thread advantage, option-1 credit. Only the shared prefix is computed once. So "rewind saves training compute" is false. |
| Q3 group shape | e9 used G=4 x 8 bundles; e10-e14 use G=8 x 4. Both give 32 episodes per step. |
| Q4 M4 | See section 3. It passes on the critical path only, and the token total is +21%. |
| Q5 cache hit | 98.65% (section 1). The earlier "97.5%" was a 10-18 synthetic KV-flag test. |

## 7. Next experiments (cheapest first)

1. **Forced-decoding dlogp** on the 145 content pairs, comparing three conditions: KV mount, the same content as text,
   and no content (`scripts/rl_analysis/read_force_dlogp.py`). This separates a broken mechanism from irrelevant content.
2. **Extend the #129 KV probe** to 5-20K-token mounts.
3. **Credit and control changes:** give no credit or control to content-free reads, and add controls for main's reads.
4. **Coupled-bundle eval** with forced sibling reads after edits.
