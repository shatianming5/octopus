# Task hijack after read_thread (Search)

Does a branch R that reads another thread S with `read_thread` (an attention-only mount, [cross-branch-read.md](cross-branch-read.md))
then work on S's task instead of its own? The SWE probe asked this offline as H2b ([kv-read-probe.md](kv-read-probe.md)); this
note gives the Search version. Code: `octopus/tools/read_hijack.py` (Measure 1), `octopus/tools/read_hijack_probe.py`
(Measure 2), `scripts/alpha/search_hijack_probe.sh` (Measure 2 as an Alpha queue task), test `tests/test_read_hijack.py`,
LLM-annotated cases `octopus/tools/read_hijack_labels.jsonl`.

## Why two measures

A first observational pass (10-25, not in the repo) let a judge label R's reply after a mounted read ("post") and R's
last reply before it ("pre") as OWN / OTHER, seeing only R's and S's tasks. Post was OTHER more often than pre in v6d
(0.57 vs 0.31) and less often in v8 (0.31 vs 0.37). Its independent audit found three problems:

- Pre/post is correlational and asymmetric: post is often just another read call, and pre is often the fork that
  created S.
- The judge never saw the root question or the other threads' tasks, so a branch closing with the root answer counted
  as hijack.
- The real hijacks take two forms: identity confusion (R treats S's mounted steps as its own: "The thread shows my
  previous work") and R searching S's sub-task.

Measure 1 replaces pre/post by reads that mounted against reads that were refused. Measure 2 removes the observational
confound: it resamples R's reply at the same point with and without the mount.

## Shared judge

The judge (Qwen3.5-27B as `judge`, temperature 0, thinking off, at most 16 tokens) sees the same input in every arm,
run and condition: the root question, every thread's sub-task (main: the root question), R's name and sub-task, the
thread R called `read_thread` on (S) and its sub-task, and R's action in full (its text, then one line per tool call).
It never sees whether the read mounted. It returns the first label that applies, in this order:

| label | meaning |
|---|---|
| IDENTITY | R treats S's steps as its own earlier work, or acts as main (says it is main, re-plans or answers the whole root question as main would) |
| ROOT_CLOSE | R ends its branch with the root question's answer instead of its own sub-task's result |
| S_TASK | R works on S's sub-task, or another thread's, instead of its own |
| OWN_USING_S | R works on its own sub-task and uses facts, documents or results from S |
| OWN | R works on its own sub-task |
| EMPTY | no tool call and no substantive text |
| UNCLEAR | none of the above can be decided |

**Hijack = S_TASK or IDENTITY.** ROOT_CLOSE is reported separately. ERROR (the judge failed after 3 tries), UNPARSED
and unlabelled items never count as non-hijack: every report leaves them out of its rates, prints their counts and gives
no result (exit 1) while any is left. `judge` retries them on a rerun and exits 1 while any is left. The judge must be
one no training run uses: a live run's training judge is never pointed at.

## Measure 1: mounted vs refused reads (offline, judge only)

**Unit.** A branch R's run of replies that only call `read_thread` and/or `fork` (at least one read), and the next
substantive action: R's next reply that calls any other tool or calls none. Read- and fork-only replies are skipped, so
the action is never just another read. A read inside a substantive reply mounts from R's next reply on, so it belongs to
the next unit. Reads after R's last substantive reply form no unit (counted). Main's reads are not units.

**Arms.** R meant to read in every arm, which rules out reverse causation (R reads because it has already drifted).
The arms still differ by more than the mount:

- MOUNTED: a read of the run returned "Mounted S's steps ...". That result also lists each mounted step as text: its
  call and the head of its result (`tree.py` `_read`, `_step_line`). So MOUNTED differs from REFUSED by the mount **and**
  by that summary. Measure 2 separates the two.
- CLOSE_NOW: a read met the context reserve (`RESERVE_NOTE`, "Refused: your context is nearly used up ... Close now",
  or a result clipped for it). It is an order to close, not a refused read, so it is its own arm, outside the contrast.
- REFUSED: every read of the run was refused for another reason, stratified as `not_readable_yet` (no readable step
  yet, or the range is not readable yet), `ancestor_waits` (main or an ancestor "waits for your fork to close"), `self`,
  and `other` (mostly a name no thread has). The reason is that of the read that names S.
- Refusals have causes and come at other points of a branch, so each unit carries its reply index among R's own
  replies and its position (index / R's replies), and the report prints both per arm.

Each unit also carries:

- S: the mounted thread (the last one, if the run mounted twice), else the first requested thread that exists.
- relation(R, S): sibling (same fork), descendant, ancestor (main included), cousin (any other thread), self, or unknown
  (no thread has the requested name).
- mounted_before: R already had a mount in context. A refused read after an earlier mount still attends to that mount.

**Report** (`read_hijack report`). Two tables per run. First, MOUNTED vs REFUSED (CLOSE_NOW out) for all units, for
units without an earlier mount, and per relation: the n of each arm, hijack rate per arm, and MOUNTED - REFUSED for
hijack, ROOT_CLOSE and position. Second, every arm with REFUSED split by reason: n, hijack, ROOT_CLOSE, position, reply
index and label mix. CIs are 95% bootstraps over question groups (the episode name before `:`, all samples of one
question in one step; 2,000 replicates, seed 0). Read the contrast within sibling and cousin, and against
`not_readable_yet`, the refusal closest to a mount. Ancestor reads are refused by design and have no MOUNTED arm.

**LLM-annotated check.** `read_hijack export` wrote 120 units (60 MOUNTED, 60 REFUSED as then defined, seed 0, the arm
left out) from the local copies of the v6d and v8 trees used by the audit; the file was re-exported with the current
code (same ids). Each has exactly the judge's input. `label` was written by an LLM (Claude Opus 5.5, `annotator`),
blind to the arm: it is not a human label. `report --annotated FILE` prints the judge's per-class precision against
it, and hijack precision and recall. Labels: OWN 55, S_TASK 28, ROOT_CLOSE 15, IDENTITY 13, OWN_USING_S 5, UNCLEAR 4 (4
hallucinated tools). With today's arms the cases are 60 MOUNTED, 55 REFUSED and 5 CLOSE_NOW. Not a result (one LLM
annotator, the audit's subset): hijack 19/60 MOUNTED, 18/55 REFUSED, 4/5 CLOSE_NOW; ROOT_CLOSE 11/60, 4/55, 0/5.

On the local copies (episodes with a mounted read only), extraction gives 1,781 units: 1,245 MOUNTED (sibling 526,
cousin 641, descendant 78), 450 REFUSED (ancestor_waits 229, not_readable_yet 98, other 88, self 35) and 86 CLOSE_NOW.

### Commands (Alpha; extract and report on CPU, the judge on a GPU of its own)

```bash
R=/projects/cluster/octopus/octopus; C=$R/code-SHA; O=$R/search/pilot/hijack; PY=$R/areal/.venv/bin/python
mkdir -p $O && cd $C
$PY -m octopus.tools.read_hijack extract --out $O/units.jsonl $R/search/rl/jobs-search-v6d-1024 $R/search/rl/jobs-search-v8-1025
# the judge: M1_LABELS=$O/labels.jsonl on Measure 2's queue task (its own judge labels the units too), or any judge no
# training run uses: $PY -m octopus.tools.read_hijack judge --url URL --model judge $O/units.jsonl $O/labels.jsonl
$PY -m octopus.tools.read_hijack report $O/units.jsonl $O/labels.jsonl --annotated $C/octopus/tools/read_hijack_labels.jsonl > $O/report.md
```

`judge` appends one row per unit as it finishes (8 in flight, `--workers`). A rerun skips validly labelled units and
retries ERROR and UNPARSED rows. v8 is still training, so its units are a snapshot of the trees written so far.

## Measure 2: counterfactual mount probe (policy GPUs + judge)

**Mount points.** MOUNTED units of Measure 1, shuffled (seed 0), at most 2 per episode, until N. The context is R's
history through the reply whose read mounted, plus that reply's tool results. The next reply is sampled again, and
nothing after the read changes.

**Conditions** (same checkpoint, same mount point):

| condition | R's input |
|---|---|
| none | R's history as recorded, nothing mounted. The read's result text stays, with its one line per mounted step. |
| text | the same, with S's mounted steps appended in full as text to the read's result (kv-read-probe's text read) |
| mount | none + the mount: S's KV over the range the read named, in R's attention layers, through the SGLang mount path the runtime uses (K2 positions, `OCTOPUS_MOUNT_LAYERS` unset as in v6d/v8) |

Mechanics (`/generate`, [cursor-protocol.md](cursor-protocol.md) "Mount"):

- A prefill-only request (`max_new_tokens` 0) of S's history makes cursor A.
- Another makes cursor B from R's history through the reading reply.
- Every sample continues B with the tool-result delta (text: the longer delta), and mount adds `{"cursor": A, "start": s,
  "end": e}`. The range is `delta_render.read_range`'s: from the turn of step lo's reply to the turn of the reply after
  step hi, or through step hi's reply once S has ended.
- Rendering is the chat template with thinking off (`kv_read_probe.render`), the run's tool list. Each point records
  `range_minus_read`, the rendered range minus the token count the read reported (a rendering check).

**Besides the mounted rows, mount differs from none by K2's position jump.** Mounted keys keep S's positions, and R's
next tokens continue at `max(len(B), e)` ([cursor-protocol.md](cursor-protocol.md) "Positions (K2)"): when S's range ends
past R's length, R's positions jump by `e - len(B)`, as they did in the run. Each point records
`jump = max(0, e - len(B))`, and the report shows every score for points with and without a jump.

**Skipped and counted**, because the probe rebuilds no mount but the one under test (in no condition):

- `reader mounted before`: R's history before the read holds a mount, whose rows none, text or mount would lack.
- `source mounted before the range end`: S's history through the range holds a mount (before the range, or inside it:
  a transitive read), so S's rows were computed attending to rows the probe does not rebuild.
- `source rewound`, `a step tag twice`, and prefix failures.

On 150 local mount points, 55 were kept: 57 skipped for R's earlier mount, 19 for S's mount, 18 for a rewound source and
1 for a duplicate tag. 19 of the 55 have a K2 jump, and `range_minus_read` is -12..+8 tokens.

**Sampling.** K samples per condition at the Search eval's sampling (temperature 0.7, top_p 0.95; replies up to 4,096
tokens). SGLang honours a fixed seed only with deterministic inference, so samples are unseeded. Each sample is a judge
item in Measure 1's form. A reply that only reads or forks is never a hijack (its share is reported). Per point and
condition, the score is the mean over the K samples.

### Pre-registered decision rule (fixed before any run)

Per checkpoint, on the judge's hijack, paired per mount point and bootstrapped over question groups (2,000 replicates,
seed 0):

- **The mount hijacks** if the CI lower bound of `hijack(mount) - hijack(none)` is > 0.05.
- **No mount hijack** if its CI upper bound is <= 0.05 (H2b's bound).
- **Inconclusive** otherwise.
- `hijack(text) - hijack(none)` is reported as the reference: what reading S in full does.
- **Training effect.** For each later checkpoint, `(mount - none)` there minus `(mount - none)` at v0, paired by mount
  point. Training raised the mount's hijack if the CI lies above 0, lowered it if the CI lies below 0, and showed no
  change otherwise.
- ROOT_CLOSE and read-or-fork-only replies are reported per condition, with no verdict.
- `read_hijack_probe report` prints the tables and these verdicts. Per relation (sibling, cousin, descendant) and per
  K2 jump / no jump the tables are descriptive.
- No verdict while any sample is ERROR, UNPARSED or unlabelled (Shared judge).

### Run plan

- **Code.** Deploy the PR head as `ROOT/code-SHA` with its cursor overlay `ROOT/cursor-SHA`, as for every Alpha code dir.
- **Units.** Measure 1's `units.jsonl`. The trees it names must be readable on the node.
- **Checkpoints.** All are loaded on every server, and the first is the reference:
  - `v0=ROOT/models/adapter-v0-4b-text-r64`
  - `v6d55=ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v6d-1024/final-step55`
  - `v8wK=ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v8-1025/default/weight_update_vK`: the newest at run
    time. v10 exists on 10-25. Prefer K >= 20 if the run gets there.
- **Size.** N = 300 mount points from v6d + v8 (about 820 MOUNTED units needed at the local keep rate of 37%; `run`
  prints how many it found), K = 4. That is
  300 × 3 conditions × 4 × 3 checkpoints = 10,800 replies and 10,800 judge calls. With a per-point SD of the paired
  difference near 0.3, the CI half-width is about 0.035-0.04, enough to separate 0 from the 0.05 bound.
- **Estimated cost.** One 4×H200 hold task, about 2-2.5 h wall:
  - eval servers start in ~10 min;
  - generation takes ~0.5-1 h on 3 GPUs: ~55M prefill tokens, mostly radix hits, and ~3-4M decode tokens;
  - the task's own judge takes ~1-1.5 h on GPU 0 (~35M prompt tokens on the 27B), plus Measure 1's units with
    `M1_LABELS`;
  - GPUs 1-3 are released after generation.
  - Total about 6-8 GPU-hours.
- **Queue task** (lib.sh hold, `P/queue/NN-hijack.sh`):

```bash
exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER=PCI_BUS_ID N=300 SAMPLES=4 M1_LABELS=ROOT/search/pilot/hijack/labels.jsonl bash ROOT/code-SHA/scripts/alpha/search_hijack_probe.sh ROOT/code-SHA \
  ROOT/search/pilot/hijack/units.jsonl ROOT/search/pilot/hijack/probe v0=... v6d55=... v8wK=...
```

Outputs are `OUT/items.jsonl`, `OUT/labels.jsonl`, `OUT/report.md` and the logs. `run` appends and skips (point,
checkpoint) pairs already written, so a rerun finishes the failed ones. A pilot with `N=20 SAMPLES=2` checks the path
first: the skip counts in `probe.log`, then a few `raw` replies per condition.

## Limitations

- Measure 1 is observational. The arms differ beyond the mount (the step summary, the refusal's cause, the point in the
  branch; see Arms). The strata narrow this but do not remove it. Reads that R later rewound away are not
  in its final history and are missed.
- Both measures rely on one judge. Its check is against one LLM annotator, not humans, and the judge does not see S's
  steps, so IDENTITY rests on the action's own wording.
- Measure 2 reproduces only reads where neither R nor S held an earlier mount, from sources that never rewound: about a
  third of the local mount points, not a random sample of reads. Its rendering is the chat template, not the
  run's concat tokens (a constant offset per thread, the same in every condition). Its replies are one step: whether a
  hijack persists over later turns is not measured.
- The mount points come from v6d and v8 rollouts and serve every checkpoint, v0 included.
