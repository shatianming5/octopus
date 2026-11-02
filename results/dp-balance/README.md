# Data-parallel balance of a training step: lockstep vs. rank totals

`replay.py` replays one training step on the CPU from its spool group files. It runs the dispatch of plan episodes to
the ranks (`octopus/rl/dp_balance.py`, longest first by tree tokens) and every rank's micro-batch split
(`actor._allocate_by_prefix`).

The only step whose spool groups are kept is search-v9-rest step 43, on node-138 (16 groups and 128 episodes in
`rollout-spool/search-v9-rest/trained/000041~*.pt`; 4 ranks; `MB_TOKENS` 65536). No other step can be replayed. The
replay reproduces the trainer log's per-rank micro-batch lists item for item (`--check-log`: `Microbatch #tokens
(rank N)` at 15:29, 87/98/95/98 micro-batches).

## Cost model

A ppo_update micro-batch's time is `b x tree tokens + a x attention query-key pairs`, the two terms of
`prefix_tree.cost`. It is calibrated on mb-256190, Qwen3.5-4B on a 4090D, chunk-major after #346:
- ppo_update takes 184.2 s (node-68, steady state; [results/chunk-major](../chunk-major/README.md) has 183-190 s).
- FlashAttention takes 1.28 s per layer for a forward and 5.65 s for the backward, over the tree's 1.594e10 pairs.
  This comes from `bench_attn.py` (eval-attn's script, cut to the FA2 timing) and its log `logs/bench_attn-256k.log`.
- One ppo_update runs 816 FlashAttention calls, which is 3.09 forwards of the 8 layers' 33 calls each. So attention
  is 76.9 s of the 184.2 s.

`KAPPA` = a / b = 1.15e-5 is fitted to this model and card only. The non-attention part also holds memory-bound work:
the GDN kernels, norms, element-wise ops, and chunk-major's host slab copies. So `KAPPA` can move between cards. It has
not been measured on an A100; `OCTOPUS_KAPPA` overrides it. It feeds only the per-call log
(`[octopus] rank R: N micro-batches, cost ...`), not any decision.

compute_logp is not modelled: since #349 (`reuse_train_logp`) it no longer runs.

## Step 43 (modelled ppo_update)

| schedule | time | idle |
|---|---|---|
| lockstep (FSDP syncs every micro-batch index: Σ_j max_r; before #353) | 134.7 min | **13.3%** |
| no_sync (parameters gathered, one reduce per call: max_r Σ_j; #353) | 119.8 min | **2.5%** |
| ideal (mean rank total) | 116.8 min | |

The rank totals are 112.3 / 119.5 / 115.6 / 119.8 min. They were already close; the idle came from the lockstep.

In the no_sync model, a rank with fewer micro-batches still runs AReaL's dummies, its shortest micro-batch each. The
single reduce-scatter of the LoRA gradients at the end is left out.

## Not adopted: other dispatch weights

Two variants were tried in earlier revisions of this PR and dropped:
- Dispatching each group as the micro-batches it makes alone.
- Weighting episodes by `cost` instead of by tree tokens.

On this one replayable step, neither beat tree-token weights once the rank totals are what counts (no_sync). Their
idle went up and down with `KAPPA`, not monotonically. 128 coarse episodes dealt longest first land within a few
percent by chance, and a rank's real cost is not the sum of its episodes' costs: same-group episodes share their
prefix on a rank, and the rank packs them again. Those variants are not reproducible with this `replay.py`.

usage: `PYTHONPATH=<areal overlay>:<code> python results/dp-balance/replay.py --check-log <the step's 4 log lines>
<spool files>`
