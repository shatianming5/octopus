# Trainer memory (`octopus/rl/actor.py`)

Where a training step's batch and activations live on one FSDP rank, and the knobs that move them. The module
docstrings named below hold the details and measurements; this page is the map.

## Micro-batches

A train call's batch (`compute_logp`, `ppo_update`) stays in host memory, and the FSDP engine moves one micro-batch at a
time to the GPU (`cpu_staged_rpc_methods` in `octopus/rl/actor.py`, `patches/areal-cpu-batch.patch`, #322). Micro-batches
are cut at `MB_TOKENS` prefix-tree tokens (`rl_run.sh` → `actor.mb_spec.max_tokens_per_mb`, default 262144; changeable
during a run through `$STATE/live.env`). Sequences that share a prefix stay together; one sequence, or a reader with its
mount sources, over the cap gets a micro-batch of its own and is never cut ([rl-plan-episodes.md](rl-plan-episodes.md)
"AReaL facts" item 4, [cross-branch-read.md](cross-branch-read.md) "Export and allocation").

Each call logs every rank's micro-batch costs in run order (`[octopus] rank R: N micro-batches, cost ...`), a cost
being tree tokens plus `KAPPA` x attention query-key pairs (`prefix_tree.cost`, `OCTOPUS_KAPPA`). With FSDP gathering
and reducing once per call (#353) a call takes the slowest rank's total: on search-v9-rest step 43, 2.5% of the
modelled ppo_update idle, against 13% when the ranks ran micro-batch j in lockstep
([results/dp-balance](../results/dp-balance/README.md)).

## GPU: layers streamed over the tree

Every decoder layer is checkpointed. On a prefix-tree micro-batch of Qwen3.5 a layer runs one checkpointed kernel call at
a time (a GDN recurrence segment, or a FlashAttention call after k, v are projected over the whole tree), so the GPU peak
follows the segment size, not the tree: units of 665K tree tokens fit a 48 GB card (`octopus/rl/seq_tiles.py`, #323). A
flat micro-batch runs its token-wise parts in tiles of `TILE` tokens.

## Host: saved activations

The tensors saved for backward (the checkpointed layer inputs) wait in one pinned host buffer, reused by every
micro-batch and sized to the largest one's need (`host_activations` in `octopus/rl/lm_head.py`, its buffer a
`PinnedSlab` of `octopus/rl/pinned_slab.py`), except the first `ACT_GPU_GB` GiB (`rl_run.sh`,
default 0, bounded per micro-batch by `ACT_GPU_TARGET_GB`), which stay on the GPU. The buffer is released at the end of
each `ppo_update` and pinned again at its size by the next call, so pinned memory is not held between steps (#325).

Block checkpointing (`octopus/rl/block_checkpoint.py`, #326, #327) keeps the input of every `OCTOPUS_BLOCK_CHECKPOINT`-th
layer only (a host setting, default 4); the backward recomputes the rest of the block. The gradients are the same
(bitwise in the CPU tests). Measured on Qwen3.5-4B, 2 FSDP2 ranks, one 199K-tree-token unit
([results/block-checkpoint](../results/block-checkpoint)): host 170 → 50 KiB per tree token, forward + backward +11%,
GPU peak 16.1 → 19.5 GiB. A micro-batch over 650K tree tokens (`BLOCK_TOKENS`) is checkpointed per layer instead (a
994K unit ran out of a 48 GB card in the block recompute); all FSDP ranks use the block size of the largest of their
current micro-batches. `OCTOPUS_BLOCK_CHECKPOINT=1` checkpoints every micro-batch per layer.

So for Qwen3.5-4B (the figures scale with hidden size x layers) a rank pins one buffer, the maximum over its
micro-batches, not their sum: about 50 KiB per tree token of a micro-batch run in blocks, 170 KiB of one run per layer.
`MEM_MAX` (`rl_run.sh`) caps the trainer's host memory as a whole.

## Train-step RPCs

The controller's `compute_logp` / `ppo_update` calls to the workers get one attempt and a 24 h timeout
(`patches/areal-train-rpc-timeout.patch`, #329). Under AReaL's default (2 h, 3 attempts) a 3 h `ppo_update` was re-sent
and the workers ran the same batch twice.

## Proximal logprobs (no `compute_logp`)

AReaL's step is `compute_logp` (a no-grad forward over the whole batch for the decoupled loss's proximal logp,
`prox_logp`), `compute_advantages`, `ppo_update`. `grpo.yaml` sets `prox_logp_method: reuse_train_logp`, so the
first is skipped and the loss takes the proximal logp from the training forward itself (`logprobs.detach()` in
`grpo_loss_fn`). With `ppo_n_minibatches: 1` (AReaL's `__post_init__` forces it under this method) one optimizer step
follows all micro-batches, so that forward runs at the weights `compute_logp` would have used. Nothing before the loss
needs `prox_logp`: the decoupled loss keeps the rollout logprobs as `old_logp`, which the KL reward also reads. Per token
the loss weight is exp(logp − old) either way (the behavior weight exp(prox − old) times the ratio exp(logp − prox));
what changes is the PPO clip, which no longer fires on the two forwards' bf16 disagreement (search-v9-rest on node-138:
|logp − prox| up to 1.15, 1.9e-4 of the tokens clipped), and the rejection test `exp(· − old) > 5`, now on the
training forward's logp. `tests/test_actor_chunk_major.py::test_reused_train_logp_is_the_recomputed_prox`: same loss
and gradients in fp64, flag off and chunk-major. The step loses `timeperf/recompute_logp`: 1057 of its 6695 s on
node-138 (search-v9-rest), ~39 min of step 43 on the 4×A100 host. Pure MOPD distillation (`mopd.loss.rl_coefficient` 0) with
rejection sampling does not take the reused logp in AReaL and fails at its first loss; set `prox_logp_method:
recompute` there.

Lost: `ppo_actor/train_infer/*` (logged by `compute_advantages` from `prox_logp`: kl_k1, kl_k2, kl_k3, nll,
`logp_abs_diff`, `ratio_outside_*`). The loss's stats keep the mismatch: `update/behave_approx_kl/avg` (logp − old)
≈ −kl_k1 (over the tokens rejection sampling keeps; equal when `rs_filtered_fraction` is 0), `update/logp_abs_diff`
(avg, max) the absolute difference (`scripts/beta/step_stats.py`, `scripts/alpha/search_step_report.py` read these, in
older logs too). `update/importance_weight`, `approx_kl` and `clip_ratio` become 1, 0, 0. On the first step after
enabling, check `behave_approx_kl/avg` ≈ −(3–6e-4) (kl_k1's historical scale) and `clip_ratio` 0. Before a run on a
host, `grep -c REUSE_TRAIN_LOGP <overlay>/areal/api/cli_args.py` checks that its AReaL has the method.

## Chunk-major (off by default)

`octopus/rl/chunk_major.py` (#330, #332, #333) runs one prefix-tree micro-batch through the whole model one chunk of tree
tokens at a time, so activations span a chunk instead of the tree. Its gradients match the whole tree's (CPU fp64 tests
and a 2-rank FSDP2 LoRA test). On a GPU its kept tensors, and their gradient sums, wait in a second `PinnedSlab` in host
memory. Measured on Qwen3.5-4B ([results/chunk-major](../results/chunk-major/README.md)): the 994K unit that block
checkpointing could not fit peaks at 31.9 GiB of GPU memory on one card in 32K chunks (27-31 GiB before the change
below), about 44 GiB pinned per rank.

The actor uses it when the train workers start with `OCTOPUS_CHUNK_MAJOR=1`, in chunks of `OCTOPUS_CHUNK_TOKENS` tree
tokens (default 32768; 64K chunks ran out of a 40 GiB cap on the 994K unit on one card, 35.4 GiB per rank on 2). Both
are host settings: export them before `rl_run.sh`. Then a prefix-tree micro-batch's `compute_logp` forward runs the
chunk loop without autograd, and its `ppo_update` forward and backward the chunk loop with it, into the same `[1,
T_flat, 2]` logits (`install_forward` in `octopus/rl/actor.py`). Invariant: with the flag on every packed micro-batch
(with `cu_seq_lens_q`) runs chunk-major, also one whose sequences share nothing (otherwise no tree), since FSDP ranks
must issue the same collectives in the same order: one rank's flat backward beside another's chunk-major one gave NaN
gradients
([results/chunk-major](../results/chunk-major/README.md) "Ranks on different paths"). It replaces the saved-activation
path (no `ACT_GPU_GB`, no blocks): `OCTOPUS_BLOCK_CHECKPOINT` must be 1, or the actor refuses to start. `ppo_update`
releases the slab at its end, also when it raised; `compute_logp` keeps it for the `ppo_update` after it (pinning it
again took ~0.5 s per GiB). On CPU the flag on and off give the same logits and
gradients within 1e-9 (fp64, `tests/test_actor_chunk_major.py`). On the real actor (bf16, Qwen3.5-4B, real spool groups)
neither 1 nor 2 GPUs meet the plan's 1e-3, which the bf16 noise floor of a changed kernel-call bound itself exceeds: on
1 GPU, 32K chunks are inside two floor samples on grad norm and mean logp and above both on loss (1.4e-3 against 6.1e-4
and 1.1e-3); on 2 GPUs (3 steps, no hang) of the floor's order, not inside it (loss 1.4x, grad norm 1.9x the SEG-changed
sample). On those small trees chunk-major was 15-25% slower in compute_logp, as fast in ppo_update, and 4-5 GiB higher
on the GPU ([results/chunk-major](../results/chunk-major/README.md) "Use in training"). In 16K chunks the bf16 LoRA
gradient's cosine to the flag-off path was 0.63, below two noise-floor samples (0.79, 0.86); in IEEE fp32 the same
micro-batch matches it within fp32 rounding (logp 1.05e-4, cosine 1.000000; one card, unsharded: the FSDP2 path and the
actor's wiring are exact in the CPU fp64 tests), so that gap is bf16 rounding. Time and memory were not measured for
chunks smaller than 32K; keep the default 32768.

In its backward a chunk keeps only each layer's own checkpoint: the tiles and kernel calls inside a layer run without
one (`seq_tiles.no_inner`; it needs HF layer checkpointing, or `chunk_major_output` refuses to train). Nested, every
checkpoint level ran the layer once more, ~3.9 forwards per layer per step instead of 3. A layer's backward then holds
its activations over the chunk plus the keys and values all of the chunk's attention calls gathered (4 KB per key,
set by the tree's paths, not by the chunk size: up to 1.7-3.9 GiB per 32K chunk on the 256K-994K units). On the 256K unit (one 48 GB card, bf16) this cut
forward + backward from 210 s to 183-190 s for 3.5 GiB more GPU peak (31.9 -> 35.4 GiB); on the 994K unit from 751 s
to 738 s (at least: the new path ran cold), peak 32.0 -> 31.9 GiB. Logp is unchanged and the
gradients equal the nested path's bit for bit in the CPU fp64 test
([results/chunk-major](../results/chunk-major/README.md) "One checkpoint per layer inside a chunk").

Under chunk-major FSDP syncs the ranks once per actor call, not per micro-batch (`chunk_major.across_micro_batches`,
`forward_backward_batch` in `octopus/rl/actor.py`): each module's parameters are gathered at the call's first forward
and stay gathered to its end, and the gradients are reduce-scattered once, in the last micro-batch's backward. Before,
every micro-batch gathered the parameters again (twice in training) and reduce-scattered its gradients, so every rank
waited for the slowest one's micro-batch. A call's sync points are now AReaL's all_reduces before its loop (the
micro-batch count; in training also the loss weight), the first all-gathers and the last reduce-scatter. The GPU peak is
unchanged, since chunk-major held the parameters gathered over a whole micro-batch already; the unsharded LoRA gradients
(fp32, 0.45 GiB for r64 on Qwen3.5-4B) wait over the call. On 2 GPUs the LoRA gradients equal the per-micro-batch path's
within bf16 run-to-run noise (cosine 0.9999, grad norms within 0.25%); with unequal micro-batch lists compute_logp took
31.7 -> 29.7 s, ppo_update 132.6 -> 125.0 s; on step 43's replay the modelled idle drops from 13.3% to 2.5%
([results/chunk-major](../results/chunk-major/README.md) "FSDP once per call").
