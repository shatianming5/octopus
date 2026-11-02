"""FSDP PPO actor whose LM head returns each token's (logprob, entropy) computed CHUNK tokens at a time
(recomputed in backward) instead of the full [tokens x vocab] logits: an 80k-token episode times the
248k vocabulary would need ~40 GB of logits, more than a 48 GB card has left.

Saved activations are kept in pinned host memory during the forward (as the SFT trainer does), except the first
act_gpu_budget(micro-batch tokens) GiB, at most OCTOPUS_ACT_GPU_GB (rl_run.sh ACT_GPU_GB, live in $OCTOPUS_LIVE_ENV),
which stay on the GPU. With OCTOPUS_CHUNK_MAJOR a prefix-tree micro-batch runs chunk-major instead (install_forward).

The model's own lm_head module keeps its weight (FSDP swaps sharded/unsharded tensors on it); only its
forward is replaced, and the labels and scored positions (scored_positions: the loss_mask) come from a pre-hook on
the model's inputs. The engine's logprob helpers then read the columns instead of reducing full logits.

Plan episodes (docs/rl-plan-episodes.md): per-token octopus_adv replaces group adv_norm.
"""

from __future__ import annotations


import contextlib
import functools
import os

import numpy as np
import torch
import torch.distributed as dist

import areal.utils.data as areal_data
from areal.engine import FSDPPPOActor
from transformers.modeling_outputs import CausalLMOutputWithPast

from octopus.rl import adapter_push, chunk_major, prefix_tree
from octopus.rl.block_checkpoint import checkpoint_blocks
from octopus.rl.config import live_env
from octopus.rl.dp_balance import take_rows
from octopus.rl.lm_head import _chunked_rows, bind_head, host_activations, release_host_activations
from octopus.rl.lora_targets import resolve
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.prefix_tree import (MOUNT_KEYS, allocate_prefix_groups, cost, patch_attention, seq_mounts, subset,
                                    tree_inputs, who)

_allocate_by_length = areal_data.allocate_balanced_mbs
_batch_seqs: list[np.ndarray] | None = None  # the batch _prepare_mb_list is splitting
_batch_mounts: dict | None = None  # its reads (docs/cross-branch-read.md), sequences = rows


def _allocate_by_prefix(mb_spec, lens):
    """AReaL's micro-batch allocation, but keeping each group's shared-prefix sequences together."""
    if _batch_seqs is None:
        return _allocate_by_length(mb_spec, lens)
    seqs = _batch_seqs + [np.empty(0, dtype=np.int64)] * (len(lens) - len(_batch_seqs))  # transport dummies
    mbs = allocate_prefix_groups(seqs, mb_spec.max_tokens_per_mb, mb_spec.n_mbs or 1, mb_spec.n_mbs_divisor,
                                 _batch_mounts)
    # each micro-batch's cost (k tree tokens), in run order: the ranks' sums and their lockstep show in the logs
    real = [[i for i in m if len(seqs[i])] for m in mbs]
    costs = [cost(*subset(seqs, _batch_mounts or {}, r)) / 1e3 if r else 0.0 for r in real]
    print(f"[octopus] {who()}: {len(mbs)} micro-batches, cost {sum(costs):.0f}k: {[round(c) for c in costs]}",
          flush=True)
    return mbs


def packed_mounts(data: dict, cu) -> dict | None:
    """The reads of packed sequences (cu offsets into data's flattened MOUNT_KEYS tensors) in tree_inputs' format;
    None without any (outside OCTOPUS_CURSOR the export adds no keys)."""
    if data.get("octopus_mount") is None or not bool(data["octopus_mount"].any()):
        return None
    return seq_mounts(list(cu), *(data[k].reshape(-1).cpu().numpy() for k in MOUNT_KEYS))


def batch_rows(input_: dict) -> tuple[list[np.ndarray], dict | None]:
    """A padded batch's rows (their tokens) and reads, for _allocate_by_prefix."""
    mask = input_["attention_mask"].bool()
    seqs = [input_["input_ids"][i][mask[i]].cpu().numpy() for i in range(mask.shape[0])]
    return seqs, packed_mounts({k: input_[k][mask] for k in MOUNT_KEYS if k in input_},
                               np.cumsum([0] + mask.sum(1).tolist()))


_mask_rolled = False  # within ppo_update (_loss_mask_rolled): its batch's loss_mask is already rolled


def scored_positions(kwargs: dict) -> torch.Tensor | None:
    """The flat positions of a packed micro-batch's model kwargs whose (logprob, entropy) AReaL reads: compute_logp's
    output (prox_logp) and ppo_update's loss and stats are all multiplied by, or reduced over, the loss_mask rolled
    onto the position that scores each token (areal/trainer/ppo/actor.py _compute_advantages, grpo_loss_fn); the head
    returns 0 at the others. compute_logp's batch has the rollout's loss_mask (1 on each trained token): its position
    is the one before. ppo_update's batch was rolled by compute_advantages or, in pure MOPD distillation,
    prepare_mopd_batch (which adds no `advantages`): ppo_update runs under _loss_mask_rolled and takes it in place.
    None without a loss_mask (every position)."""
    mask = kwargs.get("loss_mask")
    if mask is None:
        return None
    return (mask if _mask_rolled else torch.roll(mask, -1, -1)).reshape(-1).bool()


@contextlib.contextmanager
def _loss_mask_rolled():
    """Around ppo_update: scored_positions takes the loss_mask as already rolled."""
    global _mask_rolled
    _mask_rolled = True
    try:
        yield
    finally:
        _mask_rolled = False


def micro_batch_tree(kwargs: dict):
    """tree_inputs for a packed micro-batch's model kwargs, with its reads and scored positions: the training forward
    and compute_logp's."""
    cu = kwargs["cu_seq_lens_q"]
    return tree_inputs(kwargs["input_ids"], cu, packed_mounts(kwargs, cu.tolist()), scored_positions(kwargs),
                       with_tree=chunk_major.ENABLED)


areal_data.allocate_balanced_mbs = _allocate_by_prefix


ACT_GPU_GB = float(os.environ.get("OCTOPUS_ACT_GPU_GB", "0"))  # most saved activations kept on the GPU per micro-batch
ACT_GPU_TARGET_GB = float(os.environ.get("OCTOPUS_ACT_GPU_TARGET_GB", "40"))  # predicted peak the kept ones fill up to
# An 8-rank FSDP rank's peak allocated without kept activations, GiB = ACT_BASE_GB + ACT_PER_K_GB * tokens / 1024 (prefix-tree
# tokens of the micro-batch): the slope from octopus/tools/act_gpu_bench.py (17.4 -> 36.0 GiB from 65k to 262k tokens),
# the base from bundle-e9 at MB_TOKENS 131072 (17-18 GiB). A 4090D OOMs at ~45.3 GiB allocated (bundle-e9 at MB_TOKENS
# 229376 with a fixed 18 GiB kept); the default target leaves ~5 GiB for the model's error. Fewer ranks shard less:
# lower the target. Both were measured before prefix-tree layers streamed per kernel call (octopus/rl/seq_tiles.py):
# that path peaks lower (one A6000, 347k-token tree: train 24.2 GiB), so the prediction is conservative for trees.
# Block checkpointing (octopus/rl/block_checkpoint.py) holds a block's BLOCK-1 recomputed layer inputs on the GPU during
# its backward: ~0.0049 GiB per 1k tokens per input, 0.0146 for BLOCK 4's three (2 ranks, 199k-token tree: 19.5 GiB with
# BLOCK 4 against 16.1 per layer; 640k: 37.8 with BLOCK 4; 994k: BLOCK 2 and 4 ran out of a 48 GB card;
# results/block-checkpoint).
ACT_BASE_GB, ACT_PER_K_GB = 5.3, 0.095


def act_gpu_budget(tokens: int) -> float:
    """GiB of a micro-batch's saved activations to keep on the GPU: what fits under ACT_GPU_TARGET_GB beside the
    predicted peak of a micro-batch of this many tokens, at most ACT_GPU_GB. A fixed budget OOMs on the big ones."""
    return min(max(ACT_GPU_TARGET_GB - ACT_BASE_GB - ACT_PER_K_GB * tokens / 1024, 0.0), ACT_GPU_GB)


def apply_live_mb_tokens(mb_spec) -> None:
    """Live MB_TOKENS and ACT_GPU_GB (octopus/rl/config.py live_env): the micro-batch cap and the GPU share of its
    saved activations (lm_head.host_activations) for this train worker's next calls. Ranks that read it a call apart
    only get different micro-batch counts, which FSDPEngine.forward_backward_batch already evens out (all_reduce MAX,
    then dummy micro-batches), or a different host/GPU split of the same activations."""
    global ACT_GPU_GB
    live = live_env()
    mb, gb = live.get("MB_TOKENS"), live.get("ACT_GPU_GB")
    if mb is not None and int(mb) != mb_spec.max_tokens_per_mb:
        print(f"[octopus] live MB_TOKENS {mb_spec.max_tokens_per_mb} -> {int(mb)}", flush=True)
        mb_spec.max_tokens_per_mb = int(mb)
    if gb is not None and float(gb) != ACT_GPU_GB:
        print(f"[octopus] live ACT_GPU_GB {ACT_GPU_GB} -> {float(gb)}", flush=True)
        ACT_GPU_GB = float(gb)


def install_forward(model, temperature: float, block_tokens=None, dp_group=None) -> None:
    """The model's forward on AReaL's micro-batches (ChunkedHeadFSDPPPOActor.initialize): the chunked head at this
    temperature (lm_head.bind_head), its labels taken from the input_ids and its scored positions from the loss_mask
    (scored_positions). A packed micro-batch whose sequences share prefixes runs once on its prefix tree
    (octopus/rl/prefix_tree.py). Tensors saved for backward (the checkpointed
    layer inputs, or block inputs: octopus/rl/block_checkpoint.py, block_tokens its setter, the token count the max over
    dp_group) wait in pinned host memory, past the first act_gpu_budget on the GPU. With OCTOPUS_CHUNK_MAJOR the tree
    runs chunk-major instead (octopus/rl/chunk_major.py: no saved-tensor hooks, no blocks), into the same
    [1, T_flat, 2] logits."""
    if chunk_major.ENABLED and block_tokens is not None:
        raise RuntimeError("OCTOPUS_CHUNK_MAJOR=1 runs without block checkpointing: set OCTOPUS_BLOCK_CHECKPOINT=1")
    head = bind_head(model, temperature)

    def stash_labels(module, args, kwargs):
        head.octopus_labels = torch.roll(kwargs["input_ids"], shifts=-1, dims=-1).reshape(-1)
        head.octopus_scored = scored_positions(kwargs)
        # AReaL also passes cu_seqlens (= cu_seq_lens_q); transformers forwards it next to the
        # cu_seqlens it derives for the GDN kernel, so drop the duplicate
        return args, {k: v for k, v in kwargs.items() if k != "cu_seqlens"}

    model.register_forward_pre_hook(stash_labels, with_kwargs=True)
    patch_attention()
    head.octopus_tree = None
    forward = model.forward
    rows = functools.partial(_chunked_rows, head)

    def tree_offloaded_forward(*args, **kwargs):
        head.octopus_tree = None
        if kwargs.get("cu_seq_lens_q") is not None:
            tree = micro_batch_tree(kwargs)
            if tree is not None:
                head.octopus_tree, kwargs = tree
        if chunk_major.ENABLED and head.octopus_tree is not None:  # in the FSDP root's forward, as it must run
            plan = chunk_major.ChunkPlan(head.octopus_tree.tree, chunk_major.TOKENS, kwargs["input_ids"].device)
            return CausalLMOutputWithPast(logits=chunk_major.chunk_major_output(plan, model, rows))
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        if block_tokens is not None:  # one block size on every FSDP rank: that of the largest micro-batch
            tokens = torch.tensor(kwargs["input_ids"].numel(), device=kwargs["input_ids"].device)
            dist.all_reduce(tokens, op=dist.ReduceOp.MAX, group=dp_group)
            block_tokens(int(tokens))
        with host_activations(model, act_gpu_budget(kwargs["input_ids"].numel()) * 2**30):
            return forward(*args, **kwargs)

    model.forward = tree_offloaded_forward


@contextlib.contextmanager
def _chunk_major_call(slab: bool):
    """Around compute_logp and ppo_update: with OCTOPUS_CHUNK_MAJOR, afterwards (also when the call raised) no
    micro-batch's kept tensors wait for a backward and, with `slab`, their slab goes back to the OS
    (chunk_major.release_host). A call that returns with a micro-batch still waiting for its backward fails: that
    micro-batch's gradient would be lost."""
    try:
        yield
        assert not chunk_major._pending, "a chunk-major micro-batch's forward with grad ended its call without backward"
    finally:
        if chunk_major.ENABLED:
            chunk_major.release_host(slab)


INIT_LORA = os.environ.get("OCTOPUS_INIT_LORA", "")  # saved adapter the LoRA starts from (ChunkedHeadFSDPPPOActor)


class ChunkedHeadFSDPPPOActor(FSDPPPOActor):
    # These RPCs' batches stay in host memory (AReaL's engine_blueprint skips their move to the GPU and broadcasts them
    # over cpu_model_parallel_group); FSDPEngine moves one micro-batch at a time to the GPU (patches/areal-cpu-batch.patch).
    # On the GPU, a 340-row x 113k-padded rank's batch took 7.1 GiB for the whole call: its groups, their concatenation
    # and the micro-batch list. compute_advantages stays on the GPU: its float math must not move to another device.
    cpu_staged_rpc_methods = frozenset({"compute_logp", "ppo_update"})

    def _apply_peft_wrapper(self):
        """With OCTOPUS_INIT_LORA (a saved adapter dir), the LoRA starts from it instead of PEFT's B = 0 init: a run
        continued from a checkpoint. Rank 0's weights are what FSDPEngine.initialize broadcasts to every rank; the
        servers' v0 (ADAPTER0) must be the same adapter, or step 1's train/infer logp gap shows the mismatch.
        Empty target_modules (grpo.yaml's default; AReaL would read it as all-linear) resolve from the model:
        every decoder linear SGLang serves LoRA on (octopus/rl/lora_targets.py), as zero_adapter.py does."""
        if not self.config.target_modules:
            self.config.target_modules = resolve(self.model)
            print(f"[octopus] LoRA target_modules resolved: {self.config.target_modules}", flush=True)
        super()._apply_peft_wrapper()
        if INIT_LORA and self.rank == 0:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            saved = load_file(f"{INIT_LORA}/adapter_model.safetensors")
            result = set_peft_model_state_dict(self.model, saved)
            assert not result.unexpected_keys, result.unexpected_keys[:5]
            missing = [k for k in result.missing_keys if "lora_" in k]
            assert not missing, missing[:5]
            norm = sum(float(v.float().norm()) for k, v in saved.items() if "lora_B" in k)
            print(f"[octopus] LoRA initialized from {INIT_LORA}: {len(saved)} tensors, sum |lora_B| {norm:.2f}", flush=True)

    def _save_model_to_hf(self, path, tokenizer, processor):
        """A weight update's adapter also goes to the remote rollout hosts (octopus/rl/adapter_push.py) before
        _update_weights_from_disk publishes it for loading; saver checkpoints stay local."""
        super()._save_model_to_hf(path, tokenizer, processor)
        if adapter_push.DESTS and self.rank == 0 and os.path.basename(path).startswith("weight_update"):
            adapter_push.push_adapter(path, adapter_push.DESTS)

    def initialize(self, *args, **kwargs):
        result = super().initialize(*args, **kwargs)
        self.cpu_model_parallel_group = dist.new_group(  # gloo twin of the group cpu_staged_rpc_methods broadcast over
            dist.get_process_group_ranks(self.context_and_model_parallel_group), backend="gloo",
            use_local_synchronization=True)
        patch_model(self.model.config)  # packed sequences stay separate in the GDN layers
        tile_model(self.model)  # layers run token-wise parts in sequence tiles
        block_tokens = checkpoint_blocks(self.model)  # up to BLOCK_TOKENS keep every BLOCK-th layer input only
        self._check_weights_loaded()
        install_forward(self.model, self.config.temperature, block_tokens,
                        self.world_mesh["dp_sp"].get_group() if block_tokens else None)
        return result

    def compute_logp(self, data, *args, **kwargs):
        apply_live_mb_tokens(self.config.mb_spec)
        with _chunk_major_call(slab=False):  # ppo_update reuses it
            return super().compute_logp(take_rows(data), *args, **kwargs)

    def ppo_update(self, data, *args, **kwargs):
        apply_live_mb_tokens(self.config.mb_spec)
        with _chunk_major_call(slab=True), _loss_mask_rolled():
            result = super().ppo_update(take_rows(data), *args, **kwargs)
        release_host_activations()  # the step's pinned activation buffer is not held until the next step
        return result

    def forward_backward_batch(self, mb_list, *args, **kwargs):
        """With OCTOPUS_CHUNK_MAJOR, FSDP gathers the parameters and reduces the gradients once per call instead of per
        micro-batch (chunk_major.across_micro_batches), so ranks do not wait for each other between micro-batches.
        The micro-batch count is AReaL's: every rank runs the most any rank has (all_reduce MAX, then dummies). Only
        chunk-major: it keeps the parameters gathered over a whole micro-batch anyway; the flag-off path reshards each
        layer after its forward, and held gathered, the whole model would sit beside its activations."""
        if not chunk_major.ENABLED:
            return super().forward_backward_batch(mb_list, *args, **kwargs)
        if self.enable_tree_training:  # AReaL pads every rank to the MAX count only without its tree training
            raise RuntimeError("OCTOPUS_CHUNK_MAJOR needs AReaL's enable_tree_training off (once-per-call FSDP sync "
                               "counts the micro-batches as all_reduce MAX)")
        n = torch.tensor(len(mb_list), dtype=torch.int64)
        dist.all_reduce(n, op=dist.ReduceOp.MAX, group=self.cpu_group)
        with chunk_major.across_micro_batches(self.model, int(n)):
            return super().forward_backward_batch(mb_list, *args, **kwargs)

    def compute_advantages(self, data, *args, **kwargs):
        """AReaL group adv_norm, then replace plan-sequence advantages with leave-one-out A_main / A_branch. A plan
        episode dispatched alone (octopus/rl/dp_balance.py) is its own AReaL group: every row gets octopus_adv."""
        data = super().compute_advantages(take_rows(data), *args, **kwargs)
        for batch in data:
            plan, adv = batch.get("octopus_plan"), batch.get("octopus_adv")
            if plan is None or adv is None or not bool(plan.any()):
                continue
            plan = plan.bool()
            batch["advantages"] = torch.where(
                plan, adv.to(device=batch["advantages"].device, dtype=batch["advantages"].dtype), batch["advantages"])
        return data

    def _prepare_mb_list(self, input_, **kwargs):
        global _batch_seqs, _batch_mounts
        _batch_seqs, _batch_mounts = batch_rows(input_)
        try:
            return super()._prepare_mb_list(input_, **kwargs)
        finally:
            _batch_seqs = _batch_mounts = None

    def _check_weights_loaded(self):
        """Refuse to train a model whose weights did not load: a real 9B scores code at NLL ~1-2, random ~12."""
        ids = self.tokenizer("def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
                             "for i in range(10):\n    print(i, fibonacci(i))\n", return_tensors="pt")["input_ids"].cuda()
        logits = self.model(input_ids=ids, use_cache=False).logits.float()
        loss = -torch.log_softmax(logits[0, :-1], -1).gather(-1, ids[0, 1:, None]).mean()
        # the backward also runs every GDN (Triton) kernel's first-call autotuning now, on an empty card:
        # mid-step it benchmarks configs with extra buffers and OOMed inside an 88k-token micro-batch
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        for shape in {(m.in_channels, m.kernel_size[0], m.bias is not None) for m in self.model.modules()
                      if isinstance(m, torch.nn.Conv1d) and m.groups == m.in_channels}:  # GDN, Mamba-2
            prefix_tree.warm_conv(*shape, getattr(torch, self.config.dtype), ids.device)
        nll = loss.item()
        print(f"[octopus] init check: NLL {nll:.2f} on a fixed snippet", flush=True)
        if nll > 5:
            raise RuntimeError(f"actor weights look unloaded (NLL {nll:.2f})")

    def optimizer_step(self):
        # backward leaves the card full of cached blocks; NCCL allocates its buffers (grad-norm
        # all-reduce) outside PyTorch's cache, so hand the cache back first
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()  # and pinned blocks an outgrown micro-batch took (lm_head.host_activations)
        stats = super().optimizer_step()
        print(f"[octopus] step peak allocated {peak:.1f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats()
        return stats

    def _compute_logprobs_entropy(self, logits, inputs, ulysses_pad_size=0):
        return logits[..., 0], logits[..., 1]

    def _compute_logprobs(self, logits, inputs, ulysses_pad_size=0):
        return logits[..., 0]

    def _get_vocab_min_max_logits(self, logits, ulysses_pad_size=0):
        zeros = torch.zeros_like(logits[..., 0])  # logging only; full logits never exist
        return zeros, zeros
