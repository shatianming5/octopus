# Chunk-major on the GPU (P2): Qwen3.5-4B, real search-v9-rest micro-batches

`octopus/rl/chunk_major.py` runs a prefix tree through the model one chunk of tree tokens at a time. Here it is measured
against the actor's path (k1: HF per-layer checkpointing; k4: block checkpointing,
`octopus/rl/block_checkpoint.py`) on real micro-batches. The setup matches the actor's: LoRA r64 on the resolved targets
(B initialised to N(0, 1e-3)), FSDP2 per decoder layer and root (bf16 params, fp32 reduce), sequence tiles and the
chunked head. The reference paths keep saved tensors in pinned host memory (ACT_GPU_GB 0); chunk-major keeps its kept
tensors and their gradient sums there.

The runs were on node-134 (RTX 4090D 48 GB, cards shared with another user's ~1 GB processes, 40 GiB cap per process)
on 10-29. The commands are in `bench.py`, `run134.sh` and `select_mbs.py`, and the logs are in `logs/`. The 994k tree
packs one group at a 1M cap: the spool's largest real micro-batch is 665k.

## Use in training

The actor runs chunk-major when the train workers start with `OCTOPUS_CHUNK_MAJOR=1` (default off), in chunks of
`OCTOPUS_CHUNK_TOKENS` tree tokens (default 32768: below, 32k chunks fit the 994k unit in 31.9 GiB on one card;
before #346, 64k ran out of a 40 GiB cap there, and took 35.4 GiB per rank on 2). Both are host settings (`octopus/rl/config.py` HOST_ENV);
export them before `rl_run.sh`, whose train workers inherit its environment. What changes (`octopus/rl/actor.py`
install_forward):
- A prefix-tree micro-batch's forward returns `chunk_major_output` as the logits, same shape [1, T_flat, 2]: with grad
  (ppo_update) the chunk loop and its backward, without (compute_logp) the forward loop alone. No saved-tensor hooks
  (`lm_head.host_activations`, ACT_GPU_GB), no block checkpointing. A micro-batch whose sequences share nothing is a
  tree too (below, "Ranks on different paths").
- `OCTOPUS_BLOCK_CHECKPOINT` must be 1 (per-layer checkpointing inside each chunk): larger, the actor refuses to start.
- compute_logp and ppo_update end with `chunk_major.release_host()`, also when they raise: a micro-batch whose backward
  failed or never ran no longer blocks the next forward, and after ppo_update the slab goes back to the OS (compute_logp
  keeps it for the ppo_update after it, below). A call that returns with a micro-batch still waiting for its backward
  raises: that micro-batch's gradient would be lost.

Checked on CPU (`tests/test_actor_chunk_major.py`, fp64): with the flag the actor's compute_logp and train logits and
the gradients of two micro-batches plus AReaL's dummy equal the flag-off path's within 1e-9; with the flag off no
chunk-major code runs.

### The real actor on GPUs (P3, 10-29)

`actor_ab.py` runs `ChunkedHeadFSDPPPOActor` on AReaL's FSDP engine on real search-v9-rest spool groups, as the
controller calls it: compute_logp, compute_advantages, ppo_update (one optimizer step) per step; `run_ab.sh` launches
it. Every run starts from the same adapter: node-134's `adapter-v0-4b-text-r64` (r64, the resolved targets) with B drawn
from N(0, 1e-3), seed 0, as `bench.py` does (a trained adapter did not reach node-134: its LAN to node-138 moved ~130
kB/s). Qwen3.5-4B-text, bf16, node-134 (RTX 4090D 48 GB, 0.85 of a card per process; another user's ~1 GB process
shared G6 for part of the runs). k1 / k4 are the flag off (`OCTOPUS_BLOCK_CHECKPOINT` 1 / 4), cm32 / cm16 on, in
chunks of 32k / 16k; `s16` / `s8` the flag off with `prefix_tree.SEG` at 16k / 8k, the bf16 noise floor as above. The
filtered logs and comparisons are in `logs/p3-actor.log`.

Same weights, step 1, one card, group search-6806~3e9e76fc (19 rows, one micro-batch of 187,627 flat tokens):

| run | actor loss | grad norm | against k1: loss rel diff | grad-norm rel diff | logp max / mean abs diff | LoRA grad cosine (smallest per tensor) |
|---|---|---|---|---|---|---|
| k1 | 0.249209 | 0.3535 | | | | |
| k4 | 0.249209 | 0.3535 | 0 | 0 | 0 / 0 | 0.99990 (0.9998) |
| k1s16 (noise floor) | 0.249361 | 0.4531 | 6.1e-4 | 28% | 21.7 / 0.061 | 0.855 (0.68) |
| k1s8 (noise floor) | 0.248938 | 0.3281 | 1.1e-3 | 7.2% | 26.7 / 0.063 | 0.787 (0.43) |
| cm32 | 0.249560 | 0.3926 | 1.4e-3 | 11% | 24.3 / 0.054 | 0.858 (0.61) |
| cm16 | 0.249442 | 0.3398 | 9.3e-4 | 3.9% | 23.2 / 0.063 | 0.631 (0.05) |

(grad norm as AReaL logs it; logp over every position of the batch, prompt and tool output included, hence the large
maxima.) The plan's "loss and grad-norm rel diff < 1e-3" does not hold in bf16 for any path that changes kernel-call
bounds, the SEG-changed whole tree included (grad norm 7-28%). Against the two floor samples: cm32's loss rel diff
(1.4e-3) is above both (6.1e-4, 1.1e-3); its grad-norm diff (11%) and mean logp diff (0.054) are inside them, and its
gradient cosine (0.858) equals the SEG-16k sample's. cm16's loss (9.3e-4), grad norm (3.9%) and mean logp (0.063) are
inside them, its gradient cosine (0.63, smallest per tensor 0.05) below both (0.79, 0.86), so the same micro-batch ran
again in IEEE fp32: `bench.py --fp32 --ieee --no-fsdp` on one card (its own LoRA init, B from N(0, 1e-3)), against k1:
k1s8 differs by logp max |diff| 2.1e-4 on the loss positions and gradient cosine 1.000000 (smallest per tensor 1.0000),
cm16 (11 chunks) by 1.05e-4 and 1.000000 (1.0000); cm32 ran out of the 40 GiB cap there (`logs/p3-fp32-cm16.log`). So
cm16 matches the flag-off path within fp32 rounding, and its bf16 cosine of 0.63 is bf16 rounding. This backward
comparison is the one-card, unsharded path; the FSDP2 path and the actor's wiring are exact in the CPU fp64 tests
(`tests/test_chunk_major.py` on 2 FSDP2 ranks, `tests/test_actor_chunk_major.py`). The actor itself in fp32 (2 ranks)
ran out of the 0.85 cap in cm16's backward (inferred cause: the chunk loop's gathered fp32 parameters beside their
shards; not measured). Time and memory were not measured for chunks smaller than 32k; keep the default 32768.

Two FSDP ranks (G6 + G7), 3 steps each (rank 0 search-6806~5ca61d05, rank 1 search-2981~27ffb875, one micro-batch
each; rank 1's 377,870 flat tokens), real optimizer steps; "pin" is the chunk-major slab's pin + unpin seconds per step:

| run | step 1 loss / grad norm | step 2 | step 3 | compute_logp (steps 2-3) | ppo_update (steps 2-3) | GPU peak | host peak | pin |
|---|---|---|---|---|---|---|---|---|
| k4 | 0.141519 / 0.1343 | 0.132979 / 0.1495 | 0.126288 / 0.1385 | 29 s | 155-160 s | 20.0-21.3 GiB | 10.9-20.9 GiB | |
| cm32, slab released after compute_logp | 0.141565 / 0.1354 | 0.132957 / 0.1471 | 0.125969 / 0.1361 | 34.5-36.5 s | 158.5-163.4 s | 24.6-26.5 GiB | 8.7-12.2 GiB | 9.0-12.6 s |
| cm32, slab kept (this code) | 0.141565 / 0.1354 | 0.133124 / 0.1437 | 0.126223 / 0.1361 | 33.3-35.9 s | 155.4-162.0 s | 24.6-26.5 GiB | 8.7-12.2 GiB | 4.2-6.7 s |

Step 1 (same weights) on the same 2-rank batch, against k1 (0.141482 / 0.13468, one step of each run): k4 differs by
loss 2.6e-4, grad norm 2.6e-3; k1s16 (noise floor) by 4.1e-4, 2.7e-3; cm32 by 5.9e-4, 5.0e-3 (against k4: 3.2e-4,
7.7e-3). So chunk-major is of the floor's order, not inside it: 1.4x (loss) and 1.9x (grad norm) the SEG-16k sample,
2.2x and 1.9x the k4 one; above the plan's 1e-3 on grad norm, as the floor's own k1 against k1s16 is. No hang, no error;
the two chunk-major runs differ from step 2 on by GPU nondeterminism (their step 1 is identical). GPU peak is the
actor's per-step `step peak allocated`; host peak the process RSS above the call's start. On these groups (few tree
tokens per micro-batch: the slab holds 8.7 / 12.2 GiB per rank) chunk-major is 15-25% slower in compute_logp, within
noise in ppo_update, 4-5 GiB higher on the GPU and lower on the host than k4. The 994k-token units it is for are in the
tables below.

### Ranks on different paths (10-30)

FSDP ranks must issue the same collectives in the same order, so with the flag on every micro-batch runs chunk-major.
A micro-batch whose sequences share nothing, mount nothing and fit one kernel call (`prefix_tree.SEG`) used to be no
tree, and ran the HF layers flat: its backward gathers and reduce-scatters layer by layer (31 down to 0), where
chunk-major's gathers every layer for its first chunk and reduce-scatters them all in FSDP's final callback (0 up to
31). One rank flat beside another chunk-major in the same micro-batch mispairs NCCL's collectives, which NCCL does not
check: buffers of different sizes came out NaN, equal-sized ones mix another layer's shard in silently. The forward
(compute_logp, and the forward of ppo_update, run without autograd) gathers the same layers in the same order on both
paths: DETAIL passed it. With the flag off the streamed tree and the flat path both run FSDP's per-layer hooks, one
order: unaffected (the shortcut stays there).

Seen on 2 ranks of node-134 (G4 + G7; `nan_check.py`: `actor_ab.py` plus each micro-batch's path and its non-finite
gradient shards; the 4 groups at MB_TOKENS 131072, 5 micro-batches per rank; f6e72ca, twice): finite after
micro-batches 1-4, and after the 5th (rank 0 chunk-major in 4 chunks, rank 1 flat at 20,420 tokens) rank 1's shard of
every LoRA tensor of layers 0-1 and rank 0's of layers 30-31 all NaN, the other halves finite. Under
`TORCH_DISTRIBUTED_DEBUG=DETAIL` that run's compute_logp passed and its 5th backward stopped at "Detected mismatch
between collectives" (rank 0 in an all-gather, rank 1 in a reduce-scatter). The one-micro-batch-per-rank runs above,
all trees, never met it. On CPU, 2 gloo ranks (one tree, one such micro-batch; `tests/test_actor_chunk_major.py`)
abort at "Received data size doesn't match expected size".

Now `prefix_tree.tree_inputs` builds every tree with_tree (chunk-major on), and chunk-major's collectives do not depend
on its chunk count. The same run, rank 1's 5th micro-batch in 1 chunk:

| run | loss | grad norm | LoRA grad cosine to k1 (norm diff) | DETAIL |
|---|---|---|---|---|
| f6e72ca (twice) | 0.049515 | NaN | | mismatch |
| this fix (3 runs) | 0.049507 | 0.0844, 0.0846, 0.0844 | 0.767 (5.9%) | no mismatch, whole step |
| k1, the flag off (3 runs) | 0.049650 | 0.0898 (each) | k1 again: 0.99988 (0.1%) | no mismatch, whole step |
| k1s16, the flag off at SEG 16k | 0.049545 | 0.0856 | 0.807 (4.6%) | |

Per layer (`grad_layers.py`) the fix tracks k1s16: cosine to k1 0.98-0.995 from layer 12 up for both, falling to
0.54-0.84 (fix) and 0.63-0.86 (k1s16) in layers 0-7, bf16 rounding accumulating down the backward; its 8 attention
layers' norms are k1's (0.0255 / 0.0292 at layer 3, 0.0131 / 0.0130 at 31; the once-per-call FSDP variant on the same
batch, before this fix and so on mixed paths too, had 2.299 in all 8, cosine 0.003). So the fix is within the noise
floor of three SEG samples (same data, 2 GPUs, flag off: k1 / k1s8 0.819, k1s16 / k1s8 0.712, k1 / k1s16 0.807:
0.71-0.82), as is once per call on it (0.799 to k1, "FSDP once per call"), and as cm16 was on the single micro-batch
above (exact in fp32 there; here the CPU fp64 tests are the exactness check). On one card (rank 1's groups alone) that
20,420-token micro-batch took 1.8 s in compute_logp and 9.2-10.0 s in ppo_update in one chunk, 1.8 s and 9.2-9.4 s flat
(`logs/p3-mixed-paths.log`).

The slab: pinning took ~0.38 s and unpinning ~0.1 s per GiB. Released after compute_logp as well, it was unpinned and
pinned again before every ppo_update: 4.5-6.4 s per step at these sizes, ~17 s at the 665k unit's 36 GiB. So
compute_logp now keeps it for ppo_update (`release_host(slab=False)`), which releases it at its end; the pin + unpin
time per step halved (table).

## Correctness: the bf16 noise floor and fp32

Under bf16 no path matches another to a fixed tolerance. The same whole-tree path with only SEG changed (`k1s16`: other
kernel-call bounds, same arithmetic) already moves a logprob by up to 0.84 and leaves a LoRA gradient cosine of 0.83.
That is the bf16 rounding of a 4B model over a 256k-token tree. `layerdiff.py` shows it layer by layer on a 41k tree:
the GDN layers before the first attention layer are bit-identical, and from there both chunk-major and the
SEG-changed whole tree drift by the same amount. So chunk-major is held to that floor in bf16, and to fp32 for
exactness. The fp32 runs use SDPA for attention (FlashAttention has no fp32) and fla's Triton kernels, which round
`dot` products through TF32.

| micro-batch | precision | compared with k1 | logp max abs diff | LoRA grad cosine (smallest per tensor) |
|---|---|---|---|---|
| 256k, 1 GPU | bf16 | k1 again (atomics) | 0 | 0.99991 (0.9998) |
| | | k4 | 0 | 0.99991 (0.9998) |
| | | **k1s16 (noise floor)** | 0.838 | **0.825** (0.38) |
| | | cm32, fp32 sums | 0.837 | 0.814 (0.37) |
| | | cm64, fp32 sums | 0.457 | 0.856 (0.53) |
| | | cm32, bf16 sums | 0.837 | 0.814 (0.38) |
| | | cm64, bf16 sums | 0.457 | 0.857 (0.53) |
| 101k, 1 GPU, SEG 8k | fp32 | **k1s4 (noise floor)** | 0.031 | **0.99883** (0.9949) |
| | | cm16 (8 chunks) | 0.025 | 0.99843 (0.9902) |
| | | cm8 (18 chunks) | 0.029 | 0.99661 (0.9734) |
| | | cm8, kept tensors in GPU memory, against cm8 in host memory | 0 | 1.000000 (1.0000) |
| 41k, 1 GPU, SEG 4k | fp32 with TF32 (as above) | **k1s2 / k1s8 (noise floor)** | 0.0039 / 0.0049 | **0.99883 / 0.99899** |
| | | cm8 (6 chunks) / cm4 (12 chunks) | 0.0039 / 0.0030 | 0.99935 / 0.99112 (0.9384) |
| | IEEE fp32 (no TF32 anywhere) | **k1s2 / k1s8 (noise floor)** | 7.5e-5 / 4.8e-5 | **1.000000 / 1.000000** |
| | | cm8 / cm4 | 5.7e-5 / 6.7e-5 | 1.000000 / 1.000000 (1.0000) |

In fp32 cm8 at 101k (18 chunks) sits 3x further from k1 than the SEG floor, and cm4 at 41k 8x further. Two checks
(`diag134.sh`) show this is TF32 rounding, not a defect:
- With the kept tensors in GPU memory instead of the pinned slab, the forward is bit-identical and the gradients
  equal. This rules out the host copies, the slab and `_Kept`.
- With TF32 off everywhere (cuBLAS, cuDNN, and Triton's `TRITON_F32_DEFAULT=ieee`, which fla's kernels follow), the
  noise floor drops from 1-cos ~1e-3 to below 1e-6, and chunk-major drops with it, cm4 included. A defect would have
  stayed at ~1e-2.

The bf16 comparisons themselves varied by about ±0.04 between processes (the same micro-batch's cm32 against k1 gave
cosine 0.814 and 0.852) mostly because `bench.py` seeded after `get_peft_model`, so each process drew its own LoRA A
init; it now seeds before. Repeated runs of one config in one process agree to cosine 0.9999, chunk-major and k1 alike
(FlashAttention-2's atomic adds of dq), and fp32 repeats to 1e-6 (eval-determinism's matrix, node-134
`/mnt/disk4/user/tmp/cmnd/logs/`).

The CPU tests (`tests/test_chunk_major.py`, fp64) cover exactness itself: logp and every gradient within 1e-9 of the
whole tree, also under FSDP2.

## Memory and time (bf16)

"stores" is the kept tensors plus their gradient sums, carved from one exactly pinned slab. That is the host memory
chunk-major adds in steady state. It matches the bench's count from the plan (`predicted_host`): a self-consistency
check of the carving, not an independent estimate. "host peak" is process RSS above its level before the run. On a
first micro-batch it also includes torch's power-of-two pinned blocks: k1's 68 GiB is its 41.5 GiB of saved tensors
rounded up, and chunk-major's first 665k micro-batch peaks at 46.0 GiB per rank, 39% above the assessment's ~33 GiB.
The first 256k cm32 run (before `_Kept`) peaked at 42.9 GiB of GPU memory; its log was overwritten, not archived.

| micro-batch | ranks | variant | chunks | train fwd+bwd | compute_logp | GPU peak | host peak | stores |
|---|---|---|---|---|---|---|---|---|
| 256k | 1 | k1 | | 226 s (211 s rerun; warm 228 s) | 34 s | 23.3 GiB | 68.2 / 41.5 GiB | |
| | | k4 | | 237 s | 34 s | 27.9 GiB | 41.6 GiB | |
| | | cm32 fp32 | 10 | 222 s | 41 s | 31.9 GiB | 25.6 GiB | 20.2 GiB |
| | | cm64 fp32 | 5 | 224 s | 39 s | 39.3 GiB | 24.8 GiB | 14.0 GiB |
| | | cm32 bf16 | 10 | 224 s | 42 s | 31.9 GiB | 17.5 GiB | 13.9 GiB |
| | | cm64 bf16 | 5 | 223 s | 39 s | 39.3 GiB | 16.7 GiB | 9.6 GiB |
| 256k, warm | 1 | k1 / k4 / cm32 / cm64 (`--repeat 2`) | | 228 / 211 / 223 / 219 s | 34 / 34 / 40 / 34 s | | | |
| 516k | 1 | k4 | | 453 s | 70 s | 37.0 GiB | 40.3 GiB | |
| | | cm32 fp32 | 19 | 456 s | 81 s | 32.2 GiB | 44.8 GiB | 32.7 GiB |
| 994k | 1 | cm32 fp32 | 39 | 761 s | 144 s | 31.1 GiB | 54.3 GiB | 43.9 GiB |
| | | cm64 fp32 | 18 | out of the 40 GiB cap | | | | |
| 665k | 2 | k4 | | 600 s | 94 s | 39.6 GiB | 41.6 GiB | |
| | | cm32 fp32 | 26 | 559 s | 101 s | 27.3 GiB | 46.0 GiB | 36.0 GiB |
| | | k4s16 (noise floor) | | out of the 40 GiB cap | | | | |
| 994k | 2 | cm32 fp32 | 39 | 761 s | 136 s | 27.0 GiB | 50.6 GiB | 43.9 GiB |
| | | cm64 fp32 | 18 | 776 s | 144 s | 35.4 GiB | (slab reused) | 40.3 GiB |
| | | k4 | | out of a 48 GB card (results/block-checkpoint, BLOCK 2 and 4) | | | | |

Per-rank numbers; 2 ranks = FSDP2 over NCCL, both ranks on the same micro-batch. At 665k the chunk-major gradients
against k4 have cosine 0.932, against 0.814-0.857 at 256k; the 665k noise floor itself did not fit the cap. The 994k
cm32 single-rank row and the first 256k ones ran before the slab (`release_host`): their "host peak" includes the
pinned cache's rounding. The time per run varies by ~7% between identical runs (k1 at 256k: 226 / 211 s).

At 994k there is no reference that fits. Chunk-major at 64k chunks against 32k chunks gives logp max |diff| 3.3
(rank 0) and 2.23 (rank 1), and gradient cosine 0.959. The two ranks of one run see the same micro-batch, yet their
logp differences are not the same (665k: 0.454 / 0.541 against k4): the GPU kernels are not deterministic across ranks
either.

## One checkpoint per layer inside a chunk (10-29)

`run_backward` used to recompute a chunk with every checkpoint of the streamed layers still on: HF's per-layer one
(`OCTOPUS_BLOCK_CHECKPOINT=1`), `checkpoint(_tree_attention_call / _tree_gdn_call)` per kernel call and `tiled()`'s per
tile, each level running the layer's GEMMs once more. It now runs under `seq_tiles.no_inner()`: the layer checkpoint
stays, the ones inside it go (the flat and whole-tree paths keep theirs). Per step a layer runs 3 times (compute_logp
aside): the no-grad forward, the chunk's forward with autograd, and the layer checkpoint's recompute.

Forward runs counted on the tests' tiny fp64 model (as `tests/test_chunk_major.py` test_backward_runs_each_layer_twice:
2 GDN + 2 attention layers, 5 chunks; rows run over tree tokens per layer, in run_backward; early stop skips each
checkpoint's tail, hence the fractions): nested, q_proj 3.45, o_proj / gate_proj 3.25, k_proj 3.00, down_proj 2.25,
FlashAttention 2.45 (50 calls), GDN in_proj_qkv 3.90, out_proj 3.42, delta rule 2.75; now 2.00 each (down_proj 1.9,
FlashAttention 40 calls). Logp and every gradient are bit-identical to the nested path in fp64.

On node-134 (one rank, bf16, 41 GiB cap; `run134.sh`'s setup), `bench.py MB --no-logp --gpu-cap 41 --variants ...`
(`--repeat 1`: the counts are one training run's): 256k `cm32-fp32@nested cm32-fp32 cm32-fp32@nested cm32-fp32` on G7,
the first, cold run left out (`logs/one-checkpoint-256k-1gpu.log`, from a scratch copy of `bench.py` with the same
counting: its nested variant was named `@old`, renamed `@nested` in the log afterwards, and it labels the count "flash
calls" where `bench.py` prints "FlashAttention calls"); 994k `cm32-fp32 cm32-fp32@nested` on G4 (`logs/one-checkpoint-994k-1gpu.log`), the new path first, cold:

| unit | variant | train fwd+bwd | GPU peak | FlashAttention calls | MLP forwards per layer | logp max diff, grad cosine |
|---|---|---|---|---|---|---|
| 256k | nested (before) | 209.6 s | 31.9 GiB | 1008 | 3.90 | 0, 0.999901 against the first nested run |
| | one checkpoint per layer | 190.1 / 182.9 s | 35.4 GiB | 816 | 3.00 | 0, 0.999919 / 0.999923 against it |
| 994k | one checkpoint per layer (cold) | 737.5 s | 31.9 GiB | 2256 | 3.00 | |
| | nested (before) | 751.4 s | 32.0 GiB | 2696 | 3.87 | 0, 0.999947 against it |

At 256k this is 9-13% of the step; at 994k 1.8%, with the new path paying the process's warm-up (at 256k the cold run
took 486 s against 210 s warm, much of it torch.compile, whose cache the 994k run reused), so 994k's gain is a lower
bound; eval-speedups' warm runs of the same unit, 786 -> 720 s, bound it from above. The 994k peak does not come from
a layer's backward (31.9 against 32.0 GiB).

The gradient differences are the run-to-run floor of the kernels' atomics. The memory added is what one layer's
backward now holds at once: its activations over the chunk, which grow with `OCTOPUS_CHUNK_TOKENS`, and the keys and
values each of the chunk's attention calls gathered (4 KB per key), which do not: their sum over the chunk's calls
follows the tree's path lengths and fan-out. Its largest value per 32k chunk on the real units (review of #346): 3.01
GiB at 256k, 3.85 GiB at 516k, 2.06 GiB at 665k, 1.69 GiB at 994k. The earlier sections' tables were measured before
the change.

## FSDP once per call (across_micro_batches, 10-29)

`chunk_major.across_micro_batches` (wired in `actor.py` `forward_backward_batch`) keeps every FSDP module gathered from
an actor call's first forward to its end and reduce-scatters the gradients once, in the last micro-batch's backward.
`actor_ab.py --lockstep` runs AReaL's `forward_backward_batch` instead: per micro-batch, as before. The runs:
`actor_ab.py` on node-134 G1+G2 (2 ranks, bf16, chunk-major 32k, 0.84 of a card), launched by `run_ab.sh` with the
caps and groups below; logs in node-134 `/mnt/disk4/user/nosync-134/logs/`, gradients in `save/`.

**CPU.** `tests/test_chunk_major.py::test_fsdp_across_micro_batches` (2 gloo ranks, 3 micro-batches each, rank 1's last
a dummy): compute_logp gathers once and returns the per-micro-batch path's logits bit for bit; ppo_update gathers once
and reduce-scatters once, in the last micro-batch; its gradients equal the per-micro-batch path's bit for bit with mixed
precision (fp32 compute, fp64 reduce) and within 4.3e-14 (largest 2.4e2) in fp64. The same comparison on 2 GPUs with
NCCL and the pinned slab (`fsdp_gpu_check.py`: the test's tiny model and kernels): both paths within 7.6e-13 of the
whole tree in fp64.

**Gradients on 2 GPUs** (with #354: every micro-batch chunk-major; the earlier runs here, before it, mixed flat and
chunk-major micro-batches across ranks and were wrong both ways: "Ranks on different paths"). AReaL's grad norm and
`actor_ab.py --compare` of the merged LoRA gradients before the optimizer step; k1 is the flag off on the same groups and
caps:

| groups (rank 0 / rank 1), `MB_TOKENS` | per micro-batch | once per call | per micro-batch vs once per call | each vs k1 (k1 grad norm) |
|---|---|---|---|---|
| 3e9e76fc + 85d76f47 / 5ca61d05 + 27ffb875, 131072 (5 micro-batches a rank) | 0.08373169600963593 | 0.08373169600963593 (repeat: same) | logp equal; cosine 0.99990, norm 4.9e-5 apart | 0.7986 / 0.7988 (0.0889) |
| 3e9e76fc / 5ca61d05, 65536 (3 + 3 dummies / 6) | 0.1713 | 0.1709 | logp equal; cosine 0.99988 | 0.8736 / 0.8733 (0.1742) |

Once per call against its own repeat: cosine 0.99989 (FlashAttention's atomics), the same as against per micro-batch:
the two differ by bf16 run-to-run noise, from per micro-batch adding each micro-batch's reduced gradient into the bf16
sharded `.grad` where once per call sums in fp32 and casts once. Both are inside the bf16 floor from k1 (three SEG samples on the first row's data: 0.71-0.82).
IEEE fp32 (`--fp32`) does not fit at these sizes: the SDPA attention it needs ran out of a 38 GiB cap in compute_logp,
in 32k and 8k chunks; the exact comparison is the CPU test above and `fsdp_gpu_check.py`.

**Time** (same runs, the second row's lists). Seconds per micro-batch, from its forward to the next one's (the last to
the call's end):

| call | per micro-batch (rank 0 / rank 1) | call |
|---|---|---|
| compute_logp, per micro-batch | 8.8 6.8 4.3 4.6 4.1 2.7 / the same | 31.7 s |
| compute_logp, once per call | 7.3 5.5 1.7 1.7 1.7 1.7 / 8.9 5.7 4.1 4.4 3.6 3.0 | rank 0 done at 19.6 s, rank 1 29.7 s |
| ppo_update, per micro-batch | 32.4 27.0 19.1 21.4 17.0 15.7 / the same | 132.6 s |
| ppo_update, once per call | 26.0 25.0 7.7 7.6 7.6 50.9 (waits) / 30.2 25.1 18.7 20.8 16.3 13.9 | 125.0 s |

Per micro-batch, both ranks take the slower one's time plus the gathers and reduce-scatter (PCIe cards). Once per call,
rank 0 runs its own micro-batches and waits once, at the reduce-scatter. GPU peak unchanged (24.1 / 25.7 GiB per rank
both ways). On the first row's lists (sizes alike at each index) ppo_update took 297.0 s per micro-batch and 283.0 s
once per call. `results/dp-balance/replay.py` (the slowest rank's total, its dummies included) on search-v9-rest
step 43, 4 ranks: ppo_update idle 13.3% -> 2.5% (134.7 -> 119.8 min, ideal 116.8).

## The GDN convolution on fla's kernel (#348, 10-30)

`prefix_tree.segment_conv` runs a tree level's (or a packed batch's) causal convolution as one call of fla's Triton
`causal_conv1d`, each segment's context as its initial state, instead of transformers' torch fallback per segment.
`bench.py --variants cm32-fp32 cm32-fp32` on main (#346 + #354) against main + #348, back to back on one exclusive card
(`logs/conv-ab-*`; 256k on node-68, 994k on node-134, utilization logged), the second variant (steady state):

| micro-batch | main: compute_logp / train / GPU peak | #348 | change |
|---|---|---|---|
| 256k | 41.9 s / 184.2 s / 35.4 GiB | 39.3 s / 176.4 s / 35.4 GiB | -6.2% / -4.2% |
| 994k | 140.1 s / 623.2 s / 32.8 GiB | 129.7 s / 590.7 s / 32.6 GiB | -7.4% / -5.2% |

LoRA gradients against the first variant: cosine 0.99995 on both. That is bf16 rerun noise within one process, not a
main-against-fla comparison: the unit tests cover that (`tests/test_prefix_tree.py`: fp32 within 1.2e-6 of torch, bf16
no further from fp64 than torch's). The first variant
of a fresh process paid autotuning: 994k train 1281 s against main's 795 s. fla's autotune key is (channels, width,
cdiv(T, 1024), each tensor argument's dtype) and every length bucket compiles anew, so a process with an empty Triton
cache spends 25 min in `prefix_tree.warm_conv` (8192 channels: 32 buckets, with and without contexts; node-134, 2.8 GiB
peak); the results stay in `TRITON_CACHE_DIR` (default `~/.triton/cache`), and a later process on the host takes 3.2 s.
The actor runs `warm_conv` at load (`_check_weights_loaded`), so no step tunes mid-way. At 994k the train saving is
~33 s per micro-batch: the first 25 minutes on a new host pay back within a few steps.

## Against the plan's P2 acceptance

| criterion | result |
|---|---|
| bf16 logp diff of the order of the tree path's own | yes: equal to the SEG-changed whole tree's (0.84 at 256k), not test_seq_tiles' 2e-2 bound, which a 256k tree's own path exceeds |
| grad cosine >= 0.999 | not reachable in bf16 by any path (noise floor 0.825); revised (coordinator, 10-29) to two layers: fp32 against its noise floor (with TF32: 0.9966-0.9993 against 0.9988-0.9990; IEEE fp32: 1.000000 like the floor) and bf16 against the bf16 floor |
| compute_logp | 0-24% slower than k1/k4 (256k 34-42 s against 34 s; 665k 101 s against 94 s) |
| 994k GPU peak <= 40 GiB | before #346: 27.0 GiB (32k chunks) and 35.4 GiB (64k), 2 ranks; 31.1 GiB (32k) on one card. After: 31.9 GiB (32k) on one card |
| host within +-20% of the estimate | stores = the plan's prediction exactly; 665k 36.0 GiB per rank against the assessment's ~33 GiB (+9%) |
| time <= k4 + 5% | 256k warm (`--repeat 2`, both pinned slabs sized before the timed run): cm32 +5.8% (223 / 211 s), just over k4 + 5%, within the ~7% run-to-run noise; cm64 +3.7% (219 s); k1 228 s. 516k +0.6% (456 / 453 s); 665k (2 ranks) -7% (559 / 600 s), inside the same noise. The single-run rows time k1/k4 re-pinning lm_head's slab inside the run while chunk-major reused its own: not like for like |

Coordinator decisions (10-29): the (e) reference of the assessment is dropped (it is not implemented), side-stream
prefetch of the kept rows is not done (the time criterion holds within noise without it: a chunk's loads are copies of a
few runs per layer, kv at most 11 runs of 190k tokens), and the gradient-cosine criterion is the two layers above.