# Baselines

Code and reproduction scripts for the external methods we compare against. No external repository is vendored: each
method is reproduced as a condition of our own probe, on identical tokens, and this file says what each condition
does and does not reproduce of its paper.

## kv_read: Route A (cross-architecture x external methods on the KV-mount probe)

Extends the attention-only KV-mount probe ([docs/kv-read-probe.md](../docs/kv-read-probe.md), Run 1/2 on
Qwen3.5-4B) to more conditions and to three architecture families. Same probe set
(`octopus/tools/kv_read_probe_set.jsonl`: 366 mount points, 99 trees, 1,122 facts), same answer slot, metrics,
groups (all / peer ahead / peer behind) and tree-clustered bootstrap. `octopus/tools/kv_read_probe.py` is unchanged
and still reproduces Run 1/2; `baselines/kv_read/route_a.py` imports its helpers. CPU tests:
`tests/test_route_a.py`.

### Conditions

At each mount point sibling B reads sibling A's last <= 4 exchanges (the "mount"), then answers retrieval probes and
regenerates its own next reply (leakage probes).

| condition | what B gets | method | what is reproduced / not |
|---|---|---|---|
| `none` | nothing | lower bound | — |
| `text` | A's mounted turns (chat markers stripped) as one user message | AutoGen ([2308.08155](https://arxiv.org/abs/2308.08155)), CAMEL ([2303.17760](https://arxiv.org/abs/2303.17760)): agents exchange natural-language messages | the message passing: verbatim turns in B's context, every layer sees them. Not their frameworks, role prompts or conversation programs |
| `summary` | A summarizes its context (greedy, <= 192 tokens, on A's own cache after a summary-request user message); the summary goes to B as a user message | Context-Folding ([2510.11967](https://arxiv.org/abs/2510.11967)): a branch folds into a concise summary returned to the caller; ReSum ([2509.13313](https://arxiv.org/abs/2509.13313)): periodic context summarization | zero-shot summarize-then-read with the same model; tokens generated / inserted and wall ms are recorded. Not their RL training (FoldGRPO, ReSum-GRPO) or their summary prompts/tools |
| `replay` | A's mounted tokens spliced into B's context, recomputed by every layer from B's state | CacheBlend ([2405.16444](https://arxiv.org/abs/2405.16444)), EPIC ([2410.15332](https://arxiv.org/abs/2410.15332)): reuse another context's chunk at a new position | full recomputation of the spliced chunk: the quality these methods approximate. Not their partial recomputation (CacheBlend's selected tokens, EPIC's LegoLink) |
| `K3` | A's K/V rows appended in every attention layer, keys re-rotated by `p_B - a0` to B's next positions; B's recurrent state untouched | Hogwild! ([2504.06261](https://arxiv.org/abs/2504.06261)): shared cache with RoPE re-rotation; KVCOMM ([2510.12872](https://arxiv.org/abs/2510.12872)): cross-context KV reuse with offset alignment | the position shift (RoPE composes: `R(p + d) = R(d) R(p)`; identity under NoPE). Not KVCOMM's anchor-based offset correction, nor Hogwild!'s concurrent workers and prompts |
| `kvcomm_sel` | K3's re-rotated rows, only on the `--mount-layers` subset | KVComm ([2510.03346](https://arxiv.org/abs/2510.03346)): share the K/V of selected layers | the layer-subset share. Not KVComm's layer selection (attention-importance score with a Gaussian prior): the subset is given. With `--mount-layers all` it is K3, whose result is reused (`same_as`) |
| `K1` | A's K/V rows on the `--mount-layers` layers at A's own positions | ours (Run 1/2) | — |
| `K2` | as K1; B continues at `max(p_B, a0 + M)` | ours: the SGLang mount (docs/cursor-protocol.md "Mount") | — |

On granite (NoPE attention, Mamba-2 layers that ignore positions) the position rules K1, K2, K3 and
`kvcomm_sel` differ only in the layers they mount: mounts on the same layers are one computation: the first of them in
the order K1, K2, K3, kvcomm_sel runs and the others are recorded `same_as` it (the report header names a condition
that is `same_as` the same one at every mount point; on RoPE models K2 is `same_as` K1 only where the peer is behind). Equal K1 / K2 / K3 numbers on granite are therefore by
construction, not a finding.

Copy cost per mount condition is recorded as a formula, `layers x M x kv heads x head dim x 2 (K, V) x dtype bytes`
(`copy_bytes`), not measured.

### Models (configs checked on Hugging Face, 2026-10-31)

| model | layers | attention layers (0-based) | KV heads x head dim | positions in attention | post-head |
|---|---|---|---|---|---|
| Qwen3.5-4B-text (+ e11 adapter) | 32 (24 GDN) | 3, 7, ..., 31 (8) | 4 x 256 | interleaved M-RoPE [11, 11, 10] on 64 of 256 dims, theta 1e7 | — |
| Qwen3.5-9B-text | 32 (24 GDN) | 3, 7, ..., 31 (8) | 4 x 256 | same | — |
| granite-4.0-h-350m | 32 (28 Mamba-2) | 10, 13, 17, 27 | 4 x 64 | none (NoPE) | logits / 3 |
| granite-4.0-h-micro | 40 (36 Mamba-2) | 5, 15, 25, 35 | 8 x 64 | none (NoPE) | logits / 8 |
| granite-4.0-h-tiny | 40 (36 Mamba-2) | 5, 15, 25, 35 | 4 x 128 | none (NoPE) | logits / 6; MoE MLP, 64 experts, top-6 |
| Qwen3-4B | 36 dense | all | 8 x 128 | RoPE, theta 1e6 | — |
| Qwen3-8B | 36 dense | all | 8 x 128 | RoPE, theta 1e6 | — |

`--mount-layers 16-24` is inclusive: 9 of Qwen3's 36 layers. Qwen3 runs with `enable_thinking=False` (an empty think
block in each generation prompt), as all models here.

### How it runs

- Rendering is template-agnostic: B's context, generation prompt and recorded reply are `materialize`d as in Run 1/2
  (each a prefix of the longer rendering). A message a condition appends (text, summary, the probe question, the
  summary request after A's context) is rendered after a stub of the context (its system message + one user query)
  and taken as the suffix. Not after the whole context: Qwen3.5's template re-renders the assistant turns before a
  later user query (drops their empty think blocks), so the whole-context render is never a prefix, while a cache
  keeps B's tokens as they were. On Qwen3.5 B's context, generation prompt and the probe question are
  token-identical to Run 2's; the `text` message is one token shorter than Run 2's hand-written one (the template
  trims message content, dropping the mount's final newline), so the Qwen3.5-4B backfill runs `none,summary` and
  keeps Run 2's `text` rows. A mount
  point whose template breaks the prefix property is written as `{"skipped": ...}` and counted by `report`.
- Stop tokens: `octopus.serving.delta_render.terminator` (after a plain reply and after a tool call) and the eos ids.
- Families come from `octopus.rl.model_hooks`: attention layers from the config's `layer_types` (transformers maps
  granite's `mamba` / `attention` to `linear_attention` / `full_attention`; a config without them is pure attention),
  the head's post-ops from `octopus.rl.lm_head._post_head_ops`, RoPE from the family's `apply_rotary_pos_emb` and the
  model's `rotary_emb` (absent under NoPE: rotation is the identity).
- Recurrent state: with the locked transformers (5.20.2, `uv sync --extra tiny-lora`) both hybrids continue a
  multi-token chunk from the cached state (GDN through `patch_model`'s fla kernels; granite's Mamba-2 through the
  hybrid cache: its conv context is concatenated before the chunk and the SSM state is the chunked scan's initial
  state). Checked on CPU against one plain forward (max abs diff 7e-8 on a random granite, test
  `test_multi_token_continuation_equals_plain_forward`), and per run by S1. The models build their hybrid
  `DynamicCache(config)` themselves; attention rows are `cache.layers[i].keys / .values` in this version.
- A mount on a layer subset leaves the other attention layers with fewer cache rows. On the GPU flash attention
  (`flash_attention_2`, as Run 1/2) gets no mask and aligns the causal mask bottom-right per layer, which is correct
  for any row count; the CPU tests use an sdpa wrapper with the same alignment.
- Order per mount point: A's prefill, its K/V sliced, its summary decoded, its cache freed; then B (peak memory).
- Sanity checks in each output: S1 / S2, cache continuation vs one plain forward for `none` / `text` (first
  `--sanity` mount points per shard); S4, A's mount continued at shifted positions vs at its own and re-rotated (same
  chunking, first mount point per shard; exactly 0 under NoPE); S3 asserted (mount tokens decode to A's turns, cue +
  answer in the mount, answer not in B's context).
- `report` takes several results files of one model and merges rows by id (the first file holding a condition wins),
  prints each file's mount layers, renames a layer-subset run's K1 / K2 / kvcomm_sel to `K2@16-24` etc. (so the
  all-layer ones survive next to them), prints only the conditions present, summary length and latency, copy MB, and the paired EM differences: the Run 1/2
  decision pairs (on the all-layer K1 / K2 / K3), `K2 - summary`, `kvcomm_sel - K2`, `kvcomm_sel - K3`, per subset
  `X@L - X`, `X@L - K3` and `kvcomm_sel@L - K2@L`, each `K - 0.8 text`, and the positive-control
  gate `text - none`. `same_tool` parses Qwen3.5's `<function=NAME>` and the JSON `<tool_call>{"name": ...}` of
  Qwen3 / granite, and a reply without a tool call never matches (Run 1/2 counted None == None as a match, so their
  same-tool values differ when re-reported here).

### Reproduce

GPU hosts, the locked env plus `flash-attn` and `flash-linear-attention` (as Run 1/2). `ROOT_MAP` points the probe
set's `agent_dir`s (`../data/e11/...`, `/mnt/disk4/user/harbor-udocker/rl/jobs-cursor-remote-10180140/...`) at
local copies. Each command shards over the GPUs, resumes from its shard files (rerun with the same GPU count) and
writes `$OUT/results.jsonl` and `$OUT/report.md`.

```bash
S=baselines/kv_read/run_route_a.sh
RM="../data/e11=/data/e11 /mnt/disk4/user=/data/user"
# Qwen3.5-4B: summary backfill, reported with run 2's rows (node-94: /home/user/kv-read-probe/run2/)
MODEL=$M/Qwen3.5-4B-text CONDITIONS=none,summary ROOT_MAP=$RM OUT=runs/q35-4b-base \
  REPORT_WITH=$RUN2/results-base.jsonl bash $S
MODEL=$M/Qwen3.5-4B-text ADAPTER=$M/bundle-e11-step40 CONDITIONS=none,summary ROOT_MAP=$RM OUT=runs/q35-4b-e11 \
  REPORT_WITH=$RUN2/results-e11.jsonl bash $S
# Qwen3.5-9B: text-only checkpoint first
python -m octopus.rl.text_checkpoint --src $M/Qwen3.5-9B --out $M/Qwen3.5-9B-text
MODEL=$M/Qwen3.5-9B-text ROOT_MAP=$RM OUT=runs/q35-9b bash $S
# granite-4.0-h: every attention layer (NoPE: K1 runs once; K2, K3, kvcomm_sel are recorded same_as K1)
for g in 350m micro tiny; do MODEL=$M/granite-4.0-h-$g ROOT_MAP=$RM OUT=runs/granite-$g bash $S; done
# Qwen3 dense: all layers, then layers 16-24 for the mount conditions; one report pairs X@16-24 with X
for q in 4B 8B; do
  MODEL=$M/Qwen3-$q ROOT_MAP=$RM OUT=runs/qwen3-$q-all bash $S
  MODEL=$M/Qwen3-$q ROOT_MAP=$RM MOUNT_LAYERS=16-24 CONDITIONS=K1,K2,kvcomm_sel OUT=runs/qwen3-$q-16-24 \
    REPORT_WITH=runs/qwen3-$q-all/results.jsonl bash $S
done
```

Not run yet (no GPU results). Before the full runs, a smoke run (`--limit 8` per model through
`python -m baselines.kv_read.route_a run ...`) should show S1 / S2 <= 0.02 nats with argmax >= 0.98, S4 <= 0.02
(0 for granite), no skipped mount points on these three templates, and the Qwen3.5-4B `none` rows matching run 2's.
