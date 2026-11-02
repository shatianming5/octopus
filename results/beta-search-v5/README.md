# beta-search-v5: Search v5 on Cluster Beta (aarch64 GB200)

Search v5 = `rl.branch_credit=search_gate` (#254) on the Alpha Search baseline's settings (search-read-alpha-full-1022),
held Beta job 216338 (1 node, 4× GB200, b2-33-s1-dgx-02-c14). Stack: `scripts/beta/install_env.sh`
(ROOT = /projects/cluster/octopus/beta/octopus); controller `scripts/beta/search_v5.sh`.

## Stack: Alpha baseline vs Beta

| | Alpha (H200, x86, code-d57c294) | Beta (GB200 sm_100, aarch64, code-e403bbf) |
|---|---|---|
| SGLang venv torch / CUDA | 2.8.0+cu128 / 12.8 | 2.8.0+cu129 / 12.9 (no aarch64 cu128 wheel) |
| sglang | 0.5.19 | 0.5.19 |
| sgl_kernel | 0.3.5+cu128 (x86_64) | 0.3.5+cu129 (aarch64) |
| flashinfer-python (SGLang) | 0.7.0 | 0.7.0 |
| triton (SGLang) | 3.4.0 | 3.4.0 (PyTorch index; gdc.py fix applied) |
| transformers (SGLang / trainer) | 5.12.1 / 5.15.0 | 5.12.1 / 5.15.0 |
| trainer torch / CUDA | 2.9.1+cu129 / 12.9 | 2.9.1+cu129 / 12.9 |
| trainer triton | 3.5.1 | 3.5.1 |
| flash-linear-attention / fla-core (GDN) | 0.4.2 / 0.4.2 | 0.4.2 / 0.4.2 |
| flash-attn | 2.8.3+cu128torch2.9 (x86_64) | 2.8.3+cu128torch2.9 (aarch64) |
| peft / tantivy | 0.18.1 / 0.26.2 | 0.18.1 / 0.26.2 |
| AReaL | 2fad2d0 + logprobs, cursor, route-spread patches | same |
| SGLang JIT kernels (activation, rope, qknorm, kvcache, clamp) | nvcc 13.0.88 (/usr/local/cuda), libcudart.so.13 | same nvcc 13.0.88 (ROOT/cuda-13.0 from our CUDA 13.0.3 image), libcudart.so.13 preloaded |
| SGLang venv CUDA libraries | pip nvidia-*-cu12 12.8 | torch 2.8.0 aarch64's bundled 12.9 + pip nvrtc 12.9.86 (sgl_kernel needs it) |
| left out | | torchcodec (no aarch64 wheel; video only), torch-memory-saver (sdist only; AReaL falls back to its mock, offload is off) |

Every other package: Alpha's pin, `--no-deps` (the trainer's nvidia-*-cu12 too: cudnn 9.19.0.56 as on Alpha).

## Checks (10-23)

- Sources byte-identical to Alpha's (diff -r): the cursor overlay's SGLang (cursor + mount + lora_targets +
  lora_abort_release + node-94's files) vs Alpha cursor-d57c294, the venv's sglang package, sgl_kernel's Python files, the
  AReaL overlay and trainer source, fla, `patches/`. Only the compiled Rust extensions differ (aarch64 vs x86_64 `.so`).
  Re-applying the four SGLang patch scripts to a copy of the overlay changes nothing; the overlay's octopus_*.py are shims.
- GB200 kernels, trainer venv (`scripts/beta/gpu_check.py`): flash-attn varlen bf16 vs fp32 SDPA max |diff| 0.0075; fla 0.4.2 gated delta
  rule vs a plain torch fp32 recurrence (T = 64/300/1024): chunk fp32 ≤ 1.4e-4, chunk bf16 ≤ 5.3e-4, fused recurrent fp32
  1.5e-8, bf16 ≤ 3.6e-4 (|o| ≈ 3.5e-3).
- SGLang policy server on GB200 (`scripts/beta/lora_test.sh`, cursor overlay, `--enable-lora rq1-v0`): serves; hot `load_lora_adapter` and
  `unload_lora_adapter` succeed; the zero adapter loaded under a second name decodes the same greedy tokens as rq1-v0
  (first 12: 248068 198 90700 8340 25 271 16 13 220 2972 2014 53983 for "Name the capital of France and explain briefly.").

## Smoke test (216338, trial search-v5-beta-smoke-1023b, code-6c035e3, 10-23 08:49 UTC)

`STOP_STEP=1 VAL_LIMIT=8 bash scripts/beta/search_v5.sh` in the held job (srun --overlap). Its EXTRA_ADD (batch 2, no
zero-variance filter) did not reach the trainer (fixed in 7d01f38), so step 1 ran the production settings: batch 16 x 8,
filter on, offpolicyness 2, 2 rollout servers x 16 groups (EPISODES_PER_SERVER=16 counts groups for search: 128 episodes per server). BM25 service (14,878,084 docs, index copied to /dev/shm in < 3
min) and the Qwen3-32B judge up in 2.5 min; config.yaml `rl.branch_credit: search_gate` and the trainer command line
`rl.branch_credit=search_gate`; episode_settings.json identical to Alpha's (OCTOPUS_CURSOR=1, OCTOPUS_READ_TOOL=1,
qwen3_coder); config.yaml otherwise Alpha's but for the topology (32 concurrent, sglang:d2, fsdp:d1 = Alpha steps 1-5);
init check NLL 0.45.

Step 1, train/infer mismatch (`scripts/beta/step_stats.py`), Alpha search-read-alpha-full-1022 step 1 (same topology):

| | Alpha H200 | Beta GB200 |
|---|---|---|
| train_infer/kl_k1 | 3.54e-4 | 3.33e-4 |
| train_infer/kl_k3 | 3.83e-4 | 4.06e-4 |
| train_infer/logp_abs_diff avg / max | 0.00948 / 0.64 | 0.0097 / 0.639 |
| timeperf rollout / recompute_logp / train_step (s) | 788 / 134 / 551 | 601 / 228 / 529 |

Alpha's 16 steps ranged kl_k1 3.4e-4–7.7e-4, logp_abs_diff avg 0.0095–0.014: no larger mismatch on Blackwell.

search_gate on the smoke's finished episodes (`scripts/beta/gate_dist.py`, 1079 episodes, 835 branches): commit 327
(274 ok, 53 gated for no own search/open), done 225, context 165, abort 101, cut mid-call 17: every branch that is not an
ok commit keeps only min(A_main, 0) — a branch that ends with done (27% here; it answers instead of committing) gets 0 in a
won episode, by the rule's definition. A search/open that was refused or failed counts as searched.

Then (code-7e30be4: the recover dir is `generation_step%08d`) the same trial resumed with `STOP_STEP=2 VAL_LIMIT=8`:
resume from step 1's recover checkpoint, step 2 (kl_k1 3.75e-4, kl_k3 3.19e-4, logp_abs_diff avg 0.0075 / max 0.414;
rollout 1890 s after the resume dropped the in-flight groups), LoRA update to the rollout servers 2.8 s, stop at the step
boundary (225 of 3040 train qids finished), step-2 adapter from the recover checkpoint (`dcp_adapter.py`, 400 tensors, all
changed vs rq1-v0), Val v2 and Val v0 on 8 validation questions each (k=1, judge at 0.7, 3 eval servers with the cursor
overlay): 0 infra failures, y 0.375 / 0.375, `paired.py` written, `SEARCH V5 DONE (step 2)`, cleanup left every GPU at
≤ 4 MiB, no process, /dev/shm empty.

## Production run (216338, trial search-v5-beta-1023b, code-8ee793c = main 03e7f21)

- Queue task `queue/12-search-v5.sh`, started 10-23 11:00:24 UTC (services up 11:02:34). The first attempt
  (`10-search-v5.sh`, trial search-v5-beta-1023, run 10:49:43) died before training: a redeploy re-ran install_env.sh, uv
  reinstalled the sgl-kernel wheel in place and the starting judge got SIGBUS; install_env.sh now builds each venv and
  overlay once (#259).
- Checked after the start: config.yaml `rl.branch_credit: search_gate` (batch 16, offpolicyness 2,
  filter_zero_variance true), the trainer command line `rl.branch_credit=search_gate trial_name=search-v5-beta-1023b`,
  episode_settings.json identical to search-read-alpha-full-1022's; config.yaml otherwise Alpha's but for the topology.
- Time limit: `scripts/beta/search_v5_deadline.sh` on login node b7-39-s1-mgmt-02 (code-3ed6f3d = main 0365614,
  deadline 10-24 04:30 UTC, log ROOT/logs/deadline-search-v5-beta-1023b.log).
- Steps 1-6 (`step_stats.py`), 11:26-12:14 UTC; steps 2-6 average 9.6 min (train-bound, rollout wait ~0); train qids
  finished after steps 1-6: 104, 169, 213, 254, 291, 364 (+65/44/41/37/73, mean ~52 over steps 2-6), so the pass ends
  near step 58-75, ~20:30-23:20 UTC, then Val vK + v0:

| step | kl_k1 | kl_k3 | logp_abs_diff avg / max | recompute / train (s) |
|---|---|---|---|---|
| 1 | 4.87e-4 | 4.35e-4 | 0.0101 / 1.13 | 65 / 304 |
| 2 | 5.61e-4 | 4.78e-4 | 0.0115 / 1.13 | 155 / 680 |
| 3 | 6.68e-4 | 5.67e-4 | 0.0127 / 1.00 | 101 / 451 |
| 4 | 8.82e-4 | 7.45e-4 | 0.0148 / 0.98 | 74 / 347 |
| 5 | 9.16e-4 | 8.70e-4 | 0.0157 / 1.50 | 70 / 319 |
| 6 | 8.25e-4 | 9.45e-4 | 0.0171 / 1.48 | 101 / 449 |

  Alpha's steps 1-5 (H200, outcome credit): logp_abs_diff avg 0.0095 → 0.0131, kl_k1 3.5e-4 → 7.7e-4 (max over 16 steps
  0.0142 / 7.7e-4). Beta's step 1 (same weights) is within Alpha's 16-step range (kl_k1 4.87e-4 vs Alpha's step 1 3.54e-4; the smoke's
  step 1 was 3.33e-4). Steps 4-6 are ~1.0-1.2x Alpha's 16-step maximum (logp_abs_diff avg; kl_k3 1.06-1.34x), but step
  for step Beta's step 6 is ~1.7x Alpha's in logp_abs_diff avg (0.0171 vs 0.0102) and ~1.8x in kl_k1 (8.25e-4 vs 4.51e-4). Watched (report if avg > ~0.03):
  it may come from v5's different updates rather than the hardware.

Later: the checkpoint-curve evaluations (steps 10-50, Val and BCP) are set up in `scripts/beta/tasks/search-ckpt-curve.sh`.

## Result (10-24)

- Training ended at step 69 (10-24 01:53 UTC): all 3040 train qids had a finished episode (the coverage stop; the time-limit
  watcher found train.done and exited). Final adapter `ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v5-beta-1023b/final-step69`
  (step 69 is off the saver's interval: dcp_adapter.py from its recover checkpoint), r 64, alpha 64, 10 targets (down_proj,
  gate_proj, in_proj_qkv, in_proj_z, k_proj, o_proj, out_proj, q_proj, up_proj, v_proj), sha256 8818c6f1...ec7e5.
- Over the 69 steps (`step_stats.py`): logp_abs_diff avg median 0.0145, max 0.0236 (steps 11-30 mean 0.0178), last 5 steps
  ~0.014; kl_k1 median 6.4e-4, max 2.3e-3, last 5 steps 5-6e-4; behave_imp_weight avg 1 throughout. Step time grew from ~10
  to 15-26 min as calls per training episode went 10-14 → 30-49 and the fork rate 0.2 → 0.9 (steps 45-69).
- Val (search_v5.sh's own: fork arm, k=1, the 500 OpenResearcher validation questions, the same GB200 node and judge):
  v69 accuracy 0.467 (499 scored, 1 infra failure) vs v0 0.366 (500): paired +0.100 [0.066, 0.134] (66 up / 16 down / 417
  same); branches per episode 4.87 vs 0.75. (Alpha's v0, H200, k=1: 0.360.) Files: ROOT/search/pilot/search-v5-beta-1023b/
  {val-v69,val-v0}.jsonl, paired.txt. The fork k=2 / BCP / linear evaluations of final-step69: scripts/beta/tasks/
  v5final_tasks.sh and search-ckpt-curve.sh.
