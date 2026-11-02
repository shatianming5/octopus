# Bundle RL results (Qwen3.5-4B task-bundle GRPO)

Per run: `eval/` held-out eval from node-67 (`results/step<N>.jsonl` + `.done`, `steps.list`, `run.env`, `logs/`, `summary.txt`)
and `train/` from node-94 (`rl_run.bundle-<run>*.env`, `incidents.log`, `mem.log`, `steps.cache.jsonl` per-episode outcomes,
`steps.txt` per-step metrics). Refresh with `scripts/rl_analysis/sync_results.sh [RUN ...]`.

`summary.txt` = `scripts/eval/summary.py eval --base base-a941eea/eval/results/step0.jsonl` for every run. e1-82b and e2 were
evaluated on tasks-eval70, not tasks-eval-v3, so their M1 vs that base is not a paired comparison of the same tasks.
`steps.txt` = `scripts/rl_analysis/steps.py`; "steps" below = training-step windows it found in the trainer log.

## Runs

| run | dates | code | key flags (on top of OCTOPUS_BUNDLE=1, FORK_OPTIONAL=1, 8 x 4 groups) | steps |
|---|---|---|---|---|
| base-a941eea | 2026-10-14 | a941eea | eval of the start adapter (adapter-v0-4b-text-r64) | - |
| e1-82b | 2026-10-09..10 | 82b657a | MAX_STEPS 200 | 0 |
| e1b | 2026-10-10 | 82b657a | MAX_STEPS 200 | 0 |
| e1c | 2026-10-10 | c969cf6 | MAX_STEPS 200 | 0 |
| e1d | 2026-10-10..11 | 4f2a664, 8d3c7eb (2-host) | MAX_STEPS 200 | 47 |
| e2 | 2026-10-11 | 73b0f67 | PR #53, MAX_STEPS 200 | 0 |
| e2b | 2026-10-11 | 73b0f67 | e2 with 12 rollout servers | 53 |
| e3 | 2026-10-12 | f0375a3 | control, red check, random fork point | 10 |
| e4 | 2026-10-12..13 | 34842ee | non-blocking forks, MAX_STEPS 120 | 41 |
| e5 | 2026-10-13 | 9d788d3 | per-command udocker exec, C_wall | 4 |
| e6 | 2026-10-13 | daca008 | conflicting patches taken back whole | 53 |
| e7 | 2026-10-13..14 | 0adb851, 00fd097 | PR #66, CHECK_COEF 0.5, bundles-v3-valid | 17 |
| e7p | 2026-10-14 | 0bd804b | e7 pilot, logprob recompute skipped | 3 |
| e8 | 2026-10-14 | a457ffa | + TOKEN_PRICE 0.5, FORK_TASK_COEF 1 | 3 |
| e8f | 2026-10-14..15 | c8acbdc | e8 science, 8 FSDP ranks + 15 remote SGLang, EPS 6 | 53 |
| e9 | 2026-10-14..16 | f28550d → e376694 (s21+) | e8f + PR #69 DP balance + ABORT_SOLVABLE 0.5; s25+ 8 FSDP ranks, MB 131072; 10-14 remote SGLang | 53 |
| e10 | - | - | `train/e8f-replay.txt`: e10 reward design replayed on e8f episodes | - |
| e14-read | 2026-10-19..20 | 7c27967 | read_thread (#152/#153) + READ_CONTROL; report: [`results/e14-read/report.md`](../e14-read/report.md) | 40 |

## Held-out (from `eval/summary.txt`; best checkpoint row)

| run | best ckpt | y | fork | M1 paired vs base | M2 main-first | pooled late M2 | M4 wall ratio (fork faster) |
|---|---|---|---|---|---|---|---|
| base-a941eea | - | 0.180 | 24% | - | +0.017±0.067 n=26 | - | 1.14 (38%) |
| e1-82b (eval70) | s10 | 0.147 | 47% | -0.032±0.049 n=120 | - | - | - |
| e2 (eval70; e2 itself ran 16 min, so s10-s53 are e2b's) | s53 | 0.242 | 37% | +0.074±0.048 n=120 | - | - | - |
| e6 | s50 | 0.293 | 83% | +0.113±0.044 n=120 | -0.004±0.039 n=84 | +0.003±0.027 n=237 (s40,50,53) | 1.16 (43%) |
| e7 | s15 | 0.250 | 33% | +0.067±0.043 n=117 | +0.074±0.103 n=31 | +0.067±0.063 n=58 (s0,15) | 1.36 (32%) |
| e8f | s20 | 0.268 | 68% | +0.087±0.049 n=120 | +0.013±0.068 n=58 | -0.056±0.028 n=180 (s40,50,53) | 1.20 (29%) |
| **e9 (final: s53)** | s53 | 0.318 | 83% | +0.137±0.045 n=120 | +0.048±0.054 n=72 | **+0.037±0.027 n=282 (s30,40,53)** | 1.47 (24%) |

Best checkpoint = max M1 among done steps whose M2 95% CI upper bound >= 0 (else max M1). e6's base-adapter step0 under its
own runtime is its `s0` row (y 0.182). e9 s20/s30 are k=4 (240 episodes); each was split at task 30 across two lanes and
merged (dedupe on task_dir+sample). e9 is the first run meeting M1 (every checkpoint) and M2 (pooled late CI > 0); M4 not met.
