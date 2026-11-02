# Base Qwen3.5-4B on the CooperBench Train bundles (2026-10-21)

Measured on `cooperbench-v1-train-valid` under the 16k shared token budget, the protocol before #220.

The scripts as run, for the record: k=1, `--lora base`, read tool off, dev_eval of main 122f874 with
`OCTOPUS_BUNDLE=1 OCTOPUS_FORK_OPTIONAL=1 GROK_HARBOR_MAX_STEPS=120`. They ran on `cooperbench-v1-train-valid`
(311 bundles), with SGLang from `~/octo-env/eval_serve.sh` on node-136 GPUs 0-4 and node-134 GPUs 1/5/6/7, and
containers on node-134 and node-136. The env files keep `scripts/env.sh` minus UDOCKER_OVERLAY, XAI_API_KEY and GROK_*.
That recorded command line is historical, not the current main launch line: bundle mode is detected,
`OCTOPUS_FORK_OPTIONAL` is gone, the call cap is `OCTOPUS_MAIN_MAX_CALLS`, and eval is
`python -m octopus.eval.dev_eval` (`docs/rename-map-step3.md`).

- `build-tasks.py`, `split.py`, `split-pal.py`: task links and shards.
- The run scripts (`run-train6.sh`, `run-134-rest.sh`, `run-pal-134.sh`, `setup-shard.sh`, `setup-pal136.sh`,
  `retry-136.sh`, `run-eval-136.sh`) used the pre-#223 names above and only ran on 122f874; they are in git
  history before #212 and #229.
- `merge-final.sh` + `analyze-final.py`: one row per (task, sample), the last one; per-repo summary.
- `scan-secrets.sh`: key-like strings in the trajectories, reported as counts.

| repo | bundles | mean reward | both solved | >= 1 solved |
|---|---|---|---|---|
| pallets_jinja | 135 | 0.111 | 4 | 26 |
| samuelcolvin_dirty_equals | 36 | 0.028 | 0 | 2 |
| openai_tiktoken | 45 | 0.156 | 3 | 11 |
| pillow | 30 | 0.100 | 1 | 5 |
| llama_index | 39 | 0.308 | 5 | 19 |
| huggingface_datasets | 26 | 0.173 | 3 | 6 |
| all | 311 | 0.137 | 16 | 69 |

Infra failures 0 in the final rows: 65 node-136 episodes first ended with "a model request failed" (their SGLang
servers were stopped by mistake mid-run) and were rerun; the analysis takes each (task, sample)'s last row. 278 of 311 episodes ended on the 16k token budget. Output: node-134
`/home/user/harbor-udocker/evals/s1b-base-train6-122f874.jsonl`.
