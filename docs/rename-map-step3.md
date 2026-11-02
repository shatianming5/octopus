# Rename map: step 3 of the code-structure cleanup

`benchmarks/` became the package `octopus/`, by layer: `runtime/` (the fork-tree runtime), `serving/` (what the SGLang
overlay and the AReaL proxy import), `rl/`, `envs/swe/` (+ `envs/swe/data/`, the data tools), `envs/search/`, `eval/`,
`tools/` (probes and benchmarks of our own code). Imports go one way only: runtime, serving <- rl <- envs.* <- eval <-
tools (`python scripts/import_graph.py`). Model-visible text is unchanged (`python scripts/check_agent_equivalence.py
71dc20e`: identical). No old name keeps working: a run started before this change is evaluated from its own commit.

## Rebasing onto it

    git mv benchmarks/harbor_udocker_ops/<new file> octopus/envs/swe/data/<new file>   # likewise for the others below
    sed -i -e 's#benchmarks\.harbor_udocker_ops#octopus.envs.swe.data#g' \
           -e 's#benchmarks/harbor_udocker_ops/#octopus/envs/swe/data/#g' \
           -e 's#benchmarks\.harbor_taskset#octopus.envs.swe.taskset#g' \
           -e 's#benchmarks\.udocker_environment#octopus.envs.swe.udocker#g' \
           -e 's#benchmarks\.rl\.dev_eval#octopus.eval.dev_eval#g' \
           -e 's#\bGROK_HARBOR_MAX_STEPS\b#OCTOPUS_MAIN_MAX_CALLS#g' \
           -e 's#\bGROK_BRANCH_MAX_STEPS\b#OCTOPUS_BRANCH_MAX_TURNS#g' \
           -e 's#\bGROK_\(HTTP_WORKERS\|HTTP_ATTEMPTS\|HTTP_TIMEOUT\|CHAT_URL\|MODEL\|CHAT_TEMPLATE_KWARGS\|MAX_TOKENS\|SAMPLING_JSON\)\b#OCTOPUS_\1#g' \
           FILES...

## Modules and files

| old | new |
|---|---|
| `benchmarks/__init__.py` | `octopus/__init__.py` |
| `benchmarks/runtime_core.py` | `octopus/runtime/core.py` |
| `benchmarks/cursor_runtime.py` | `octopus/runtime/tree.py` (the fork tree's runtime: CursorRuntime) |
| `benchmarks/settings.py` | `octopus/runtime/settings.py` |
| `benchmarks/{delta_render,cursor_store,cursor_audit,kv_mount}.py` | `octopus/serving/` (same names) |
| `benchmarks/grok_harbor_agent.py` | `octopus/envs/swe/agent.py` (class `GrokHarborAgent` unchanged) |
| `benchmarks/harbor_runtime.py` | `octopus/envs/swe/prompt.py` (the SWE system prompt, TOOLS, fork/commit/abort texts) |
| `benchmarks/harbor_tools.py` | `octopus/envs/swe/container_tools.py` (still uploaded to `/tmp/harbor_tools.py`) |
| `benchmarks/harbor_taskset.py` | `octopus/envs/swe/taskset.py` |
| `benchmarks/udocker_environment.py`, `udocker_nokill.c` | `octopus/envs/swe/udocker.py`, `udocker_nokill.c` |
| `benchmarks/singularity_environment.py` | `octopus/envs/swe/singularity.py` |
| `benchmarks/rl/harbor_agent.py` | split: `octopus/rl/workflow.py` (AReaL glue every environment shares: `EpisodeAgent` = the old class's `__init__`/`model_env`/`run`, `trial_outcome` (now takes the builder), `GROUP_STEPS`, `plan_group_rewards`, the export patches, spool, `has_variance`, `_run_group`, `HARBOR`, `CODE`, `STOP_GRACE`) and `octopus/envs/swe/rl.py` (`HarborEpisodeAgent(EpisodeAgent)` with `run_episode`, `Slots`, `_ensure_plan_verifier`, `EPISODE_MEM`; registers `GROUP_STEPS["plan"]`) |
| `benchmarks/rl/plan.py` | `octopus/envs/swe/reward.py` (`plan_outcome`, `code_read_rewards`, `plan_group_step`, `DISHONEST_COMMIT`); `plan_group_stats` -> `octopus/rl/group.py` |
| `benchmarks/harbor_udocker_ops/` (everything, `scripts/`, `pool-build/`) | `octopus/envs/swe/data/` |
| `benchmarks/rl/acceptance_rewrite.py` | `octopus/envs/swe/data/acceptance_rewrite.py` |
| `benchmarks/harbor_emit.py`, `harbor_oracles.py` | `octopus/envs/swe/data/emit_tasks.py`, `oracles.py` |
| `benchmarks/search/{agent,service,data,reward,rl,dry_run}.py` | `octopus/envs/search/` (same names) |
| `benchmarks/rl/dev_eval.py` | `octopus/eval/dev_eval.py` |
| `benchmarks/search/dev_eval.py` | `octopus/eval/search_dev_eval.py` |
| `benchmarks/search/eval_bcp.py` | `octopus/eval/bcp.py` |
| `benchmarks/rl/{train,config,run_config,group,actor,prefix_tree,lm_head,seq_tiles,model_hooks,gdn_varlen,mamba2_varlen,dp_balance,spool,proxy,lora_targets,zero_adapter,adapter_push,text_checkpoint}.py` | `octopus/rl/` (same names) |
| `benchmarks/rl/rq1_grpo.yaml` | `octopus/rl/grpo.yaml` (values unchanged: `lora_name: rq1`, experiment `rq1-rl`) |
| `benchmarks/rl/{tree_check,mount_check,lora_parity,replay_bench,act_gpu_bench,step_stats,rollout_load}.py` | `octopus/tools/` (same names) |
| `benchmarks/{kv_read_probe.py,kv_read_probe_set.jsonl,mount_load.py,delta_render_bench.py}` | `octopus/tools/` (same names) |
| `benchmarks/tree_sft/` | `octopus/tools/tree_sft/` (kept: `tools/kv_read_probe.py` imports its `encode._for_template`) |

## Entry points

| old | new |
|---|---|
| `python -m benchmarks.rl.train --config benchmarks/rl/rq1_grpo.yaml` | `python -m octopus.rl.train --config octopus/rl/grpo.yaml` |
| `python -m benchmarks.rl.config OUT` (`rl_run.sh settings`) | `python -m octopus.rl.config OUT` |
| `python -m benchmarks.rl.dev_eval` | `python -m octopus.eval.dev_eval` |
| `python -m benchmarks.search.dev_eval` | `python -m octopus.eval.search_dev_eval` |
| `python -m benchmarks.search.eval_bcp` | `python -m octopus.eval.bcp` |
| `python -m benchmarks.search.{agent,service,data,dry_run}` | `python -m octopus.envs.search.{agent,service,data,dry_run}` |
| `python -m benchmarks.harbor_udocker_ops.{build_bundles,statement_coverage,cooperbench,...}` | `python -m octopus.envs.swe.data.{...}` |
| `python -m benchmarks.rl.{text_checkpoint,zero_adapter}` | `python -m octopus.rl.{text_checkpoint,zero_adapter}` |
| `python -m benchmarks.rl.{tree_check,mount_check,lora_parity,replay_bench,act_gpu_bench,step_stats,rollout_load}` | `python -m octopus.tools.{...}` |
| `python -m benchmarks.{kv_read_probe,mount_load,delta_render_bench}`, `benchmarks.tree_sft.*` | `python -m octopus.tools.{...}`, `octopus.tools.tree_sft.*` |
| config.workflow `benchmarks.rl.harbor_agent.HarborEpisodeAgent` | `octopus.envs.swe.rl.HarborEpisodeAgent` |
| config.workflow `benchmarks.search.rl.SearchEpisodeAgent` | `octopus.envs.search.rl.SearchEpisodeAgent` |
| dynamic_filter_fn `benchmarks.rl.harbor_agent.has_variance` | `octopus.rl.workflow.has_variance` |
| Harbor `-a benchmarks.grok_harbor_agent:GrokHarborAgent` | `-a octopus.envs.swe.agent:GrokHarborAgent` |
| Harbor `-e benchmarks.udocker_environment:UdockerEnvironment` | `-e octopus.envs.swe.udocker:UdockerEnvironment` |
| Harbor `-e benchmarks.singularity_environment:SingularityEnvironment` | `-e octopus.envs.swe.singularity:SingularityEnvironment` |
| SGLang overlay shims `from benchmarks import {delta_render,cursor_store,kv_mount}` | `from octopus.serving import ...` |

## Environment variables

| old | new | why |
|---|---|---|
| `GROK_HARBOR_MAX_STEPS` | `OCTOPUS_MAIN_MAX_CALLS` | main's tool-call cap (and its control's), both environments |
| `GROK_BRANCH_MAX_STEPS` | `OCTOPUS_BRANCH_MAX_TURNS` | a branch's turn cap (it counts replies) |
| `GROK_CHAT_URL`, `GROK_MODEL` | `OCTOPUS_CHAT_URL`, `OCTOPUS_MODEL` | the policy endpoint and request model |
| `GROK_CHAT_TEMPLATE_KWARGS`, `GROK_MAX_TOKENS`, `GROK_SAMPLING_JSON` | `OCTOPUS_CHAT_TEMPLATE_KWARGS`, `OCTOPUS_MAX_TOKENS`, `OCTOPUS_SAMPLING_JSON` | per-request settings |
| `GROK_HTTP_ATTEMPTS`, `GROK_HTTP_TIMEOUT`, `GROK_HTTP_WORKERS` | `OCTOPUS_HTTP_ATTEMPTS`, `OCTOPUS_HTTP_TIMEOUT`, `OCTOPUS_HTTP_WORKERS` | the chat client (`OCTOPUS_HTTP_WORKERS` stays a host setting, not an episode one) |
| `XAI_API_KEY` (an episode's policy-proxy key) | `OCTOPUS_API_KEY` | AReaL hands the key to `EpisodeAgent.run` as an argument and our agent reads it itself; neither AReaL nor Harbor reads the variable. A credential: never in `episode_settings.json` |
| `GROK_API_KEY` (a test's credential example) | `OCTOPUS_API_KEY` | |
| `GROK_MODELS_BASE_URL`, `XAI_API_KEY`, `REWRITE_MODEL` of `envs/swe/data/statement_coverage.py` and `acceptance_rewrite.py` | unchanged | they address the real xAI Grok gateway (the Grok judge of the data tools) |

`episode_settings.json` and `dev_eval --run` read `OCTOPUS_*` only (`rl.config.episode_key`).

## After deploying a commit with this change

- Host env files (`~/octo-env/*.env`, `$HARBOR/scripts/env.sh`, NVwulf env): rename the `GROK_*` names above (e.g.
  `GROK_HTTP_WORKERS=256` in `$HARBOR/scripts/env.sh`) and any `OCTOPUS_HARBOR_ENV=benchmarks.*` value.
- Every AReaL overlay built from `patches/areal-cursor.patch` (`$CURSOR/areal`, the eval lanes' on node-67, NVwulf's
  `$ROOT/areal`) and every SGLang overlay (`$CURSOR/sglang`, from a fresh SGLang copy with `patches/apply_sglang_cursor.py`
  and `apply_sglang_mount.py`; docs/cursor-protocol.md "Deployment"): rebuild them, the old ones import `benchmarks.*`.
- `dev_eval --run` and training refuse pre-rename names: a run's `episode_settings.json` with `GROK_*` keys, or an
  environment that still exports a `GROK_*` name other than `GROK_MODELS_BASE_URL`.
- Data tools' deployed copy (`$N/code`, `PYTHONPATH`): the `octopus/` tree in place of `benchmarks/`.
- `scripts/eval/cooperbench_base/README.md` records that run's pre-rename command line (it ran the pinned `code-122f874`);
  its run scripts with the old names were removed in #212 and #229.
