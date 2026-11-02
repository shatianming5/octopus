#!/bin/bash
# search-v9-rest trainer on node-67 (the env staged there): GPUs 4-5, FSDP d2, max_tokens_per_mb 65536, the rollout
# servers in $SERVERS (all on the SGLang cursor overlay), AReaL from the overlay with ba8072c's logprobs + cursor +
# route-spread patches, and the episode settings Alpha's search-v9-1025 trained with (its episode_settings.json:
# OCTOPUS_CURSOR, READ_TOOL, REWIND_TOOL, REWIND_MAX=2; the 10-26 node-138 relaunch had dropped them), plus admission
# sized to the 7 node-134/68 4090D servers: ROUTE_KV 0.5, CAP 16, WAIT 800, TREE_TOKENS 40000 (a 684K pool holds
# max(1, 0.5 x 684K / 40K) = 8 active episode trees, the 10-28 load; 68000, the time-averaged P75 tree footprint,
# gives 5; recalibrate from the route: log lines, docs/search-env.md "Admission"; the 10-28 run had GAP 540 instead,
# gone since), 7 concurrent groups. It refuses to start while the last trainer (train-rest.pid) lives or a trainer GPU is busy.
set -e
for h in user@198.51.100.134 user@10.0.1.68; do  # a mount there would make the push rm -rf hit this host dirs
  ssh -n -o BatchMode=yes $h 'mountpoint -q /home/user/search-v9' && { echo "$h: run dir is a mount"; exit 1; }
done
D=/home/user/search-v9
G=4,5  # the trainer GPUs
P=$(cat $D/logs/train-rest.pid 2>/dev/null || true)
if [ -n "$P" ] && tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q octopus.rl.train; then echo "trainer $P still running"; exit 1; fi
for g in ${G//,/ }; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits || true)
  [ -n "$used" ] && [ "$used" -le 4000 ] || { echo "GPU $g busy (${used:-unreadable} MiB)"; exit 1; }
done
SERVERS=$1
NS=$(awk -F, '{print NF}' <<< "$SERVERS")
NR=/home/user/search-v9/areal-exp/name_resolve/user/rq1-rl/search-v9-rest
if [ -d "$NR" ]; then mv "$NR" "$NR.bak-$(date +%H%M%S)"; fi
PY=$D/areal-venv/.venv/bin/python
SGL=/home/user/octo-env/sglang-venv/bin/python
C=$D/runtime/code-local
F=/home/user/search-v9/areal-exp
for s in ${SERVERS//,/ }; do curl -sf -m 4 http://$s/v1/models >/dev/null || { echo "down $s"; exit 1; }; done
curl -sf -m 4 http://127.0.0.1:32600/health >/dev/null
curl -sf -m 4 http://198.51.100.134:32500/v1/models >/dev/null
export CUDA_VISIBLE_DEVICES=$G NCCL_P2P_DISABLE=1 PYTHONPATH=$D/areal-overlay:$C PYTHONUNBUFFERED=1
# OCTOPUS_ROUTE_TREE_TOKENS needs the overlay's /rl/end_session and /rl/start_session hooks (#351): without the end
# no tree frees its slot, without the start a reused session id is refused
grep -qF 'route.start(session_id)' $D/areal-overlay/areal/experimental/openai/proxy/proxy_rollout_server.py \
  || { echo "$D/areal-overlay: no route hooks: rebuild it with patches/areal-route-spread.patch (#351)"; exit 1; }
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
export OCTOPUS_SGLANG_PY=$SGL
export OCTOPUS_KEEP_WEIGHT_UPDATES=1
export OCTOPUS_SEARCH_URL=http://127.0.0.1:32600
export OCTOPUS_JUDGE_URL=http://198.51.100.134:32500/v1
export OCTOPUS_JUDGE_MODEL=judge OCTOPUS_JUDGE_TIMEOUT=120 OCTOPUS_FORK_DEPTH=0
export OCTOPUS_CURSOR=1 OCTOPUS_READ_TOOL=1 OCTOPUS_REWIND_TOOL=1 OCTOPUS_REWIND_MAX=2 LITELLM_LOCAL_MODEL_COST_MAP=True
export OCTOPUS_ROUTE_SERVERS=$SERVERS
export OCTOPUS_ROLLOUT_SPOOL=$D/rollout-spool/search-v9-rest  # finished groups survive a trainer restart (octopus/rl/spool.py)
# each weight_update_vN goes to node-134 and node-68 local disk (same path) before the servers load it (adapter_push.py)
export OCTOPUS_ADAPTER_PUSH="user@198.51.100.134:22 user@10.0.1.68:22"
export OCTOPUS_ROUTE_CAP=16 OCTOPUS_ROUTE_KV=0.5 OCTOPUS_ROUTE_WAIT=800 OCTOPUS_ROUTE_TREE_TOKENS=40000 OCTOPUS_LIVE_ENV=$D/live.env
: > $D/logs/train-rest.log
setsid "$PY" -m octopus.rl.train --config /home/user/octo-env-data/areal/octopus_rl_grpo.yaml \
  experiment_name=rq1-rl trial_name=search-v9-rest \
  cluster.fileroot=$F cluster.n_gpus_per_node=2 \
  total_train_epochs=4 +total_train_steps=61 \
  actor.path=$D/models/Qwen3.5-4B-text \
  actor.lora_rank=64 actor.lora_alpha=64 \
  rl.lora_keep_versions=40 \
  rl.servers="'$SERVERS'" \
  rl.jobs_dir=$D/jobs-search-v9-rest \
  rl.episodes_per_server=1 \
  rollout.backend=sglang:d$NS actor.backend=fsdp:d2 \
  rollout.max_concurrent_rollouts=7 \
  workflow=octopus.envs.search.rl.SearchEpisodeAgent \
  rl.search_questions=$D/runtime/search/openresearcher-questions.jsonl \
  rl.search_split=$D/runtime/search/rest2-qids.txt \
  actor.mb_spec.max_tokens_per_mb=65536 gconfig.n_samples=8 train_dataset.batch_size=16 \
  saver.freq_steps=10 recover.freq_steps=1 \
  > $D/logs/train-rest.log 2>&1 < /dev/null &
echo $! > $D/logs/train-rest.pid
echo train_pid $(cat $D/logs/train-rest.pid)
