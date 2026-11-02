#!/bin/bash
# newhost.sh CMD: resume search-v9-rest on one host that runs everything (trainer, judge, rollout servers, search
# service), all on localhost, from the recover checkpoint in FILEROOT. Settings: newhost.env beside this script, or the
# file NEWHOST_ENV names (newhost-134.env: node-134); the code dir is the checkout this script is in. README "New host
# (10-29)" and "node-134 (10-30)".
#   spool      download the spool archived after step 43 (SPOOL_REPO's spool/) into $RUN/rollout-spool/search-v9-rest;
#              refuses once a trainer has used that dir (its resume/ exists): the groups would train twice
#   preflight  check (read only) every path, the checkpoint (every DCP shard file whole) and its adapter, the spool, the
#              input files' md5, the code (#338, the overlay's train-rpc-timeout patch), free GPUs; lists what listens on
#              all interfaces
#   serve      search service (openresearcher BM25, :32600), judge (JUDGE_GPUS, :32500) and the rollout servers
#              (ROLLOUTS) with rq1-v(N+1), N the checkpoint's global_step; each one that answers or is still loading
#              (its pid file's process, command line checked) is kept, so a rerun starts only what is down; waits
#              until all answer. SHARED=1 (env file): a server starts at BUDGET = its card's free memory - 6.5 GiB.
#              A ROLLOUTS entry SSH:GPU:PORT (SSH user@addr) runs on that host over ssh: the same paths there (ROOT,
#              ROOT/code-<sha>, SGLANG_PY, MODEL_VL), CUDA R_CU (default CU), TMPDIR R_TMPDIR (if set), the adapter pushed
#              first, a tunnel to 127.0.0.1:PORT. TUNNELS entries PORT:SSH:ADDR:APORT start first (127.0.0.1:PORT ->
#              ADDR:APORT as host SSH sees it, reconnected when ssh exits): a search service or judge elsewhere; empty
#              SEARCH_ROOT: no local search. Waits for JUDGE_URL too
#   train      the trainer on TRAIN_GPUS (refuses while a trainer or the watcher runs, or, SHARED=1, without TRAIN_MIB
#              + 6 GiB free on each, or, RANK_GIB set, with less MemAvailable than RANK_GIB per trainer GPU + SERVER_GIB
#              per rollout server; preflight checks it too), then pass_stop.py, which TERMs
#              the trainer once the dataloader's epoch is trained (one pass), and the watcher: once the trainer exits
#              (that TERM, +total_train_steps=61, a crash) it stops the AReaL workers, then, after any running
#              search_dev_eval, the search service, judge, rollout servers (each pid checked against its command line;
#              a remote one over ssh, then its tunnel), the TUNNELS and the RELEASE lines. Adapter pushes: ADAPTER_PUSH
#              and every remote ROLLOUTS host
#   status     our processes, GPU memory, the last trainer, watcher and pass_stop.py log lines
# To restart the trainer: TERM the watcher (watch.pid) first, then the trainer's process group (train-rest.pid), then
# `train` again; the servers stay up. pass_stop.py exits with its trainer; logs/pass-stop.json carries the pass to
# the next one (delete it only for a new pass).
set -uo pipefail
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); CODE=$(cd "$K/../../.." && pwd)
# absolute, so the watcher (started from $CODE) finds the same file
[ -n "${NEWHOST_ENV:-}" ] && export NEWHOST_ENV=$(realpath "$NEWHOST_ENV")
source "${NEWHOST_ENV:-$K/newhost.env}"
U=$(id -un); CK=$FILEROOT/checkpoints/$U/rq1-rl/search-v9-rest; L=$RUN/logs; PIDF=$L/train-rest.pid
mkdir -p "$L"; cd "$CODE"
say() { echo "$(date '+%F %T %Z') $*"; }
die() { say "$*"; exit 1; }
n() { awk -F, '{print NF}' <<< "$1"; }
step() { sed -n 's/.*"global_step": *\([0-9]*\).*/\1/p' "$CK/LATEST" 2>/dev/null; }
servers() { for pg in $ROLLOUTS; do printf '127.0.0.1:%s,' "${pg##*:}"; done | sed 's/,$//'; }
# a ROLLOUTS entry GPU:PORT here or SSH:GPU:PORT on host SSH: sets g, p and h (SSH, empty here)
rp() { p=${1##*:}; g=${1%:*}; h=; [[ $g == *:* ]] && { h=${g%:*}; g=${g##*:}; }; }
# on SSH CMD: the shell command CMD on host SSH (empty: here), with say, alive and pgkill
on() { local h=$1; shift; if [ -n "$h" ]; then ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" "$(declare -f say alive pgkill); $*"
  else eval "$*"; fi; }
alive() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
pgkill() { alive "$1" "$2" || { say "skip ${1:-none} ($2)"; return; }
  kill -TERM -- "-$1" 2>/dev/null || kill "$1"; say "TERM $1 ($2)"
  for _ in $(seq 30); do kill -0 "$1" 2>/dev/null || return; sleep 2; done; kill -KILL -- "-$1" 2>/dev/null; say "KILL $1"; }
mib() { on "${3:-}" nvidia-smi -i "$1" --query-gpu="memory.$2" --format=csv,noheader,nounits; }
# GPUS NEED [SSH]: each card (of host SSH) has <= 4000 MiB used (a host of our own) or, SHARED=1, NEED MiB + 6 GiB free
gpus_free() { local g f used; for g in ${1//,/ }; do
  if [ -n "${SHARED:-}" ]; then f=$(mib "$g" free "${3:-}"); [ -n "$f" ] && [ "$f" -ge $(($2 + 6144)) ] \
    || { echo "${3:+$3 }GPU $g: ${f:-unreadable} MiB free, needs $2 + 6144"; return 1; }
  else used=$(mib "$g" used "${3:-}"); [ -n "$used" ] && [ "$used" -le 4000 ] \
    || { echo "${3:+$3 }GPU $g busy (${used:-unreadable} MiB)"; return 1; }
  fi; done; }
hosts() { for pg in $ROLLOUTS; do rp "$pg"; echo "$h"; done | sort -u; }  # the remote rollout hosts
# PORT SSH DEST [loop]: an ssh tunnel 127.0.0.1:PORT -> DEST (as host SSH sees it), kept while its pid file's process
# lives; loop: reconnected 5 s after ssh exits (a TUNNELS one: search or judge lost stalls every episode), in one
# process group, so pgkill stops both
tunnel() { alive "$(cat "$L/tunnel-$1.pid" 2>/dev/null)" "ssh -N .*-L 127.0.0.1:$1:" && return
  local t=(ssh -N -S none -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "127.0.0.1:$1:$3" "$2")
  if [ -n "${4:-}" ]; then setsid bash -c "while :; do ${t[*]}; sleep 5; done" > "$L/tunnel-$1.log" 2>&1 < /dev/null &
  else setsid "${t[@]}" > "$L/tunnel-$1.log" 2>&1 < /dev/null & fi
  echo $! > "$L/tunnel-$1.pid"; }
# host RAM (README "Trainer on node-68 (10-30)"): RANK_GIB per trainer rank (its pinned slab follows its largest micro-batch),
# SERVER_GIB per rollout server (its AReaL rollout worker and proxy run on the trainer host); RANK_GIB unset: no check
ram() { [ -z "${RANK_GIB:-}" ] && return; local d s need avail; d=$(n "$TRAIN_GPUS"); s=$(wc -w <<< "$ROLLOUTS")
  need=$((d * RANK_GIB + s * SERVER_GIB)); avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  [ "$avail" -ge "$need" ] || { echo "host RAM: $avail GiB available, d$d + $s servers need $need"; return 1; }; }

SP=$RUN/rollout-spool/search-v9-rest
case ${1:-} in
spool)
  [ -d "$SP/resume" ] && die "$SP/resume exists: a trainer already recovered this spool"
  T=$RUN/rollout-spool/hf  # the repo's spool/{ready,taken,trained}, then moved under their names into $SP
  HF_HUB_DISABLE_XET=1 "$AREAL_PY" -c 'import sys; from huggingface_hub import snapshot_download as d
d(sys.argv[1], local_dir=sys.argv[2], allow_patterns=["spool/ready/*", "spool/taken/*", "spool/trained/*"])' "$SPOOL_REPO" "$T" \
    || die "spool download failed"
  for d in ready taken trained; do [ -d "$T/spool/$d" ] || continue
    mkdir -p "$SP/$d" && mv "$T/spool/$d"/*.pt "$SP/$d/" || die "spool: moving $d failed"; done
  say "spool: $(for d in ready taken trained; do printf '%s %s ' $d "$(ls "$SP/$d" 2>/dev/null | wc -l)"; done)($(du -sh "$SP" | cut -f1))"
  ;;
preflight)
  bad=0; ok() { echo "ok   $*"; }; no() { echo "FAIL $*"; bad=$((bad + 1)); }
  N=$(step); V=$((N + 1)); G=$CK/checkpoint_generations/$(printf 'generation_step%08d' "${N:-0}")
  for f in "$AREAL_PY" "$SGLANG_PY" "$SEARCH_PY" "$CU/bin/nvcc" "$HARBOR/scripts/env.sh" \
      "$CURSOR/areal/areal/utils/checkpoint_pointer.py" "$CURSOR/sglang/sglang/srt" "$MODEL_VL/config.json" \
      "$MODEL_TEXT/config.json" ${JUDGE_GPUS:+"$JUDGE_MODEL/config.json"} ${SEARCH_ROOT:+"$TOKENIZER" "$SEARCH_ROOT/search-scratch/openresearcher/docs.arrow"} \
      "$CK/LATEST" "$CK/actor/initial_lora/adapter_model.safetensors" "$G/manifest/step_info.json" \
      "$CK/default/weight_update_v$V/adapter_config.json"; do [ -e "$f" ] && ok "$f" || no "missing $f"; done
  # torch DCP: the shards load at any FSDP degree (README "node-134 (10-30)", d4 -> d2); every file .metadata names, whole
  out=$("$AREAL_PY" -c 'import os, sys; from torch.distributed.checkpoint import FileSystemReader
p, end = sys.argv[1], {}
for i in FileSystemReader(p).read_metadata().storage_data.values():
    end[i.relative_path] = max(end.get(i.relative_path, 0), i.offset + i.length)
bad = [f for f, e in end.items() if not os.path.isfile(f"{p}/{f}") or os.path.getsize(f"{p}/{f}") < e]
print(len(end), *bad[:3]); sys.exit(bool(bad) or not end)' "$G/payloads/default" 2>&1 | tail -n 1) \
    && ok "$out DCP shard files, loaded at d$(n "$TRAIN_GPUS")" || no "$G/payloads/default: shard files missing or short: $out"
  c=$(for d in ready taken trained; do printf '%s ' "$(ls "$SP/$d"/*.pt 2>/dev/null | wc -l)"; done)
  [ "$c" = "9 1 0 " ] && ok "spool ready/taken/trained $c" || no "spool ready/taken/trained: $c(want 9 1 0: newhost.sh spool)"
  grep -q 'http_timeout=86400.0' "$CURSOR/areal/areal/infra/controller/train_controller.py" 2>/dev/null \
    && ok "overlay has areal-train-rpc-timeout.patch" || no "$CURSOR/areal lacks patches/areal-train-rpc-timeout.patch"
  grep -qF 'route.start(session_id)' "$CURSOR/areal/areal/experimental/openai/proxy/proxy_rollout_server.py" 2>/dev/null \
    && ok "overlay starts and ends route trees (#351)" || no "$CURSOR/areal lacks #351's patches/areal-route-spread.patch (ROUTE_TREE_TOKENS)"
  grep -q REUSE_TRAIN_LOGP "$CURSOR/areal/areal/api/cli_args.py" 2>/dev/null \
    && ok "overlay AReaL has prox_logp_method reuse_train_logp (#349)" || no "$CURSOR/areal lacks REUSE_TRAIN_LOGP (grpo.yaml's prox_logp_method)"
  grep -qF '# octopus-top-k-all' "$CURSOR/sglang/sglang/srt/sampling/sampling_params.py" 2>/dev/null \
    && ok "overlay SGLang reads top_k >= vocab as all (#352)" || no "$CURSOR/sglang lacks patches/apply_sglang_top_k.py (#352)"
  for p in "$SPLIT abb8f1a678c7cf10fda3916785c763ec" "$QUESTIONS dcadeb557f32d4dcf75f195d1c0ce402"; do
    set -- $p; [ "$(md5sum < "$1" 2>/dev/null | cut -d' ' -f1)" = "$2" ] && ok "md5 $1" || no "$1 is not node-138's (md5 $2)"; done
  [ "$CHUNK_MAJOR" != 1 ] || { grep -q OCTOPUS_CHUNK_MAJOR octopus/rl/config.py \
    && grep -qF 'not with_tree and not mounts' octopus/rl/prefix_tree.py; } && ok "code $(git rev-parse --short HEAD 2>/dev/null)" \
    || no "CHUNK_MAJOR=1 but the code lacks #338 or #354 (every micro-batch chunk-major)"
  "$SEARCH_PY" -c 'import harbor' 2>/dev/null && ok "harbor importable" || no "$SEARCH_PY cannot import harbor"
  for h in $(hosts); do out=$(on "$h" "ls -d $SGLANG_PY $MODEL_VL/config.json $CURSOR/sglang/sglang/srt $ROOT/${CODE##*/}/octopus \
    ${R_CU:-$CU}/bin/nvcc 2>&1 > /dev/null") && ok "$h: SGLang, model, overlay, code, nvcc" || no "$h: $out"; done
  out=$(gpus_free "$TRAIN_GPUS" "${TRAIN_MIB:-}" && for pg in $ROLLOUTS; do rp "$pg"; gpus_free "$g" "${ROLLOUT_MIB:-}" "$h" || exit; done \
    && SHARED= gpus_free "$JUDGE_GPUS") && ok "GPUs free" || no "$out"  # the judge: ours alone (MEM)
  out=$(ram) && ok "host RAM${RANK_GIB:+ (RANK_GIB $RANK_GIB, SERVER_GIB $SERVER_GIB)}" || no "$out"
  echo "info listening on all interfaces now (a public host: AReaL's RPC and proxy servers bind them too; the proxy"
  echo "     needs $RUN/admin_api_key; keep the cloud firewall closed except ssh):"
  ss -ltnH 2>/dev/null | awk '$4 ~ /^(0\.0\.0\.0|\[::\]|\*):/ {print "     " $4}'
  [ "$bad" = 0 ] && echo "preflight passed (checkpoint step $N, rollouts start on rq1-v$V)" || { echo "preflight FAILED ($bad)"; exit 1; }
  ;;
serve)
  N=$(step); [ -n "$N" ] || die "no $CK/LATEST"; V=$((N + 1))
  # kept: what answers, or what our pid file names while its command line matches (still loading)
  up() { curl -sf -m 4 "http://$1" > /dev/null || on "${4:-}" "alive \"\$(cat $L/$2 2>/dev/null)\" '$3'"; }
  for t in ${TUNNELS:-}; do a=${t#*:}; tunnel "${t%%:*}" "${a%%:*}" "${a#*:}" loop; done
  if [ -n "$SEARCH_ROOT" ] && ! up 127.0.0.1:32600/health search.pid octopus.envs.search.service; then
    setsid "$AREAL_PY" -m octopus.envs.search.service serve --corpus openresearcher --root "$SEARCH_ROOT" --backend bm25 \
      --tokenizer "$TOKENIZER" --port 32600 --workers 8 > "$L/search.log" 2>&1 < /dev/null &
    echo $! > "$L/search.pid"
  fi
  if [ -n "$JUDGE_GPUS" ] && ! up 127.0.0.1:32500/v1/models judge.pid 'sglang.launch_server.*--port 32500 '; then
    TP=$(n "$JUDGE_GPUS") MEM=$JUDGE_MEM MAMBA_SLOTS=$JUDGE_MAMBA_SLOTS SGLANG_PY=$SGLANG_PY CU=$CU \
      bash "$K/serve_judge.sh" "$L" "$JUDGE_MODEL" "$JUDGE_GPUS" || die "judge"
  fi
  todo=$(for pg in $ROLLOUTS; do rp "$pg"  # a remote server's tunnel first (SGLang binds 127.0.0.1 there too)
    [ -z "$h" ] || tunnel "$p" "$h" "127.0.0.1:$p"
    up "127.0.0.1:$p/v1/models" "rollout-g$g.pid" "sglang.launch_server.*--port $p " "$h" || echo "$pg"; done)
  # a remote server starts with rq1-v$V at the same path there: pushed once per host (it replaces weight_update_v*)
  for h in $(for pg in $todo; do rp "$pg"; echo "$h"; done | sort -u); do
    "$AREAL_PY" -c 'import sys; from octopus.rl.adapter_push import push_adapter; push_adapter(sys.argv[1], [sys.argv[2] + ":22"])' \
      "$CK/default/weight_update_v$V" "$h" || die "adapter push to $h"; done
  # SHARED: BUDGET = free - 6.5 GiB; serve_f2.sh re-reads free 5 s later and needs BUDGET + 6 GiB (512 MiB of slack)
  for pg in $todo; do rp "$pg"; b=; c=$CODE; cu=$CU; tmp=; [ -z "$h" ] || { c=$ROOT/${CODE##*/}; cu=${R_CU:-$CU}; tmp=${R_TMPDIR:-}; }
    if [ -n "${SHARED:-}" ]; then gpus_free "$g" "$ROLLOUT_MIB" "$h" || die "rollout $pg"; b=$(($(mib "$g" free "$h") - 6656))
      [ "$b" -gt 0 ] || die "rollout $pg: free memory unreadable"; fi
    # KV_TOKENS, MAMBA_SLOTS from the caller's environment: serve_f2.sh's fixed pools
    on "$h" "${tmp:+TMPDIR=$tmp }BUDGET=$b KV_TOKENS=${KV_TOKENS:-} MAMBA_SLOTS=${MAMBA_SLOTS:-} SGLANG_PY=$SGLANG_PY MODEL=$MODEL_VL \
      SGL_PATH=$CURSOR/sglang:$c CU=$cu LORAS=rq1-v$V=$CK/default/weight_update_v$V \
      bash $c/results/search-v9-rest-1028/scripts/serve_f2.sh $L $g:$p" || die "rollout $pg"
  done
  for _ in $(seq 180); do  # up to 30 min
    down=$(for s in $(servers | tr , ' '); do curl -sf -m 4 "http://$s/v1/models" > /dev/null || echo -n "$s "; done
      curl -sf -m 4 "$JUDGE_URL/models" > /dev/null || echo -n "judge "  # ours (JUDGE_GPUS), a tunnel's or another host's
      curl -sf -m 4 http://127.0.0.1:32600/health > /dev/null || echo -n "search")
    [ -z "$down" ] && break; sleep 10
  done
  [ -z "$down" ] && say "all up: rollouts $(servers) (rq1-v$V), judge ${JUDGE_URL}, search :32600" || die "not up: $down (logs in $L)"
  ;;
train)
  [ -n "$(step)" ] || die "no $CK/LATEST (FILEROOT: the ROOT symlink?)"  # else AReaL trains from step 0
  P=$(cat "$PIDF" 2>/dev/null); [ -n "$P" ] && alive "$P" octopus.rl.train && die "trainer $P still running"
  P=$(cat "$L/watch.pid" 2>/dev/null); [ -n "$P" ] && alive "$P" "newhost.sh _watch" && die "watcher $P runs: TERM it first"
  gpus_free "$TRAIN_GPUS" "${TRAIN_MIB:-}" || die "trainer GPUs $TRAIN_GPUS: not starting (README \"node-134 (10-30)\")"
  ram || die "not starting"
  SERVERS=$(servers)${EXTRA_SERVERS:+,$EXTRA_SERVERS}; NS=$(n "$SERVERS")
  for s in ${SERVERS//,/ }; do curl -sf -m 4 "http://$s/v1/models" > /dev/null || die "down $s"; done
  curl -sf -m 4 http://127.0.0.1:32600/health > /dev/null || die "search service down"
  curl -sf -m 4 "$JUDGE_URL/models" > /dev/null || die "judge down"
  NR=$FILEROOT/name_resolve/$U/rq1-rl/search-v9-rest; [ -d "$NR" ] && mv "$NR" "$NR.bak-$(date +%H%M%S)"
  if [ "$CHUNK_MAJOR" = 1 ]; then
    grep -q OCTOPUS_CHUNK_MAJOR octopus/rl/config.py || die "CHUNK_MAJOR=1 but the code lacks #338"
    # #354: before it, a short micro-batch sharing nothing ran flat beside chunk-major ones: NaN gradients
    grep -qF 'not with_tree and not mounts' octopus/rl/prefix_tree.py || die "CHUNK_MAJOR=1 but the code lacks #354"
    export OCTOPUS_CHUNK_MAJOR=1 OCTOPUS_BLOCK_CHECKPOINT=1 OCTOPUS_CHUNK_TOKENS=$CHUNK_TOKENS
  fi
  push=$(echo $ADAPTER_PUSH $(for h in $(hosts); do echo "$h:22"; done))  # and every remote rollout host
  [ -n "$push" ] && export OCTOPUS_ADAPTER_PUSH=$push
  # train-138-h.sh's settings (Alpha search-v9-1025's episode settings, admission, AReaL overrides) on this host's paths
  export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE PYTHONPATH=$CURSOR/areal:$CODE PYTHONUNBUFFERED=1
  # the proxy's admin key (no AREAL_ALLOW_DEFAULT_ADMIN_KEY on a public host): one random key, kept in a mode-600 file,
  # given through the environment (hydra's oc.env below), not the command line that ps shows every user
  [ -s "$RUN/admin_api_key" ] || (umask 077; openssl rand -hex 24 > "$RUN/admin_api_key") || die "admin key"
  export OCTOPUS_ADMIN_KEY=$(cat "$RUN/admin_api_key")  # *_KEY: not an episode setting (octopus/rl/config.py)
  export OCTOPUS_SGLANG_PY=$SGLANG_PY OCTOPUS_KEEP_WEIGHT_UPDATES=1
  export OCTOPUS_HARBOR=$HARBOR OCTOPUS_CODE=$CODE OCTOPUS_SEARCH_PYTHON=$SEARCH_PY
  export OCTOPUS_SEARCH_URL=http://127.0.0.1:32600 OCTOPUS_JUDGE_URL=$JUDGE_URL
  export OCTOPUS_JUDGE_MODEL=judge OCTOPUS_JUDGE_TIMEOUT=600 OCTOPUS_FORK_DEPTH=0  # the code's default (tp 1 judge)
  export OCTOPUS_CURSOR=1 OCTOPUS_READ_TOOL=1 OCTOPUS_REWIND_TOOL=1 OCTOPUS_REWIND_MAX=2 LITELLM_LOCAL_MODEL_COST_MAP=True
  export OCTOPUS_ROUTE_SERVERS=$SERVERS OCTOPUS_ROLLOUT_SPOOL=$RUN/rollout-spool/search-v9-rest
  export OCTOPUS_ROUTE_CAP=$ROUTE_CAP OCTOPUS_ROUTE_KV=$ROUTE_KV OCTOPUS_ROUTE_WAIT=$ROUTE_WAIT OCTOPUS_ROUTE_TREE_TOKENS=$ROUTE_TREE_TOKENS
  export OCTOPUS_LIVE_ENV=$RUN/live.env
  [ -e "$L/train-rest.log" ] && mv "$L/train-rest.log" "$L/train-rest-$(date +%m%d%H%M).log"
  setsid "$AREAL_PY" -m octopus.rl.train --config "$CODE/octopus/rl/grpo.yaml" \
    experiment_name=rq1-rl trial_name=search-v9-rest \
    cluster.fileroot="$FILEROOT" cluster.n_gpus_per_node=$(n "$TRAIN_GPUS") \
    total_train_epochs=4 +total_train_steps=61 '+rollout.agent.admin_api_key=${oc.env:OCTOPUS_ADMIN_KEY}' \
    actor.path="$MODEL_TEXT" actor.lora_rank=64 actor.lora_alpha=64 rl.lora_keep_versions=40 \
    rl.servers="'$SERVERS'" rl.jobs_dir="$RUN/jobs-search-v9-rest" rl.episodes_per_server=1 \
    rollout.backend=sglang:d$NS actor.backend=fsdp:d$(n "$TRAIN_GPUS") rollout.max_concurrent_rollouts=$CONCURRENT \
    workflow=octopus.envs.search.rl.SearchEpisodeAgent rl.search_questions="$QUESTIONS" rl.search_split="$SPLIT" \
    actor.mb_spec.max_tokens_per_mb=$MB_TOKENS gconfig.n_samples=8 train_dataset.batch_size=16 \
    saver.freq_steps=10 recover.freq_steps=1 > "$L/train-rest.log" 2>&1 < /dev/null &
  echo $! > "$PIDF"; say "trainer $(cat "$PIDF") on GPUs $TRAIN_GPUS, servers $SERVERS, code $(git rev-parse --short HEAD 2>/dev/null)"
  setsid nohup bash "$K/newhost.sh" _watch >> "$L/watch.log" 2>&1 < /dev/null & echo $! > "$L/watch.pid"
  say "watcher $(cat "$L/watch.pid") ($L/watch.log)"
  setsid "$AREAL_PY" "$K/pass_stop.py" "$RUN/jobs-search-v9-rest" "$SP" "$PIDF" "$CK" "$L/pass-stop.json" \
    >> "$L/pass-stop.log" 2>&1 < /dev/null &
  say "pass_stop.py $! ($L/pass-stop.log)"
  ;;
_watch)
  while kill -0 "$(cat "$PIDF")" 2>/dev/null; do sleep 60; done
  say "trainer $(cat "$PIDF") exited; releasing"
  w=$(ps -u "$U" -o pid=,args= | awk '/areal\.infra\.rpc\.rpc_server|proxy\.proxy_rollout_server/ && /search-v9-rest/ {print $1}')
  [ -n "$w" ] && { kill $w; sleep 10; kill -9 $w 2>/dev/null; }
  while pgrep -u "$U" -f '^[^ ]*python[^ ]* -m octopus\.eval\.search_dev_eval' > /dev/null; do say "eval running; release waits"; sleep 300; done
  pgkill "$(cat "$L/search.pid" 2>/dev/null)" octopus.envs.search.service
  pgkill "$(cat "$L/judge.pid" 2>/dev/null)" 'sglang.launch_server.*--port 32500 '
  for pg in $ROLLOUTS; do rp "$pg"; on "$h" "pgkill \"\$(cat $L/rollout-g$g.pid 2>/dev/null)\" 'sglang.launch_server.*--port $p '"
    [ -z "$h" ] || pgkill "$(cat "$L/tunnel-$p.pid" 2>/dev/null)" "ssh -N .*-L 127.0.0.1:$p:"; done
  for t in ${TUNNELS:-}; do pgkill "$(cat "$L/tunnel-${t%%:*}.pid" 2>/dev/null)" "ssh -N .*-L 127.0.0.1:${t%%:*}:"; done
  [ -n "$RELEASE" ] && while read -r h f p; do [ -z "$h" ] || [[ $h == \#* ]] && continue
    say "$h: $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" "$(declare -f say alive pgkill); pgkill \$(cat $f) '$p'" 2>&1)"
  done < "$RELEASE"
  say released; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  ;;
status)
  ps -u "$U" -o pid=,etime=,args= | grep -E 'octopus\.rl\.train|sglang\.launch_server|octopus\.envs\.search\.service|newhost\.sh _watch|pass_stop\.py|ssh -N' \
    | grep -v grep | cut -c1-180
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  tail -n 3 "$L/train-rest.log" 2>/dev/null; tail -n 2 "$L/watch.log" 2>/dev/null; tail -n 1 "$L/pass-stop.log" 2>/dev/null
  ;;
*) sed -n 2,31p "$0"; exit 1 ;;
esac
