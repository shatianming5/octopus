#!/bin/bash
# RQ1 RL phase 1: start (or resume) a run, stop it, or show it. Every run is a trial under $FILEROOT; AReaL saves a
# recover checkpoint after each training step, so `stop` then `start <same trial>` continues after the last finished step
# (the rollouts in flight at the stop are lost and redone).
#
#   rl_run.sh start [TRIAL]   new trial (phase1-MMDDHHMM) or resume TRIAL      rl_run.sh stop    rl_run.sh status
#   rl_run.sh settings OUT_DIR [key=value ...]   a run dir for dev_eval --run without training (octopus/rl/config.py)
#
# Remote rollout hosts (REMOTE, optional): more SGLang servers on machines the trainer host cannot reach directly. Each
# remote host runs `rl_run.sh tunnel` once (its own env file: SGLANG_PY MODEL_VL ADAPTER0 STATE INFER_GPUS PORT0
# TRAINER_SSH SSH_RPORT and the server settings below): reverse ssh tunnels that put its sshd on the trainer's
# 127.0.0.1:SSH_RPORT and each server port on the same trainer port (`rl_run.sh untunnel` ends them). Then the trainer
# env lists REMOTE="user@127.0.0.1:SSH_RPORT ..." and REMOTE_ENV (that env file's path on the remote hosts): start and
# stop run this script's serve/unserve there, and the actor pushes each adapter there before the servers load it
# (octopus/rl/adapter_push.py; FILEROOT must be a writable path on the remote hosts too).
#
# Machine settings come from $RL_ENV_FILE (default ~/octo-env/rl_run.env): AREAL_PY SGLANG_PY CODE MODEL_TEXT MODEL_VL
# ADAPTER0 FILEROOT STATE INFER_GPUS TRAIN_GPUS STEPS EXCLUDE, optional CUDA_HOME, EPISODES_PER_SERVER (8), ROUTE_CAP (0 = none; see OCTOPUS_ROUTE_CAP below), OFFPOLICY (the yaml's),
# MB_TOKENS (training micro-batch cap in prefix-tree tokens, 262144), LORA_RANK (32; alpha = rank, ADAPTER0 must match),
# MEM_FRAC (SGLang --mem-fraction-static, 0.85), SGLANG_ARGS (more SGLang server flags, word-split; a remote host reads its own env), BUSY_MIB (MiB a GPU may already use at serve, 1000), PORT0 (first server port, 30200), LORA_KEEP (adapter versions kept
# loaded, 10; raise it when steps are fast: an episode still on a version due for unloading stalls update_weights), EXTRA (more Hydra overrides,
# e.g. actor.optimizer.lr=1e-5) and MEM_MAX (trainer host-memory cap, 380G: node-94 has 503 GB; what each training rank
# pins per micro-batch token: docs/trainer-memory.md; the rest is for the
# SGLang servers, ~10 GB each, the Harbor episodes, each capped at OCTOPUS_EPISODE_MEM, and the OS; MEM_MAX=job: no
# user systemd, a Slurm job's cgroup is the cap), JOBS_ROOT (Harbor jobs dir parent, $OCTOPUS_HARBOR/rl; OCTOPUS_HARBOR defaults to /home/user/harbor-udocker) and
# ROLLOUT_SPOOL (a dir: finished, untrained groups are kept in ROLLOUT_SPOOL/TRIAL and a resume trains on them first,
# one group per task, the others deleted and logged; octopus/rl/spool.py). While a run goes, KEY=VALUE lines in $STATE/live.env set EPISODES_PER_SERVER (at most the
# launch value; SWE only, octopus/envs/swe/rl.py episode slots: search has no slots, its concurrency is the launch
# rollout.max_concurrent_rollouts) at the next episode start and MB_TOKENS at the next train-worker call; the logs show each change.
# ACT_GPU_GB (0): at most this many GB of each micro-batch's saved activations kept on the GPU instead of pinned host
# memory (more GPU, less MEM_MAX), also live in $STATE/live.env; per micro-batch only what fits under ACT_GPU_TARGET_GB
# (40) beside its token-predicted peak (octopus/rl/actor.py act_gpu_budget).
# CURSOR (a dir, off when unset): the cursor protocol (docs/cursor-protocol.md "Deployment"): the trainer on the AReaL
# overlay $CURSOR/areal (logprobs + cursor + route-spread + cpu-batch + train-rpc-timeout patches, in place of AREAL_OVERLAY), every SGLang server, here and on the
# REMOTE hosts (the same paths there; serve refuses an overlay with copies, and a REMOTE CODE whose shimmed files differ from
# the trainer's), on the overlay $CURSOR/sglang + $CODE, and the episodes with OCTOPUS_CURSOR=1. The read
# tool (docs/cross-branch-read.md "Slice 4 run") adds no knob: `export OCTOPUS_READ_TOOL=1` in the env file (the
# episodes inherit it; the rollout workers export every cursor run's reads), with the same SGLANG_ARGS as without it (the mount works on any mamba strategy, overlap
# on); $CURSOR/sglang carries the mount patch (docs/cursor-protocol.md "Deployment").
# OCTOPUS_READ_CONTROL=1: start refuses a $CURSOR/areal built before the retry-orphan key holds the mount.
# OCTOPUS_MOUNT_LAYERS (unset: every attention layer): the decoder layers whose attention reads a mount, e.g. 3,7,11
# (docs/cursor-protocol.md "Mount"); export it in the env file: the servers here and on REMOTE get this value, and the
# trainer refuses servers that report another.
# SERVE_HOST (serve; default 127.0.0.1): the address the servers listen on and are listed under, e.g. this node's LAN
# address for a trainer on another node of one cluster (no ssh, no adapter push: FILEROOT on a filesystem both share);
# that trainer's start then takes them as EXTRA_SERVERS="host:port,..." beside its own INFER_GPUS servers (none: INFER_GPUS=).
set -uo pipefail
source "${RL_ENV_FILE:-$HOME/octo-env/rl_run.env}"
HARBOR=${OCTOPUS_HARBOR:-/home/user/harbor-udocker}  # the episodes' Harbor dir (octopus/rl/workflow.py)
mkdir -p "$STATE/serve"; cd "$STATE"
IFS=, read -ra IG <<< "$INFER_GPUS"
code_sum() { cat "$CODE"/octopus/serving/{delta_render,cursor_store,kv_mount}.py | md5sum | cut -d' ' -f1; }  # what the overlay's shims load
# trial $1's AReaL processes: training workers (rpc_server) and proxy-rollout servers (the trainer leaves them orphaned)
trial_workers() { ps -u "$USER" -o pid=,args= | awk -v t="--trial-name $1" '$0 ~ /areal\.infra\.rpc\.rpc_server|proxy\.proxy_rollout_server/ && (index($0, t " ") || substr($0, length($0) - length(t) + 1) == t) {print $1}'; }

# what this script adds to the env file's exports for the trainer, its workers and the episodes (start; settings)
episode_exports() {
  export LITELLM_LOCAL_MODEL_COST_MAP=True  # proxy workers and Harbor episodes: no GitHub fetch at import
  [ -z "${CURSOR:-}" ] || export OCTOPUS_CURSOR=1  # episodes inherit it, remote ones too
}

here() { [ "$(cat trainer.host 2>/dev/null || hostname -s)" = "$(hostname -s)" ]; }  # trainer.pid is this host's
stop_servers() {  # this host's servers of this state dir (serve/sglang-HOST-g*.pid: STATE may be on a filesystem other
  # hosts share, e.g. Alpha's /projects) and nothing else; waits until they are gone
  local f p pids=""; for f in serve/sglang-$(hostname -s)-g*.pid; do [ -f "$f" ] && pids="$pids $(cat "$f")"; rm -f "$f"; done
  pids=${pids# }; [ -n "$pids" ] || return 0
  # each server runs in its own session (setsid): signal the whole group, so its scheduler children go even when the
  # launcher pid is already gone (bundle-e9's last stop left 8 orphaned schedulers holding node-68/node-134). Any other
  # process on a rollout GPU, this user's too, is left alone: serve refuses that GPU as busy
  for p in $pids; do pgrep -g "$p" -f sglang > /dev/null && kill -- "-$p"; done  # still an SGLang group (no reused pid)
  for _ in $(seq 60); do pgrep -g "${pids// /,}" > /dev/null || break; sleep 2; done
  for p in $pids; do pgrep -g "$p" -f sglang > /dev/null && kill -9 -- "-$p"; done; sleep 3
}

serve() {  # this host's SGLang servers, one per INFER_GPUS card on port PORT0 + i; last line "servers <host:port,...>"
  stop_servers  # our own servers from an earlier start must not count as someone else's
  local g gpus="$INFER_GPUS,${TRAIN_GPUS:-}"
  for g in ${gpus//,/ }; do  # never start on GPUs someone else is using
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -gt "${BUSY_MIB:-1000}" ] && { echo "GPU $g busy (${used} MiB)"; return 1; }
  done
  [ -n "${CUDA_HOME:-}" ] && export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
  export PATH="$(dirname "$SGLANG_PY"):$PATH" SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_CACHE_DIR="$STATE/sglang-cache"
  if [ -n "${CURSOR:-}" ]; then  # the cursor-patched SGLang ahead of the venv's; its shims import CODE's octopus/
    old=$(grep -L 'sys.modules\[__name__\]' "$CURSOR"/sglang/sglang/srt/octopus_*.py 2>&1)
    [ -z "$old" ] || { echo "$CURSOR/sglang has copies, not shims ($old): re-run patches/apply_sglang_cursor.py and apply_sglang_mount.py on it"; return 1; }
    [ -z "${CODE_SUM:-}" ] || [ "$(code_sum)" = "$CODE_SUM" ] || {  # a REMOTE host: its CODE (env file's or the trainer's path) = the trainer's
      echo "$(hostname): CODE $CODE's octopus/serving/{delta_render,cursor_store,kv_mount}.py differ from the trainer's"; return 1; }
    export PYTHONPATH=$CURSOR/sglang:$CODE
  fi
  local servers=""
  # LORA_KEEP + 4 adapter slots: v0 + the last LORA_KEEP versions. SGLang 0.5.19 needs
  # patches/apply_sglang_lora_abort_release.py: requests aborted in the queue at a pause leaked their LoRA count, so
  # unloading that version later held the LoRA lock forever and update_weights timed out (bundle-e1d steps 29/38/48)
  for i in "${!IG[@]}"; do  # one SGLang per rollout GPU, LoRA hot-loading; AReaL loads rq1-v<step> after each step / on resume
    PORT=$((${PORT0:-30200} + i)); servers="$servers${servers:+,}${SERVE_HOST:-127.0.0.1}:$PORT"
    CUDA_VISIBLE_DEVICES=${IG[$i]} setsid bash -c 'echo $$ > "$0"; exec "$@"' "serve/sglang-$(hostname -s)-g${IG[$i]}.pid" "$SGLANG_PY" -m sglang.launch_server --model-path "$MODEL_VL" --host "${SERVE_HOST:-127.0.0.1}" \
      --port $PORT --tp-size 1 --mem-fraction-static ${MEM_FRAC:-0.85} --attention-backend triton --sampling-backend pytorch \
      --context-length 262144 --disable-prefill-cuda-graph --enable-cache-report --enable-lora --lora-paths rq1-v0="$ADAPTER0" \
      --max-lora-rank ${LORA_RANK:-32} --max-loaded-loras $(( ${LORA_KEEP:-10} + 4 )) --max-loras-per-batch 4 --lora-backend triton --lora-strict-loading --mm-attention-backend triton_attn ${SGLANG_ARGS:-} \
      > "serve/sglang-$(hostname -s)-g${IG[$i]}.log" 2>&1 < /dev/null &
    for _ in $(seq 100); do [ -s "serve/sglang-$(hostname -s)-g${IG[$i]}.pid" ] && break; sleep 0.1; done  # written by the session leader itself (setsid forks under job control, so $! can name an exited pid)
  done
  for i in "${!IG[@]}"; do  # a server that exits while loading fails the start (and stops the others)
    until curl -sf -m 3 "http://${SERVE_HOST:-127.0.0.1}:$((${PORT0:-30200} + i))/v1/models" > /dev/null; do
      f="serve/sglang-$(hostname -s)-g${IG[$i]}.pid"; [ ! -s "$f" ] || kill -0 "$(cat "$f")" 2>/dev/null || {  # no pid file yet: still starting
        echo "SGLang on GPU ${IG[$i]} exited: $(tail -n 5 "serve/sglang-$(hostname -s)-g${IG[$i]}.log" | tr '\n' '|')"; stop_servers; return 1; }
      sleep 10
    done
  done
  echo "servers $servers"
}

stop_episodes() {  # Harbor episodes of trial $1 still running here, and their containers
  local c d hp JD=${JOBS_ROOT:-$HARBOR/rl}/jobs-$1
  hp=$(ps -u "$USER" -o pid=,args= | awk -v d="-o $JD " 'index($0, d) && $0 ~ /harbor run/ {print $1}')
  [ -n "$hp" ] && { kill -INT $hp; sleep 20; kill -9 $hp 2>/dev/null; }
  for c in $(cat "$JD"/*.console.log 2>/dev/null | grep -o "udocker started [0-9a-f-]*" | awk '{print $3}'); do
    d=$HARBOR/containers/$c  # overlay trials (UDOCKER_OVERLAY): unmount before removing
    mountpoint -q "$d/ROOT" && fusermount3 -u -z "$d/ROOT"; rm -rf "$d" "$d.ovl"
  done
}

on_remote() {  # on_remote user@host:port CMD: this script's CMD on that remote rollout host, with its env file; ssh's
  # own failure (255: e.g. a remote sshd whose unauthenticated slots are full drops the handshake) is tried 4 times
  local i rc
  for i in 1 2 3 4; do
    ssh -p "${1##*:}" -o BatchMode=yes -o ConnectTimeout=20 "${1%:*}" "CURSOR=${CURSOR:-} CODE=$CODE CODE_SUM=$(code_sum) OCTOPUS_READ_TOOL=${OCTOPUS_READ_TOOL:-} OCTOPUS_MOUNT_LAYERS=${OCTOPUS_MOUNT_LAYERS:-} RL_ENV_FILE=$REMOTE_ENV bash -s $2" < "$0"; rc=$?
    [ "$rc" = 255 ] || return "$rc"
    sleep $((i * 3))
  done
  return "$rc"
}

case "${1:-}" in
start)
  [ -d "$CODE/octopus" ] || { echo "CODE $CODE: no code dir"; exit 1; }  # before any server starts
  # a read control's first request repeats its reader's mounted one: an AReaL overlay without the mount in the retry-orphan
  # key (patches/areal-cursor.patch) drops the readers that closed right after their read (docs/rl-plan-episodes.md)
  if [ "${OCTOPUS_READ_CONTROL:-}" = 1 ] && ! grep -qF 'entry.metadata.get("mount")' "${CURSOR:-/nonexistent}/areal/areal/experimental/openai/cache.py" 2>/dev/null; then
    echo "OCTOPUS_READ_CONTROL=1 needs CURSOR's AReaL overlay rebuilt from patches/areal-cursor.patch with the mount in the retry-orphan key"; exit 1
  fi
  # a tree budget counts each episode from AReaL's /rl/start_session to /rl/end_session: the AReaL that start runs
  # (CURSOR's overlay, else AREAL_OVERLAY, else the venv's) needs #351's hooks, or no tree frees its slot
  if [ "${OCTOPUS_ROUTE_TREE_TOKENS:-0}" != 0 ]; then
    a=$(PYTHONPATH=${CURSOR:+$CURSOR/areal:}${AREAL_OVERLAY:+$AREAL_OVERLAY:}$CODE "$AREAL_PY" -c 'import importlib.util as u; print(u.find_spec("areal").submodule_search_locations[0])')
    grep -qF 'route.start(session_id)' "$a/experimental/openai/proxy/proxy_rollout_server.py" 2>/dev/null \
      || { echo "OCTOPUS_ROUTE_TREE_TOKENS: $a lacks patches/areal-route-spread.patch's route hooks (#351): rebuild it"; exit 1; }
  fi
  if [ -f trainer.pid ] && here && kill -0 "$(cat trainer.pid)" 2>/dev/null; then echo "already running: $(cat trial)"; exit 1; fi
  rm -f serve/*.out; k=0; rp=""
  for d in ${REMOTE:-}; do on_remote "$d" serve > "serve/remote-$k.out" 2>&1 & rp="$rp $!"; k=$((k + 1)); done
  serve > serve/local.out; [ -n "$rp" ] && wait $rp  # the remote hosts' servers start in parallel (not `wait`: the local servers are jobs too)
  SERVERS=$(cat serve/*.out | awk -v n=$((k + 1)) '/^servers / {if ($2 != "") {printf "%s%s", s, $2; s = ","}; n--} END {exit n != 0}') || {
    cat serve/*.out; stop_servers; for d in ${REMOTE:-}; do on_remote "$d" unserve; done; exit 1; }
  for s in ${EXTRA_SERVERS:+${EXTRA_SERVERS//,/ }}; do  # servers another job started (rl_run.sh serve with SERVE_HOST): they must answer
    m=$(curl -sf -m 10 "http://$s/v1/models") || { echo "EXTRA_SERVERS: $s down"; stop_servers; exit 1; }
    # they outlive a trainer: unload what an earlier one loaded (rq1-v<N> names carry no trial; a resume's same-named
    # load would be refused as already loaded and the server would keep the old weights), all but rq1-v0
    old=$("$AREAL_PY" -c 'import json, sys
print(" ".join(m["id"] for m in json.load(sys.stdin)["data"] if m.get("parent") and m["id"] != "rq1-v0"))' <<< "$m") \
      || { echo "EXTRA_SERVERS: $s: unreadable /v1/models"; stop_servers; exit 1; }
    for a in $old; do
      curl -sf -m 120 -X POST "http://$s/unload_lora_adapter" -H 'Content-Type: application/json' -d "{\"lora_name\": \"$a\"}" > /dev/null \
        || { echo "EXTRA_SERVERS: $s cannot unload $a (an unload can hang on a count a disconnected request leaked: restart that server)"; stop_servers; exit 1; }
    done
  done
  SERVERS=$SERVERS${EXTRA_SERVERS:+${SERVERS:+,}$EXTRA_SERVERS}
  echo "$SERVERS" > serve/servers; NS=$(awk -F, '{print NF}' <<< "$SERVERS"); NT=$(awk -F, '{print NF}' <<< "$TRAIN_GPUS")
  TRIAL=${2:-phase1-$(date +%m%d%H%M)}; echo "$TRIAL" > trial
  rm -rf "$FILEROOT/name_resolve/$USER/rq1-rl/$TRIAL"  # stale keys from a previous process of this trial (adapter-ready signals)
  episode_exports
  # every server this start brought up (patches/areal-route-spread.patch, where the AReaL overlay has it): a session
  # stays on one of them (one server per episode tree, as a read's mount needs), a new session goes to the one with the
  # fewest requests in flight; ROUTE_CAP (> 0): only below that many, else it waits (per proxy worker, octopus/serving/route.py)
  export OCTOPUS_ROUTE_SERVERS=$SERVERS
  export OCTOPUS_ROUTE_CAP=${ROUTE_CAP:-0}
  export OCTOPUS_ADAPTER_PUSH="${REMOTE:-}"
  export OCTOPUS_LIVE_ENV="$STATE/live.env"  # EPISODES_PER_SERVER (<= the launch value) and MB_TOKENS without a restart
  export OCTOPUS_ACT_GPU_GB=${ACT_GPU_GB:-0} OCTOPUS_ACT_GPU_TARGET_GB=${ACT_GPU_TARGET_GB:-40}  # train workers inherit them
  export OCTOPUS_SGLANG_PY="$SGLANG_PY"  # its SGLang decides the default LoRA targets (octopus/rl/lora_targets.py)
  [ -n "${ROLLOUT_SPOOL:-}" ] && export OCTOPUS_ROLLOUT_SPOOL="$ROLLOUT_SPOOL/$TRIAL"  # finished groups survive a resume
  echo "$(date +%T) $NS servers up ($SERVERS); trial $TRIAL"
  cd "$CODE"  # AReaL resolves --config relative to its own install path: keep both under the same symlinked prefix
  [ -n "${CURSOR:-}" ] && AREAL_OVERLAY=$CURSOR/areal
  # AREAL_OVERLAY: a run-local AReaL copy with the repo's patches/areal-*.patch, first
  export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS PYTHONPATH=${AREAL_OVERLAY:+$AREAL_OVERLAY:}$CODE NCCL_P2P_DISABLE=1
  KEY=$(openssl rand -hex 24)  # per-run proxy admin secret (the proxy listens on a routable address)
  # hard host-memory cap (cgroup): over it the kernel kills this run, never the machine (OOM-starved node-94 on 10-01)
  CAP=(systemd-run --user --scope -q -p MemoryMax="${MEM_MAX:-380G}" -p MemorySwapMax=0); [ "${MEM_MAX:-}" = job ] && CAP=()
  setsid nohup "${CAP[@]}" "$AREAL_PY" -m octopus.rl.train \
    --config "$CODE/octopus/rl/grpo.yaml" scheduler.type=local cluster.fileroot="$FILEROOT" \
    actor.path="$MODEL_TEXT" actor.lora_rank=${LORA_RANK:-32} actor.lora_alpha=${LORA_RANK:-32} rl.lora_keep_versions=${LORA_KEEP:-10} "rl.servers='$SERVERS'" rollout.backend=sglang:d$NS actor.backend=fsdp:d$NT \
    rollout.max_concurrent_rollouts=$((${EPISODES_PER_SERVER:-8} * NS)) rl.episodes_per_server=${EPISODES_PER_SERVER:-8} \
    actor.mb_spec.max_tokens_per_mb=${MB_TOKENS:-262144} ${OFFPOLICY:+rollout.max_head_offpolicyness=$OFFPOLICY} rl.exclude="$EXCLUDE" \
    rl.jobs_dir=${JOBS_ROOT:-$HARBOR/rl}/jobs-$TRIAL +total_train_steps=$STEPS trial_name=$TRIAL \
    +rollout.agent.admin_api_key=$KEY ${EXTRA:-} >> "$STATE/train-$TRIAL.log" 2>&1 < /dev/null &
  echo $! > "$STATE/trainer.pid"; hostname -s > "$STATE/trainer.host"; echo "trainer pid $(cat "$STATE/trainer.pid"), log $STATE/train-$TRIAL.log"
  ;;
stop)
  TRIAL=$(cat trial); p=$(cat trainer.pid 2>/dev/null || echo 0)
  here || p=0  # another host's trainer (shared STATE)
  [ "$p" = 0 ] || { kill "$p" 2>/dev/null; for _ in $(seq 20); do kill -0 "$p" 2>/dev/null || break; sleep 3; done; kill -9 "$p" 2>/dev/null; }
  w=$(trial_workers "$TRIAL"); [ -n "$w" ] && kill $w; sleep 10; w=$(trial_workers "$TRIAL"); [ -n "$w" ] && kill -9 $w
  stop_servers; stop_episodes "$TRIAL"; for d in ${REMOTE:-}; do on_remote "$d" unserve; done
  echo "stopped $TRIAL; resume with: rl_run.sh start $TRIAL"
  ;;
status)
  TRIAL=$(cat trial 2>/dev/null); echo "trial $TRIAL"
  [ -f trainer.pid ] && here && kill -0 "$(cat trainer.pid)" 2>/dev/null && echo "trainer running" || echo "trainer not running here"
  grep -hoE "Epoch [0-9]+/[0-9]+ Step [0-9]+/[0-9]+ Train step [0-9]+/[0-9]+ done" "$FILEROOT/logs/$USER/rq1-rl/$TRIAL/main.log" 2>/dev/null | tail -1
  ls "$FILEROOT/checkpoints/$USER/rq1-rl/$TRIAL" 2>/dev/null | tr '\n' ' '; echo
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
  for s in $(tr , ' ' < serve/servers 2>/dev/null); do curl -sf -m 3 "http://$s/v1/models" > /dev/null && echo -n "$s up " || echo -n "$s DOWN "; done; echo
  ;;
settings)  # rl_run.sh settings OUT_DIR [key=value ...]: what dev_eval --run takes from a run (config.yaml and
  # episode_settings.json) for a training never started: these exports and config overrides (e.g. an untrained base
  # model: rollout.use_lora=false)
  OUT=$(realpath -m "$2"); shift 2; episode_exports; cd "$CODE" || exit 1
  PYTHONPATH=${CURSOR:+$CURSOR/areal:}$CODE "$AREAL_PY" -m octopus.rl.config "$OUT" actor.path="$MODEL_TEXT" "$@" \
    && echo "settings in $OUT" ;;
serve) serve ;;
unserve) stop_servers ;;
tunnel)  # on a remote rollout host: reverse tunnels to TRAINER_SSH, re-opened whenever ssh exits (link drop, trainer reboot)
  ports="$SSH_RPORT"; fw="-R 127.0.0.1:$SSH_RPORT:127.0.0.1:22"
  for i in "${!IG[@]}"; do p=$((${PORT0:-30200} + i)); ports="$ports $p"; fw="$fw -R 127.0.0.1:$p:127.0.0.1:$p"; done
  # the trainer's sshd keeps a dead session's forwards (ClientAliveInterval 0): free those ports before reconnecting
  setsid bash -c "while true; do ssh -o BatchMode=yes -o ConnectTimeout=20 $TRAINER_SSH 'fuser -k -n tcp $ports' 2>/dev/null
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 $fw $TRAINER_SSH
    echo \"\$(date '+%F %T') tunnel exited \$?\"; sleep 10; done" >> serve/tunnel.log 2>&1 < /dev/null &
  echo $! > tunnel.pid; echo "tunnel pid $(cat tunnel.pid) to $TRAINER_SSH: ports $ports"
  ;;
untunnel) kill -- -"$(cat tunnel.pid)"; rm -f tunnel.pid ;;
*) sed -n 2,38p "$0"; exit 1 ;;
esac
