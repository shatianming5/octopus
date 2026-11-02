# Sourced by pilot.sbatch (SWE) and search.sbatch after their env file (C, TRIAL, P, STATE, FILEROOT set): the stage
# helpers and the hold loop. User, 10-22: an Alpha allocation is kept until its time limit, never released early; each
# stage still stops its own processes (trainer, SGLang servers) when it ends.
say() { echo "=== $(date '+%F %T') $*"; }
# alphagpu22 does not resolve its own short name (getaddrinfo AF_UNSPEC: EBUSY), which AReaL's gethostip does not catch:
# map it to the cluster FQDN for every process of the job (glibc HOSTALIASES)
if ! getent hosts "$(hostname)" > /dev/null && getent hosts "$(hostname).cm.cluster" > /dev/null; then
  echo "$(hostname) $(hostname).cm.cluster" > "/tmp/$USER-hostaliases-${SLURM_JOB_ID:-0}"; export HOSTALIASES=/tmp/$USER-hostaliases-${SLURM_JOB_ID:-0}
fi
BG=""
bgrun() {  # bgrun DIR LOG CMD...: CMD in DIR, waited on in the background so that a TERM runs the traps at once
  (cd "$1" && exec "${@:3}") > "$2" 2>&1 & BG=$!; wait $BG; local rc=$?; BG=""; return $rc; }
stop_stage() {  # whatever a stage may have left: the waited-on command, the trainer and its servers, eval servers
  [ -n "$BG" ] && kill "$BG" 2>/dev/null; BG=""
  [ -f "$STATE/trial" ] && bash "$C/scripts/rl_env/rl_run.sh" stop > /dev/null
  bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null
}
takeover() {  # takeover FILE SCRIPT: FILE's first line names a code dir; when it is not C, its copy of SCRIPT (this job's
  # script, scripts/alpha/...) replaces this one in the same allocation, same TRIAL (a fix made while the job queued)
  local R; R=$(head -1 "$1" 2>/dev/null); [ -d "$R/octopus" ] && [ "$(realpath "$R")" != "$(realpath "$C")" ] || return 0
  say "code $R takes over"; trap - EXIT; exec bash "$R/$2" "$R" "$TRIAL"
}
run_clean() {  # run_clean SCRIPT CODE TRIAL [VAR=VALUE ...]: another pilot's controller (CODE/SCRIPT) in this allocation,
  # from a clean environment plus the VAR=VALUEs (e.g. STEPS=40 SPLIT_ROLLOUT=3): this pilot's exports (its OCTOPUS_* limits)
  # would otherwise become that one's episode settings
  local keep; mapfile -t keep < <(env | grep -E '^(SLURM_[A-Z0-9_]*|CUDA_VISIBLE_DEVICES|HOME|USER|LOGNAME|LANG|TERM)=')
  exec env -i "${keep[@]}" PATH=/usr/local/bin:/usr/bin:/bin "${@:4}" bash "$2/$1" "$2" "$3"
}
up() { local i; for i in $(seq "$3"); do curl -sf -m 5 "$1" > /dev/null && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 10; done; return 1; }
judge_start() {  # judge_start HOST LOG: the search judge (JUDGE_MODEL, served as "judge") on JUDGE_GPU at HOST:JUDGE_PORT, its own
  # process group (setsid; JP = its leader): search.sbatch's local judge and serve_rollout's
  CUDA_VISIBLE_DEVICES=$JUDGE_GPU PATH="$(dirname "$SGLANG_PY"):$CUDA_HOME/bin:$PATH" CUDA_HOME=$CUDA_HOME SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_CACHE_DIR="$STATE/sglang-cache-judge" setsid "$SGLANG_PY" -m sglang.launch_server --model-path "$JUDGE_MODEL" --served-model-name judge \
    --host "$1" --port "$JUDGE_PORT" --tp-size 1 --mem-fraction-static 0.85 --context-length 32768 --attention-backend triton \
    --sampling-backend pytorch --disable-prefill-cuda-graph > "$2" 2>&1 < /dev/null & JP=$!
}
serve_rollout() {  # serve_rollout P [judge]: a rollout job's hold task (search_launch.sh): rl_run.sh serve (INFER_GPUS at
  # SERVE_HOST) and, with "judge", the judge on JUDGE_GPU at SERVE_HOST:JUDGE_PORT (the trainer's JUDGE_HOST); then sleeps.
  # search_stop_at.sh ends exactly P/judge-JOBID.pid's group and P/rollout-JOBID.pid (this shell, then `sleep infinity`)
  local JP=""
  [ "${2:-}" = judge ] && { judge_start "$SERVE_HOST" "$1/judge-$SLURM_JOB_ID.log"; echo "$JP" > "$1/judge-$SLURM_JOB_ID.pid"; }
  bash "$CODE/scripts/rl_env/rl_run.sh" serve && { [ -z "$JP" ] || { up "http://$SERVE_HOST:$JUDGE_PORT/v1/models" "$JP" 180 && say "judge up"; }; } \
    || { [ -n "$JP" ] && kill -- "-$JP"; return 1; }  # the servers: the hold's stop_stage
  echo $BASHPID > "$1/rollout-$SLURM_JOB_ID.pid"; exec sleep infinity
}
adapter() { ls -d "$FILEROOT/checkpoints/$USER/rq1-rl/$TRIAL/default/"*globalstep$(($1 - 1)) 2>/dev/null | head -1; }
final_adapter() {  # final_adapter K: prints the step-K adapter dir of a stopped TRIAL: the saver's, else made from that
  # step's recover checkpoint (the saver skips steps off saver.freq_steps) by dcp_adapter.py as CK/final-stepK
  local ck=$FILEROOT/checkpoints/$USER/rq1-rl/$TRIAL A g like; A=$(adapter "$1")
  [ -f "$A/adapter_model.safetensors" ] && { echo "$A"; return 0; }
  g=$ck/checkpoint_generations/generation_step$(printf %08d $(($1 - 1)))/payloads/default
  [ -d "$g" ] || { echo "no recover checkpoint of step $1 in $ck/checkpoint_generations" >&2; return 1; }
  like=$(ls -d "$ck"/default/*globalstep* 2>/dev/null | tail -1)
  "$AREAL_PY" "$C/scripts/rl_analysis/dcp_adapter.py" "$g" "${like:-$ADAPTER0}" "$ck/final-step$1" >&2 && echo "$ck/final-step$1"
}
owner_running() {  # owner_running JOBID: 0 unless sacct names a final state (an sacct error or no answer counts as running)
  local st; st=$(/cm/shared/apps/slurm/current/bin/sacct -n -X -P -j "$1" -o State 2>/dev/null) && [ -n "$st" ] || return 0
  case $st in RUNNING*|PENDING*|REQUEUE*|SUSPENDED*|RESIZING*|CONFIGURING*|COMPLETING*) return 0;; esac; return 1
}
train_to() {  # train_to N: trains (or resumes) TRIAL until its step-N adapter is saved (N a multiple of saver.freq_steps);
  # first waits while another running Slurm job owns STATE (STATE/owner: a continuation job queued beside the first), and
  # refuses a trial whose batch / saver interval (P/train-settings, written at its first start, with its steps then) differ
  [ -f "$(adapter "$1")/adapter_model.safetensors" ] && return 0
  local want o
  want="batch=$(grep -oE 'train_dataset\.batch_size=[0-9]+' <<< "$EXTRA" | cut -d= -f2) saver=$(grep -oE 'saver\.freq_steps=[0-9]+' <<< "$EXTRA" | cut -d= -f2) steps=$1"
  [ -f "$P/train-settings" ] || echo "$want" > "$P/train-settings"
  [ "$(cut -d' ' -f1,2 "$P/train-settings")" = "$(cut -d' ' -f1,2 <<< "$want")" ] \
    || { say "trial $TRIAL was started with '$(cat "$P/train-settings")', not '$want': refused"; return 1; }
  while o=$(cat "$STATE/owner" 2>/dev/null) && [ "$o" != "$SLURM_JOB_ID" ] && owner_running "$o"; do  # another job trains this STATE
    say "waiting: job $o owns $STATE"; sleep 300 & BG=$!; wait $BG; BG=""; done
  echo "$SLURM_JOB_ID" > "$STATE/owner"
  STEPS=$1 bash "$C/scripts/rl_env/rl_run.sh" start "$TRIAL" || return 1
  bgrun "$STATE" /dev/null tail --pid="$(cat "$STATE/trainer.pid")" -f /dev/null  # the trainer exits after step N
  bash "$C/scripts/rl_env/rl_run.sh" stop > /dev/null
  [ -f "$(adapter "$1")/adapter_model.safetensors" ] || { say "no step-$1 adapter ($STATE/train-$TRIAL.log)"; return 1; }
  say "step-$1 adapter $(adapter "$1")"
}
hold() {  # until the time limit: each P/queue/*.sh (bash run in a subshell with these helpers and the job's val, e.g.
  # `train_to 40 && val v40 "$(adapter 40)"`) in name order, .running-JOBID while it runs (a later job runs it again
  # when that job ended first), .done / .failed after, output in .log
  mkdir -p "$P/queue"; say "holding the allocation until the time limit; tasks: $P/queue/*.sh"
  local t rc
  for t in "$P"/queue/*.sh.running-*; do  # cut off by an earlier job's end (time limit): runs again here
    # (only when that job has ended: two jobs may hold on one queue, e.g. a Search trial's)
    [ -f "$t" ] && [ "${t##*.running-}" != "$SLURM_JOB_ID" ] && ! owner_running "${t##*.running-}" && mv "$t" "${t%.running-*}" \
      && say "task ${t%.running-*} again"
  done
  while true; do
    for t in "$P"/queue/*.sh; do
      [ -f "$t" ] || continue
      # renamed first: a task that execs a newer controller (takeover, same job id) must not run again in that one's hold
      mv "$t" "$t.running-$SLURM_JOB_ID"; say "task $t"
      (source "$t.running-$SLURM_JOB_ID") > "$t.log" 2>&1 & BG=$!; wait $BG; rc=$?; BG=""; stop_stage
      mv "$t.running-$SLURM_JOB_ID" "$t.$([ $rc = 0 ] && echo done || echo failed)"; say "task $t exit $rc"
    done
    sleep 60 & BG=$!; wait $BG; BG=""
  done
}
