#!/bin/bash
# One held-out eval step of an RL run on one lane, run on node-67 by pool.py: copies the step's adapter to the lane host,
# starts SGLang there (eval_serve.sh, one server per GPU, the lane's own env file and state dir), tunnels the ports to
# node-67 and runs octopus.eval.dev_eval --run $E/run into $E/results/step$N.jsonl (a rerun resumes it). Servers are stopped
# and the claim results/step$N.lockdir removed on exit; success writes step$N.done, failure appends to step$N.failed.
# Env from pool.py: E (eval dir), N (step), K (samples per task; 0 = main pass already done), KS (0, or K_SMALL from run.env:
# then a second pass runs the 1-3-task bundles in $E/tasks-small with k=KS into results/step$N-small.jsonl, own .done), SRC (adapter dir on node-94, or zero = base adapter),
# LANE (lane id), GH (ssh target of the host; run from HOP = node-94 when set), RH (lane root there), SERVE (eval_serve
# script there), BASEENV (host env file there: SGLANG_PY MODEL_VL CUDA_HOME), GPUS, PB (port base), PAR (episodes per
# server). The run: $E/run, the trial's config.yaml and episode_settings.json (pool.py --trial copies them from node-94): its
# episodes' settings, adapter name and proxy (dev_eval --run), plus rl.plan_control=true on both passes (the no-fork
# control: summary.py's M2 / M4; training never runs it); the adapter is served as <gconfig.lora_name>-v<step>
# (v0: the zero adapter). Host settings from $E/run.env: C (code dir on node-67), A (node-67's AReaL overlay built from C's
# patches/areal-{logprobs,cursor,route-spread}.patch, as rl_run.sh's $CURSOR/areal less the trainer-only cpu-batch and train-rpc-timeout:
# dev_eval runs the proxy in-process),
# TOK (the policy's tokenizer dir on node-67), K_SMALL (read by pool.py). DEV_EVAL_ARGS: extra dev_eval arguments of the
# main pass, e.g. "--offset 30" to run part of a step's tasks as its own N, or "--env OCTOPUS_READ_EVENT=1" for an arm.
set -u
R=/home/user/octo-env; H=/home/user/harbor-udocker; G94=user@10.0.1.94
source $E/run.env
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
# node-94's sshd drops ~30-40% of new handshakes when public scanners fill its MaxStartups (10:30:100) slots: an ssh
# that fails to connect exits 255, so retry it (every remote step here is idempotent)
retry() { local i rc; for i in 1 2 3 4 5 6; do "$@"; rc=$?; [ $rc = 255 ] || return $rc; sleep $((i * 5)); done; return $rc; }
rcmd1() { if [ -n "${HOP:-}" ]; then $SSH $HOP "$SSH $GH $(printf %q "$1")"; else $SSH $GH "$1"; fi; }
rcmd() { retry rcmd1 "$1"; }  # never with piped stdin: a retry would read an already consumed pipe (copy() retries whole)
RUN=$(basename $E); LENV=$RH/eval-lane-$LANE.env; ST=$RH/eval-state-lane-$LANE; ADH=$RH/adapters/$RUN-step$N
log() { echo "$(date '+%F %T') step$N [L$LANE ${HOST:-$GH}] $*" | tee -a $E/logs/driver.log; }
TP=""
cleanup() { [ -n "$TP" ] && kill $TP 2>/dev/null; rcmd "RL_ENV_FILE=$LENV bash $SERVE stop" > /dev/null 2>&1; rmdir $E/results/step$N.lockdir 2>/dev/null; }
fail() { log "FAILED: $1"; echo "$(date '+%F %T') L$LANE: $1" >> $E/results/step$N.failed; exit 1; }  # rows kept: a rerun resumes them
trap cleanup EXIT
mkdir -p $E/logs
[ -n "${A:-}" ] && [ -n "${TOK:-}" ] || fail "run.env needs A (AReaL overlay) and TOK (tokenizer dir)"
[ -f $E/run/episode_settings.json ] || fail "no $E/run/episode_settings.json (pool.py --trial copies it)"
LORA=$($R/areal/.venv/bin/python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["gconfig"]["lora_name"])' \
  $E/run/config.yaml) || fail "no run config $E/run/config.yaml (pool.py --trial copies it)"

rcmd "mkdir -p $ST/tmp && { cat $BASEENV; echo STATE=$ST; echo export TMPDIR=$ST/tmp; } > $LENV" || fail "lane env on $GH failed"
untar="mkdir -p $ADH && tar -C $ADH -x"
copy() {
  if [ "$SRC" = zero ]; then NAME=$LORA-v0; set -o pipefail; tar -C $R/models/adapter-v0-4b-text-r64 -c . | rcmd1 "$untar"; local rc=$?; set +o pipefail; return $rc
  elif [ -n "${HOP:-}" ]; then NAME=$LORA-v$N; $SSH $HOP "tar -C $SRC -c . | $SSH $GH $(printf %q "$untar")"  # HOP is node-94
  else NAME=$LORA-v$N; set -o pipefail; $SSH $G94 "tar -C $SRC -c ." | rcmd1 "$untar"; local rc=$?; set +o pipefail; return $rc; fi
}
ok=0; for i in 1 2 3 4 5 6; do copy && { ok=1; break; }; sleep $((i * 5)); done  # a dropped handshake on either hop
[ $ok = 1 ] || fail "adapter copy $SRC -> $GH:$ADH failed"

SV=$(rcmd "RL_ENV_FILE=$LENV PORT_BASE=$PB LORA_RANK=64 timeout 2400 bash $SERVE start $GPUS '$NAME=$ADH' 2>&1 | tail -1")
case "$SV" in 127.0.0.1:*) ;; *) fail "server start failed: $SV";; esac
PORTS=$(echo "$SV" | tr ',' '\n' | sed 's/.*://')
FW=""; for p in $PORTS; do FW="$FW -L 127.0.0.1:$p:127.0.0.1:$p"; done
KA="-o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
for i in 1 2 3 4 5 6; do  # a dropped handshake ends the tunnel ssh at once: start it again
  if [ -n "${HOP:-}" ]; then  # node-94 forwards to the host through its reverse-tunnelled sshd; -tt: the inner ssh dies with ours
    $SSH -tt $KA $FW $HOP "$SSH -N $KA $FW $GH" < /dev/null > /dev/null 2>&1 & TP=$!
  else
    $SSH -N $KA $FW $GH & TP=$!
  fi
  sleep 5; kill -0 $TP 2>/dev/null && break; TP=""; sleep $((i * 5))
done
[ -n "$TP" ] || fail "tunnel failed"
for p in $PORTS; do curl -sf -m 10 127.0.0.1:$p/v1/models > /dev/null || fail "port $p not reachable"; done
log "servers $SV (GPUs $GPUS, PAR $PAR, k $K, k_small ${KS:-0}) via tunnel pid $TP, adapter $NAME=$ADH"

cd $C
evalpass() {  # $1 results/ stem, $2 k, $3 tasks dir, rest: extra dev_eval args
  local o=$1 k=$2 td=$3 rc rows errs; shift 3
  env RL_ENV_FILE=$R/pv1-base/eval.env OCTOPUS_CODE=$C \
    UDOCKER_OVERLAY=$R/forkpay/bin/fuse-overlayfs PYTHONPATH=$A:$C LITELLM_LOCAL_MODEL_COST_MAP=True \
    $R/areal/.venv/bin/python -m octopus.eval.dev_eval --servers "$SV" --lora "$NAME" --k $k --parallel $PAR \
    --run $E/run --tokenizer $TOK --tasks-dir $td --out $E/results/$o.jsonl --jobs-dir $H/rl/jobs-$RUN-$o-L$LANE \
    rl.plan_control=true "$@" \
    >> $E/logs/$o.log 2>&1 < /dev/null
  rc=$?
  # an episode's last row counts: dev_eval runs an error row's episode again
  read rows errs < <($R/areal/.venv/bin/python -c 'import json, sys
last = {(r["task_dir"], r["sample"]): r for r in map(json.loads, open(sys.argv[1]))}
print(len(last), sum("error" in r for r in last.values()))' $E/results/$o.jsonl 2>/dev/null || echo 0 0)
  [ $rc = 0 ] || fail "$o dev_eval rc=$rc rows=$rows"
  [ $(( ${errs:-0} * 10 )) -le $rows ] || { mv $E/results/$o.jsonl $E/results/$o.jsonl.failed-$(date +%s); fail "$o rows=$rows error rows=$errs > 10%, moved aside"; }
  echo "k=$k rc=$rc rows=$rows errors=$errs" > $E/results/$o.done
  log "$o finished k=$k rc=$rc rows=$rows error rows=$errs"
}
[ $K = 0 ] || evalpass step$N $K $E/tasks-eval-v3 ${DEV_EVAL_ARGS:-}
[ ${KS:-0} = 0 ] || evalpass step$N-small $KS $E/tasks-small
log "eval finished; servers stopped"
