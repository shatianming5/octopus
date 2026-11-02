#!/bin/bash
# search_launch.sh SHA TRIAL TJ RJ IP: deploy SHA to Alpha and start a fresh Search TRIAL over two held jobs (layout A):
# RJ = 4 rollout servers (32300-32303, 8 groups each) at IP (RJ's node: `srun --jobid=RJ --overlap -n1 -c1 hostname -I`),
# TJ = judge G0 + trainer G1-3 (SPLIT_ROLLOUT=0, EXTRA_SERVERS); BATCH=16 STEPS=190 SAVE_EVERY=10, env scripts/alpha/search.env.
# search_launch.sh SHA TRIAL TJ RJ IP [OCTOPUS_X=VALUE ...]: episode settings (e.g. OCTOPUS_FORK_DEPTH=0) exported for the settings
# step (P/run/episode_settings.json) and passed on the trainer task's run_clean line. EPISODES_PER_SERVER (default 8):
# the rollout task's and the trainer's.
# JUDGE_ON_RJ=1: RJ = judge G0 (IP:32500) + 3 rollout servers (32300-32302), TJ = trainer G0-3 (JUDGE_HOST=IP).
# Servers outside these jobs, e.g. on Cluster Beta (scripts/beta/serve_rollout.sh), started fresh for this trainer:
# EXTRA_ROLLOUT="host:port,..." adds rollout servers to the trainer's EXTRA_SERVERS; JUDGE_HOST=H uses the judge at
# H:32500 (search.env JUDGE_PORT), so TJ starts none and trains on all 4 GPUs. RJ = TJ (IP -): no Alpha rollout job
# (no rollout task, P/jobs "TJ TJ"), the rollout is EXTRA_ROLLOUT alone; give the guards JUDGE_LOG=<the judge's log>
# and the step-3 check SERVER_LOGS=<the remote servers' log dirs>.
# The 8 groups per server are an upper bound: search.env's OCTOPUS_ROUTE_KV admits a new episode only to a server below
# that KV usage (docs/search-env.md "Admission"), so it may be set higher (16) after a pilot (route peak lines).
# Both jobs hold (search.sbatch, ROOT/search/HOLD) on ROOT/search/pilot/hold-JOB/queue; the tasks: RJ 01-rollout-TRIAL.sh, TJ
# 20-TRIAL.sh. P=ROOT/search/pilot/TRIAL/jobs records "TJ RJ" for the guards (more rollout jobs may follow: "TJ RJ J2 ...",
# each with an 01-rollout-TRIAL.sh task as RJ's and its servers in EXTRA_SERVERS). Run from the lab host (repo checkout).
# Full run, in this order (the guards run ON Alpha from the deployed code, `nohup setsid ... < /dev/null`, one log each):
#   1. bash scripts/alpha/search_launch.sh SHA TRIAL 145533 146854 10.0.4.28            (lab host)
#   2. C=ROOT/code-SHA/scripts/alpha                                                         (Alpha, right after 1)
#      bash $C/search_judge_gate.sh TRIAL      # exit 1: the judge failed: bash $C/search_stop_at.sh TRIAL 0
#      bash $C/search_step3_check.sh TRIAL     # PASS: the run continues; else stops it at the next step boundary
#      bash $C/search_epoch_stop.sh TRIAL $(date -d '2026-10-24 21:45' +%s) [BCP_K=1]
#   epoch_stop: one pass over the train qids -> stop -> Val 500 k=1 and BCP 830 k=BCP_K queued on RJ; at DEADLINE (epoch s,
#   0 = none) it stops at the next step boundary instead, recover checkpoint kept, no eval. Every guard exits once
#   P/stopped exists (search_stop_at.sh), TJ has ended (sacct) or no longer runs 20-TRIAL.sh, so none can stop a later
#   resume. Resuming TRIAL (not done by this script, which wants a fresh one): move P/stopped away, rewrite P/jobs if the jobs
#   changed, and keep the trainer task named 20-TRIAL.sh, or the guards exit at once.
#   Restart a stopped trainer's rollout servers (RJ's task, EXTRA_ROLLOUT's) fresh, never unload adapters from them: requests
#   the trainer left in flight leak their LoRA count, and the unload (rl_run.sh start) hangs holding the LoRA lock (v9, 3x).
set -euo pipefail
SHA=${1:?SHA}; T=${2:?TRIAL}; TJ=${3:?trainer job}; RJ=${4:?rollout job or TJ}; IPB=${5:?IP of the rollout job or -}
shift 5; X="$*"; EP=${EPISODES_PER_SERVER:-8}; JH=${JUDGE_HOST:-}; XR=${EXTRA_ROLLOUT:-}
for a in "$@"; do [[ $a =~ ^OCTOPUS_[A-Z0-9_]+=[A-Za-z0-9_.,:/+-]*$ ]] || { echo "not an OCTOPUS_X=VALUE: $a"; exit 1; }; done
[[ $XR =~ ^([A-Za-z0-9.-]+:[0-9]+(,|$))*$ && $JH =~ ^[A-Za-z0-9.-]*$ ]] || { echo "EXTRA_ROLLOUT: host:port,...; JUDGE_HOST: a host"; exit 1; }
REPO=$(cd "$(dirname "$0")/../.." && pwd); R=/projects/cluster/octopus/octopus; N=$R/code-$SHA
JW=""; IG=0,1,2,3; NS=4; [ "${JUDGE_ON_RJ:-0}" = 1 ] && JW=judge IG=1,2,3 NS=3 JH=$IPB  # lib.sh serve_rollout's judge: on JUDGE_GPU (0)
[ "$RJ" != "$TJ" ] || { [ -z "$JW" ] || { echo "JUDGE_ON_RJ=1 needs a rollout job"; exit 1; }; NS=0; }
PS=$(seq -s ' ' 32300 $((32299 + NS))); ES=""; for p in $PS; do ES=$ES$IPB:$p,; done; ES=$ES$XR; ES=${ES%,}
[ -n "$ES" ] || { echo "no rollout servers: RJ = TJ needs EXTRA_ROLLOUT"; exit 1; }
QT=$R/search/pilot/hold-$TJ/queue; QR=$R/search/pilot/hold-$RJ/queue; ST=$R/search/rl-state-search; P=$R/search/pilot/$T
r() { ssh -n cluster-login "$@"; }
echo "== preconditions"
r "set -e; if compgen -G \"$QT/*.running-*\" || compgen -G \"$QR/*.running-*\"; then exit 1; fi; [ ! -e $P ]; grep -q Qwen3_5ForCausalLM $R/models/Qwen3.5-27B-text/config.json
  for j in $TJ $RJ; do srun --jobid=\$j --overlap -n1 -c1 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v j=\$j '\$1>2000{print \"job \" j \" GPU busy: \" \$1; bad=1} END{exit bad}'
    srun --jobid=\$j --overlap -n1 -c1 ss -ltnH | awk -v j=\$j '{n=split(\$4,a,\":\"); if (a[n]>=32300 && a[n]<=32699) {print \"job \" j \" port in use: \" a[n]; bad=1}} END{exit bad}'; done" \
  || { echo "precondition failed (a queue task is running, GPUs busy, or trial dir exists)"; exit 1; }
echo "== deploy $SHA"
if ! r "[ -d $N/octopus ]"; then
  git -C "$REPO" archive "$SHA" -- . ':!results' | ssh cluster-login "mkdir -p $N.tmp && tar x -C $N.tmp && mv $N.tmp $N"
fi
if ! r "[ -e $R/cursor-$SHA ]"; then
  git -C "$REPO" diff --quiet 508e235 "$SHA" -- patches || { echo "patches changed since 508e235: rebuild the cursor overlay (CURSOR=$R/cursor-$SHA scripts/nvwulf/install_env.sh ROOT BUNDLE_DIR CODE \"\") before launch"; exit 1; }
  r "ln -s cursor-508e235 $R/cursor-$SHA"
fi
r "grep -qE '^JUDGE_MODEL=.*models/Qwen3.5-27B-text' $N/scripts/alpha/search.env" || { echo "search.env of $SHA: JUDGE_MODEL is not models/Qwen3.5-27B-text"; exit 1; }
echo "== state: old trial's STATE files aside, owner $TJ, CODE $N"
r "set -e; o=$ST/old-\$(date +%m%d%H%M); mkdir -p \$o; for f in trial owner trainer.pid trainer.host judge-cache.jsonl; do if [ -f $ST/\$f ]; then mv $ST/\$f \$o/; fi; done; cp $R/search/CODE $R/search/CODE.pre-$T; cp $R/search/HOLD $R/search/HOLD.pre-$T; grep -vx $TJ $R/search/HOLD.pre-$T > $R/search/HOLD || true;  # a job id in HOLD makes search.sbatch (even via run_clean) only hold
    echo $N > $R/search/CODE; echo $TJ > $ST/owner; mkdir -p $P; touch $P/val-v0.done; echo '$TJ $RJ' > $P/jobs"
echo "== run settings dir (for Val/BCP of this trial)"
r "srun --jobid=$TJ --overlap -n1 -c2 env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin bash -c 'export RL_ENV_FILE=$N/scripts/alpha/search.env BATCH=16 SAVE_EVERY=10 CUDA_DEVICE_ORDER=PCI_BUS_ID $X; source \$RL_ENV_FILE; bash \$CODE/scripts/rl_env/rl_run.sh settings $P/run \$EXTRA' > $P/settings.log 2>&1; tail -n 2 $P/settings.log"
if [ "$RJ" != "$TJ" ]; then echo "== rollout servers on $RJ"
r "for x in log failed done; do [ -e $QR/01-rollout-$T.sh.\$x ] && mv $QR/01-rollout-$T.sh.\$x $QR/01-rollout-$T.sh.\$x-old-\$(date +%H%M%S); done; true;  cat > $QR/01-rollout-$T.sh <<'EOT'
exec env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=\$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES bash -c 'export RL_ENV_FILE=$N/scripts/alpha/search.env INFER_GPUS=$IG TRAIN_GPUS= SERVE_HOST=$IPB EPISODES_PER_SERVER=$EP CUDA_DEVICE_ORDER=PCI_BUS_ID; source \$RL_ENV_FILE; source \$CODE/scripts/alpha/lib.sh; serve_rollout $P $JW'  # search_stop_at.sh ends its exact pids
EOT"
until r "grep -q '^servers $IPB' $QR/01-rollout-$T.sh.log && { [ -z '$JW' ] || grep -q 'judge up' $QR/01-rollout-$T.sh.log; }" 2>/dev/null; do
  r "[ ! -e $QR/01-rollout-$T.sh.failed ]" || { echo "servers failed:"; r "tail -n 15 $QR/01-rollout-$T.sh.log"; exit 1; }; sleep 20; done; fi
for s in ${ES//,/ }; do echo "$s: $(r "srun --jobid=$TJ --overlap -n1 -c1 curl -s -m 10 http://$s/v1/models | python3 -c 'import json,sys; print([m[\"id\"] for m in json.load(sys.stdin)[\"data\"]])'")"; done
echo "== trainer task on $TJ"
r "cat > $QT/20-$T.sh <<'EOT'
echo \$BASHPID > $P/controller.pid  # the controller (run_clean execs): search_stop_at.sh ends it by this pid
source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/search.sbatch $N $T BATCH=16 STEPS=190 SAVE_EVERY=10 SPLIT_ROLLOUT=0 EPISODES_PER_SERVER=$EP EXTRA_SERVERS=$ES${JH:+ JUDGE_HOST=$JH}${X:+ $X}
EOT"
echo "queued $QT/20-$T.sh; log $QT/20-$T.sh.log, trial log $P"
