#!/bin/bash
# search_eval.sh CODE NAME ADAPTER SET RUN [K] [LINEAR] [LABEL]: one Search eval of one adapter on Cluster Beta: scripts/alpha/
# search_eval.sh with Beta's stack (scripts/beta/search.env: ROOT = .../beta/octopus, its venvs, CUDA_HOME = ROOT/cuda-13.0
# for SGLang's JIT kernels, libcudart.so.13 preloaded by the SGLang venv; install_env.sh), and a clean environment.
# Everything else as Alpha's (EVAL_PARALLEL, default 32, an upper bound under OCTOPUS_ROUTE_KV and OCTOPUS_ROUTE_CAP):
#   NAME: v<N> (the LoRA name rq1-NAME; proxy.py parses rsplit("-v") + int). ADAPTER: the adapter dir.
#   SET val: the OpenResearcher validation questions (500); corpus openresearcher, port SEARCH_PORT (32600); the settings and the
#     output of scripts/alpha/val_search.sh and search.sbatch's val() (k=1 there; K here). Full log: OUT/[LABEL-]val-NAME-MODETAG-kK.log.
#   SET bcp: the 830 BrowseComp-Plus queries, decrypted in memory only; corpus browsecomp-plus, port 32601. Summary lines only
#     persist: OUT/[LABEL-]test-NAME-MODETAG-kK.log; the full log stays on /dev/shm with the trials, removed at exit.
#   LABEL: a prefix for the output names (two adapters can share NAME v40: LABEL ext40 or v5-40, the LoRA name is unchanged).
#   K: samples per question (default 1; the external baseline protocol uses 2). LINEAR=1: a single agent (OCTOPUS_LINEAR=1, no
#   read tool); the call cap of the main thread is the caller's OCTOPUS_MAIN_MAX_CALLS (env, read by octopus/runtime/settings.py).
#   Rows carry linear and main_max_calls (search_dev_eval.py), and per question: calls, completion tokens, wall_s.
# Both: RUN (required): the evaluated run's settings dir, ROOT/search/pilot/TRIAL/run (search_dev_eval applies its
#   episode_settings.json; no default: another run's would silently change the protocol). The judge (JUDGE_MODEL: Qwen3.5-27B, JUDGE_PORT,
# started here only when none answers at OCTOPUS_JUDGE_URL) grades at temperature 0; the policy samples at temperature 0.7,
# top_p 0.95 (search_dev_eval: search rl.EVAL_SAMPLING), replies capped at 4096 tokens. Eval servers on GPUs 1-3.
# Guard: eval servers of another run on this host and STATE are never stopped; this script stops only the ones it started
# (ES=1 before its own start: a failed start is cleaned up too). Anything it started (judge, service, eval, /dev/shm copy) is stopped at exit.
# Usage, as a queue task of a held Beta job (scripts/beta/hold.sbatch; it re-runs itself in a clean environment):
#   exec bash ROOT/code-SHA/scripts/beta/search_eval.sh ROOT/code-SHA v40 ADAPTER bcp ROOT/search/pilot/TRIAL/run 2 0 ext40
[ -n "${SEARCH_EVAL_CLEAN:-}" ] || exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  SLURM_JOB_ID="$SLURM_JOB_ID" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  ${OCTOPUS_MAIN_MAX_CALLS:+OCTOPUS_MAIN_MAX_CALLS=$OCTOPUS_MAIN_MAX_CALLS} ${OCTOPUS_THREAD_BUDGET:+OCTOPUS_THREAD_BUDGET=$OCTOPUS_THREAD_BUDGET} \
  ${EVAL_PARALLEL:+EVAL_PARALLEL=$EVAL_PARALLEL} ${OCTOPUS_ROUTE_CAP+OCTOPUS_ROUTE_CAP=$OCTOPUS_ROUTE_CAP} \
  ${OCTOPUS_ROUTE_KV+OCTOPUS_ROUTE_KV=$OCTOPUS_ROUTE_KV} ${OCTOPUS_ROUTE_WAIT+OCTOPUS_ROUTE_WAIT=$OCTOPUS_ROUTE_WAIT} \
  ${OCTOPUS_ROUTE_TREE_TOKENS+OCTOPUS_ROUTE_TREE_TOKENS=$OCTOPUS_ROUTE_TREE_TOKENS} \
  SEARCH_EVAL_CLEAN=1 bash "$0" "$@"  # not the shared home's profile (Alpha's PYTHONPATH, TRITON_CACHE_DIR, SLURM_*)
set -uo pipefail
C=$1; NAME=$2; ADAPTER=$3; SET=${4:-}; RUN=${5:-}; K=${6:-1}; LINEAR=${7:-0}; LABEL=${8:-}
[[ $NAME =~ ^v[0-9]+$ ]] || { echo "NAME must be v<N> (proxy.py rsplit('-v')), got '$NAME'"; exit 1; }
[[ $K =~ ^[1-9][0-9]*$ ]] || { echo "K must be a positive integer, got '$K'"; exit 1; }
[[ ${EVAL_PARALLEL:-32} =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_PARALLEL must be a positive integer, got '$EVAL_PARALLEL'"; exit 1; }
case $SET in val|bcp) ;; *) echo "SET must be val or bcp, got '$SET'"; exit 1 ;; esac
case $RUN in ''|-) echo "RUN (5th argument) must be the evaluated run's settings dir, ROOT/search/pilot/TRIAL/run, got '$RUN'"; exit 1 ;; esac
export RL_ENV_FILE=$C/scripts/beta/search.env
source "$RL_ENV_FILE"; source "$C/scripts/alpha/lib.sh"
export OCTOPUS_ROUTE_CAP=${OCTOPUS_ROUTE_CAP:-32}  # the eval's hard cap per server (rl_run.sh sets training's)
if [ "$SET" = val ]; then CORPUS=openresearcher; SPORT=$SEARCH_PORT; PFX=val; else CORPUS=browsecomp-plus; SPORT=32601; PFX=test; fi
[ -f "$RUN/episode_settings.json" ] || { echo "RUN $RUN has no episode_settings.json"; exit 1; }
# the arm is passed as --env (search_dev_eval, config.py run_episode_settings): the process environment must not carry any
# per-arm episode setting (search.env exports READ_TOOL=1, REWIND_TOOL=1 and REWIND_MAX=2; the run's own settings supply
# them)
CAP_IN=${OCTOPUS_MAIN_MAX_CALLS:-}; TB_IN=${OCTOPUS_THREAD_BUDGET:-}
unset OCTOPUS_LINEAR OCTOPUS_READ_TOOL OCTOPUS_REWIND_TOOL OCTOPUS_REWIND_MAX OCTOPUS_MAIN_MAX_CALLS OCTOPUS_THREAD_BUDGET
ARM=(); if [ "$LINEAR" = 1 ]; then
  ARM=(--env OCTOPUS_LINEAR=1 --env OCTOPUS_READ_TOOL=); MODETAG=linear-c${CAP_IN:-30}-t${TB_IN:-default}
  [ -n "$CAP_IN" ] && ARM+=(--env OCTOPUS_MAIN_MAX_CALLS=$CAP_IN)
  [ -n "$TB_IN" ] && ARM+=(--env OCTOPUS_THREAD_BUDGET=$TB_IN)
else MODETAG=fork; fi
TAG=${LABEL:+$LABEL-}$NAME-$MODETAG-k$K
OUT=$PILOT/eval; mkdir -p "$OUT"; SHM=/dev/shm/$USER-eval-$SLURM_JOB_ID; mkdir -p "$SHM"
[ "$SET" = val ] && LOGS=$OUT || LOGS=$SHM  # val: judge and service logs persist too (no decrypted text); bcp: /dev/shm only
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"; [ ${#G[@]} = 4 ] || { say "want 4 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }
D=$STATE/eval-serve-$(hostname -s); ES=0; JG=""; SP=""; BG=""
for p in $(cat "$D"/*.pid 2>/dev/null); do  # eval_serve.sh start stops every server of this host and STATE: refuse instead
  kill -0 -- "-$p" 2>/dev/null && { say "eval servers of another run are up in $D: not starting over them"; exit 1; }; done
up() { local i; for i in $(seq "$3"); do curl -sf -m 5 "$1" > /dev/null && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 10; done; return 1; }  # as search.sbatch
stopjob() { [ -n "$1" ] && { kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 1; done; kill -9 -- "-$1" 2>/dev/null; kill -9 "$1" 2>/dev/null; }; }
cleanup() {  # the eval first (SIGTERM: search_dev_eval's finally removes its trials), then what this script started, then /dev/shm
  [ -n "$BG" ] && { kill -TERM "$BG" 2>/dev/null; for _ in $(seq 120); do kill -0 "$BG" 2>/dev/null || break; sleep 1; done; kill -9 "$BG" 2>/dev/null; }
  BG=""; [ "$ES" = 1 ] && bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null 2>&1
  stopjob "$SP"; stopjob "$JG"
  rm -rf "$SHM"; say "exit; GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM
if ! curl -sf -m 10 "$OCTOPUS_JUDGE_URL/models" > /dev/null; then  # the judge as search.sbatch serves it
  CUDA_VISIBLE_DEVICES=${G[0]} PATH="$(dirname "$SGLANG_PY"):$CUDA_HOME/bin:$PATH" CUDA_HOME=$CUDA_HOME SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_CACHE_DIR="$STATE/sglang-cache-judge" setsid "$SGLANG_PY" -m sglang.launch_server --model-path "$JUDGE_MODEL" --served-model-name judge \
    --host 127.0.0.1 --port "$JUDGE_PORT" --tp-size 1 --mem-fraction-static 0.85 --context-length 32768 --attention-backend triton \
    --sampling-backend pytorch --disable-prefill-cuda-graph > "$LOGS/judge-$TAG-$SLURM_JOB_ID.log" 2>&1 < /dev/null & JG=$!
  up "$OCTOPUS_JUDGE_URL/models" $JG 180 || { say "judge down"; exit 1; }
fi
if ! curl -sf -m 5 "http://127.0.0.1:$SPORT/health" | grep -q "\"corpus\": \"$CORPUS\""; then  # this set's corpus, not another's
  mkdir -p "$SHM/search-scratch" && cp -r "$SEARCH_DATA/search-scratch/$CORPUS" "$SHM/search-scratch/" || { say "copy to $SHM failed"; exit 1; }
  (cd "$C" && exec setsid "$AREAL_PY" -m octopus.envs.search.service serve --corpus "$CORPUS" --root "$SHM" --backend bm25 \
    --tokenizer "$OCTO_ENV/models/Qwen3-0.6B-tokenizer" --port "$SPORT" --workers 8) > "$LOGS/service-$TAG-$SLURM_JOB_ID.log" 2>&1 < /dev/null & SP=$!
  up "http://127.0.0.1:$SPORT/health" $SP 120 || { say "$CORPUS service down"; exit 1; }
fi
say "$SET $TAG ($ADAPTER): judge and $CORPUS service on $SPORT up; arm ${ARM[*]:-fork}"
ES=1  # any eval server of this start is stopped at exit; the guard above found none of another run
S=$(timeout 1800 bash "$C/scripts/rl_env/eval_serve.sh" start "${G[1]},${G[2]},${G[3]}" "rq1-$NAME=$ADAPTER" | tail -1)
[[ $S == 127.0.0.1:* ]] || { say "eval_serve: $S"; exit 1; }
say "$SET $TAG on $S"
if [ "$SET" = val ]; then
  DS="--set validation --dataset-dir $SEARCH_DATA/datasets/OpenResearcher__OpenResearcher-Corpus-Gold-Doc --jobs-dir $OUT/jobs-val-$TAG"; LOG=$OUT/val-$TAG.log
else
  DS="--set browsecomp-plus --dataset-dir $SEARCH_DATA/datasets/Tevatron__browsecomp-plus"; LOG=$SHM/eval.log
fi
bgrun "$C" "$LOG" env OCTOPUS_SEARCH_URL=http://127.0.0.1:$SPORT PYTHONPATH="$CURSOR/areal:$C" "$AREAL_PY" -m octopus.eval.search_dev_eval \
  $DS "${ARM[@]}" --run "$RUN" --servers "$S" --lora "rq1-$NAME" --tokenizer "$MODEL_TEXT" --judge-url "$OCTOPUS_JUDGE_URL" --judge-model judge \
  --k "$K" --parallel "${EVAL_PARALLEL:-32}" --out "$OUT/$PFX-$TAG.jsonl"; rc=$?
[ "$SET" = bcp ] && grep -E '^(browsecomp-plus: |n=[0-9]+ accuracy|infra failures)' "$LOG" > "$OUT/$PFX-$TAG.log"  # summary lines only
say "$SET $TAG exit $rc: $(tail -n 1 "$OUT/$PFX-$TAG.log" 2>/dev/null)"; exit $rc
