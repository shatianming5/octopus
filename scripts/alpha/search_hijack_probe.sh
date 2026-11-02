#!/bin/bash
# search_hijack_probe.sh CODE UNITS OUT NAME=ADAPTER [NAME=ADAPTER ...]: Measure 2 of docs/search-read-hijack.md, the
# counterfactual read probe, as a queue task of a job that holds 4 GPUs (lib.sh hold, as search_eval.sh tasks run).
#   GPU 0: its own judge (JUDGE_MODEL served as "judge", lib.sh judge_start) on 127.0.0.1:HIJACK_JUDGE_PORT (32510), never a
#     training run's judge; a busy port stops the task.
#   GPUs 1-3: mount-capable eval servers (eval_serve.sh on this code's cursor overlay CURSOR, OCTOPUS_MOUNT_LAYERS unset as in
#     the v6d/v8 runs) with every adapter loaded under its NAME; the first NAME is the report's reference checkpoint.
# Then read_hijack_probe run (N mount points, default 300, SAMPLES per condition, default 4, from UNITS: read_hijack
# extract's output; the trees it names must be readable here), the eval servers stop (GPUs 1-3 free), read_hijack judge,
# and read_hijack_probe report > OUT/report.md. M1_LABELS=FILE (optional): the same judge also labels UNITS for Measure
# 1 into FILE, after the probe's items (read_hijack report then runs on the login node). OUT keeps items.jsonl, labels.jsonl and the logs. Whatever this script
# started is stopped at exit (the hold's stop_stage stops eval servers too). Eval servers of another run on this host are
# never stopped: the script refuses to start then.
# Usage, as a queue task (env -i; CUDA_VISIBLE_DEVICES lists the job's 4 GPUs):
#   exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID \
#     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID bash ROOT/code-SHA/scripts/alpha/search_hijack_probe.sh \
#     ROOT/code-SHA UNITS OUT rq1-v0=ADAPTER0 rq1-v55=.../final-step55 rq1-v8x=.../weight_update_vK
set -uo pipefail
C=$1; UNITS=$2; OUT=$3; shift 3
[ $# -ge 1 ] || { echo "want NAME=ADAPTER_DIR ..."; exit 1; }
export RL_ENV_FILE=$C/scripts/alpha/search.env
source "$RL_ENV_FILE"; source "$C/scripts/alpha/lib.sh"
mkdir -p "$OUT"
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"; [ ${#G[@]} = 4 ] || { say "want 4 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }
D=$STATE/eval-serve-$(hostname -s); ES=0; JP=""
for p in $(cat "$D"/*.pid 2>/dev/null); do  # eval_serve.sh start stops every server of this host and STATE: refuse instead
  kill -0 -- "-$p" 2>/dev/null && { say "eval servers of another run are up in $D: not starting over them"; exit 1; }; done
cleanup() {
  [ "$ES" = 1 ] && bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null 2>&1
  [ -n "$JP" ] && { kill -TERM -- "-$JP" 2>/dev/null; for _ in $(seq 60); do kill -0 "$JP" 2>/dev/null || break; sleep 1; done
    kill -9 -- "-$JP" 2>/dev/null; }
  say "exit; GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM
JUDGE_PORT=${HIJACK_JUDGE_PORT:-32510}; URL=http://127.0.0.1:$JUDGE_PORT/v1
curl -sf -m 5 "$URL/models" > /dev/null && { say "port $JUDGE_PORT answers already: not our judge"; exit 1; }
JUDGE_GPU=${G[0]} judge_start 127.0.0.1 "$OUT/judge-server-$SLURM_JOB_ID.log"
up "$URL/models" "$JP" 180 || { say "judge down"; exit 1; }
ES=1  # any eval server of this start is stopped at exit; the guard above found none of another run
S=$(timeout 1800 bash "$C/scripts/rl_env/eval_serve.sh" start "${G[1]},${G[2]},${G[3]}" "$@" | tail -1)
[[ $S == 127.0.0.1:* ]] || { say "eval_serve: $S"; exit 1; }
NAMES=$(printf '%s\n' "$@" | cut -d= -f1 | paste -sd,)
say "probe $NAMES on $S, judge $URL"
bgrun "$C" "$OUT/probe.log" env PYTHONPATH="$C" "$AREAL_PY" -m octopus.tools.read_hijack_probe run --tokenizer "$MODEL_TEXT" \
  --servers "$S" --lora "$NAMES" --n "${N:-300}" --samples "${SAMPLES:-4}" "$UNITS" "$OUT/items.jsonl" || { say "probe failed"; exit 1; }
bash "$C/scripts/rl_env/eval_serve.sh" stop > /dev/null 2>&1; ES=0
bgrun "$C" "$OUT/judge.log" env PYTHONPATH="$C" "$AREAL_PY" -m octopus.tools.read_hijack judge --url "$URL" \
  "$OUT/items.jsonl" "$OUT/labels.jsonl" || { say "judge left invalid labels ($OUT/judge.log)"; exit 1; }
if [ -n "${M1_LABELS:-}" ]; then
  bgrun "$C" "$OUT/judge-m1.log" env PYTHONPATH="$C" "$AREAL_PY" -m octopus.tools.read_hijack judge --url "$URL" \
    "$UNITS" "$M1_LABELS" || { say "Measure 1 judge left invalid labels ($OUT/judge-m1.log)"; exit 1; }
fi
bgrun "$C" "$OUT/report.md" env PYTHONPATH="$C" "$AREAL_PY" -m octopus.tools.read_hijack_probe report "$OUT/items.jsonl" \
  "$OUT/labels.jsonl"; rc=$?
say "done exit $rc: $OUT/report.md"; exit $rc
