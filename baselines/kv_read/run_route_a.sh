#!/usr/bin/env bash
# Route A for one model on one host: one shard per GPU (resumable: a rerun skips the ids a shard file holds), then
# the shards merged and reported (baselines/README.md has the per-model commands).
#   MODEL=/models/Qwen3-4B OUT=runs/qwen3-4b-all ROOT_MAP="../data/e11=/data/e11 /mnt/disk4/user=/srv/z" \
#   [NGPUS=8 | GPUS="0 3 5"] [ADAPTER=lora_dir] [CONDITIONS=none,summary] [MOUNT_LAYERS=all|16-24] \
#   [PROBES=octopus/tools/kv_read_probe_set.jsonl] [REPORT_WITH="earlier/results.jsonl ..."] [PYTHON=python] \
#   bash baselines/kv_read/run_route_a.sh
# REPORT_WITH: results of the same model merged after this run's (a condition both hold is this run's; a run on a
# layer subset reports its K1 / K2 / kvcomm_sel as K2@16-24 etc., next to the all-layer ones).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL:?}" "${OUT:?}"
PY=${PYTHON:-python}
GPUS=(${GPUS:-$(seq 0 $((${NGPUS:-8} - 1)))})
args=(--model "$MODEL" --probes "${PROBES:-octopus/tools/kv_read_probe_set.jsonl}"
      --conditions "${CONDITIONS:-none,text,summary,replay,K1,K2,K3,kvcomm_sel}" --mount-layers "${MOUNT_LAYERS:-all}")
[ -n "${ADAPTER:-}" ] && args+=(--adapter "$ADAPTER")
for pair in ${ROOT_MAP:-}; do args+=(--root-map "$pair"); done
mkdir -p "$OUT"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" -m baselines.kv_read.route_a run "${args[@]}" --shard "$i/${#GPUS[@]}" \
    --out "$OUT/shard$i.jsonl" > "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
cat "$OUT"/shard*.jsonl > "$OUT/results.jsonl"
"$PY" -m baselines.kv_read.route_a report "$OUT/results.jsonl" ${REPORT_WITH:-} > "$OUT/report.md"
exit $status
