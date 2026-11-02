#!/usr/bin/env bash
# Re-copy bundle-RL experiment results into results/bundle-rl/<run>/ and regenerate their summaries (sources read-only).
# usage: scripts/rl_analysis/sync_results.sh [RUN ...]    RUN = base-a941eea | e1-82b | ... | e9 | e10 (default: all)
#   eval/  <- node-67 evals/{bundle-<run>,<run>}: results/ (step<N>.jsonl only with its step<N>.done; no *.lockdir,
#             *.claimed, *.failed-*), steps.list, run.env, logs/{driver,pool}.log; summary.txt = scripts/eval/summary.py
#             against base-a941eea step0 (every run, including those evaluated under another runtime or eval set)
#   train/ <- node-94 octopus-sft: rl_run.bundle-<run>[-*].env, rl-state-bundle-<run>/{incidents.log,mem.log},
#             <run>/*-replay.txt; steps.cache.jsonl + steps.txt = scripts/rl_analysis/steps.py over rl/jobs-bundle-<run>,
#             run on node-94 against a scratch copy of the run's live <run>_steps.cache.jsonl in /tmp/bundle-rl-sync-<run>
set -euo pipefail
EVALS=/home/user/harbor-udocker/evals
SFT=/home/user/octopus-sft
JOBS=/home/user/harbor-udocker/rl
CODE=/home/user/harbor-udocker/code-f28550d
PY=/mnt/disk1/user/areal/.venv/bin/python
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$REPO/results/bundle-rl

runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
  mapfile -t runs < <({ ssh node-67 "ls $EVALS" | grep -E '^(base-|bundle-)' | sed 's/^bundle-//'
    ssh node-94 "cd $SFT && ls -d rl-state-bundle-e* e[0-9]*" | sed 's/^rl-state-bundle-//'; } | sort -uV)
fi

for run in "${runs[@]}"; do
  dst=$OUT/$run
  edir=$(ssh node-67 "cd $EVALS && ls -d bundle-$run $run 2>/dev/null | head -1" || true)
  if [ -n "$edir" ]; then
    mkdir -p "$dst/eval"
    ssh node-67 bash -s -- "$EVALS/$edir" <<'EOF' | tar -xf - -C "$dst/eval"
cd "$1"
{ for f in steps.list run.env logs/driver.log logs/pool.log; do [ -e "$f" ] && echo "$f"; done
  for f in results/*; do
    case $f in *.lockdir|*.claimed|*.failed-*) continue;; esac
    [[ $f =~ ^results/step[0-9]+\.jsonl$ && ! -e ${f%.jsonl}.done ]] && continue
    echo "$f"
  done; } | tar -cf - -T -
EOF
    [ "$run" = base-a941eea ] || python3 "$REPO/scripts/eval/summary.py" "$dst/eval" \
      --base "$OUT/base-a941eea/eval/results/step0.jsonl" > "$dst/eval/summary.txt"
  fi
  S=/tmp/bundle-rl-sync-$run
  ssh node-94 "mkdir -p $S/train && cat > $S/steps.py" < "$REPO/scripts/rl_analysis/steps.py"
  mkdir -p "$dst/train"
  ssh node-94 bash -s -- "$run" "$S" "$SFT" "$JOBS/jobs-bundle-$run" "$CODE" "$PY" <<'EOF' | tar -xf - -C "$dst/train"
run=$1 S=$2 st=$3/rl-state-bundle-$1 jobs=$4
cd "$3"
for f in rl_run.bundle-$run.env rl_run.bundle-$run-*.env $st/incidents.log $st/mem.log $run/*-replay.txt; do
  [ -e "$f" ] && cp "$f" "$S/train/"
done
[ -e "$st/${run}_steps.cache.jsonl" ] && cp "$st/${run}_steps.cache.jsonl" "$S/train/steps.cache.jsonl"
if [ -d "$jobs" ] && [ -e "$st/train-bundle-$run.log" ]; then
  PYTHONPATH=$5 "$6" "$S/steps.py" "$jobs" "$st/train-bundle-$run.log" "$S/train/steps.cache.jsonl" > "$S/train/steps.txt" || exit 1
fi
tar -C "$S/train" -cf - .
EOF
  rmdir "$dst/train" 2>/dev/null || true
  echo "synced $run: $(du -sh "$dst" | cut -f1)"
done
