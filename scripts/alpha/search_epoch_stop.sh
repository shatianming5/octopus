#!/bin/bash
# search_epoch_stop.sh TRIAL DEADLINE [BCP_K=1]: on Alpha, the one guard that ends a Search full trial (search_launch.sh).
# The trial covers its 3040 train questions once (user 10-23; the zero-variance filter draws ~60 questions a step, so 190
# steps would be ~3.8 passes): at the step boundary after every train qid has a finished episode, or after 2 step boundaries
# in a row at which more train qids were re-drawn than newly drawn since the one before (job dirs search-QID-GROUP:*; a qid
# with a second GROUP is re-drawn: the dataloader is in its next pass), search_stop_at.sh stops the run (recover kept), the
# step's adapter is taken (lib.sh final_adapter: the saver's, else made from that step's recover checkpoint), and Val 500 k=1
# then BrowseComp-Plus 830 k=BCP_K (search_eval.sh, same settings as P/run, its own judge on G0) are queued on RJ's hold (TJ
# may end first; RJ = TJ: on TJ): 30-val-TRIAL.sh, 31-bcp-TRIAL.sh (BCP_K=0: none), with EVAL_PARALLEL when the guard's env has it (search_eval.sh: 32).
# Not a count against 3040 or a total re-draw count: a resume skips or repeats prompts the rollout had already pulled
# (v6d: the step-46 stop skipped 41 in a row, 3 came up in the next pass; the step-37 one re-drew 10).
# DEADLINE (epoch seconds, 0 = none): once past it, stop at the next step boundary instead, recover kept, no eval. Exits when P/stopped exists, TJ has ended (sacct) or no longer runs 20-TRIAL.sh. Pairing against v0 is the caller's (paired.py).
T=${1:?}; DEADLINE=${2:?DEADLINE epoch seconds or 0}; BK=${3:-1}
N=$(cd "$(dirname "$0")/../.." && pwd); R=$(dirname "$N"); P=$R/search/pilot/$T; read -r TJ RJ _ < "$P/jobs" || exit 1
M=$R/search/areal-exp/logs/user/rq1-rl/$T/main.log; J=$R/search/rl/jobs-$T; QT=$R/search/pilot/hold-$TJ/queue
QR=$R/search/pilot/hold-$RJ/queue; C=$R/search/analysis-covered-$T.txt; Q=$R/search-data/search/train-qids.txt
source "$N/scripts/alpha/lib.sh"  # owner_running: sacct; an error or no answer counts as running, only a final state does not
live() { [ ! -e "$P/stopped" ] && { [ -e "$QT/20-$T.sh" ] || [ -e "$QT/20-$T.sh.running-$TJ" ]; } && owner_running "$TJ"; }  # a TERM'd TJ leaves .running-TJ
want=$(wc -l < "$Q"); d0=0; r0=0; over=0; steps=0
until [ -f "$M" ]; do live || { echo "$(date +%T) $T stopped or not running: guard ends"; exit 0; }; sleep 60; done
while true; do
  live || { echo "$(date +%T) $T stopped or not running: guard ends"; exit 0; }
  n=$(python3 "$N/scripts/alpha/search_finished_qids.py" "$J" "$C" "$Q" 2>/dev/null); n=${n:-0}
  read -r d r < <(ls "$J" 2>/dev/null | sed -n 's/^search-\([^-]*\)-\([0-9]*\):.*/\1 \2/p' | sort -u |  # train qids drawn, re-drawn
    awk 'NR == FNR { t[$1]; next } $1 in t { g[$1]++ } END { for (q in g) { d++; r += g[q] > 1 } print d + 0, r + 0 }' "$Q" -)
  s=$(grep -c 'Train step' "$M"); s=${s:-0}
  if [ "$s" -gt "$steps" ]; then steps=$s; if [ $((r - r0)) -gt $((d - d0)) ]; then over=$((over + 1)); else over=0; fi; d0=$d; r0=$r
    echo "$(date +%T) $(grep -oE 'Train step [0-9]+' "$M" | tail -1): $n of $want train qids finished, $d drawn, $r re-drawn"; fi
  [ "$n" -ge "$want" ] && break; [ "$over" -ge 2 ] && break
  [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date +%T) deadline: stop after step $((s + 1)), no eval"; exec bash "$N/scripts/alpha/search_stop_at.sh" "$T" $((s + 1)); }
  sleep 120
done
echo "$(date +%T) pass over ($n of $want finished, $d drawn, $r re-drawn): stop after step $((s + 1))"
bash "$N/scripts/alpha/search_stop_at.sh" "$T" $((s + 1)); [ $? = 1 ] && exit 1  # 2 = stopped but NOT CLEAN: eval still queued (RJ's task fails on its own if RJ is not clean)
K=$(grep -c 'Train step' "$M")
A=$(srun --jobid="$RJ" --overlap -n1 -c4 --mem=64G bash -c "C=$N TRIAL=$T; source $N/scripts/alpha/search.env; source $N/scripts/alpha/lib.sh; final_adapter $K") || exit 1
echo "$(date +%T) step-$K adapter $A"
q() {  # q NN SET K: a task of RJ's hold (name order: Val first), clean env as in search_eval.sh's usage
  echo "exec env -i HOME=\$HOME USER=\$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=\$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID ${EVAL_PARALLEL:+EVAL_PARALLEL=$EVAL_PARALLEL} bash $N/scripts/alpha/search_eval.sh $N v$K $A $2 $P/run $3 0 $T" > "$QR/$1-$2-$T.sh"; }
q 30 val 1; [ "$BK" = 0 ] || q 31 bcp "$BK"
echo "$(date +%T) queued $QR/30-val-$T.sh and, unless BCP_K=0, 31-bcp-$T.sh (step $K, Val k=1 + BCP k=$BK); results $R/search/pilot/eval/{val,test}-$T-v$K-fork-k*.jsonl"
