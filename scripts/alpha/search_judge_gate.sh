#!/bin/bash
# search_judge_gate.sh TRIAL: on Alpha right after search_launch.sh queued the trainer task. Waits (up to 60 min) for the
# controller's "judge up at URL" (or "judge down"), then, from TJ, checks that URL's judge (search.sbatch's own, or with
# JUDGE_HOST the one in RJ, lib.sh serve_rollout): /model_info model_path ends with Qwen3.5-27B-text, sends one real chat
# request (T=0, enable_thinking false) and requires HTTP 200 + a reply, and finds a 200 POST /v1/chat/completions in the judge's
# log (polled for 60 s: SGLang writes its access lines late; JUDGE_LOG=FILE for a judge outside TJ and RJ, e.g. on Beta). Exit 1 = stop the trial (search_stop_at.sh TRIAL 0).
T=${1:?}; N=$(cd "$(dirname "$0")/../.." && pwd); R=$(dirname "$N"); P=$R/search/pilot/$T; read -r TJ RJ _ < "$P/jobs" || exit 1
Q=$R/search/pilot/hold-$TJ/queue
t=0; until L=$(ls -t "$Q"/*-"$T".sh.log 2>/dev/null | head -1); [ -n "$L" ] && grep -q "judge up\|judge down\|search service down" "$L" || [ $t -ge 360 ]; do sleep 10; t=$((t+1)); done
grep -q "judge up" "$L" 2>/dev/null || { echo "JUDGE GATE FAILED: $(tail -n 3 "$L" 2>/dev/null | cut -c1-200)"; exit 1; }
U=$(grep -o 'judge up at [^ ]*' "$L" | tail -1 | cut -d' ' -f4); U=${U:-http://127.0.0.1:32500/v1}; B=${U%/v1}  # an older controller logs only "judge up"
case $B in http://127.0.0.1:*) JL=$P/judge-$TJ.log ;; *) JL=$P/judge-$RJ.log ;; esac; JL=${JUDGE_LOG:-$JL}  # JUDGE_LOG: a judge outside TJ and RJ
out=$(B=$B srun --jobid="$TJ" --overlap -n1 -c1 bash -c 'curl -s -m 10 $B/model_info | python3 -c "import json,sys; print(json.load(sys.stdin)[\"model_path\"])"; curl -s -m 120 -o /tmp/gate.json -w "HTTP %{http_code}\n" $B/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"judge\",\"temperature\":0,\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: correct: yes\"}]}"; head -c 300 /tmp/gate.json; rm -f /tmp/gate.json' < /dev/null)
echo "$out"
for _ in $(seq 12); do grep -q 'POST /v1/chat/completions.* 200' "$JL" && break; sleep 5; done
echo "$out" | sed -n 1p | grep -q 'Qwen3.5-27B-text$' && echo "$out" | grep -q '^HTTP 200' && grep -q 'POST /v1/chat/completions.* 200' "$JL" \
  && echo "JUDGE GATE OK: $(grep -m1 'POST /v1/chat/completions.* 200' "$JL" | cut -c1-120)" || { echo "JUDGE GATE FAILED"; tail -n 15 "$JL"; exit 1; }
