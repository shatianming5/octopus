#!/bin/bash
# lora_test.sh CODE GPU [ADAPTER] (srun in the held job): one policy SGLang (eval_serve.sh: cursor overlay, LoRA rq1-v0) on GPU; greedy 48 tokens
# with rq1-v0, hot-load ADAPTER (default ADAPTER0) as rq1-x, greedy with it, unload it, list loras; stop the server.
[ -n "${LT_CLEAN:-}" ] || exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  SLURM_JOB_ID="$SLURM_JOB_ID" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LT_CLEAN=1 bash "$0" "$@"
C=$1; GPU=$2; export RL_ENV_FILE=$C/scripts/beta/search.env PORT_BASE=32700; source $RL_ENV_FILE; X=${3:-$ADAPTER0}
trap 'bash $C/scripts/rl_env/eval_serve.sh stop' EXIT
S=$(timeout 1800 bash $C/scripts/rl_env/eval_serve.sh start $GPU "rq1-v0=$ADAPTER0" | tail -1); echo "server $S"
[[ $S == 127.0.0.1:* ]] || { tail -30 $STATE/eval-serve-$(hostname -s)/sglang-g$GPU.log; exit 1; }
gen() { curl -s -m 300 http://$S/generate -H 'Content-Type: application/json' -d "{\"text\": \"<|im_start|>user\nName the capital of France and explain briefly.<|im_end|>\n<|im_start|>assistant\n\", \"sampling_params\": {\"temperature\": 0, \"max_new_tokens\": 48}, \"lora_path\": \"$1\", \"return_logprob\": true}" | "$AREAL_PY" -c 'import json,sys; d=json.load(sys.stdin); m=d["meta_info"]; print(repr(d["text"][:120]), [round(x[0],4) for x in m["output_token_logprobs"][:6]], [x[1] for x in m["output_token_logprobs"][:12]])'; }
echo "v0: $(gen rq1-v0)"
echo "load: $(curl -s -m 300 http://$S/load_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\": \"rq1-x\", \"lora_path\": \"$X\"}")"
echo "x:  $(gen rq1-x)"
echo "unload: $(curl -s -m 300 http://$S/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "rq1-x"}')"
echo "models: $(curl -s -m 30 http://$S/v1/models | "$AREAL_PY" -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"
grep -iE "error|traceback" $STATE/eval-serve-$(hostname -s)/sglang-g$GPU.log | head -5
