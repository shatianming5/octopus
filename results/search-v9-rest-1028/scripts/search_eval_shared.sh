#!/bin/bash
# search_eval_shared.sh NAME ADAPTER SET [search_dev_eval args...]: one Search eval (k=1, fork arm) of one adapter on node-138
# (no GPU of its own), against the 4B rollout servers in SERVERS and the judge at JUDGE_URL, both required: the live
# search-v9-rest run's servers (shared with training) or the eval's own (search_eval_queue.sh). The lab-cluster counterpart
# of scripts/alpha/search_eval.sh, with the same protocol.
#   SERVERS: comma-separated host:port as node-138 reaches them (e.g. ssh-tunnel ends 127.0.0.1:32410,127.0.0.1:32411);
#     JUDGE_URL: the Qwen3.5-27B judge's base URL, served as "judge" (e.g. http://127.0.0.1:32500/v1).
#   NAME: v<N>; the servers serve ADAPTER as rq1-NAME (proxy.py parses rsplit("-v") + int). ADAPTER: the adapter dir, at the
#     same path on every server host (the servers load it there). Training loads rq1-vV and unloads rq1-v(V-40)
#     (rl.lora_keep_versions=40; AReaL sglang_remote.build_disk_weight_update_requests): from v42 to its last version (61 at
#     most) it unloads v2-v21 only, so v0 and v41 stay. rq1-NAME stays loaded after the eval: an unload on a live server can
#     hang (patches/apply_sglang_lora_abort_release.py), and it takes 1 of the 44 slots.
#   SET bcp: the 830 BrowseComp-Plus queries, decrypted in memory only (trials in search_dev_eval's mode-700 /dev/shm dir,
#     removed at exit), against a BM25 service of the browsecomp-plus corpus on 32601. Only summary and route-peak lines
#     persist: OUT/test-NAME-fork-k1.log; the full eval and service logs stay in SHM.
#   SET val: the 500 OpenResearcher validation questions against a BM25 service of the openresearcher corpus on 32600
#     (training's index); log, rows and trials (jobs-val-NAME) in OUT.
#   Search service: the one already answering on that port for that corpus (training's, for val), else one started here
#     and stopped at exit.
# Settings: RUN (training's log dir) supplies config.yaml and episode_settings.json, the same six settings as Alpha's
# search-v9-1025 Val (CURSOR, READ_TOOL, REWIND_TOOL, REWIND_MAX=2, FORK_DEPTH=0, TOOL_FORMAT=qwen3_coder); the process
# environment carries no episode setting (env -i). Judge: Qwen3.5-27B at temperature 0 (search_dev_eval); policy:
# rl.EVAL_SAMPLING (temperature 0.7, top_p 0.95), replies capped at 4096 tokens. Code: training's code-local and AReaL overlay.
# Load: at most EVAL_PARALLEL x servers eval episodes at once in all (search_dev_eval --parallel; default 1), not a
# per-server bound: this process's router admits a new episode only on a server below EVAL_ROUTE_KV (default 0.4,
# under training's 0.5) holding fewer active trees (training's and the eval's) than max(1, EVAL_ROUTE_KV x pool /
# EVAL_ROUTE_TREE_TOKENS) (default 40000, training's), fewer than training's (none reserved for it), and draws among the open
# servers by free slots; a new episode that has waited OCTOPUS_ROUTE_WAIT (800 s) takes a server under the cap anyway,
# past the KV target and the slots. Training's weight-update pauses abort and resume eval requests like training's own.
# Extra args go to search_dev_eval (a pilot: --limit 10, with EVAL_OUT=its own dir). OUT (EVAL_OUT, default D/eval) holds
# the rows and launch-SET-TAG.log; rerunning the same NAME and SET resumes OUT's rows file.
#   SERVERS=127.0.0.1:32410,127.0.0.1:32411 JUDGE_URL=http://127.0.0.1:32500/v1 setsid nohup bash search_eval_shared.sh v0 \
#     /home/user/octo-env/models/adapter-v0-4b-text-r64 bcp > /dev/null 2>&1 < /dev/null &
set -uo pipefail
NAME=$1; ADAPTER=$2; SET=$3; shift 3
[[ $NAME =~ ^v[0-9]+$ ]] || { echo "NAME must be v<N>, got '$NAME'"; exit 1; }
case $SET in val|bcp) ;; *) echo "SET must be val or bcp, got '$SET'"; exit 1 ;; esac
D=/mnt/ssd4/user/search-v9; C=$D/runtime/code-local; PY=$D/areal-venv/.venv/bin/python
RUN=/home/user/search-v9-run/areal-exp/logs/user/rq1-rl/search-v9-rest
SERVERS=${SERVERS:?comma-separated host:port of the rollout servers}; JUDGE=${JUDGE_URL:?the judge base URL}
OUT=${EVAL_OUT:-$D/eval}; TAG=$NAME-fork-k1; LORA=rq1-$NAME
mkdir -p "$OUT"; SHM=$(mktemp -d /dev/shm/search-eval-XXXXXX)  # mode 700
say() { echo "$(date '+%F %T %Z') $*" >> "$OUT/launch-$SET-$TAG.log"; }
SP=""; BG=""; OLD=" $(echo /dev/shm/bcp-trials-*) "  # tmpfs trial dirs of other evals, left alone
cleanup() {  # the eval first (SIGTERM: search_dev_eval's finally removes its trials), then the service, then SHM
  [ -n "$BG" ] && { kill -TERM "$BG" 2>/dev/null; for _ in $(seq 120); do kill -0 "$BG" 2>/dev/null || break; sleep 1; done; kill -9 "$BG" 2>/dev/null; }
  # a stopped episode can write its requests.jsonl after that removal (or SIGKILL skipped it): remove the trial dir this
  # eval made (one started meanwhile by another BCP eval on this host would go too: run one at a time)
  for t in /dev/shm/bcp-trials-*; do [ -O "$t" ] && [[ $OLD != *" $t "* ]] && rm -rf "$t"; done
  [ -n "$SP" ] && { kill -TERM -- "-$SP" 2>/dev/null; sleep 3; kill -9 -- "-$SP" 2>/dev/null; }
  rm -rf "$SHM"; say "exit"
}
trap cleanup EXIT; trap 'say "TERM"; exit 143' TERM
if [ "$SET" = bcp ]; then
  SPORT=32601; CORPUS=browsecomp-plus; SROOT=/home/user/octo-env
  DS=(--set browsecomp-plus --dataset-dir /home/user/octo-env/datasets/Tevatron__browsecomp-plus)
  LOG=$SHM/eval.log; SUM=$OUT/test-$TAG.log; ROWS=$OUT/test-$TAG.jsonl
else
  SPORT=32600; CORPUS=openresearcher; SROOT=/mnt/ssd4/user/search; LOG=$OUT/val-$TAG.log; SUM=$LOG; ROWS=$OUT/val-$TAG.jsonl
  DS=(--set validation --dataset-dir /mnt/ssd4/user/octo-env/search/hf/OpenResearcher__OpenResearcher-Corpus-Gold-Doc --jobs-dir "$OUT/jobs-val-$TAG")
fi
if ! curl -sf -m 5 "http://127.0.0.1:$SPORT/health" | grep -q "\"corpus\": \"$CORPUS\""; then
  (cd "$C" && exec setsid "$PY" -m octopus.envs.search.service serve --corpus $CORPUS --root $SROOT \
    --backend bm25 --tokenizer "$D/runtime/Qwen3-0.6B-tokenizer" --port $SPORT --workers 4) > "$SHM/service.log" 2>&1 < /dev/null & SP=$!
  for _ in $(seq 120); do curl -sf -m 5 "http://127.0.0.1:$SPORT/health" > /dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -sf -m 5 "http://127.0.0.1:$SPORT/health" > /dev/null || { say "$CORPUS service down"; exit 1; }
fi
# proxy.episode (code-local) ends each tree itself; its session start goes through the overlay's /rl/start_session
grep -qF 'route.start(session_id)' $D/areal-overlay/areal/experimental/openai/proxy/proxy_rollout_server.py \
  || { say "$D/areal-overlay: no route start hook (OCTOPUS_ROUTE_TREE_TOKENS): rebuild it with patches/areal-route-spread.patch (#351)"; exit 1; }
for s in ${SERVERS//,/ }; do  # load the adapter (400 when the name is already loaded), then check the name holds ADAPTER
  curl -s -m 300 -X POST "http://$s/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$LORA\", \"lora_path\": \"$ADAPTER\"}" > /dev/null
  got=$(curl -sf -m 10 "http://$s/v1/models" | "$PY" -c "import sys, json  # get_server_info lists only the startup adapters
print(*[m['root'] for m in json.load(sys.stdin)['data'] if m['id'] == '$LORA'])")
  [ "$got" = "$ADAPTER" ] || { say "$s serves $LORA as '${got:-nothing}', not $ADAPTER"; exit 1; }
done
say "$SET $TAG ($ADAPTER) on $SERVERS: search $SPORT, parallel ${EVAL_PARALLEL:-1} x servers, route kv ${EVAL_ROUTE_KV:-0.4} tree tokens ${EVAL_ROUTE_TREE_TOKENS:-40000}"
(cd "$C" && exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin LANG=${LANG:-C.UTF-8} CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=$D/areal-overlay:$C PYTHONUNBUFFERED=1 AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  OCTOPUS_SGLANG_PY=$D/sglang-venv/bin/python OCTOPUS_SEARCH_URL=http://127.0.0.1:$SPORT OCTOPUS_JUDGE_TIMEOUT=120 \
  OCTOPUS_ROUTE_CAP=16 OCTOPUS_ROUTE_KV=${EVAL_ROUTE_KV:-0.4} OCTOPUS_ROUTE_WAIT=800 OCTOPUS_ROUTE_TREE_TOKENS=${EVAL_ROUTE_TREE_TOKENS:-40000} \
  "$PY" -m octopus.eval.search_dev_eval "${DS[@]}" --run "$RUN" --servers "$SERVERS" --lora "$LORA" \
  --tokenizer "$D/models/Qwen3.5-4B-text" --judge-url "$JUDGE" --judge-model judge --k 1 --parallel "${EVAL_PARALLEL:-1}" \
  --out "$ROWS" "$@") > "$LOG" 2>&1 < /dev/null & BG=$!  # exec: BG is the eval itself, which TERM reaches
wait $BG; rc=$?; BG=""
[ "$SET" = bcp ] && grep -aE '^(browsecomp-plus: |n=[0-9]+ accuracy|infra failures)|route: peak' "$LOG" > "$SUM"  # summary lines only
say "$SET $TAG exit $rc: $(grep -a '^n=' "$SUM" | tail -n 1)"; exit $rc
