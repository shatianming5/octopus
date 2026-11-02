#!/bin/bash
# The SWE coupled environment (training and Gate 0 evaluation) on one Linux x86_64 GPU host, from this repo and the HF
# dataset Tommysha/octopus-train. Everything goes under ROOT (an absolute path on a big local disk):
#   deploy.sh ROOT fetch [--search]   code (this checkout's HEAD, results/ left out) -> ROOT/code-SHA; HF branch `code`
#                                     (task dirs + env-bundle) -> ROOT/harbor/datasets; --search: branch `search` -> ROOT/search-data
#   deploy.sh ROOT install            scripts/nvwulf/install_env.sh (uv, AReaL, SGLang, Harbor venvs, Qwen3.5-4B, zero adapter,
#                                     the cursor overlay ROOT/cursor-SHA), udocker + fuse-overlayfs + ROOT/harbor/scripts/env.sh
#   deploy.sh ROOT goldens DATASET [N]   pull the images and build the goldens of the first N (default all) task dirs of
#                                     ROOT/harbor/datasets/DATASET (JOBS images at once, default 4)
#   deploy.sh ROOT check [--gpu G]    acceptance (last line "check passed"; after fetch --search also the BM25 search service on
#                                     browsecomp-plus); --gpu: one SGLang server on GPU G (base model) and one
#                                     dev_eval episode (k=1), the server is stopped at the end whatever happens
# Needs git, curl, patch, gcc, fusermount3 + /dev/fuse; no root, no Docker. Run fetch from a git checkout, the later steps from it
# too or from ROOT/code-SHA. Blocked networks: HF_ENDPOINT=https://hf-mirror.com (huggingface.co),
# DOCKER_REGISTRY=https://docker.1ms.run (Docker Hub). Afterwards `source ROOT/harbor/scripts/env.sh`
# (OCTO_ENV=ROOT, OCTOPUS_HARBOR) before RL_ENV_FILE=ROOT/code-SHA/scripts/rl_env/rl_run.coupled.env rl_run.sh / gate0_shard.sh.
set -uo pipefail
ROOT=$(realpath -m "${1:?usage: deploy.sh ROOT fetch|install|goldens|check}"); CMD=${2:-}
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if git -C "$C" rev-parse --git-dir > /dev/null 2>&1; then CODE=$ROOT/code-$(git -C "$C" rev-parse --short=7 HEAD); else CODE=$C; fi
H=$ROOT/harbor; B=$H/datasets/env-bundle; E=$CODE/scripts/rl_env/rl_run.coupled.env; CURSOR=$ROOT/cursor-${CODE##*/code-}
export OCTO_ENV=$ROOT OCTOPUS_HARBOR=$H UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/python HF_HOME=$ROOT/cache/hf
UV=$ROOT/bin/uv; HPY=$ROOT/harbor-venv/bin/python
step() { echo "=== $(date +%T) $*"; }
hf() {  # hf REVISION DIR: one branch of the dataset, by git (through a mirror's per-file limits: 29k files on `code`), then its
  # LFS files by URL (sha256-checked, resumable; not git-lfs: it fails on the mirror's CDN)
  local ep=${HF_ENDPOINT:-https://huggingface.co} f oid
  [ -d "$2/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 -b "$1" --single-branch "$ep/datasets/Tommysha/octopus-train" "$2" || return 1
  find "$2" -path "$2/.git" -prune -o -type f -size -1024c -print0 | { xargs -0 -r grep -l '^version https://git-lfs' || true; } | while read -r f; do
    oid=$(sed -n 's/^oid sha256://p' "$f")
    curl -fsSL -C - --retry 10 -o "$f.part" "$ep/datasets/Tommysha/octopus-train/resolve/$1/${f#"$2"/}" \
      && echo "$oid  $f.part" | sha256sum -c --quiet && mv "$f.part" "$f" || { echo "LFS file $f failed"; return 1; }
  done
}

case "$CMD" in
fetch)
  step code "$CODE"
  if [ "$CODE" != "$C" ] && [ ! -d "$CODE/octopus" ]; then
    mkdir -p "$CODE.tmp" && git -C "$C" archive HEAD -- . ':!results' | tar x -C "$CODE.tmp" && mv "$CODE.tmp" "$CODE" || exit 1
  fi
  step "HF code branch -> $H/datasets"
  hf code "$H/datasets" || exit 1
  if [ "${3:-}" = --search ]; then step "HF search branch -> $ROOT/search-data"; hf search "$ROOT/search-data" || exit 1; fi
  du -sh "$H/datasets" ${3:+"$ROOT/search-data"}; step fetch done
  ;;
install)
  [ -f "$B/areal-commit" ] || { echo "no $B: run deploy.sh $ROOT fetch first"; exit 1; }
  CURSOR=$CURSOR bash "$CODE/scripts/nvwulf/install_env.sh" "$ROOT" "$B" "$CODE" || exit 1  # with the cursor overlay
  step udocker
  "$UV" pip install -q --python "$HPY" udocker==1.3.17 || exit 1
  mkdir -p "$H"/{bin,scripts,golden,containers,ipc}
  # static fuse-overlayfs (the lab hosts' binary): trial containers are overlays over their golden (UDOCKER_OVERLAY)
  if ! echo "4817a8896a9e6f0433080f88f5b71dec931e8829a89d64c71af94b0630ccb4a9  $ROOT/bin/fuse-overlayfs" | sha256sum -c --quiet > /dev/null 2>&1; then
    curl -fsSL -m 600 -o "$ROOT/bin/fuse-overlayfs" https://github.com/containers/fuse-overlayfs/releases/download/v1.14/fuse-overlayfs-x86_64 \
      && echo "4817a8896a9e6f0433080f88f5b71dec931e8829a89d64c71af94b0630ccb4a9  $ROOT/bin/fuse-overlayfs" | sha256sum -c --quiet \
      && chmod +x "$ROOT/bin/fuse-overlayfs" || exit 1
  fi
  ln -sfn "$ROOT/harbor-venv/bin/harbor" "$H/bin/harbor"
  cat > "$H/bin/udocker" <<EOF
#!$HPY
import os
os.environ.setdefault("UDOCKER_DIR", "$H/udocker")
from udocker import docker
from udocker.maincmd import main
# udocker 1.3.17 cuts WWW-Authenticate at its last space, so a registry answering 'Bearer realm="...", service="..."'
# (docker.1ms.run) never gets a token
_auth = docker.DockerIoAPI._get_v2_auth
docker.DockerIoAPI._get_v2_auth = lambda self, www, retry: _auth(self, www.replace(", ", ","), retry)
main()
EOF
  chmod +x "$H/bin/udocker"
  cat > "$H/scripts/env.sh" <<EOF
# Harbor/udocker settings of this deployment (deploy.sh; the lab hosts' /home/user/harbor-udocker/scripts/env.sh).
# Every Harbor episode sources it (octopus/envs/swe/rl.py); source it before rl_run.sh, gate0_shard.sh and dev_eval.
export N=$H OCTO_ENV=$ROOT OCTOPUS_HARBOR=$H
export PATH="\$N/bin:\$PATH"
export UDOCKER_DIR="\$N/udocker" UDOCKER_GOLDEN_DIR="\$N/golden" UDOCKER_CONTAINERS="\$N/containers" UDOCKER_IPC_INDEX="\$N/ipc"
export UDOCKER_CREATE_SLOTS=4 UDOCKER_THREADS=256 HARBOR_SETUP_TIMEOUT=180 HARBOR_EXEC_TIMEOUT=180
export UDOCKER_OVERLAY=$ROOT/bin/fuse-overlayfs  # trial containers as an overlay over the golden
EOF
  # PRoot engines into UDOCKER_DIR, from a sha256-checked local copy of udocker's tarball (config.py: UDOCKER_TARBALL)
  U=$ROOT/cache/udocker-englib-1.2.11.tar.gz
  echo "2a4804ba82e087ca3e99305fce9887227925d2b56aaefcad6474c4cb86b7a157  $U" | sha256sum -c --quiet > /dev/null 2>&1 \
    || curl -fsSL -m 600 -o "$U" https://download.a.incd.pt/udocker/udocker-englib-1.2.11.tar.gz || exit 1
  echo "2a4804ba82e087ca3e99305fce9887227925d2b56aaefcad6474c4cb86b7a157  $U" | sha256sum -c --quiet \
    && UDOCKER_TARBALL=$U "$H/bin/udocker" install > /dev/null || exit 1
  du -sh "$ROOT" | cut -f1; step install done
  ;;
goldens)
  DS=${3:?usage: deploy.sh ROOT goldens DATASET [N]}; T=$ROOT/check/goldens-$DS
  [ -d "$H/datasets/$DS" ] && source "$H/scripts/env.sh" || exit 1
  rm -rf "$T"; mkdir -p "$T"
  find "$H/datasets/$DS" -mindepth 1 -maxdepth 1 -type d | sort | head -n "${4:--0}" | xargs -I{} ln -s {} "$T/"
  pull() {  # task image $1: on Docker Hub a local transport tag swebench/swesmith.x86_64.<repo>:transport-* is jyangballin/...:latest
    local hub; hub=$(sed -E 's#^swebench/(swesmith\.x86_64\.[^:]+):transport-.*#jyangballin/\1:latest#' <<< "$1")
    case $1 in akhatua/*) echo "skip $1: CooperBench images come from octopus/envs/swe/data/cooperbench.py fetch"; return;; esac
    udocker images | grep -qF "$1" && return
    udocker pull ${DOCKER_REGISTRY:+--registry=$DOCKER_REGISTRY} "$hub" > /dev/null && { [ "$hub" = "$1" ] || udocker tag "$hub" "$1"; } \
      && echo "pulled $1" || echo "FAIL pull $hub"
  }
  export -f pull
  step "images of $(ls "$T" | wc -l) task dirs of $DS"
  grep -h '^docker_image' "$T"/*/task.toml | sed -E 's/.*"(.*)"/\1/' | sort -u | xargs -P "${JOBS:-4}" -I{} bash -c 'pull "$@"' _ {} | tee "$T.log"
  step goldens
  (cd "$CODE" && PYTHONPATH=$CODE nice "$HPY" -m octopus.envs.swe.data.prebuild_goldens "$T" "${JOBS:-4}" || echo "FAIL prebuild_goldens exit $?") | grep -v '^ok\|INFO' | tee -a "$T.log"
  du -sh "$H/udocker" "$H/golden"; ! grep -q '^FAIL' "$T.log" || { echo "goldens FAILED: $T.log"; exit 1; }; step goldens done
  ;;
check)
  O=$ROOT/check; mkdir -p "$O"; bad=0
  ok() { echo "ok   $*"; }; no() { echo "FAIL $*"; bad=$((bad + 1)); }
  source "$E"; source "$H/scripts/env.sh"
  for t in gcc fusermount3 curl patch; do command -v $t > /dev/null && ok "$t" || no "$t not on PATH (gcc: the trial kill guard; fusermount3: overlay trials)"; done
  [ -r /dev/fuse ] && [ -w /dev/fuse ] && ok /dev/fuse || no "/dev/fuse not usable: overlay trials need FUSE"
  if unshare --user --map-current-user true 2> /dev/null; then ok "unprivileged user namespaces"; else
    USERNS="unprivileged user namespaces are blocked (kernel.apparmor_restrict_unprivileged_userns=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2> /dev/null), Ubuntu 24.04's default); udocker's PRoot mode does not need them, but if the oracle trial below fails, ask an admin for /etc/sysctl.d/99-userns.conf with kernel.apparmor_restrict_unprivileged_userns = 0; this script changes no system setting"
    echo "WARN $USERNS"; fi
  # versions, patches, the cursor overlay (check_env.sh), udocker
  out=$(bash "$CODE/scripts/check_env.sh" "$ROOT" "$CODE" "$CURSOR" "$AREAL_PY" "$SGLANG_PY" "$HPY:$B/harbor-freeze.txt")
  echo "$out"; bad=$((bad + $(grep -c '^FAIL' <<< "$out")))
  "$HPY" -c 'import importlib.metadata as m; assert m.version("udocker") == "1.3.17"' 2> /dev/null && ok "udocker 1.3.17" || no "udocker 1.3.17 not in $HPY"
  # models
  # Qwen/Qwen3.5-4B at revision 851bf6e, as the lab hosts serve it; the env bundle's model-sha256.txt holds the same sums
  # since 10-22 (before, it listed Qwen3.5-9B's 4 shards)
  (cd "$MODEL_VL" && sha256sum -c --quiet) > /dev/null 2>&1 <<'SUMS' && ok "sha256 $MODEL_VL" || no "sha256 of $MODEL_VL differs from Qwen3.5-4B's"
26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61  model.safetensors-00001-of-00002.safetensors
cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188  model.safetensors-00002-of-00002.safetensors
ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670  config.json
5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42  tokenizer.json
SUMS
  [ -f "$MODEL_TEXT/config.json" ] && [ -f "$ADAPTER0/adapter_config.json" ] && ok "$MODEL_TEXT, $ADAPTER0" || no "missing $MODEL_TEXT or $ADAPTER0"
  # data: each manifest = its dataset's task dirs (names and images); goldens per dataset; the oracle task
  out=$("$HPY" - "$H/datasets" "$H/golden" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
data, golden = Path(sys.argv[1]), Path(sys.argv[2])
def image(d):
    return re.search(r'^docker_image\s*=\s*"([^"]+)"', (d / "task.toml").read_text(), re.M).group(1)
def built(img):
    rec = golden / ".golden" / f"{hashlib.sha1(img.encode()).hexdigest()[:16]}.cid"
    return rec.is_file() and (golden / rec.read_text().split()[0] / "ROOT").is_dir()
oracle = ""
for man in sorted(data.glob("*.manifest.jsonl")):
    ds = data / man.name.removesuffix(".manifest.jsonl")
    want = {r["bundle"]: r["image"] for r in map(json.loads, man.read_text().splitlines())}
    have = {d.name: image(d) for d in ds.iterdir() if (d / "task.toml").is_file()} if ds.is_dir() else {}
    diff = sorted(set(want) ^ set(have)) + sorted(b for b in set(want) & set(have) if want[b] != have[b])
    print(f"FAIL {man.name}: {len(diff)} task dirs differ from {ds}, e.g. {diff[:3]}" if diff else
          f"ok   {man.name} = {len(have)} task dirs")
    imgs = sorted(set(have.values()))
    print(f"info goldens {ds.name}: {sum(map(built, imgs))}/{len(imgs)} images")
    if ds.name == "coupled-a-val":
        oracle = next((str(ds / b) for b in sorted(have) if built(have[b])), "")
print(f"task {oracle}")
PY
)
  grep -v '^task ' <<< "$out"; bad=$((bad + $(grep -c '^FAIL' <<< "$out"))); TASK=$(sed -n 's/^task //p' <<< "$out")
  [ -n "$TASK" ] && ok "a coupled-a-val task with its golden: $TASK" || no "no coupled-a-val golden: deploy.sh $ROOT goldens coupled-a-val 1"
  # settings for dev_eval / Gate 0 (rl_run.sh settings, the env file's EXTRA, the base model)
  RL_ENV_FILE=$E "$CODE/scripts/rl_env/rl_run.sh" settings "$O/run" $EXTRA rollout.use_lora=false > "$O/settings.log" 2>&1 \
    && [ -f "$O/run/episode_settings.json" ] && ok "settings $O/run" || no "rl_run.sh settings: $O/settings.log"
  # one oracle Harbor trial: the reference fixes in a udocker container must score 1
  if [ -n "$TASK" ]; then
    J=oracle-$(date +%m%d%H%M%S)
    (export PYTHONPATH=$CODE; cd "$H" && nice -n 19 harbor run -p "$TASK" -a oracle -e octopus.envs.swe.udocker:UdockerEnvironment -n 1 \
      -o "$O/jobs" --job-name "$J" --environment-build-timeout-multiplier 6 -y) > "$O/$J.log" 2>&1
    r=$("$HPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["verifier_result"]["rewards"]["reward"])' "$O/jobs/$J"/*/result.json 2>/dev/null)
    [ "$r" = 1.0 ] && ok "oracle trial reward 1.0 ($O/jobs/$J)" || no "oracle trial reward '${r}' ($O/$J.log, $O/jobs/$J)${USERNS:+; $USERNS}"
  fi
  if [ "${3:-}" = --gpu ] && [ -n "$TASK" ]; then  # one SGLang server (base model) + one dev_eval episode
    G=${4:?--gpu G}; export PORT_BASE=${PORT_BASE:-30390}
    trap 'RL_ENV_FILE=$E "$CODE/scripts/rl_env/eval_serve.sh" stop > /dev/null; echo "info GPU $G after stop: $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader)"' EXIT
    if S=$(RL_ENV_FILE=$E timeout 1800 "$CODE/scripts/rl_env/eval_serve.sh" start "$G" | tail -1) && [[ $S == *:* ]]; then
      ok "SGLang $S on GPU $G"
      rm -rf "$O/tasks" "$O/dev_eval.jsonl"; mkdir -p "$O/tasks"; ln -s "$TASK" "$O/tasks/"
      (export PYTHONPATH=$CURSOR/areal:$CODE LITELLM_LOCAL_MODEL_COST_MAP=True; cd "$CODE" && nice -n 10 "$AREAL_PY" -m octopus.eval.dev_eval \
        --run "$O/run" --tasks-dir "$O/tasks" --k 1 --parallel 1 --servers "$S" --env OCTOPUS_AUTO_FORK=1 --out "$O/dev_eval.jsonl" \
        --jobs-dir "$O/jobs-eval") > "$O/dev_eval.log" 2>&1
      row=$(tail -1 "$O/dev_eval.jsonl" 2>/dev/null)
      [ -n "$row" ] && ! grep -q '"error"' <<< "$row" && ok "dev_eval episode: $(cut -c1-200 <<< "$row")" || no "dev_eval episode: ${row:-no row} ($O/dev_eval.log)"
    else no "eval_serve.sh start $G: $S ($STATE/eval-serve-$(hostname -s))"; fi
  fi
  if [ -f "$ROOT/search-data/search-scratch/browsecomp-plus/docs.arrow" ]; then  # fetch --search: the BM25 service answers
    P=${SEARCH_PORT:-8199}
    (cd "$CODE" && PYTHONPATH=$CODE exec nice "$AREAL_PY" -m octopus.envs.search.service serve --corpus browsecomp-plus \
      --root "$ROOT/search-data" --backend bm25 --port "$P") > "$O/search.log" 2>&1 & sp=$!
    h=; for _ in $(seq 120); do h=$(curl -sf -m 5 "http://127.0.0.1:$P/health") && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
    r=$(curl -sf -m 60 -X POST "http://127.0.0.1:$P/search" -H 'Content-Type: application/json' -d '{"query": "photoelectric effect"}')
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    [[ $h == *'"n_docs": 100195'* && $r == *'"docid"'* ]] && ok "search service (browsecomp-plus bm25): $h" || no "search service ($O/search.log)"
  fi
  [ "$bad" = 0 ] && echo "check passed" || { echo "check FAILED ($bad)"; exit 1; }
  ;;
*) sed -n 2,16p "$0"; exit 1 ;;
esac
